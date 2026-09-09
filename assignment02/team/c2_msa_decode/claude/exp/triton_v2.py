"""Improved-Triton variant of the upstream split-K decode kernel (alternative to the CUDA kernel):
  (1) FUSE_MERGE: the last-arriving CTA of each (token, kv_head) merges the partials in-kernel
      (atomic counter, acq_rel) -> no second kernel launch, counter self-resets for CUDA-graph replay.
  (2) num_warps = 8 instead of 4 (more warps per scheduler to hide the ldmatrix/mma/exp chain latency).
Everything else (layout, chunking, exp2 softmax, masks) is the upstream kernel verbatim."""
import torch
import triton
import triton.language as tl
from common import sa, default_chunks


@triton.jit
def _decode_fused_kernel(
    q_ptr, kv_cache_ptr, t_ptr, o_ptr, lse_ptr, out_ptr, cnt_ptr, block_table_ptr, seq_lens,
    total_q, gqa_group_size, head_dim, max_topk, sm_scale, decode_query_len,
    stride_qn, stride_qh, stride_qd,
    stride_kv_blk, stride_kv_h, stride_kv_pos, stride_kv_d,
    stride_th, stride_tn, stride_tk,
    stride_o_c, stride_o_b, stride_o_h, stride_o_d,
    stride_l_c, stride_l_b, stride_l_h,
    stride_out_n, stride_out_h, stride_out_d,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr, NUM_TOPK_CHUNKS: tl.constexpr, BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr, FUSE_MERGE: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_bc, pid_kh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bc % total_q
    pid_c = pid_bc // total_q
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len
    pid_h = pid_kh * gqa_group_size
    chunk_size_topk = (max_topk + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS
    chunk_start_topk = pid_c * chunk_size_topk
    chunk_end_compiletime = chunk_start_topk + chunk_size_topk

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    kv_len = tl.maximum(query_pos + 1, 0)
    idx_base = t_ptr + pid_kh * stride_th + pid_b * stride_tn
    num_blocks = (kv_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    real_topk = tl.minimum(max_topk, num_blocks)
    chunk_end_topk = tl.minimum(chunk_end_compiletime, real_topk)

    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    bt_row = block_table_ptr + req_id * stride_bt_b

    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_D), dtype=tl.float32)
    off_h = tl.arange(0, BLOCK_SIZE_H)
    h_mask = off_h < gqa_group_size
    q = tl.load(q_ptr + pid_b * stride_qn + (pid_h + off_h[:, None]) * stride_qh + off_d[None, :] * stride_qd,
                mask=h_mask[:, None] & d_mask[None, :], other=0.0)

    cur_idx_ptr = idx_base + chunk_start_topk * stride_tk
    for _ in tl.range(chunk_start_topk, chunk_end_topk):
        blk = tl.load(cur_idx_ptr).to(tl.int32)
        cur_idx_ptr = cur_idx_ptr + stride_tk
        c = blk * BLOCK_SIZE_K
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = c + off_n
        pos_mask = pos < kv_len
        k = tl.load(kv_cache_ptr + page * stride_kv_blk + pid_kh * stride_kv_h + off_n[None, :] * stride_kv_pos
                    + off_d[:, None] * stride_kv_d, mask=d_mask[:, None] & pos_mask[None, :], other=0.0)
        qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.where(pos_mask[None, :], 0, float("-inf"))
        qk += tl.dot(q, k) * sm_scale_log2e
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None]
        v = tl.load(kv_cache_ptr + page * stride_kv_blk + pid_kh * stride_kv_h + off_n[:, None] * stride_kv_pos
                    + (head_dim + off_d[None, :]) * stride_kv_d, mask=pos_mask[:, None] & d_mask[None, :], other=0.0)
        acc_o += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)

    scale = tl.where(lse_i > float("-inf"), tl.exp2(m_i - lse_i), tl.zeros_like(lse_i))
    acc_o = acc_o * scale[:, None]
    o_ptrs = o_ptr + pid_c * stride_o_c + pid_b * stride_o_b + (pid_h + off_h[:, None]) * stride_o_h + off_d[None, :] * stride_o_d
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), mask=h_mask[:, None] & d_mask[None, :])
    lse_ptrs = lse_ptr + pid_c * stride_l_c + pid_b * stride_l_b + (pid_h + off_h) * stride_l_h
    tl.store(lse_ptrs, lse_i, mask=h_mask)

    if FUSE_MERGE:
        # release our partial, count arrivals; the last CTA of this (token, kv_head) merges all chunks
        cnt = cnt_ptr + pid_b * tl.num_programs(1) + pid_kh
        arrived = tl.atomic_add(cnt, 1, sem="acq_rel", scope="gpu")
        if arrived == NUM_TOPK_CHUNKS - 1:
            tl.atomic_xchg(cnt, 0, sem="relaxed", scope="gpu")  # self-reset for the next call / graph replay
            off_c = tl.arange(0, NUM_TOPK_CHUNKS)
            lse_all = tl.load(lse_ptr + off_c[:, None] * stride_l_c + pid_b * stride_l_b + (pid_h + off_h[None, :]) * stride_l_h,
                              mask=h_mask[None, :], other=float("-inf"), volatile=True)  # [C, H]
            lse_max = tl.max(lse_all, axis=0)
            w = tl.exp2(lse_all - lse_max[None, :])
            w = w / tl.sum(w, axis=0)[None, :]
            o_all = tl.load(o_ptr + off_c[:, None, None] * stride_o_c + pid_b * stride_o_b + (pid_h + off_h[None, :, None]) * stride_o_h
                            + off_d[None, None, :] * stride_o_d, mask=h_mask[None, :, None] & d_mask[None, None, :], other=0.0,
                            volatile=True)  # [C, H, D]
            o_m = tl.sum(o_all.to(tl.float32) * w[:, :, None], axis=0)
            tl.store(out_ptr + pid_b * stride_out_n + (pid_h + off_h[:, None]) * stride_out_h + off_d[None, :] * stride_out_d,
                     o_m.to(out_ptr.dtype.element_ty), mask=h_mask[:, None] & d_mask[None, :])


class TritonV2:
    def __init__(self, case, chunks=None, fuse_merge=True, num_warps=8):
        q = case["q"]
        self.case = case
        self.total_q, self.num_heads, self.head_dim = q.shape
        self.nkv = case["num_kv_heads"]
        self.gqa = self.num_heads // self.nkv
        self.max_topk = case["topk_idx"].shape[-1]
        self.chunks = chunks or default_chunks(self.total_q, self.nkv, self.max_topk)
        self.fuse, self.num_warps = fuse_merge, num_warps
        self.o_partial = torch.empty(self.chunks, self.total_q, self.num_heads, self.head_dim, dtype=q.dtype, device=q.device)
        self.lse_partial = torch.empty(self.chunks, self.total_q, self.num_heads, dtype=torch.float32, device=q.device)
        self.cnt = torch.zeros(self.total_q * self.nkv, dtype=torch.int32, device=q.device)
        self.out = torch.empty_like(q)

    def __call__(self):
        c = self.case
        q, kv, t, bt = c["q"], c["kv_cache"], c["topk_idx"], c["block_table"]
        _decode_fused_kernel[(self.total_q * self.chunks, self.nkv)](
            q, kv, t, self.o_partial, self.lse_partial, self.out, self.cnt, bt, c["seq_lens"],
            self.total_q, self.gqa, self.head_dim, self.max_topk, c["sm_scale"], c["decode_query_len"],
            q.stride(0), q.stride(1), q.stride(2), kv.stride(0), kv.stride(1), kv.stride(2), kv.stride(3),
            t.stride(0), t.stride(1), t.stride(2), *self.o_partial.stride(), *self.lse_partial.stride(),
            *self.out.stride(), bt.stride(0),
            BLOCK_SIZE_K=128, NUM_TOPK_CHUNKS=self.chunks, BLOCK_SIZE_H=max(16, triton.next_power_of_2(self.gqa)),
            BLOCK_SIZE_D=triton.next_power_of_2(self.head_dim), FUSE_MERGE=self.fuse, num_warps=self.num_warps)
        if not self.fuse:
            sa._merge_topk_attn_out_kernel[(self.total_q, self.num_heads)](
                self.o_partial, self.lse_partial, self.out, self.head_dim, *self.o_partial.stride(),
                *self.lse_partial.stride(), *self.out.stride(), NUM_TOPK_CHUNKS=self.chunks, USE_PDL=False)
        return self.out
