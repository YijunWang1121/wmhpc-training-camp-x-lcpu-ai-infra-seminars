"""P3 (a) — fused single-kernel block-sparse GQA decode for the small-batch
regime. No split-K, no merge kernel: one CTA per (query token, kv_head) walks
all <=16 selected blocks with a software-pipelined K/V load and online softmax,
writing the final output directly.

Semantics match vllm_msa_ref/sparse_attn.py's decode path:
  - logical block -> physical page via block_table
  - only the first real_topk = min(topk, cdiv(kv_len,128)) slots
  - base-2 (exp2/log2) softmax
  - per-token causal mask (pos < kv_len)
  - optional fp8 KV cache with scalar or [kv_head, token] scales (mode 0/1/2)
"""
import os
import sys

HARNESS = os.path.join(os.path.dirname(__file__), "..", "harness")
sys.path.insert(0, HARNESS)

import torch
import triton
import triton.language as tl

SPARSE_BLOCK_SIZE = 128
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


@triton.heuristics({
    "BLOCK_SIZE_H": lambda a: max(16, triton.next_power_of_2(a["gqa_group_size"])),
    "BLOCK_SIZE_D": lambda a: triton.next_power_of_2(a["head_dim"]),
})
@triton.jit(do_not_specialize=["decode_query_len"])
def _fused_sparse_decode_kernel(
    q_ptr, kv_cache_ptr, k_scale_ptr, v_scale_ptr, t_ptr, o_ptr,
    block_table_ptr, seq_lens,
    gqa_group_size, head_dim, max_topk, sm_scale, decode_query_len,
    stride_qn, stride_qh, stride_qd,
    stride_kv_blk, stride_kv_h, stride_kv_pos, stride_kv_d,
    stride_ks_h, stride_ks_t, stride_vs_h, stride_vs_t,
    stride_th, stride_tn, stride_tk,
    stride_on, stride_oh, stride_od,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_D: tl.constexpr,
    USE_FP8: tl.constexpr, KV_SCALE_MODE: tl.constexpr, NUM_STAGES: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950408889634
    pid_b = tl.program_id(0)      # query token index in [0, total_q)
    pid_kh = tl.program_id(1)
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len
    pid_h = pid_kh * gqa_group_size

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks = (kv_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    real_topk = tl.minimum(max_topk, num_blocks)

    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    bt_row = block_table_ptr + req_id * stride_bt_b
    idx_base = t_ptr + pid_kh * stride_th + pid_b * stride_tn

    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qn + pid_h * stride_qh,
        shape=(gqa_group_size, head_dim), strides=(stride_qh, stride_qd),
        offsets=(0, 0), block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D), order=(1, 0))
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")

    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_D), dtype=tl.float32)

    cur = idx_base
    for _ in tl.range(0, real_topk, num_stages=NUM_STAGES):
        blk = tl.load(cur).to(tl.int32)
        cur = cur + stride_tk
        c = blk * BLOCK_SIZE_K
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = c + off_n
        pos_mask = pos < kv_len
        k = tl.load(
            kv_cache_ptr + page * stride_kv_blk + pid_kh * stride_kv_h
            + off_n[None, :] * stride_kv_pos + off_d[:, None] * stride_kv_d,
            mask=d_mask[:, None] & pos_mask[None, :], other=0.0)
        if USE_FP8:
            k = k.to(q.dtype)
            if KV_SCALE_MODE == 1:
                k = (k * tl.load(k_scale_ptr)).to(q.dtype)
            elif KV_SCALE_MODE == 2:
                ks = tl.load(k_scale_ptr + pid_kh * stride_ks_h
                             + (page * BLOCK_SIZE_K + off_n) * stride_ks_t,
                             mask=pos_mask, other=1.0)
                k = (k * ks[None, :]).to(q.dtype)
        qk = tl.dot(q, k) * sm_scale_log2e
        qk += tl.where(pos_mask[None, :], 0, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        v = tl.load(
            kv_cache_ptr + page * stride_kv_blk + pid_kh * stride_kv_h
            + off_n[:, None] * stride_kv_pos + (head_dim + off_d[None, :]) * stride_kv_d,
            mask=pos_mask[:, None] & d_mask[None, :], other=0.0)
        if USE_FP8:
            v = v.to(q.dtype)
            if KV_SCALE_MODE == 1:
                v = (v * tl.load(v_scale_ptr)).to(q.dtype)
            elif KV_SCALE_MODE == 2:
                vs = tl.load(v_scale_ptr + pid_kh * stride_vs_h
                             + (page * BLOCK_SIZE_K + off_n) * stride_vs_t,
                             mask=pos_mask, other=1.0)
                v = (v * vs[:, None]).to(q.dtype)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    acc = acc / l_safe[:, None]
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_b * stride_on + pid_h * stride_oh,
        shape=(gqa_group_size, head_dim), strides=(stride_oh, stride_od),
        offsets=(0, 0), block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D), order=(1, 0))
    tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))


@torch.no_grad()
def fused_sparse_decode(q, kv_cache, topk_idx, block_table, seq_lens,
                        num_kv_heads, sm_scale, output, decode_query_len,
                        k_scale=None, v_scale=None, num_warps=4, num_stages=3):
    total_q, num_heads, head_dim = q.shape
    max_topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    use_fp8 = kv_cache.dtype in _FP8_DTYPES
    mode = 0
    ks = vs = output
    s_ksh = s_kst = s_vsh = s_vst = 0
    if use_fp8 and k_scale is not None:
        if k_scale.numel() == 1:
            mode, ks, vs = 1, k_scale, v_scale
        else:
            mode, ks, vs = 2, k_scale, v_scale
            s_ksh, s_kst = k_scale.stride(0), k_scale.stride(1)
            s_vsh, s_vst = v_scale.stride(0), v_scale.stride(1)
    grid = (total_q, num_kv_heads)
    _fused_sparse_decode_kernel[grid](
        q, kv_cache, ks, vs, topk_idx, output, block_table, seq_lens,
        gqa_group_size, head_dim, max_topk, sm_scale, decode_query_len,
        q.stride(0), q.stride(1), q.stride(2),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2), kv_cache.stride(3),
        s_ksh, s_kst, s_vsh, s_vst,
        topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE, USE_FP8=use_fp8, KV_SCALE_MODE=mode,
        NUM_STAGES=num_stages, num_warps=num_warps,
    )
