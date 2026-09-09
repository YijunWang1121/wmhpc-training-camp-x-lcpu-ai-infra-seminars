"""C2 P1 profiling helpers: split the decode wrapper into its two Triton
launches so we can time / profile them independently, and expose a clean
end-to-end timer. Mirrors minimax_m3_sparse_attn_decode() in
vllm_msa_ref/sparse_attn.py exactly (no FP8, no PDL).
"""
import os
import sys

HARNESS = os.path.join(os.path.dirname(__file__), "..", "harness")
sys.path.insert(0, HARNESS)

import torch  # noqa: E402
import vllm_shim  # noqa: E402
from synth import make_case  # noqa: E402
from ref_sdpa import sdpa_ref  # noqa: E402

sa = vllm_shim.load_sparse_attn()
SPARSE_BLOCK_SIZE = sa.SPARSE_BLOCK_SIZE


def plan(case):
    """Return everything needed to launch the two kernels for `case`."""
    q = case["q"]
    kv_cache = case["kv_cache"]
    topk_idx = case["topk_idx"]
    block_table = case["block_table"]
    seq_lens = case["seq_lens"]
    num_kv_heads = case["num_kv_heads"]
    sm_scale = case["sm_scale"]
    decode_query_len = case["decode_query_len"]

    total_q, num_heads, head_dim = q.shape
    max_topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads

    TARGET_GRID = 256
    target = max(1, min(max_topk, TARGET_GRID // max(1, total_q * num_kv_heads)))
    num_topk_chunks = 1 << (target.bit_length() - 1)

    o_partial = torch.empty(num_topk_chunks, total_q, num_heads, head_dim,
                            dtype=q.dtype, device=q.device)
    lse_partial = torch.empty(num_topk_chunks, total_q, num_heads,
                              dtype=torch.float32, device=q.device)
    output = torch.empty_like(q)

    decode_grid = (total_q * num_topk_chunks, num_kv_heads)
    merge_grid = (total_q, num_heads)

    def run_decode():
        sa._gqa_sparse_decode_kernel[decode_grid](
            q, kv_cache, output, output, topk_idx, o_partial, lse_partial,
            block_table, seq_lens, total_q, gqa_group_size, head_dim, max_topk,
            sm_scale, decode_query_len,
            q.stride(0), q.stride(1), q.stride(2),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2), kv_cache.stride(3),
            0, 0, 0, 0,
            topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
            o_partial.stride(0), o_partial.stride(1), o_partial.stride(2), o_partial.stride(3),
            lse_partial.stride(0), lse_partial.stride(1), lse_partial.stride(2),
            block_table.stride(0),
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE, NUM_TOPK_CHUNKS=num_topk_chunks,
            USE_FP8=False, KV_SCALE_MODE=0, USE_PDL=False,
        )

    def run_merge():
        sa._merge_topk_attn_out_kernel[merge_grid](
            o_partial, lse_partial, output, head_dim,
            o_partial.stride(0), o_partial.stride(1), o_partial.stride(2), o_partial.stride(3),
            lse_partial.stride(0), lse_partial.stride(1), lse_partial.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            NUM_TOPK_CHUNKS=num_topk_chunks, USE_PDL=False,
        )

    return dict(run_decode=run_decode, run_merge=run_merge, output=output,
                o_partial=o_partial, lse_partial=lse_partial,
                num_topk_chunks=num_topk_chunks, decode_grid=decode_grid,
                merge_grid=merge_grid)


def time_fn(fn, iters=200, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # us


def err_ratio(x, ref):
    x, ref = x.double().flatten(), ref.double().flatten()
    return ((x - ref).norm() / ref.norm().clamp_min(1e-30)).item()
