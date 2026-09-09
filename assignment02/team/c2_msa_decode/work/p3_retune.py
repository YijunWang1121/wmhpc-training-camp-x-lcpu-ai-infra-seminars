"""Can the *baseline* Triton kernels be retuned for small batch without a
rewrite? Sweep:
  - num_warps on _gqa_sparse_decode_kernel  (via kernel.run kwarg override)
  - NUM_TOPK_CHUNKS override
  - num_warps on _merge_topk_attn_out_kernel
CUDA-graph e2e timing, batch 1..16.
"""
import torch
from lib import make_case, plan, sa
from p1_graph import graph_time
import triton

SPARSE_BLOCK_SIZE = 128


def custom_decode(case, num_topk_chunks, dec_warps, mrg_warps):
    q = case["q"]; kv = case["kv_cache"]; topk = case["topk_idx"]
    bt = case["block_table"]; sl = case["seq_lens"]
    nkv = case["num_kv_heads"]; scale = case["sm_scale"]; dql = case["decode_query_len"]
    total_q, num_heads, hd = q.shape
    G = num_heads // nkv
    o_part = torch.empty(num_topk_chunks, total_q, num_heads, hd, dtype=q.dtype, device=q.device)
    lse_part = torch.empty(num_topk_chunks, total_q, num_heads, dtype=torch.float32, device=q.device)
    out = torch.empty_like(q)
    dg = (total_q * num_topk_chunks, nkv)
    mg = (total_q, num_heads)

    def run():
        sa._gqa_sparse_decode_kernel[dg](
            q, kv, out, out, topk, o_part, lse_part, bt, sl, total_q, G, hd,
            topk.shape[-1], scale, dql,
            q.stride(0), q.stride(1), q.stride(2),
            kv.stride(0), kv.stride(1), kv.stride(2), kv.stride(3),
            0, 0, 0, 0, topk.stride(0), topk.stride(1), topk.stride(2),
            o_part.stride(0), o_part.stride(1), o_part.stride(2), o_part.stride(3),
            lse_part.stride(0), lse_part.stride(1), lse_part.stride(2), bt.stride(0),
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE, NUM_TOPK_CHUNKS=num_topk_chunks,
            USE_FP8=False, KV_SCALE_MODE=0, USE_PDL=False, num_warps=dec_warps)
        sa._merge_topk_attn_out_kernel[mg](
            o_part, lse_part, out, hd,
            o_part.stride(0), o_part.stride(1), o_part.stride(2), o_part.stride(3),
            lse_part.stride(0), lse_part.stride(1), lse_part.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            NUM_TOPK_CHUNKS=num_topk_chunks, USE_PDL=False, num_warps=mrg_warps)
    return run


print("# baseline retune — CUDA-graph e2e us")
for n in (1, 2, 4, 8, 16):
    case = make_case(num_reqs=n, seq_range=(8192, 8192), seed=0)
    pl = plan(case)

    def base():
        sa.minimax_m3_sparse_attn_decode(
            case["q"], case["kv_cache"], case["topk_idx"], case["block_table"],
            case["seq_lens"], case["num_kv_heads"], case["sm_scale"], pl["output"],
            case["decode_query_len"])
    tb = graph_time([base])
    results = []
    for chunks in (4, 8, 16):
        for dw in (2, 4, 8):
            for mw in (2, 4):
                try:
                    t = graph_time([custom_decode(case, chunks, dw, mw)])
                    results.append((t, chunks, dw, mw))
                except Exception:
                    pass
    results.sort()
    best = results[0]
    print(f"batch={n:>2}  base={tb:6.2f}  best={best[0]:6.2f} "
          f"(chunks={best[1]}, dec_w={best[2]}, mrg_w={best[3]})  "
          f"speedup={tb/best[0]:.2f}x")
    for t, c, dw, mw in results[:4]:
        print(f"      {t:6.2f}  c={c} dw={dw} mw={mw}")
