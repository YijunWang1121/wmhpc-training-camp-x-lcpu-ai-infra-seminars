"""Sweep num_warps x num_stages for the fused kernel; also dump the baseline
e2e for reference. CUDA-graph timing."""
import torch
from lib import make_case, sa
from v1_fused import fused_sparse_decode
from p1_graph import graph_time

CFGS = [(w, s) for w in (2, 4, 8) for s in (1, 2, 3, 4)]

print(f"# baseline vs fused(num_warps,num_stages), CUDA-graph e2e us")
for n in (1, 2, 4, 8, 16, 32, 64):
    case = make_case(num_reqs=n, seq_range=(8192, 8192), seed=0)
    out = torch.empty_like(case["q"])

    def base():
        sa.minimax_m3_sparse_attn_decode(
            case["q"], case["kv_cache"], case["topk_idx"], case["block_table"],
            case["seq_lens"], case["num_kv_heads"], case["sm_scale"], out,
            case["decode_query_len"])
    tb = graph_time([base])
    best = (1e9, None)
    row = []
    for (w, s) in CFGS:
        def f(w=w, s=s):
            fused_sparse_decode(
                case["q"], case["kv_cache"], case["topk_idx"], case["block_table"],
                case["seq_lens"], case["num_kv_heads"], case["sm_scale"], out,
                case["decode_query_len"], num_warps=w, num_stages=s)
        try:
            t = graph_time([f])
        except Exception as e:
            t = float("nan")
        row.append(f"{w}w{s}s={t:.1f}")
        if t < best[0]:
            best = (t, (w, s))
    print(f"batch={n:>2} base={tb:6.2f}  best={best[0]:6.2f} @ {best[1]}  "
          f"speedup={tb/best[0]:.2f}x")
    print("   " + "  ".join(row))
