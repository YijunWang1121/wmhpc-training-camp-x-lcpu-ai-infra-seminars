"""P3 acceptance: correctness (3 tiers x shape matrix x 5 seeds) + perf vs baseline.
    python p3_test.py check
    python p3_test.py bench
"""
import sys
import torch

from lib import make_case, sdpa_ref, plan, time_fn, err_ratio, sa
from v1_fused import fused_sparse_decode
from p1_graph import graph_time


def run_baseline(case):
    out = torch.empty_like(case["q"])
    sa.minimax_m3_sparse_attn_decode(
        case["q"], case["kv_cache"], case["topk_idx"], case["block_table"],
        case["seq_lens"], case["num_kv_heads"], case["sm_scale"], out,
        case["decode_query_len"])
    return out


def run_fused(case, **kw):
    out = torch.empty_like(case["q"])
    fused_sparse_decode(
        case["q"], case["kv_cache"], case["topk_idx"], case["block_table"],
        case["seq_lens"], case["num_kv_heads"], case["sm_scale"], out,
        case["decode_query_len"], **kw)
    return out


SHAPES = [
    ("b1 seq8192",       dict(num_reqs=1, seq_range=(8192, 8192))),
    ("b2 seq8192",       dict(num_reqs=2, seq_range=(8192, 8192))),
    ("b4 seq8192",       dict(num_reqs=4, seq_range=(8192, 8192))),
    ("b8 seq8192",       dict(num_reqs=8, seq_range=(8192, 8192))),
    ("b16 seq8192",      dict(num_reqs=16, seq_range=(8192, 8192))),
    ("b4 mixed 1k-8k",   dict(num_reqs=4, seq_range=(1024, 8192))),
    ("b3 short 50-300",  dict(num_reqs=3, seq_range=(50, 300))),
    ("b2 dql2",          dict(num_reqs=2, seq_range=(2048, 4096), decode_query_len=2)),
    ("b2 dql4",          dict(num_reqs=2, seq_range=(3000, 6000), decode_query_len=4)),
    ("b3 exact 128k",    dict(num_reqs=3, seq_range=(128*20, 128*20))),
    ("b3 128k+1",        dict(num_reqs=3, seq_range=(128*20+1, 128*20+1))),
]


def check():
    allok = True
    print(f"{'shape':20s} {'seed':>4}  {'vs SDPA':>10} {'vs baseline':>12}  verdict")
    for name, cfg in SHAPES:
        worst1 = worst2 = 0.0
        for seed in range(5):
            case = make_case(seed=seed, **cfg)
            got = run_fused(case)
            ref = sdpa_ref(case)
            base = run_baseline(case)
            e1 = err_ratio(got, ref)
            e2 = err_ratio(got, base)
            worst1 = max(worst1, e1)
            worst2 = max(worst2, e2)
        ok = worst1 < 2e-2 and worst2 < 5e-3
        allok &= ok
        print(f"{name:20s} {'5x':>4}  {worst1:>10.2e} {worst2:>12.2e}  "
              f"{'PASS' if ok else 'FAIL'}")
    print("\nALL PASS" if allok else "\nSOME FAILED")
    return allok


def bench():
    print(f"{'batch':>6} {'base_e2e':>9} {'fused_e2e':>10} {'speedup':>8} "
          f"{'base_gpu(nsys sep)':>18}")
    for n in (1, 2, 4, 8, 16, 32):
        case = make_case(num_reqs=n, seq_range=(8192, 8192), seed=0)
        pl = plan(case)
        out = torch.empty_like(case["q"])

        def base_e2e():
            sa.minimax_m3_sparse_attn_decode(
                case["q"], case["kv_cache"], case["topk_idx"], case["block_table"],
                case["seq_lens"], case["num_kv_heads"], case["sm_scale"], out,
                case["decode_query_len"])

        def fused():
            fused_sparse_decode(
                case["q"], case["kv_cache"], case["topk_idx"], case["block_table"],
                case["seq_lens"], case["num_kv_heads"], case["sm_scale"], out,
                case["decode_query_len"])

        tb = graph_time([base_e2e])
        tf = graph_time([fused])
        print(f"{n:>6} {tb:>9.2f} {tf:>10.2f} {tb/tf:>7.2f}x")


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "check"
    if m == "check":
        sys.exit(0 if check() else 1)
    else:
        bench()
