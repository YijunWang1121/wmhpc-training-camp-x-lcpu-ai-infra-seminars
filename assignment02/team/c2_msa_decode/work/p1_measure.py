"""P1 measurement driver. Run inside a GPU allocation.

  python p1_measure.py            # everything below
  python p1_measure.py check      # just correctness
  python p1_measure.py sweep      # end-to-end + split (decode vs merge) sweep
"""
import sys

import torch

from lib import make_case, sdpa_ref, plan, time_fn, err_ratio, sa


def gpu_info():
    p = torch.cuda.get_device_properties(0)
    print(f"# GPU: {p.name}  SMs={p.multi_processor_count}  "
          f"mem={p.total_memory/2**30:.0f}GiB  cc={p.major}.{p.minor}")
    print(f"# torch {torch.__version__}  triton {sa.tl.__name__ and __import__('triton').__version__}")


def check():
    cases = [
        ("常规 batch=4", dict(num_reqs=4, seq_range=(1024, 8192), seed=0)),
        ("短序列+尾块", dict(num_reqs=3, seq_range=(50, 300), seed=1)),
        ("投机 decode dql=2", dict(num_reqs=2, seq_range=(2048, 4096),
                                   decode_query_len=2, seed=2)),
    ]
    ok = True
    for name, cfg in cases:
        case = make_case(**cfg)
        pl = plan(case)
        pl["run_decode"]()
        pl["run_merge"]()
        torch.cuda.synchronize()
        ref = sdpa_ref(case)
        e = err_ratio(pl["output"], ref)
        good = e < 2e-2
        ok &= good
        print(f"{name:24s} err_ratio={e:.3e}  {'PASS' if good else 'FAIL'}")
    return ok


def sweep():
    print(f"\n{'batch':>6} {'chunks':>7} {'d_grid':>10} {'m_grid':>10} "
          f"{'e2e_us':>9} {'decode_us':>10} {'merge_us':>9} {'d+m':>7} {'launch_gap':>11}")
    for n in (1, 2, 4, 8, 16, 32, 64):
        case = make_case(num_reqs=n, seq_range=(8192, 8192), seed=0)
        pl = plan(case)

        def e2e():
            sa.minimax_m3_sparse_attn_decode(
                case["q"], case["kv_cache"], case["topk_idx"], case["block_table"],
                case["seq_lens"], case["num_kv_heads"], case["sm_scale"],
                pl["output"], case["decode_query_len"])

        t_e2e = time_fn(e2e)
        t_dec = time_fn(pl["run_decode"])
        t_mrg = time_fn(pl["run_merge"])
        dg = pl["decode_grid"]
        mg = pl["merge_grid"]
        print(f"{n:>6} {pl['num_topk_chunks']:>7} {str(dg):>10} {str(mg):>10} "
              f"{t_e2e:>9.1f} {t_dec:>10.1f} {t_mrg:>9.1f} {t_dec+t_mrg:>7.1f} "
              f"{t_e2e-t_dec-t_mrg:>11.1f}")


def bytes_flops():
    """Analytic bytes / FLOPs per decode step, seq=8192, for the roofline."""
    print("\n# analytic (seq=8192, topk=16, page=128, hd=128, kv_heads=4, gqa=16, bf16):")
    for n in (1, 4, 8, 16):
        blocks = 16
        kv_bytes = n * 4 * blocks * 128 * 128 * 2 * 2   # reqs*kvh*blk*128*hd*(K,V)*2B
        q_bytes = n * 64 * 128 * 2
        flops = n * 4 * blocks * (2 * 16 * 128 * 128) * 2  # QK + PV, per (req,kvh)
        print(f"  batch={n:>2}: KV_read={kv_bytes/2**20:7.2f} MiB  "
              f"Q={q_bytes/2**10:6.1f} KiB  FLOPs={flops/1e6:7.1f} M  "
              f"AI={flops/kv_bytes:5.1f} FLOP/B")


if __name__ == "__main__":
    gpu_info()
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("all", "check"):
        ok = check()
    if mode in ("all", "sweep"):
        bytes_flops()
        sweep()
    if mode == "check":
        sys.exit(0 if ok else 1)
