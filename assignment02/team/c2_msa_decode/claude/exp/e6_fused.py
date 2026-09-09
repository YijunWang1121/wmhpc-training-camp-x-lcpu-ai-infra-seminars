"""E6: fused kernel — correctness (ACCEPTANCE.md R0/R1/R2) + cluster/stages sweep vs Triton baseline."""
import sys, json, itertools
import torch
from common import *
from fused import fused_decode

mode = sys.argv[1] if len(sys.argv) > 1 else "all"
CONFIGS = [(1, 3), (1, 2), (1, 1), (2, 3), (2, 2), (4, 3), (4, 2), (4, 1)]


def per_head_err(x, ref):
    x, ref = x.double(), ref.double()
    num = (x - ref).flatten(2).norm(dim=2)
    den = ref.flatten(2).norm(dim=2).clamp_min(1e-30)
    return (num / den).max().item()


if mode in ("check", "all"):
    print("== correctness (R0 = fp32 SDPA, R1 = Triton) ==")
    allok = True
    cases = []
    for b in (1, 2, 3, 4, 8, 16):
        for seq in ((50, 300), (1024, 8192), (32768, 32768)):
            for dql in (1, 2):
                for seed in (0, 1, 2):
                    cases.append(dict(num_reqs=b, seq_range=seq, decode_query_len=dql, seed=seed))
    for cfg in cases:
        case = make_case(**cfg)
        ref = sdpa_ref(case)
        tri = run_decode(case)
        e_tri = err_ratio(tri, ref)
        for cl, st in CONFIGS:
            out = fused_decode(case, cluster=cl, stages=st)
            torch.cuda.synchronize()
            e0 = err_ratio(out, ref); eh = per_head_err(out, ref)
            e2 = (out.float() - tri.float()).abs().max().item(); lim2 = tri.float().abs().max().item() / 128 + 1e-2
            ok = e0 < 2e-2 and eh < 5e-2 and e2 <= lim2 and torch.isfinite(out.float()).all().item()
            allok &= ok
            if not ok or (cl, st) == CONFIGS[0]:
                print(f"b={cfg['num_reqs']:2d} seq={cfg['seq_range']} dql={cfg['decode_query_len']} seed={cfg['seed']} cl={cl:2d} st={st} | "
                      f"err_R0={e0:.2e} (triton {e_tri:.2e}) maxhead={eh:.2e} maxabs_vs_R1={e2:.3e} (lim {lim2:.3e}) {'PASS' if ok else 'FAIL'}")
    print("ALL PASS" if allok else "SOME FAIL")

if mode in ("bench", "all"):
    print("\n== bench (CUDA graph, us/step; seq=8192) ==")
    hdr = f"{'b':>3} {'triton':>8} " + " ".join(f"cl{cl}s{st}".rjust(8) for cl, st in CONFIGS)
    print(hdr)
    rows = {}
    for b in (1, 2, 4, 8, 16, 32, 64):
        case = make_case(num_reqs=b, seq_range=(8192, 8192), seed=0)
        sd = SplitDecode(case)
        t_tri = time_graph(sd)
        outs = []
        for cl, st in CONFIGS:
            out = torch.empty_like(case["q"])
            outs.append(time_graph(lambda: fused_decode(case, out=out, cluster=cl, stages=st)))
        rows[b] = dict(triton=t_tri, fused=dict(zip([f"cl{cl}s{st}" for cl, st in CONFIGS], outs)))
        print(f"{b:>3} {t_tri:8.1f} " + " ".join(f"{x:8.1f}" for x in outs))
    print("\n== eager us/step ==")
    for b in (1, 4, 16):
        case = make_case(num_reqs=b, seq_range=(8192, 8192), seed=0)
        out = torch.empty_like(case["q"])
        print(f"b={b:2d} triton {time_eager(lambda: run_decode(case)):7.1f}  fused cl2s3 {time_eager(lambda: fused_decode(case, out=out, cluster=2, stages=3)):7.1f}")
    json.dump(rows, open("logs/e6_rows.json", "w"), indent=1)
