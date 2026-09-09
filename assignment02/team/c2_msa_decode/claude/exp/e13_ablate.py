"""E13: ablation + in-kernel phase timing of the fused kernel (b=1 and b=4, cl1s3 / cl4s2)."""
import sys, os, importlib, torch
from common import *
lib = os.environ.get("MSA_LIB"); import fused
for b in (1, 4):
    case = make_case(num_reqs=b, seq_range=(8192, 8192), seed=0)
    for cl, st in ((1, 3), (1, 1), (4, 2), (4, 1)):
        out = torch.empty_like(case["q"]); dbg = torch.zeros(b * 4 * cl * 3 * 5, dtype=torch.int64, device="cuda")
        t = time_graph(lambda: fused.fused_decode(case, out=out, cluster=cl, stages=st))
        fused.fused_decode(case, out=out, cluster=cl, stages=st, dbg=dbg); torch.cuda.synchronize()
        d = dbg.view(-1, 5).cpu(); d = d[d[:, 4] > 0].double()
        m = d.mean(0)
        print(f"{lib:28s} b={b} cl{cl}s{st}: {t:6.1f} us | cycles per CTA-group: prologue {m[0]:7.0f} wait {m[1]:7.0f} compute {m[2]:7.0f} epilogue {m[3]:7.0f} total {m[4]:7.0f}  (@2.03GHz total={m[4]/2032:.1f} us)")
