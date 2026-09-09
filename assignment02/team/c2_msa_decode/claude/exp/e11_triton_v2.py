"""E11: improved-Triton variant (in-kernel last-CTA merge, num_warps 8) — correctness + graph timing vs baseline."""
import torch
from common import *
from triton_v2 import TritonV2
ok = True
for b in (1, 3, 4, 8, 16):
    for seq in ((50, 300), (1024, 8192)):
        for dql in (1, 2):
            case = make_case(num_reqs=b, seq_range=seq, decode_query_len=dql, seed=b)
            ref = sdpa_ref(case)
            for fuse, nw in ((True, 4), (True, 8), (False, 8)):
                tv = TritonV2(case, fuse_merge=fuse, num_warps=nw)
                e = err_ratio(tv(), ref); torch.cuda.synchronize()
                e2 = err_ratio(tv(), ref)  # second call: counter self-reset must work
                good = e < 2e-2 and e2 < 2e-2 and torch.isfinite(tv.out.float()).all().item()
                ok &= good
                if not good: print(f"FAIL b={b} seq={seq} dql={dql} fuse={fuse} nw={nw} err={e:.2e}/{e2:.2e}")
print("triton_v2 correctness:", "ALL PASS" if ok else "SOME FAIL")
print(f"{'b':>3} {'baseline':>9} {'fuse,nw4':>9} {'fuse,nw8':>9} {'split,nw8':>9}")
for b in (1, 2, 4, 8, 16, 32, 64):
    case = make_case(num_reqs=b, seq_range=(8192, 8192), seed=0)
    t0 = time_graph(SplitDecode(case))
    ts = [time_graph(TritonV2(case, fuse_merge=f, num_warps=n)) for f, n in ((True, 4), (True, 8), (False, 8))]
    print(f"{b:>3} {t0:9.1f} " + " ".join(f"{x:9.1f}" for x in ts))
