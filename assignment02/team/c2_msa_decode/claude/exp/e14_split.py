"""E14: (1) floor probes: empty kernel / index chain / index+TMA in the baseline's launch shape (graph us);
(2) our CTA + Triton merge (split-K without cluster) vs baseline, per-kernel split; correctness on the acceptance matrix."""
import torch
from common import *
from fused import FusedSplit, probe
print("== floor probes (graph, us; grid = total_q*4*16 CTAs of 128 threads) ==")
for b in (1, 4, 16):
    case = make_case(num_reqs=b, seq_range=(8192, 8192), seed=0)
    lse = torch.zeros(b * 64 * 16, device="cuda")
    ts = [time_graph(lambda m=m: probe(case, m, lse)) for m in (0, 1, 2)]
    sd = SplitDecode(case, chunks=16)
    print(f"b={b:2d}: empty {ts[0]:5.1f} | index chain {ts[1]:5.1f} | index+TMA 64KiB {ts[2]:5.1f} | triton decode(16 chunks) {time_graph(sd.decode):5.1f} merge {time_graph(sd.merge):5.1f}")
print("\n== correctness: FusedSplit vs R0 ==")
ok = True
for b in (1, 3, 4, 8, 16):
    for seq in ((50, 300), (1024, 8192)):
        for dql in (1, 2):
            case = make_case(num_reqs=b, seq_range=seq, decode_query_len=dql, seed=b)
            ref = sdpa_ref(case)
            for sp in (16, 8, 4, 2):
                fs = FusedSplit(case, split=sp); out = fs(); torch.cuda.synchronize()
                e = err_ratio(out, ref); good = e < 2e-2 and torch.isfinite(out.float()).all().item(); ok &= good
                if not good: print(f"FAIL b={b} seq={seq} dql={dql} split={sp} err={e:.2e}")
print("FusedSplit correctness:", "ALL PASS" if ok else "SOME FAIL")
print("\n== timing (graph, us) ==")
print(f"{'b':>3} | {'triton':>7} {'dec':>6} {'mrg':>6} | " + " | ".join(f"{'split'+str(sp):>7} {'dec':>6} {'mrg':>6}" for sp in (16, 8, 4)))
for b in (1, 2, 4, 8, 16, 32):
    case = make_case(num_reqs=b, seq_range=(8192, 8192), seed=0)
    sd = SplitDecode(case)
    row = f"{b:>3} | {time_graph(sd):7.1f} {time_graph(sd.decode):6.1f} {time_graph(sd.merge):6.1f} | "
    for sp in (16, 8, 4):
        fs = FusedSplit(case, split=sp)
        row += f"{time_graph(fs):7.1f} {time_graph(fs.decode):6.1f} {time_graph(fs.merge):6.1f} | "
    print(row)
