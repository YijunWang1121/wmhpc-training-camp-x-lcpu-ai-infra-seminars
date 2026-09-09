"""Average nsys GPU-metrics samples over decode-kernel vs merge-kernel windows."""
import sqlite3
import sys

for tag in sys.argv[1:] or ["b1", "b16"]:
    db = f"profiles/nsysm_{tag}.sqlite"
    c = sqlite3.connect(db)
    mid = {r[1]: r[0] for r in c.execute(
        "select metricId,metricName from TARGET_INFO_GPU_METRICS")}
    want = ["SMs Active [Throughput %]", "SM Issue [Throughput %]",
            "Tensor Active [Throughput %]", "DRAM Read Bandwidth [Throughput %]",
            "DRAM Write Bandwidth [Throughput %]",
            "Compute Warps in Flight [Avg]", "GR Active [Throughput %]",
            "GPC Clock Frequency [MHz]"]
    sids = {r[0]: r[1] for r in c.execute("select id,value from StringIds")}
    kerns = list(c.execute(
        "select start,end,demangledName from CUPTI_ACTIVITY_KIND_KERNEL order by start"))
    # skip first 5 iters (warmup already done, but be safe)
    wins = {"decode": [], "merge": []}
    for s, e, nm in kerns:
        name = sids.get(nm, str(nm))
        if "decode_kernel" in name:
            wins["decode"].append((s, e))
        elif "merge" in name:
            wins["merge"].append((s, e))
    for k in wins:
        wins[k] = wins[k][5:]

    def avg_over(windows, metric):
        m = mid[metric]
        vals = []
        for (s, e) in windows:
            for (v,) in c.execute(
                "select value from GPU_METRICS where metricId=? and timestamp>=? and timestamp<=?",
                    (m, s, e)):
                vals.append(v)
        return (sum(vals) / len(vals) if vals else float("nan")), len(vals)

    print(f"\n===== {tag}  (decode windows={len(wins['decode'])}, "
          f"merge windows={len(wins['merge'])}) =====")
    dd = [e - s for s, e in wins["decode"]]
    mm = [e - s for s, e in wins["merge"]]
    if dd:
        print(f"  decode dur avg={sum(dd)/len(dd):.0f} ns   merge dur avg={sum(mm)/len(mm):.0f} ns")
    print(f"  {'metric':40s} {'decode':>10s} {'merge':>10s}")
    for w in want:
        d, dn = avg_over(wins["decode"], w)
        mr, mn = avg_over(wins["merge"], w)
        print(f"  {w:40s} {d:10.1f} {mr:10.1f}   (n={dn}/{mn})")
