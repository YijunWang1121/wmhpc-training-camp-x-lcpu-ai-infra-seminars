"""E1/E2/E5: baseline correctness + batch sweep (eager vs CUDA graph) + per-kernel split
+ NUM_TOPK_CHUNKS sweep. Everything printed is raw measurement."""
import json, sys
import torch
from common import *

info = device_info()
print("DEVICE", json.dumps(info))

# --- correctness (harness check semantics) ---
for name, cfg in [("b4 regular", dict(num_reqs=4, seq_range=(1024, 8192), seed=0)),
                  ("short+tail", dict(num_reqs=3, seq_range=(50, 300), seed=1)),
                  ("spec dql=2", dict(num_reqs=2, seq_range=(2048, 4096), decode_query_len=2, seed=2))]:
    case = make_case(**cfg)
    e = err_ratio(run_decode(case), sdpa_ref(case))
    print(f"CHECK {name:12s} err_ratio={e:.3e} {'PASS' if e < 2e-2 else 'FAIL'}")

# --- E1: batch sweep, eager vs graph; E2: decode/merge split (graph, each kernel alone) ---
print("\nE1/E2 batch sweep (seq=8192, topk=16, kvh=4, gqa=16, dql=1). times in us")
print(f"{'b':>3} {'chunks':>6} {'grid':>5} | {'eager':>8} {'graph':>8} | {'dec':>8} {'merge':>8} {'dec+mrg':>8} | {'KV MiB':>7} {'GB/s(graph)':>11}")
rows = []
for b in (1, 2, 4, 8, 16, 32, 64):
    case = make_case(num_reqs=b, seq_range=(8192, 8192), seed=0)
    sd = SplitDecode(case)
    t_eager = time_eager(lambda: run_decode(case))
    t_graph = time_graph(sd)
    t_dec = time_graph(sd.decode)
    t_mrg = time_graph(sd.merge)
    kvb = kv_bytes(case)
    row = dict(b=b, chunks=sd.chunks, grid=b * sd.chunks * 4, eager=t_eager, graph=t_graph,
               dec=t_dec, merge=t_mrg, kv_MiB=kvb / 2**20, GBps=kvb / (t_graph * 1e-6) / 1e9)
    rows.append(row)
    print(f"{b:>3} {sd.chunks:>6} {row['grid']:>5} | {t_eager:8.1f} {t_graph:8.1f} | {t_dec:8.1f} {t_mrg:8.1f} {t_dec+t_mrg:8.1f} | {row['kv_MiB']:7.1f} {row['GBps']:11.0f}")
json.dump(rows, open(sys.argv[1] if len(sys.argv) > 1 else "/dev/null", "w"), indent=1) if len(sys.argv) > 1 else None

# --- E5: chunks sweep at small batch: parallelism vs latency chain ---
print("\nE5 NUM_TOPK_CHUNKS sweep (graph, decode-only / merge / total us)")
print(f"{'b':>3} {'chunks':>6} {'grid':>5} {'blk/CTA':>7} | {'dec':>8} {'merge':>8} {'total':>8}")
for b in (1, 4, 8, 16):
    case = make_case(num_reqs=b, seq_range=(8192, 8192), seed=0)
    for ch in (1, 2, 4, 8, 16):
        sd = SplitDecode(case, chunks=ch)
        assert err_ratio(sd(), sdpa_ref(case)) < 2e-2 if b <= 4 else True
        t_dec, t_mrg, t_tot = time_graph(sd.decode), time_graph(sd.merge), time_graph(sd)
        print(f"{b:>3} {ch:>6} {b*ch*4:>5} {16//ch:>7} | {t_dec:8.1f} {t_mrg:8.1f} {t_tot:8.1f}")

# --- seq-length independence check (data comes only from topk blocks) ---
print("\nseq-length sensitivity at b=4 (graph total us)")
for lo, hi in ((256, 256), (2048, 2048), (8192, 8192), (32768, 32768)):
    case = make_case(num_reqs=4, seq_range=(lo, hi), seed=0)
    print(f"  seq={hi:>6}  {time_graph(SplitDecode(case)):8.1f}")
