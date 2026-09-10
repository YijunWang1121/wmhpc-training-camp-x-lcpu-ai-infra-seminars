"""Ablation of the tcgen05 K2 (v3): remove thread-side epilogues to measure the floor of the async-MMA chain."""
import os, sys, glob, importlib.util, torch
sys.argv = [sys.argv[0]]
os.environ.setdefault("K2TC", "k2_tc2")
import e7_k2tc as e
from torch.profiler import profile, ProfilerActivity
HERE = os.path.dirname(os.path.abspath(__file__))
def load(name):
    so = glob.glob(f"{HERE}/k2tc/{name}.*.so")[0]
    spec = importlib.util.spec_from_file_location(name, so); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
mods = {n: load(n) for n in ["k2_tc2", "k2_tc2_noE1", "k2_tc2_noE3", "k2_tc2_noE4", "k2_tc2_noE134"]}
for (T, H) in [(8192, 96), (8192, 12)]:
    p = e.setup(T, H); e.run_ref(p); torch.cuda.synchronize(); p["beta_t"] = p["beta"][0].t().contiguous()
    def prof(fn, key, iters=20):
        for _ in range(3): fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as pr:
            for _ in range(iters): fn()
            torch.cuda.synchronize()
        return sum(ev.time_range.elapsed_us() for ev in pr.events() if ev.device_type.name == "CUDA" and key in ev.name) / iters
    mhz = e.sm_mhz(); n = T // 16
    print(f"=== T={T} H={H} @ {mhz} MHz ===")
    t0 = prof(lambda: e.run_ref(p), "recurrence"); print(f"  {'FlashKDA K2 (mma.sync)':28s} {t0:8.1f} us  {t0/n*mhz:6.0f} cyc/chunk")
    for name, m in mods.items():
        out = torch.zeros_like(p["v"]); hT = torch.zeros_like(p["h0"]); dbg = torch.zeros(H * 16, dtype=torch.int64, device="cuda"); dump = torch.zeros(0, device="cuda")
        fn = lambda: m.k2_tc_fwd(p["ws"], T // 16, p["v"], p["beta_t"], p["h0"], hT, out, dbg, False, dump)
        t1 = prof(fn, "k2_tc"); print(f"  {name:28s} {t1:8.1f} us  {t1/n*mhz:6.0f} cyc/chunk   ({t0/t1:.2f}x vs mma.sync)")
