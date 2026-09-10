"""E4b: K2 attribution by ablation.  Loads each flash_kda_C_<variant>.so built by build_variants.py and times
K1 / K2 separately (torch.profiler CUDA events).  Ablated variants produce WRONG outputs on purpose; only timing
is meaningful.  Shapes: official fixed T=8192,H=96 and TP8-per-card H=12."""
import math, glob, os, importlib.machinery, importlib.util, torch, torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clk import sm_mhz

HERE = os.path.dirname(os.path.abspath(__file__))

def load(name):
    so = glob.glob(f"{HERE}/variants/flash_kda_C_{name}.*.so")
    if not so:
        return None
    spec = importlib.util.spec_from_file_location(f"flash_kda_C_{name}", so[0])
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def kernel_times(mod, T, H, D=128, iters=30):
    q = F.normalize(torch.randn(1, T, H, D, device="cuda"), dim=-1).bfloat16()
    k = F.normalize(torch.randn(1, T, H, D, device="cuda"), dim=-1).bfloat16()
    v = torch.randn(1, T, H, D, device="cuda").bfloat16(); g = torch.randn(1, T, H, D, device="cuda").bfloat16()
    beta = torch.randn(1, T, H, device="cuda").bfloat16(); A_log = torch.rand(H, device="cuda"); dt = torch.rand(H, D, device="cuda")
    h0 = (torch.randn(1, H, D, D, device="cuda") * 0.1).bfloat16(); hT = torch.zeros_like(h0); out = torch.zeros_like(v)
    ws = torch.empty(mod.get_workspace_size(T, H, 1), dtype=torch.uint8, device="cuda")
    fn = lambda: mod.fwd(q, k, v, g, beta, 1 / math.sqrt(D), out, ws, A_log, dt, -5.0, initial_state=h0, final_state=hT, cu_seqlens=None)
    for _ in range(5): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters): fn()
        torch.cuda.synchronize()
    t = {"K1": 0.0, "K2": 0.0, "other": 0.0}
    for e in prof.events():
        if e.device_type.name != "CUDA": continue
        key = "K1" if "prepare" in e.name else "K2" if "recurrence" in e.name else "other"
        t[key] += e.time_range.elapsed_us() / iters
    return t

if __name__ == "__main__":
    names = sys.argv[1:] or ["stock", "notma", "notma_noP1", "notma_noP6", "notma_noP34", "notma_noMMA", "noP6"]
    torch.manual_seed(0)
    for (T, H) in [(8192, 96), (8192, 12), (32768, 12)]:
        print(f"\n=== T={T} H={H} (K2 grid = {H} CTAs, {T//16} chunks each) ===")
        for n in names:
            m = load(n)
            if m is None: print(f"  {n:14s} (not built)"); continue
            t = kernel_times(m, T, H); mhz = sm_mhz()
            print(f"  {n:14s} K1 {t['K1']:8.1f} us   K2 {t['K2']:8.1f} us  ({t['K2']/(T//16)*1e3:7.1f} ns = {t['K2']/(T//16)*mhz:6.0f} cyc per chunk @{mhz}MHz)   other {t['other']:6.1f} us")
