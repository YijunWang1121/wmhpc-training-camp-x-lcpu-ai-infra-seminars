"""E7 (challenge): tcgen05 K2 vs FlashKDA's mma.sync K2 on the identical K1 workspace.
1. run flash_kda_C.fwd with an explicit workspace -> K1 fills it, K2 produces reference out/hT (bf16 state)
2. run k2_tc on the same workspace -> out2/hT2; compare with flash_kda (should be ~bf16-identical) and with the
   fp64 naive recurrence (must be at the same error level as flash_kda)
3. time both K2 kernels with torch.profiler (fixed T=8192, H=96 and H=12; T=32768 H=12); k2_tc timing mode gives
   per-phase cycle attribution (dbg) for the 'profile -> next step' loop."""
import sys, os, math, glob, importlib.util, torch, torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
import flash_kda_C
from clk import sm_mhz
from fla_kda_ref.naive import naive_recurrent_kda

HERE = os.path.dirname(os.path.abspath(__file__))
VER = os.environ.get("K2TC", "k2_tc2")
so = glob.glob(f"{HERE}/k2tc/{VER}.*.so")[0]
print("using", so)
spec = importlib.util.spec_from_file_location(VER, so); k2_tc = importlib.util.module_from_spec(spec); spec.loader.exec_module(k2_tc)
print("k2_tc smem bytes:", k2_tc.SMEM_BYTES)

def rel(a, b):
    a = a.double(); b = b.double()
    return ((a - b).flatten().square().mean().sqrt() / (b.flatten().square().mean().sqrt() + 1e-30)).item()

def setup(T, H, D=128, seed=0, lb=-5.0):
    g_ = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, device="cuda", generator=g_)
    q = F.normalize(r(1, T, H, D), dim=-1).bfloat16(); k = F.normalize(r(1, T, H, D), dim=-1).bfloat16()
    v = r(1, T, H, D).bfloat16(); g = r(1, T, H, D).bfloat16(); beta = r(1, T, H).bfloat16()
    A_log = torch.rand(H, device="cuda", generator=g_); dt = torch.rand(H, D, device="cuda", generator=g_)
    h0 = (r(1, H, D, D) * 0.1).bfloat16()
    ws = torch.empty(flash_kda_C.get_workspace_size(T, H, 1), dtype=torch.uint8, device="cuda")
    return dict(q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt=dt, h0=h0, ws=ws, T=T, H=H, D=D, lb=lb, scale=1 / math.sqrt(D))

def run_ref(p):
    out = torch.zeros_like(p["v"]); hT = torch.zeros_like(p["h0"])
    flash_kda_C.fwd(p["q"], p["k"], p["v"], p["g"], p["beta"], p["scale"], out, p["ws"], p["A_log"], p["dt"], p["lb"],
                    initial_state=p["h0"], final_state=hT, cu_seqlens=None)
    return out, hT

def run_tc(p, timing=False):
    out = torch.zeros_like(p["v"]); hT = torch.zeros_like(p["h0"])
    dbg = torch.zeros(p["H"] * 16, dtype=torch.int64, device="cuda")
    if "beta_t" not in p: p["beta_t"] = p["beta"][0].t().contiguous()
    dump = torch.zeros(65536, dtype=torch.float32, device="cuda") if p.get("dump") else torch.zeros(0, device="cuda")
    k2_tc.k2_tc_fwd(p["ws"], p["T"] // 16, p["v"], p["beta_t"], p["h0"], hT, out, dbg, timing, dump)
    p["dump_out"] = dump
    return out, hT, dbg

def gold(p):
    H, D = p["H"], p["D"]; f = lambda x: x.double()
    g_act = p["lb"] * torch.sigmoid(torch.exp(f(p["A_log"])).view(1, 1, H, 1) * (f(p["g"]) + f(p["dt"]).view(1, 1, H, D)))
    o, S = naive_recurrent_kda(f(p["q"]), f(p["k"]), f(p["v"]), g_act, torch.sigmoid(f(p["beta"])), p["scale"],
                               initial_state=f(p["h0"]).transpose(-1, -2).contiguous(), output_final_state=True)
    return o, S.transpose(-1, -2)

def debug_chunk0():
    """stage-by-stage check of chunk 0 / head 0 against torch computed from the raw workspace."""
    p = setup(256, 4); p["dump"] = True
    o1, h1 = run_ref(p)
    o2, h2, _ = run_tc(p); torch.cuda.synchronize()
    d = p["dump_out"]; ws = p["ws"]; D = 128
    n_ht = p["H"] * (p["T"] // 16)
    def region(base, size, idx, shape, dt=torch.bfloat16):
        return ws[base + idx * size: base + (idx + 1) * size].view(dt).float().view(*shape)
    kd = region(0, 4096, 0, (16, D)); qd = region(n_ht * 4096, 4096, 0, (16, D)); kr = region(2 * n_ht * 4096, 4096, 0, (16, D))
    gt = region(3 * n_ht * 4096, 512, 0, (D,), torch.float32); inv = region(n_ht * (3 * 4096 + 512), 512, 0, (16, 16)); mqk = region(n_ht * (3 * 4096 + 1024), 512, 0, (16, 16))
    S0 = p["h0"][0, 0].float()                               # [v][k]
    vt = p["v"][0, :16, 0].float()                           # [t][v]
    beta = torch.sigmoid(p["beta"][0, :16, 0].float())
    uT = S0 @ kd.T; outT = S0 @ qd.T                         # [v][t]
    r = lambda a, b: ((a - b).flatten().double().square().mean().sqrt() / (b.flatten().double().square().mean().sqrt() + 1e-30)).item()
    print("  G1 uT   rel", r(d[:2048].view(128, 16), uT), " outT rel", r(d[2048:4096].view(128, 16), outT))
    u = ((vt - uT.T.bfloat16().float()).bfloat16().float() * beta[:, None]).bfloat16().float()   # [t][v]
    u2 = inv @ u                                             # [t'][v]
    print("  G2 u2T  rel", r(d[6144:8192].view(128, 16), u2.T))
    U = u2.bfloat16().float()
    print("  G2' out2 rel", r(d[4096:6144].view(128, 16), (mqk @ U).T))
    kU = U.T @ kr                                            # [v][k]
    print("  G3 kU   rel", r(d[8192:8192 + 16384].view(128, 128), kU))
    Snew = (S0 * gt[None, :] + kU).bfloat16().float()
    print("  E4 Snew rel", r(d[24576:40960].view(128, 128), Snew))
    out_ref = ((d[2048:4096].view(128, 16).bfloat16().float() + d[4096:6144].view(128, 16).bfloat16().float()).bfloat16().float()).T   # [t][v]
    print("  E3 staged out vs ref rel", r(d[63488:65536].view(16, 128), out_ref), " gmem out chunk0 vs staged rel", r(o2[0, :16, 0].float(), d[63488:65536].view(16, 128)),
          " gmem out vs flash rel", r(o2[0, :16, 0].float(), o1[0, :16, 0].float()))
    # chunk 1 inputs
    kd1 = region(0, 4096, 1, (16, D)); qd1 = region(n_ht * 4096, 4096, 1, (16, D))
    print("  chunk1 S(smem) vs Snew rel", r(d[40960:57344].view(128, 128), Snew), " KQ1 kd rel", r(d[57344:57344 + 2048].view(16, D), kd1), " qd rel", r(d[57344 + 2048:61440].view(16, D), qd1),
          " V1 rel", r(d[61440:63488].view(16, D), p["v"][0, 16:32, 0].float()))
    nanmap = torch.isnan(o2.float())[0].view(-1, 16, 4, 128).any(-1).any(1)   # [chunk][head]
    print("  NaN tiles (chunk x head):", [(c, hh) for c in range(nanmap.shape[0]) for hh in range(4) if nanmap[c, hh]][:20], " hT nan:", torch.isnan(h2.float()).sum().item())

def correctness():
    for (T, H) in [(256, 4), (1024, 8), (2048, 4)]:
        p = setup(T, H)
        o1, h1 = run_ref(p)          # fills workspace (K1) and gives the mma.sync K2 result
        o2, h2, _ = run_tc(p)
        torch.cuda.synchronize()
        og, Sg = gold(p)
        print(f"[T={T} H={H}] tc-vs-flash: out rel {rel(o2, o1):.2e} hT rel {rel(h2, h1):.2e} | vs fp64 naive: out flash {rel(o1, og):.2e} tc {rel(o2, og):.2e} | hT flash {rel(h1, Sg):.2e} tc {rel(h2, Sg):.2e}")

def timing():
    for (T, H) in [(8192, 96), (8192, 12), (32768, 12)]:
        p = setup(T, H)
        run_ref(p); torch.cuda.synchronize()
        iters = 20
        def prof(fn, key):
            for _ in range(3): fn()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as pr:
                for _ in range(iters): fn()
                torch.cuda.synchronize()
            us = 0.0
            for e in pr.events():
                if e.device_type.name == "CUDA" and key in e.name: us += e.time_range.elapsed_us() / iters
            return us
        t_ref = prof(lambda: run_ref(p), "recurrence")
        t_tc = prof(lambda: run_tc(p), "k2_tc")
        nchunk = T // 16
        for _ in range(5): run_ref(p)
        mhz = sm_mhz(); torch.cuda.synchronize()
        print(f"[T={T} H={H}] SM clock now {mhz} MHz | K2 mma.sync {t_ref:8.1f} us ({t_ref/nchunk*mhz:6.0f} cyc/chunk)   K2 tcgen05 {t_tc:8.1f} us ({t_tc/nchunk*mhz:6.0f} cyc/chunk)   speedup {t_ref/t_tc:.2f}x")
        _, _, dbg = run_tc(p, timing=True); torch.cuda.synchronize()
        d = dbg.view(H, 16).double().mean(0) / nchunk
        names = ["G1 issue+wait", "E1 sync", "G2 issue+wait", "E2 U", "G2'/G3 issue+wait", "E4 rest", "out store", "relayout+sync+issue", "G1 issue only(thr0)", "TMA wait", "E3", "E4 ld b0", "E1 tmem ld", "E1 compute+st", "E1 fence"]
        print("   per-chunk cycles (head-avg, timing build):", " | ".join(f"{n} {c:6.0f}" for n, c in zip(names, d.tolist()) if n != "-"), f"| sum {d.sum():.0f}")

if __name__ == "__main__":
    debug_chunk0()
    correctness()
    timing()
