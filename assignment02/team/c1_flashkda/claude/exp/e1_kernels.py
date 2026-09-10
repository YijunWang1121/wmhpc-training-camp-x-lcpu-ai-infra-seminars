"""E1b: per-kernel breakdown of flash_kda.fwd via torch.profiler (fixed + varlen, H=96/64, state modes)."""
import sys, math, torch, torch.nn.functional as F
import flash_kda
from torch.profiler import profile, ProfilerActivity

def make_inputs(seq_lens, H, D, device="cuda"):
    T = sum(seq_lens); N = len(seq_lens)
    q = F.normalize(torch.randn(1, T, H, D, device=device), dim=-1).bfloat16()
    k = F.normalize(torch.randn(1, T, H, D, device=device), dim=-1).bfloat16()
    v = torch.randn(1, T, H, D, device=device).bfloat16()
    g = torch.randn(1, T, H, D, device=device).bfloat16()
    beta = torch.randn(1, T, H, device=device).bfloat16()
    A_log = torch.rand(H, device=device); dt_bias = torch.rand(H, D, device=device)
    h0 = (torch.arange(N*H*D*D, device=device, dtype=torch.float32).reshape(N,H,D,D)/1e6).bfloat16()
    cu = None
    if N > 1:
        cu = torch.tensor([0]+list(torch.cumsum(torch.tensor(seq_lens),0).tolist()), dtype=torch.long, device=device)
    return q,k,v,g,beta,A_log,dt_bias,h0,cu

def run(seq_lens, H, D=128, state="bf16", iters=50):
    q,k,v,g,beta,A_log,dt_bias,h0,cu = make_inputs(seq_lens,H,D)
    out = torch.zeros_like(v)
    extra = {"cu_seqlens": cu} if cu is not None else {}
    if state == "bf16":
        st = dict(initial_state=h0, final_state=torch.zeros_like(h0))
    elif state == "fp32":
        st = dict(initial_state=h0.float(), final_state=torch.zeros_like(h0).float())
    else:
        st = {}
    fn = lambda: flash_kda.fwd(q,k,v,g,beta,1/math.sqrt(D),out,A_log=A_log,dt_bias=dt_bias,lower_bound=-5.0,**st,**extra)
    for _ in range(10): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters): fn()
        torch.cuda.synchronize()
    agg = {}
    for e in prof.events():
        if e.device_type.name == "CUDA":
            nm = e.name.split("<")[0][:60]
            a = agg.setdefault(nm, [0,0.0]); a[0]+=1; a[1]+=e.time_range.elapsed_us()
    tot = sum(a[1] for a in agg.values())/iters
    print(f"\n[{'varlen' if cu is not None else 'fixed'} T={sum(seq_lens)} N={len(seq_lens)} H={H} state={state}] GPU sum per call = {tot:.1f} us")
    for nm,(cnt,us) in sorted(agg.items(), key=lambda x:-x[1][1]):
        print(f"  {us/iters:9.1f} us/call  x{cnt//iters:2d}  {nm}")

if __name__ == "__main__":
    torch.manual_seed(0)
    for H in (96, 64):
        for seq in ([8192], [1300,547,2048,963,271,3063], [1024]*8):
            run(seq, H, state="bf16")
    run([8192], 96, state="fp32"); run([8192], 96, state="none")
    # TP-8 per-card head count and a long-context sample
    run([8192], 12); run([32768], 12); run([32768], 96)
