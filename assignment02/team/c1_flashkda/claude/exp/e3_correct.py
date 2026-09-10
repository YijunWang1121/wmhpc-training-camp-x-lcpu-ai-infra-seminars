"""E3: correctness of flash_kda vs (a) pure-PyTorch naive recurrence (fla_kda_ref/naive.py, fp32/fp64) and
(b) fla Triton chunk_kda (installed fla 0.5.2; snapshot a3edffc differs only by int32/int64 index casts).
Conventions (from FlashKDA tests/test_fwd.py + csrc): q,k L2-normalised in-kernel; beta = sigmoid(beta_raw);
g_act = lower_bound * sigmoid(exp(A_log) * (g_raw + dt_bias)); state layout [N,H,V,K] (state_v_first)."""
import sys, math, torch, torch.nn.functional as F
import flash_kda
from fla_kda_ref.naive import naive_recurrent_kda
from fla.ops.kda import chunk_kda

def err_stats(name, got, ref):
    got = got.float(); ref = ref.float()
    d = (got-ref).abs()
    rel = (d.flatten().square().mean().sqrt() / (ref.flatten().square().mean().sqrt()+1e-12)).item()
    print(f"  {name:28s} max_abs={d.max().item():.3e} mean_abs={d.mean().item():.3e} rel_rms={rel:.3e}")
    return rel

def case(T, H, D=128, N=1, lower_bound=-5.0, seed=0, device="cuda", ref_dtype=torch.float64):
    torch.manual_seed(seed)
    q = F.normalize(torch.randn(N,T,H,D,device=device),dim=-1).bfloat16()
    k = F.normalize(torch.randn(N,T,H,D,device=device),dim=-1).bfloat16()
    v = torch.randn(N,T,H,D,device=device).bfloat16()
    g = torch.randn(N,T,H,D,device=device).bfloat16()
    beta = torch.randn(N,T,H,device=device).bfloat16()
    A_log = torch.rand(H,device=device); dt_bias = torch.rand(H,D,device=device)
    h0 = (torch.randn(N,H,D,D,device=device)*0.1).bfloat16()   # [N,H,V,K]
    scale = 1/math.sqrt(D)
    out = torch.zeros_like(v); hT = torch.zeros_like(h0)
    flash_kda.fwd(q,k,v,g,beta,scale,out,A_log=A_log,dt_bias=dt_bias,lower_bound=lower_bound,initial_state=h0,final_state=hT)
    torch.cuda.synchronize()
    # (a) naive reference in ref_dtype
    gd = lambda x: x.to(ref_dtype)
    g_act = lower_bound*torch.sigmoid(torch.exp(gd(A_log)).view(1,1,H,1)*(gd(g)+gd(dt_bias).view(1,1,H,D)))
    o_ref, S_ref = naive_recurrent_kda(gd(q),gd(k),gd(v),g_act,torch.sigmoid(gd(beta)),scale,
                                       initial_state=gd(h0).transpose(-1,-2).contiguous(),output_final_state=True)
    S_ref = S_ref.transpose(-1,-2)
    print(f"[T={T} H={H} N={N} lb={lower_bound} seed={seed}] vs naive({ref_dtype})")
    r1 = err_stats("out  flash_kda vs naive", out, o_ref)
    r2 = err_stats("hT   flash_kda vs naive", hT, S_ref)
    # (b) fla Triton chunk_kda (fp32 state), same raw inputs
    with torch.inference_mode():
        import os; os.environ["FLA_FLASH_KDA"]="0"
        o_tri, S_tri = chunk_kda(q=q,k=k,v=v,g=g,beta=beta,scale=scale,initial_state=h0.float(),output_final_state=True,
                                 use_gate_in_kernel=True,use_qk_l2norm_in_kernel=True,use_beta_sigmoid_in_kernel=True,
                                 A_log=A_log,dt_bias=dt_bias,lower_bound=lower_bound,transpose_state_layout=True,safe_gate=True)
    err_stats("out  fla-triton vs naive", o_tri, o_ref)
    err_stats("hT   fla-triton vs naive", S_tri, S_ref)
    err_stats("out  flash_kda vs triton", out, o_tri)
    err_stats("hT   flash_kda vs triton", hT, S_tri)
    return r1, r2

if __name__ == "__main__":
    ok = True
    for (T,H,N) in [(64,4,1),(500,4,2),(1024,8,1),(2048,4,1)]:
        for lb in (-5.0, -1.0):
            r1,r2 = case(T,H,N=N,lower_bound=lb)
            ok &= r1 < 2e-2 and r2 < 2e-2
    print("E3 RESULT:", "PASS" if ok else "FAIL", "(rel_rms < 2e-2 vs fp64 naive)")
