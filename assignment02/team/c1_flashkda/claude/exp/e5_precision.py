"""E5 (discussion point 5): validation design + data for the bf16 on-chip recurrent state.

Reference chain: fp64 naive recurrence (fla_kda_ref/naive.py, token-by-token, exact algorithm) is the gold.
Contrast: fla Triton chunk_kda keeps the state in fp32 (same bf16 inputs), so (flash_kda - gold) vs
(triton - gold) isolates the cost of bf16 state storage from bf16 input rounding.

Experiments (all with L2-normalised q/k, sigmoid beta, gate = lb*sigmoid(exp(A_log)*(g+dt_bias)) as in-kernel):
  A. sequence-length sweep T in {512 .. 16384}, lb=-5 (strong decay) : does error grow with T?
  B. decay-strength sweep: lb in {-5,-1,-0.1,-0.01} at T=4096 (weak decay = long memory = worst case for a
     bf16 accumulator, the state is a sum of ~T rank-1 updates).
  C. windowed error along T (first 256 tokens vs last 256) to see whether error is amortised or accumulates.
  D. chained calls (prefill in 512-token pieces, state passed bf16 vs fp32 between calls, 16 pieces) vs one call.
  E. adversarial scale: v magnitudes 1 vs 64 (bf16 has 8 mantissa bits -> relative, not absolute, error).
Output: rel_rms and max_abs for out / final_state, plus a PASS/FAIL against a tolerance derived from the
Triton-fp32-state contrast (flash_kda error must stay within 3x of Triton's error).
"""
import math, torch, torch.nn.functional as F, os
import flash_kda
from fla_kda_ref.naive import naive_recurrent_kda
os.environ["FLA_FLASH_KDA"] = "0"
from fla.ops.kda import chunk_kda

dev = "cuda"

def rel(a, b):
    a = a.double(); b = b.double()
    return ((a - b).flatten().square().mean().sqrt() / (b.flatten().square().mean().sqrt() + 1e-30)).item()

def inputs(T, H, D=128, N=1, seed=0, vscale=1.0, gate_center=0.0):
    g_ = torch.Generator(device=dev).manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=dev, generator=g_)
    q = F.normalize(r(N, T, H, D), dim=-1).bfloat16(); k = F.normalize(r(N, T, H, D), dim=-1).bfloat16()
    v = (r(N, T, H, D) * vscale).bfloat16(); g = (r(N, T, H, D) + gate_center).bfloat16(); beta = r(N, T, H).bfloat16()
    A_log = torch.rand(H, device=dev, generator=g_); dt_bias = torch.rand(H, D, device=dev, generator=g_)
    h0 = (r(N, H, D, D) * 0.1).bfloat16()
    return q, k, v, g, beta, A_log, dt_bias, h0

def gold(q, k, v, g, beta, A_log, dt_bias, h0, lb, scale):
    H, D = q.shape[2], q.shape[3]
    f = lambda x: x.double()
    g_act = lb * torch.sigmoid(torch.exp(f(A_log)).view(1, 1, H, 1) * (f(g) + f(dt_bias).view(1, 1, H, D)))
    o, S = naive_recurrent_kda(f(q), f(k), f(v), g_act, torch.sigmoid(f(beta)), scale,
                               initial_state=f(h0).transpose(-1, -2).contiguous(), output_final_state=True)
    return o, S.transpose(-1, -2)

def run_flash(q, k, v, g, beta, A_log, dt_bias, h0, lb, scale, state_dtype=torch.bfloat16):
    out = torch.zeros_like(v); hT = torch.zeros_like(h0).to(state_dtype)
    flash_kda.fwd(q, k, v, g, beta, scale, out, A_log=A_log, dt_bias=dt_bias, lower_bound=lb,
                  initial_state=h0.to(state_dtype), final_state=hT)
    return out, hT

def run_triton(q, k, v, g, beta, A_log, dt_bias, h0, lb, scale):
    with torch.inference_mode():
        o, S = chunk_kda(q=q, k=k, v=v, g=g, beta=beta, scale=scale, initial_state=h0.float(), output_final_state=True,
                         use_gate_in_kernel=True, use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                         A_log=A_log, dt_bias=dt_bias, lower_bound=lb, transpose_state_layout=True, safe_gate=True)
    return o, S

def compare(tag, T, H, lb, seed=0, vscale=1.0, gate_center=0.0, windows=False):
    q, k, v, g, beta, A_log, dt_bias, h0 = inputs(T, H, seed=seed, vscale=vscale, gate_center=gate_center)
    scale = 1 / math.sqrt(128)
    og, Sg = gold(q, k, v, g, beta, A_log, dt_bias, h0, lb, scale)
    of, Sf = run_flash(q, k, v, g, beta, A_log, dt_bias, h0, lb, scale)
    ot, St = run_triton(q, k, v, g, beta, A_log, dt_bias, h0, lb, scale)
    r = dict(out_flash=rel(of, og), out_tri=rel(ot, og), hT_flash=rel(Sf, Sg), hT_tri=rel(St, Sg),
             hT_maxabs_flash=(Sf.double() - Sg).abs().max().item(), hT_maxabs_tri=(St.double() - Sg).abs().max().item())
    line = f"{tag:34s} T={T:6d} lb={lb:6.2f} | out rel: flash {r['out_flash']:.2e} tri {r['out_tri']:.2e} | hT rel: flash {r['hT_flash']:.2e} tri {r['hT_tri']:.2e} | hT maxabs flash {r['hT_maxabs_flash']:.2e} tri {r['hT_maxabs_tri']:.2e}"
    if windows:
        w = 256
        wf = [rel(of[:, s:s + w], og[:, s:s + w]) for s in (0, T // 2, T - w)]
        wt = [rel(ot[:, s:s + w], og[:, s:s + w]) for s in (0, T // 2, T - w)]
        line += f" | out windows(first/mid/last) flash {wf[0]:.2e}/{wf[1]:.2e}/{wf[2]:.2e} tri {wt[0]:.2e}/{wt[1]:.2e}/{wt[2]:.2e}"
    print(line, flush=True)
    return r

def chained(T=8192, piece=512, H=8, lb=-1.0, seed=0):
    q, k, v, g, beta, A_log, dt_bias, h0 = inputs(T, H, seed=seed)
    scale = 1 / math.sqrt(128)
    og, Sg = gold(q, k, v, g, beta, A_log, dt_bias, h0, lb, scale)
    of1, Sf1 = run_flash(q, k, v, g, beta, A_log, dt_bias, h0, lb, scale)
    for sd in (torch.bfloat16, torch.float32):
        h = h0.to(sd); outs = []
        for s in range(0, T, piece):
            sl = slice(s, s + piece)
            out = torch.zeros_like(v[:, sl]); hT = torch.zeros_like(h)
            flash_kda.fwd(q[:, sl].contiguous(), k[:, sl].contiguous(), v[:, sl].contiguous(), g[:, sl].contiguous(),
                          beta[:, sl].contiguous(), scale, out, A_log=A_log, dt_bias=dt_bias, lower_bound=lb,
                          initial_state=h, final_state=hT)
            outs.append(out); h = hT
        o = torch.cat(outs, 1)
        print(f"chained {T//piece}x{piece} state-io={str(sd)[6:]:8s} lb={lb} | out rel {rel(o, og):.2e} (single call {rel(of1, og):.2e}) | hT rel {rel(h, Sg):.2e} (single {rel(Sf1, Sg):.2e})", flush=True)

if __name__ == "__main__":
    torch.manual_seed(0)
    H = 8
    print("== A. T sweep, lb=-5 ==")
    resA = [compare("A", T, H, -5.0, windows=True) for T in (512, 2048, 8192, 16384)]
    print("== B. decay strength sweep, T=4096 ==")
    resB = [compare("B", 4096, H, lb, windows=True) for lb in (-5.0, -1.0, -0.1, -0.01)]
    print("== B'. weak decay + gate pushed towards 0 (g_raw center +6 -> sigmoid~1 -> gate~lb) vs -6 (gate~0, no decay) ==")
    for gc in (6.0, -6.0):
        compare(f"B' gate_center={gc:+.0f}", 4096, H, -1.0, gate_center=gc, windows=True)
    print("== E. value scale ==")
    for vs in (1.0, 64.0):
        compare(f"E vscale={vs}", 4096, H, -1.0, vscale=vs)
    print("== D. chained calls ==")
    chained(lb=-1.0); chained(lb=-0.01)
    worst = max(r["hT_flash"] / max(r["hT_tri"], 1e-12) for r in resA + resB)
    print(f"E5 RESULT: max(flash hT err / triton hT err) over A+B = {worst:.2f}  ->", "PASS (<3x)" if worst < 3 else "FAIL")
