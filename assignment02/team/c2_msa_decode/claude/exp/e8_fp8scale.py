"""E8 (discussion point 4): where to apply the FP8 KV scale.
Upstream semantics (test_sparse_attn_fp8_scale.py): K = kv_fp8[..., :128] * k_scale, V = kv_fp8[..., 128:] * v_scale.
Compare three placements against the dequantized-bf16 reference run through the same Triton kernel:
  (A) in-kernel, per element after load  = Triton USE_FP8 + KV_SCALE_MODE=1 (upstream)
  (B) host fold: q' = q * k_scale, out = v_scale * kernel(q', kv_fp8, no scale)   [zero kernel cost, scalar mode only]
  (C) per-token scales (mode 2) via Triton (upstream) vs the same folded onto S/P (emulated in torch fp32 on the SDPA path)
Tolerance: upstream's atol=rtol=2e-2 against the dequantized reference."""
import torch, math
from common import *
torch.manual_seed(0)
K_SCALE, V_SCALE = 0.3, 0.7   # deliberately not powers of two so rounding differences show
fp8 = torch.float8_e4m3fn

def fp8_case(b, seq, seed):
    case = make_case(num_reqs=b, seq_range=seq, seed=seed)
    kv = case["kv_cache"].float()
    kv8 = torch.empty(kv.shape, dtype=fp8, device=kv.device)
    kv8[..., :128] = (kv[..., :128] / K_SCALE).to(fp8)
    kv8[..., 128:] = (kv[..., 128:] / V_SCALE).to(fp8)
    deq = torch.empty_like(case["kv_cache"])
    deq[..., :128] = (kv8[..., :128].float() * K_SCALE).to(torch.bfloat16)
    deq[..., 128:] = (kv8[..., 128:].float() * V_SCALE).to(torch.bfloat16)
    return case, kv8, deq

def run(case, kv, k_scale=None, v_scale=None, q=None):
    out = torch.empty_like(case["q"])
    sa.minimax_m3_sparse_attn_decode(case["q"] if q is None else q, kv, case["topk_idx"], case["block_table"],
                                     case["seq_lens"], case["num_kv_heads"], case["sm_scale"], out,
                                     case["decode_query_len"], k_scale=k_scale, v_scale=v_scale)
    return out

for b, seq in ((1, (1024, 8192)), (4, (50, 300)), (8, (8192, 8192))):
    case, kv8, deq = fp8_case(b, seq, 0)
    ref = run(case, deq)                                            # dequantized bf16 through the same kernel
    ks = torch.tensor(K_SCALE, device="cuda"); vs = torch.tensor(V_SCALE, device="cuda")
    A = run(case, kv8, ks, vs)                                      # upstream in-kernel scalar mode
    B = V_SCALE * run(case, kv8, q=(case["q"].float() * K_SCALE).to(torch.bfloat16)).float()   # host fold
    U = run(case, kv8)                                              # no scale at all (must differ)
    def chk(x, name):
        ok = torch.allclose(x.float(), ref.float(), rtol=2e-2, atol=2e-2)
        print(f"  b={b} seq={seq} {name:34s} max|diff|={(x.float()-ref.float()).abs().max().item():.3e} err_ratio={err_ratio(x, ref):.2e} {'OK' if ok else 'FAIL'}")
    chk(A, "(A) in-kernel scalar (upstream)"); chk(B, "(B) host fold into q / epilogue"); chk(U, "unscaled (expected FAIL)")
    # (C) per-token/head scales: upstream mode 2
    nblk = kv8.shape[0]
    kst = torch.full((case["num_kv_heads"], nblk * 128), K_SCALE, device="cuda"); vst = torch.full_like(kst, V_SCALE)
    C = run(case, kv8, kst, vst)
    chk(C, "(C) in-kernel per-token (upstream)")
    # timing: does the in-kernel scale cost anything? (graph, us)
    t_bf16 = time_graph(lambda: run(case, deq)); t_A = time_graph(lambda: run(case, kv8, ks, vs)); t_C = time_graph(lambda: run(case, kv8, kst, vst)); t_U = time_graph(lambda: run(case, kv8))
    print(f"  b={b} timing us: bf16 {t_bf16:.1f} | fp8 no-scale {t_U:.1f} | fp8 scalar {t_A:.1f} | fp8 per-token {t_C:.1f}")
