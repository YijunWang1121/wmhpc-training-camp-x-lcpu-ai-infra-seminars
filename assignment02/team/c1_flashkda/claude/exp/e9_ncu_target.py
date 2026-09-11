"""ncu 目标:baseline K2(B=1,H=12,T=8192 -> 12 个 CTA,512 chunk 串行)vs 分层版 phase 3 的 K2(192 个 CTA,各 32 chunk)。
用法: ncu -k regex:_flash_kda_fwd_recurrence --launch-skip 0 -c 2 ... python e9_ncu_target.py
第 1 次 recurrence launch = baseline,第 2 次 = 分层版 phase 1a(与 phase 3 同形,都是 varlen 192 CTA)。"""
import sys, math, torch
sys.argv = [sys.argv[0]]
import e9_hier as e
B, T, H, G = 1, 8192, 12, 32
q, k, v, g, beta, A_log, dt_bias, h0 = e.make_inputs(B, T, H)
scale = 1 / math.sqrt(e.D); lb = -1.0
out = torch.zeros_like(v); hT = torch.zeros_like(h0)
e.baseline(q, k, v, g, beta, A_log, dt_bias, h0, scale, lb, out, hT)
torch.cuda.synchronize()
hier = e.Hier(B, T, H, G, lb); hier.p2_mode = "tree"
out2 = torch.zeros_like(v)
hier.run(q, k, v, g, beta, A_log, dt_bias, h0, scale, out2)
torch.cuda.synchronize()
print("done")
