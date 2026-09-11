"""E9: Design B — 分层(blocked)扫描版 K2,零 CUDA 改动的最小可行原型。

原理:chunk 间递推 S_i = M_i S_{i-1} + N_i 对 S 线性 => 一组 G 个 chunk 的复合也是仿射 T_g(S) = A_g S + B_g。
FlashKDA K2 支持 varlen + 每序列 initial_state,于是:
  phase 1a  B_g  = K2(group g, S_in = 0)            末状态          [一次 varlen 调用,NG 组并行]
  phase 1b  A_g  = K2(group g, v = 0, S_in = I)     末状态 (= A_g·I)[一次 varlen 调用]   -- 不显式 compose
  phase 2   S_in[g+1] = S_in[g] · A_g + B_g          NG-1 步 batched bmm(fp32)
  phase 3   out, S_T = K2(group g, S_in = S_in[g])   [一次 varlen 调用]
状态布局按 FlashKDA 的 [N,H,V,K](state_v_first):S^T 右乘 A^T,所以 phase 2 直接用 K2 吐出的矩阵右乘即可。
依赖深度 NT -> ≈ 2G + NG,CTA 数 ×NG,总算力 ≈ 3× 基线 + NG 条 128^3。
"""
import sys, math, time, torch, torch.nn.functional as F
import flash_kda

D = 128; CHUNK = 16

def make_inputs(B, T, H, device="cuda", seed=0):
    torch.manual_seed(seed)
    q = F.normalize(torch.randn(B, T, H, D, device=device), dim=-1).bfloat16()
    k = F.normalize(torch.randn(B, T, H, D, device=device), dim=-1).bfloat16()
    v = torch.randn(B, T, H, D, device=device).bfloat16()
    g = torch.randn(B, T, H, D, device=device).bfloat16()
    beta = torch.randn(B, T, H, device=device).bfloat16()
    A_log = torch.rand(H, device=device); dt_bias = torch.rand(H, D, device=device)
    h0 = (torch.randn(B, H, D, D, device=device) * 0.1).bfloat16()
    return q, k, v, g, beta, A_log, dt_bias, h0

class Hier:
    """把 [B,T] 切成 B*NG 条虚拟序列;缓冲区预分配,便于 CUDA graph / 计时。"""
    def __init__(self, B, T, H, G, lb=-5.0):
        self.B, self.T, self.H, self.G, self.lb = B, T, H, G, lb
        NT = math.ceil(T / CHUNK)
        NG = math.ceil(NT / G)
        self.NG = NG
        bounds = [min(g * G * CHUNK, T) for g in range(NG + 1)]
        # varlen 打包:序列 b 的组 g 在扁平 T 轴上的偏移 b*T + bounds[g]
        cu = []
        for b in range(B):
            for g in range(NG):
                cu.append(b * T + bounds[g])
        cu.append(B * T)
        self.cu = torch.tensor(cu, dtype=torch.long, device="cuda")
        N = B * NG
        self.N = N
        dev = "cuda"
        self.zero_state = torch.zeros(N, H, D, D, device=dev, dtype=torch.bfloat16)
        self.eye_state = torch.eye(D, device=dev, dtype=torch.bfloat16).expand(N, H, D, D).contiguous()
        self.hB = torch.zeros(N, H, D, D, device=dev, dtype=torch.bfloat16)
        self.hA = torch.zeros(N, H, D, D, device=dev, dtype=torch.bfloat16)
        self.hF = torch.zeros(N, H, D, D, device=dev, dtype=torch.bfloat16)
        self.S_in = torch.zeros(N, H, D, D, device=dev, dtype=torch.bfloat16)
        self.out_scratch = None
        self.p2_mode = "seq"          # "seq" | "tree"
        self.p2_dtype = torch.float32 # phase-2 GEMM dtype: float32 (SIMT) | bfloat16 (tensor core, 与基线 bf16 状态口径一致)
        self.merge_p1 = False         # True: 1a+1b 合成一次 2H-head 调用
        self.lookback = None          # None: phase 2 走链/树;W>=1: 只回看前 W 组,所有组并行(零跨组依赖)
        self.p2_graph = None
        self.h0_static = torch.zeros(B, H, D, D, device=dev, dtype=torch.bfloat16)
        self.state2 = torch.cat([self.zero_state, self.eye_state], 1).contiguous()   # [N, 2H, D, D]
        self.hAB = torch.zeros(N, 2 * H, D, D, device=dev, dtype=torch.bfloat16)

    def _phase2(self):
        B, H, NG, D_ = self.B, self.H, self.NG, D
        dt = self.p2_dtype
        if self.merge_p1:
            hB = self.hAB.view(B, NG, 2 * H, D_, D_)[:, :, :H].to(dt); hA = self.hAB.view(B, NG, 2 * H, D_, D_)[:, :, H:].to(dt)
        else:
            hA = self.hA.view(B, NG, H, D_, D_).to(dt); hB = self.hB.view(B, NG, H, D_, D_).to(dt)
        S_in = self.S_in.view(B, NG, H, D_, D_)
        if self.p2_mode == "seq":
            S = self.h0_static.to(dt)
            S_in[:, 0].copy_(S)
            for gi in range(NG - 1):
                S = torch.baddbmm(hB[:, gi].reshape(B * H, D_, D_), S.reshape(B * H, D_, D_), hA[:, gi].reshape(B * H, D_, D_)).view(B, H, D_, D_)
                S_in[:, gi + 1].copy_(S)
        else:
            # Hillis-Steele inclusive scan over (A_g, B_g),复合规则(右乘布局): (A1,B1)∘(A2,B2) = (A1@A2, B1@A2 + B2)
            A = hA.transpose(0, 1).reshape(NG, B * H, D_, D_).clone()   # [NG, BH, D, D]
            Bm = hB.transpose(0, 1).reshape(NG, B * H, D_, D_).clone()
            off = 1
            while off < NG:
                A2 = A[off:]; B2 = Bm[off:]; A1 = A[:-off]; B1 = Bm[:-off]
                nA = torch.matmul(A1, A2)
                nB = torch.baddbmm(B2.reshape(-1, D_, D_), B1.reshape(-1, D_, D_), A2.reshape(-1, D_, D_)).view_as(B2)
                A = torch.cat([A[:off], nA], 0); Bm = torch.cat([Bm[:off], nB], 0)
                off *= 2
            # 此时 (A[g], Bm[g]) = 组 0..g 的复合;S_in[g+1] = h0 @ A[g] + Bm[g]
            h0f = self.h0_static.to(dt).reshape(1, B * H, D_, D_)
            S_next = torch.matmul(h0f, A[:-1]) + Bm[:-1]                  # [NG-1, BH, D, D]
            S_in[:, 0].copy_(self.h0_static)
            S_in[:, 1:].copy_(S_next.view(NG - 1, B, H, D_, D_).transpose(0, 1))

    def capture_p2(self):
        """把 phase 2 抓成 CUDA graph(纯 torch 算子,输入是静态缓冲 hA/hB/h0_static)。"""
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(2): self._phase2()
        torch.cuda.current_stream().wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._phase2()
        self.p2_graph = g

    def run(self, q, k, v, g, beta, A_log, dt_bias, h0, scale, out, timing=None):
        B, T, H, G, NG, N = self.B, self.T, self.H, self.G, self.NG, self.N
        qf, kf, vf, gf, bf = [x.reshape(1, B * T, *x.shape[2:]) for x in (q, k, v, g, beta)]
        if self.out_scratch is None:
            self.out_scratch = torch.zeros_like(vf)
            self.v0 = torch.zeros_like(vf)
        outf = out.reshape(1, B * T, H, D)
        ev = {}
        def mark(name):
            if timing is not None:
                e = torch.cuda.Event(enable_timing=True); e.record(); ev[name] = e
        mark("t0")
        if self.lookback == 1:
            # W=1:S_in[g] = B_{g-1}(前一组从零状态跑出的末状态),不需要 A_g,也没有 phase 2 的链
            flash_kda.fwd(qf, kf, vf, gf, bf, scale, self.out_scratch, A_log=A_log, dt_bias=dt_bias, lower_bound=self.lb,
                          initial_state=self.zero_state, final_state=self.hB, cu_seqlens=self.cu)
            mark("p1a"); mark("p1b")
            S_in = self.S_in.view(B, NG, H, D, D); hB = self.hB.view(B, NG, H, D, D)
            S_in[:, 0].copy_(h0); S_in[:, 1:].copy_(hB[:, :-1])
            mark("p2")
            flash_kda.fwd(qf, kf, vf, gf, bf, scale, outf, A_log=A_log, dt_bias=dt_bias, lower_bound=self.lb,
                          initial_state=self.S_in, final_state=self.hF, cu_seqlens=self.cu)
            mark("p3")
            if timing is not None:
                torch.cuda.synchronize()
                timing["p1a"] = ev["t0"].elapsed_time(ev["p1a"]); timing["p1b"] = 0.0
                timing["p2"] = ev["p1b"].elapsed_time(ev["p2"]);  timing["p3"] = ev["p2"].elapsed_time(ev["p3"])
            return self.hF.view(B, NG, H, D, D)[:, -1]
        if self.merge_p1:
            # phase 1a+1b 合并:2H 个 head,前 H 个 (v, S_in=0) 给 B_g,后 H 个 (v=0, S_in=I) 给 A_g;一次调用,CTA 数 x2
            if not hasattr(self, "q2"):
                self.q2 = torch.cat([qf, qf], 2).contiguous(); self.k2 = torch.cat([kf, kf], 2).contiguous()
                self.g2 = torch.cat([gf, gf], 2).contiguous(); self.b2 = torch.cat([bf, bf], 2).contiguous()
                self.v2 = torch.cat([vf, torch.zeros_like(vf)], 2).contiguous()
                self.A2 = torch.cat([A_log, A_log]); self.dt2 = torch.cat([dt_bias, dt_bias], 0).contiguous()
                self.out2 = torch.zeros_like(self.v2)
            flash_kda.fwd(self.q2, self.k2, self.v2, self.g2, self.b2, scale, self.out2, A_log=self.A2, dt_bias=self.dt2,
                          lower_bound=self.lb, initial_state=self.state2, final_state=self.hAB, cu_seqlens=self.cu)
            mark("p1a"); mark("p1b")
        else:
            # phase 1a: B_g
            flash_kda.fwd(qf, kf, vf, gf, bf, scale, self.out_scratch, A_log=A_log, dt_bias=dt_bias, lower_bound=self.lb,
                          initial_state=self.zero_state, final_state=self.hB, cu_seqlens=self.cu)
            mark("p1a")
            # phase 1b: A_g (v=0 => u=0 => S_out = A_g I)
            flash_kda.fwd(qf, kf, self.v0, gf, bf, scale, self.out_scratch, A_log=A_log, dt_bias=dt_bias, lower_bound=self.lb,
                          initial_state=self.eye_state, final_state=self.hA, cu_seqlens=self.cu)
            mark("p1b")
        # phase 2: 组间前缀(fp32),状态布局 [V,K]:S_next^T = S^T A^T + B^T -> 右乘
        if self.lookback is not None and self.lookback >= 2:
            # W>=2:S_in[g] = B_{g-1} + B_{g-2}·A_{g-1} + ... + B_{g-W}·(A_{g-W+1}⋯A_{g-1}),g > W 的组之间无依赖(全并行);
            # 前 W 组按精确串行公式算(只有 W 步,带 h0)。右乘布局:S_in[g] = S_in[g-1]·hA[g-1] + hB[g-1]
            W = self.lookback; dt = self.p2_dtype
            hB = (self.hAB.view(B, NG, 2 * H, D, D)[:, :, :H] if self.merge_p1 else self.hB.view(B, NG, H, D, D)).to(dt)
            hA = (self.hAB.view(B, NG, 2 * H, D, D)[:, :, H:] if self.merge_p1 else self.hA.view(B, NG, H, D, D)).to(dt)
            S_in = self.S_in.view(B, NG, H, D, D)
            S = h0.to(dt); S_in[:, 0].copy_(S)
            for g in range(1, min(W, NG - 1) + 1):
                S = torch.matmul(S, hA[:, g - 1]) + hB[:, g - 1]; S_in[:, g].copy_(S)
            if NG - 1 > W:
                # g = W+1 .. NG-1(共 n = NG-1-W 个),各项切片长度都是 n
                acc = hB[:, W:NG - 1].clone()                          # w=1: B_{g-1}
                P = None
                for w in range(2, W + 1):
                    P = hA[:, W:NG - 1].clone() if w == 2 else torch.matmul(hA[:, W - w + 2:NG - w + 1], P)   # P_g = hA[g-w+1]⋯hA[g-1]
                    acc += torch.matmul(hB[:, W - w + 1:NG - w], P)    # B_{g-w} · P_g
                S_in[:, W + 1:].copy_(acc)
            mark("p2")
        elif self.p2_graph is not None:
            self.h0_static.copy_(h0)
            self.p2_graph.replay()
        else:
            self.h0_static.copy_(h0)
            self._phase2()
        mark("p2")
        # phase 3: replay
        flash_kda.fwd(qf, kf, vf, gf, bf, scale, outf, A_log=A_log, dt_bias=dt_bias, lower_bound=self.lb,
                      initial_state=self.S_in, final_state=self.hF, cu_seqlens=self.cu)
        mark("p3")
        if timing is not None:
            torch.cuda.synchronize()
            timing["p1a"] = ev["t0"].elapsed_time(ev["p1a"]); timing["p1b"] = ev["p1a"].elapsed_time(ev["p1b"])
            timing["p2"] = ev["p1b"].elapsed_time(ev["p2"]);  timing["p3"] = ev["p2"].elapsed_time(ev["p3"])
        return self.hF.view(B, NG, H, D, D)[:, -1]

def err(name, got, ref):
    got = got.float(); ref = ref.float()
    d = (got - ref).abs()
    rel = d / (ref.abs() + 1e-6)
    print(f"  {name:36s} max_abs={d.max().item():.3e} mean_abs={d.mean().item():.3e} max_rel={rel.max().item():.3e} rel_rms={(d.pow(2).mean().sqrt()/(ref.pow(2).mean().sqrt()+1e-30)).item():.3e}")

def baseline(q, k, v, g, beta, A_log, dt_bias, h0, scale, lb, out, hT):
    flash_kda.fwd(q, k, v, g, beta, scale, out, A_log=A_log, dt_bias=dt_bias, lower_bound=lb, initial_state=h0, final_state=hT)

def timeit(fn, warm=5, iters=50):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters

def check(B, T, H, G, lb=-5.0, p2="seq"):
    q, k, v, g, beta, A_log, dt_bias, h0 = make_inputs(B, T, H)
    scale = 1 / math.sqrt(D)
    out_ref = torch.zeros_like(v); hT_ref = torch.zeros_like(h0)
    baseline(q, k, v, g, beta, A_log, dt_bias, h0, scale, lb, out_ref, hT_ref)
    hier = Hier(B, T, H, G, lb)
    hier.merge_p1 = p2.startswith("merged:"); spec = p2.split(":")[-1]
    hier.p2_mode = "tree" if "tree" in spec else "seq"; hier.p2_dtype = torch.bfloat16 if "bf" in spec else torch.float32
    if "lb1" in spec: hier.lookback = 1
    if "lb2" in spec: hier.lookback = 2
    if "lb3" in spec: hier.lookback = 3
    out = torch.zeros_like(v)
    hT = hier.run(q, k, v, g, beta, A_log, dt_bias, h0, scale, out)
    torch.cuda.synchronize()
    # 衰减诊断:每组每维累积门控 exp(sum g_act),取 max(最弱衰减) -> 回看窗口判据
    g_act = lb * torch.sigmoid(torch.exp(A_log.float()).view(1, 1, H, 1) * (g.float() + dt_bias.float().view(1, 1, H, D)))
    NT = math.ceil(T / CHUNK); gsz = G * CHUNK
    worst = max(g_act[:, b0:b0 + gsz].sum(1).max().item() for b0 in range(0, T, gsz))
    print(f"[check B={B} T={T} H={H} G={G} NG={hier.NG} lb={lb} p2={p2}]  最弱衰减: 一组内 max_dim exp(sum g) = {math.exp(worst):.2e} (阈值 2^-9≈1.95e-3)")
    err("out  hier vs flash_kda", out, out_ref)
    err("hT   hier vs flash_kda", hT, hT_ref)
    # 对 fp64 朴素递推(小 T):看两者谁离真值更近
    if T <= 2048:
        from fla_kda_ref.naive import naive_recurrent_kda
        gd = lambda x: x.double()
        g_act = lb * torch.sigmoid(torch.exp(gd(A_log)).view(1, 1, H, 1) * (gd(g) + gd(dt_bias).view(1, 1, H, D)))
        o64, S64 = naive_recurrent_kda(gd(q), gd(k), gd(v), g_act, torch.sigmoid(gd(beta)), scale,
                                       initial_state=gd(h0).transpose(-1, -2).contiguous(), output_final_state=True)
        S64 = S64.transpose(-1, -2)
        err("out  flash_kda vs fp64", out_ref, o64); err("out  hier vs fp64", out, o64)
        err("hT   flash_kda vs fp64", hT_ref, S64);  err("hT   hier vs fp64", hT, S64)

def bench(B, T, H, Gs, lb=-5.0, iters=50, p2_modes=("seq32", "seqbf+graph", "merged:seqbf+graph")):
    q, k, v, g, beta, A_log, dt_bias, h0 = make_inputs(B, T, H)
    scale = 1 / math.sqrt(D)
    out_ref = torch.zeros_like(v); hT_ref = torch.zeros_like(h0)
    t_base = timeit(lambda: baseline(q, k, v, g, beta, A_log, dt_bias, h0, scale, lb, out_ref, hT_ref), iters=iters)
    print(f"\n===== B={B} T={T} H={H}  (B*H={B*H} CTA in baseline K2, NT={T//CHUNK}) =====")
    print(f"  baseline flash_kda.fwd(K1+K2): {t_base:8.1f} us")
    for G in Gs:
        for p2 in p2_modes:
            hier = Hier(B, T, H, G, lb)
            hier.merge_p1 = p2.startswith("merged:")
            spec = p2.split(":")[-1]
            hier.p2_mode = "tree" if "tree" in spec else "seq"
            hier.p2_dtype = torch.bfloat16 if "bf" in spec else torch.float32
            if "lb1" in spec: hier.lookback = 1
            if "lb2" in spec: hier.lookback = 2
            if "lb3" in spec: hier.lookback = 3
            out = torch.zeros_like(v)
            hier.run(q, k, v, g, beta, A_log, dt_bias, h0, scale, out)      # 分配缓冲
            if "graph" in spec and hier.lookback is None: hier.capture_p2()
            tm = {}
            hier.run(q, k, v, g, beta, A_log, dt_bias, h0, scale, out, timing=tm)
            t = timeit(lambda: hier.run(q, k, v, g, beta, A_log, dt_bias, h0, scale, out), iters=iters)
            d = (out.float() - out_ref.float()).abs()
            print(f"  G={G:3d} NG={hier.NG:4d} CTAs={B*hier.NG*H:6d} {p2:19s}: {t:8.1f} us  speedup {t_base/t:5.2f}x  "
                  f"| p1a {tm['p1a']*1e3:7.1f} p1b {tm['p1b']*1e3:7.1f} p2 {tm['p2']*1e3:7.1f} p3 {tm['p3']*1e3:7.1f} us"
                  f"  | out max_abs {d.max().item():.2e} mean {d.mean().item():.2e}")


def precision_sweep():
    """精度随 T 增长(hier vs baseline vs fp64)、分段调用状态传递、弱衰减下的计时不变性。"""
    from fla_kda_ref.naive import naive_recurrent_kda
    scale = 1 / math.sqrt(D)
    print("== 精度随 T(H=4, G=32, 与 fp64 递推对拍;lb=-0.01 组间携带 ~70% 状态) ==")
    for lb in (-0.01, -0.1):
        for T in (1024, 4096, 8192, 16384):
            B, H = 1, 4
            q, k, v, g, beta, A_log, dt_bias, h0 = make_inputs(B, T, H)
            out_ref = torch.zeros_like(v); hT_ref = torch.zeros_like(h0)
            baseline(q, k, v, g, beta, A_log, dt_bias, h0, scale, lb, out_ref, hT_ref)
            hier = Hier(B, T, H, 32, lb); hier.merge_p1 = True; hier.p2_dtype = torch.bfloat16
            out = torch.zeros_like(v); hT = hier.run(q, k, v, g, beta, A_log, dt_bias, h0, scale, out)
            gd = lambda x: x.double()
            g_act = lb * torch.sigmoid(torch.exp(gd(A_log)).view(1, 1, H, 1) * (gd(g) + gd(dt_bias).view(1, 1, H, D)))
            o64, S64 = naive_recurrent_kda(gd(q), gd(k), gd(v), g_act, torch.sigmoid(gd(beta)), scale,
                                           initial_state=gd(h0).transpose(-1, -2).contiguous(), output_final_state=True)
            S64 = S64.transpose(-1, -2)
            def rr(a, b):
                a = a.double(); b = b.double(); d = (a - b)
                return (d.pow(2).mean().sqrt() / (b.pow(2).mean().sqrt() + 1e-30)).item(), d.abs().max().item()
            ro, mo = rr(out, out_ref); rh, mh = rr(hT, hT_ref)
            bo64, _ = rr(out_ref, o64); ho64, _ = rr(out, o64); bh64, _ = rr(hT_ref, S64); hh64, _ = rr(hT, S64)
            print(f"  lb={lb:6} T={T:6d}: hier-vs-base out rel_rms {ro:.2e} (max {mo:.1e}), hT {rh:.2e} (max {mh:.1e}) | vs fp64: out base {bo64:.3e} hier {ho64:.3e}, hT base {bh64:.3e} hier {hh64:.3e}")
    print("== 分段调用:hier(T1) 的末状态作为 hier(T2) 的 h0,对比一次 baseline(T1+T2) ==")
    for lb in (-0.01, -0.1):
        B, H, T1, T2 = 1, 4, 2048, 2048
        q, k, v, g, beta, A_log, dt_bias, h0 = make_inputs(B, T1 + T2, H)
        out_ref = torch.zeros_like(v); hT_ref = torch.zeros_like(h0)
        baseline(q, k, v, g, beta, A_log, dt_bias, h0, scale, lb, out_ref, hT_ref)
        out = torch.zeros_like(v)
        h1 = Hier(B, T1, H, 32, lb); h1.merge_p1 = True; h1.p2_dtype = torch.bfloat16
        S1 = h1.run(q[:, :T1].contiguous(), k[:, :T1].contiguous(), v[:, :T1].contiguous(), g[:, :T1].contiguous(), beta[:, :T1].contiguous(), A_log, dt_bias, h0, scale, out[:, :T1])
        o1 = out[:, :T1].clone()
        h2 = Hier(B, T2, H, 32, lb); h2.merge_p1 = True; h2.p2_dtype = torch.bfloat16
        o2 = torch.zeros(B, T2, H, D, device="cuda", dtype=torch.bfloat16)
        S2 = h2.run(q[:, T1:].contiguous(), k[:, T1:].contiguous(), v[:, T1:].contiguous(), g[:, T1:].contiguous(), beta[:, T1:].contiguous(), A_log, dt_bias, S1.contiguous(), scale, o2)
        o_cat = torch.cat([o1, o2], 1)
        err(f"  lb={lb} out  seg(hier) vs baseline(T1+T2)", o_cat, out_ref)
        err(f"  lb={lb} hT   seg(hier) vs baseline(T1+T2)", S2, hT_ref)
    print("== 计时与 lb 无关(B=1 H=12 T=8192 G=32,lb=-1 vs -0.01) ==")
    for lb in (-1.0, -0.01):
        B, T, H = 1, 8192, 12
        q, k, v, g, beta, A_log, dt_bias, h0 = make_inputs(B, T, H)
        out_ref = torch.zeros_like(v); hT_ref = torch.zeros_like(h0)
        tb = timeit(lambda: baseline(q, k, v, g, beta, A_log, dt_bias, h0, scale, lb, out_ref, hT_ref))
        hier = Hier(B, T, H, 32, lb); hier.merge_p1 = True; hier.p2_dtype = torch.bfloat16
        out = torch.zeros_like(v); hier.run(q, k, v, g, beta, A_log, dt_bias, h0, scale, out); hier.capture_p2()
        th = timeit(lambda: hier.run(q, k, v, g, beta, A_log, dt_bias, h0, scale, out))
        print(f"  lb={lb}: baseline {tb:.1f} us, hier {th:.1f} us, {tb/th:.2f}x")

def precision_lookback():
    """W=1/W=2 回看版的精度:对 fp64 真值,与原版、树版逐项对比;随 T 与 lb 扫;含衰减判据、分段调用。"""
    from fla_kda_ref.naive import naive_recurrent_kda
    scale = 1 / math.sqrt(D)
    def rr(a, b):
        a = a.double(); b = b.double(); d = (a - b)
        return (d.pow(2).mean().sqrt() / (b.pow(2).mean().sqrt() + 1e-30)).item()
    def mk(B, T, H, G, lb, lookback=None):
        h = Hier(B, T, H, G, lb); h.merge_p1 = True; h.p2_dtype = torch.bfloat16; h.lookback = lookback; return h
    print("== 对 fp64 真值:原版 / 树版 / W=1 / W=2(H=4,随 T 与 lb;'衰减'= 所有组里最弱的一组 max_dim exp(sum g)) ==")
    print(f"  {'lb':>6} {'G':>3} {'T':>6} {'衰减':>9} | out: base / tree / W1 / W2      | hT: base / tree / W1 / W2      | W1-vs-base out / hT")
    for lb in (-5.0, -1.0, -0.1, -0.01):
        for G in (32, 64):
            for T in (1024, 4096, 8192, 16384):
                if G == 64 and T not in (4096, 16384): continue
                B, H = 1, 4
                q, k, v, g, beta, A_log, dt_bias, h0 = make_inputs(B, T, H)
                out_ref = torch.zeros_like(v); hT_ref = torch.zeros_like(h0)
                baseline(q, k, v, g, beta, A_log, dt_bias, h0, scale, lb, out_ref, hT_ref)
                res = {}
                for name, lk in (("tree", None), ("W1", 1), ("W2", 2)):
                    h = mk(B, T, H, G, lb, lk); o = torch.zeros_like(v); S = h.run(q, k, v, g, beta, A_log, dt_bias, h0, scale, o)
                    torch.cuda.synchronize(); res[name] = (o, S)
                gd = lambda x: x.double()
                g_act = lb * torch.sigmoid(torch.exp(gd(A_log)).view(1, 1, H, 1) * (gd(g) + gd(dt_bias).view(1, 1, H, D)))
                o64, S64 = naive_recurrent_kda(gd(q), gd(k), gd(v), g_act, torch.sigmoid(gd(beta)), scale,
                                               initial_state=gd(h0).transpose(-1, -2).contiguous(), output_final_state=True)
                S64 = S64.transpose(-1, -2)
                gsz = G * CHUNK
                worst = max(g_act[:, b0:b0 + gsz].sum(1).max().item() for b0 in range(0, T, gsz))
                eo = [rr(out_ref, o64)] + [rr(res[n][0], o64) for n in ("tree", "W1", "W2")]
                eh = [rr(hT_ref, S64)] + [rr(res[n][1], S64) for n in ("tree", "W1", "W2")]
                flag = "" if math.exp(worst) < 2 ** -9 else "  <- 判据不成立,W=1 不该启用"
                print(f"  {lb:6} {G:3d} {T:6d} {math.exp(worst):9.2e} | " + " / ".join(f"{x:.2e}" for x in eo) + " | " +
                      " / ".join(f"{x:.2e}" for x in eh) + f" | {rr(res['W1'][0], out_ref):.1e} / {rr(res['W1'][1], hT_ref):.1e}{flag}")
    print("== 分段调用:W=1(T1) 末状态作为 W=1(T2) 的 h0,对比一次 baseline(T1+T2) 与 fp64 ==")
    for lb in (-1.0, -0.1):
        B, H, T1, T2 = 1, 4, 2048, 2048
        q, k, v, g, beta, A_log, dt_bias, h0 = make_inputs(B, T1 + T2, H)
        out_ref = torch.zeros_like(v); hT_ref = torch.zeros_like(h0)
        baseline(q, k, v, g, beta, A_log, dt_bias, h0, scale, lb, out_ref, hT_ref)
        c = lambda x: x.contiguous()
        out = torch.zeros_like(v)
        S1 = mk(B, T1, H, 32, lb, 1).run(c(q[:, :T1]), c(k[:, :T1]), c(v[:, :T1]), c(g[:, :T1]), c(beta[:, :T1]), A_log, dt_bias, h0, scale, out[:, :T1])
        o1 = out[:, :T1].clone(); o2 = torch.zeros(B, T2, H, D, device="cuda", dtype=torch.bfloat16)
        S2 = mk(B, T2, H, 32, lb, 1).run(c(q[:, T1:]), c(k[:, T1:]), c(v[:, T1:]), c(g[:, T1:]), c(beta[:, T1:]), A_log, dt_bias, S1.contiguous(), scale, o2)
        o_cat = torch.cat([o1, o2], 1)
        gd = lambda x: x.double()
        g_act = lb * torch.sigmoid(torch.exp(gd(A_log)).view(1, 1, H, 1) * (gd(g) + gd(dt_bias).view(1, 1, H, D)))
        o64, S64 = naive_recurrent_kda(gd(q), gd(k), gd(v), g_act, torch.sigmoid(gd(beta)), scale,
                                       initial_state=gd(h0).transpose(-1, -2).contiguous(), output_final_state=True)
        S64 = S64.transpose(-1, -2)
        print(f"  lb={lb}: out seg(W1) vs base {rr(o_cat, out_ref):.2e}, vs fp64 {rr(o_cat, o64):.2e} (base vs fp64 {rr(out_ref, o64):.2e}) | "
              f"hT seg(W1) vs base {rr(S2, hT_ref):.2e}, vs fp64 {rr(S2, S64):.2e} (base vs fp64 {rr(hT_ref, S64):.2e})")
    print("== 逐位一致性:lb=-5(模型真实门控),B=2 H=12 T=8192 G=32/64,W=1 vs 原版 ==")
    for G in (32, 64):
        B, T, H = 2, 8192, 12
        q, k, v, g, beta, A_log, dt_bias, h0 = make_inputs(B, T, H)
        out_ref = torch.zeros_like(v); hT_ref = torch.zeros_like(h0)
        baseline(q, k, v, g, beta, A_log, dt_bias, h0, scale, -5.0, out_ref, hT_ref)
        o = torch.zeros_like(v); S = mk(B, T, H, G, -5.0, 1).run(q, k, v, g, beta, A_log, dt_bias, h0, scale, o); torch.cuda.synchronize()
        print(f"  G={G}: out 不同元素数 {(o != out_ref).sum().item()} / {o.numel()}, hT 不同元素数 {(S != hT_ref).sum().item()} / {S.numel()}")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "check":
        # lb=-5 时每 token 衰减 ~e^-3,A_g 在一个组内就下溢到 0,测不出 phase 1b/2 对不对;必须加弱衰减用例
        for lb in (-5.0, -1.0, -0.1, -0.01):
            for (B, T, H, G) in [(1, 2048, 4, 8), (1, 8192, 12, 32)]:
                for p2 in ("merged:seqbf", "lb1", "merged:lb2", "merged:lb3"):
                    check(B, T, H, G, lb=lb, p2=p2)
    elif mode == "precision":
        precision_sweep()
    elif mode == "precision_lb":
        precision_lookback()
    elif mode == "bench":
        Gs = [32, 64]
        for (B, T, H) in [(1, 8192, 12), (1, 32768, 12), (1, 65536, 16), (1, 16384, 32)]:
            bench(B, T, H, Gs, lb=-1.0, p2_modes=("merged:seqbf+graph", "lb1", "merged:lb2"))
    elif mode == "sweep":   # brief 里的负载矩阵
        for B in (1, 2, 4):
            for H in (16, 32, 64):
                for T in (4096, 16384, 65536):
                    if B * T * H * D * 2 * 5 > 40e9: continue
                    bench(B, T, H, [32, 64], iters=20, lb=-1.0, p2_modes=("merged:seqbf+graph",))
