"""E8: 把 K2 的 chunk 间状态递推改成前缀扫描(REPORT §3.3.1 的实现与验证)。

数学(记号同 fla_kda_ref/naive.py naive_chunk_kda 第 160-163 行):
    v_i = u_i - w_i @ S_{i-1}
    S_i = diag(exp(g_i[-1])) @ S_{i-1} + K_i'^T @ v_i,  K_i' = exp(g_i[-1]-g_i) * k_i
  => S_i = M_i @ S_{i-1} + N_i,  M_i = diag(exp(g_i[-1])) - K_i'^T @ w_i,  N_i = K_i'^T @ u_i
M_i / N_i 只依赖本 chunk,可对全部 chunk 一次性并行算;递推是仿射变换复合,满足结合律,
用两级分块扫描代替 NT 步串行:
    level-1: NG 组 × 组内 G 步串行 batched compose(batch = B*HV*NG)
    level-2: NG 步串行,把 h0 沿组传播(batch = B*HV)
    level-3: 一次 batched apply 得到每个 chunk 之后的状态
深度 (G-1)+(NG-1)+1 而不是 NT;总算力 ≈ 3×NT 条 128^3 GEMM(串行版从不材料化 M_i)。

用法:
    python e8_scan.py check          # CPU/GPU 小形状 fp64 对拍 naive_chunk_kda / naive_recurrent_kda
    python e8_scan.py bench          # GPU:对拍 flash_kda + 计时(需要 flash_kda)
"""
import sys, math, time, torch
from einops import rearrange
from fla_kda_ref.naive import naive_chunk_kda, naive_recurrent_kda

# ----------------------------------------------------------------------------- 预计算(= naive_chunk_kda 循环之前的部分,K1 的工作)
def precompute(q, k, v, g, beta, scale, BT):
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    G = HV // H
    NT = T // BT
    q, k = [rearrange(x, 'b (n c) h ... -> b h n c ...', c=BT) for x in [q, k]]
    v, g, beta = [rearrange(x, 'b (n c) h ... -> b h n c ...', c=BT) for x in [v, g, beta]]
    q = q.repeat_interleave(G, dim=1) * scale
    k = k.repeat_interleave(G, dim=1)
    g = g.cumsum(-2)
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)
    A = torch.zeros(*g.shape[:-1], BT, dtype=q.dtype, device=q.device)
    for i in range(BT):
        A[..., i] = torch.einsum('... c d, ... d -> ... c', k * (g - g[..., i:i+1, :]).exp(), k[..., i, :])
    A = A * beta[..., None]
    A = -A.masked_fill(mask, 0)
    for i in range(1, BT):
        A[..., i, :i] = A[..., i, :i].clone() + (A[..., i, :, None].clone() * A[..., :, :i].clone()).sum(-2)
    A = (A + torch.eye(BT, dtype=q.dtype, device=q.device)) * beta[..., None, :]
    w = A @ (g.exp() * k)
    u = A @ v
    # Aqk(chunk 内 causal 注意力矩阵,与 S 无关)
    mask1 = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)
    Aqk = torch.zeros(B, HV, NT, BT, BT, dtype=q.dtype, device=q.device)
    for j in range(BT):
        Aqk[..., j] = torch.einsum('... c d, ... d -> ... c', q * (g - g[..., j:j+1, :]).exp(), k[..., j, :])
    Aqk = Aqk.masked_fill(mask1, 0)
    return q, k, v, g, w, u, Aqk

# ----------------------------------------------------------------------------- M_i, N_i
def build_MN(k, g, w, u, dtype=None):
    """k,g,w:[B,HV,NT,BT,K], u:[B,HV,NT,BT,V] -> M:[B,HV,NT,K,K], N:[B,HV,NT,K,V]"""
    g_last = g[..., -1:, :]
    Kp = (g_last - g).exp() * k                              # [.., BT, K]
    KpT = Kp.transpose(-1, -2)                               # [.., K, BT]
    if dtype is not None:
        KpT, w, u = KpT.to(dtype), w.to(dtype), u.to(dtype)
    M = -(KpT @ w)                                           # [.., K, K]
    D = g_last[..., 0, :].exp().to(M.dtype)                  # [.., K]
    M = M + torch.diag_embed(D)
    N = KpT @ u                                              # [.., K, V]
    return M, N

# ----------------------------------------------------------------------------- 扫描
def scan_states(M, N, h0, G):
    """M:[B,HV,NT,K,K], N:[B,HV,NT,K,V], h0:[B,HV,K,V] -> S_after:[B,HV,NT,K,V](每个 chunk 之后的状态)"""
    B, HV, NT, K, _ = M.shape
    V = N.shape[-1]
    assert NT % G == 0
    NG = NT // G
    Mg = M.reshape(B * HV * NG, G, K, K)
    Ng = N.reshape(B * HV * NG, G, K, V)
    # level-1: 组内 inclusive scan(串行 G-1 步,batched)
    P = Mg[:, 0]; Q = Ng[:, 0]
    Ps = [P]; Qs = [Q]
    for j in range(1, G):
        Mj = Mg[:, j]
        P = Mj @ P
        Q = Mj @ Q + Ng[:, j]
        Ps.append(P); Qs.append(Q)
    P_in = torch.stack(Ps, 1)                                # [B*HV*NG, G, K, K]
    Q_in = torch.stack(Qs, 1)                                # [B*HV*NG, G, K, V]
    # level-2: 组间把 h0 串行传播(NG-1 步,batch B*HV)
    Pt = P_in[:, -1].reshape(B * HV, NG, K, K)
    Qt = Q_in[:, -1].reshape(B * HV, NG, K, V)
    Sg = h0.reshape(B * HV, K, V).to(M.dtype)
    Sgs = [Sg]
    for gi in range(NG - 1):
        Sg = Pt[:, gi] @ Sg + Qt[:, gi]
        Sgs.append(Sg)
    Sg_start = torch.stack(Sgs, 1).reshape(B * HV * NG, 1, K, V)   # 每组起始状态
    # level-3: batched apply
    S_after = P_in @ Sg_start + Q_in                         # [B*HV*NG, G, K, V]
    return S_after.reshape(B, HV, NT, K, V)

def scan_states_v2(M, N, h0, G, work=None):
    """同 scan_states,但去掉 torch 层面的低效:level-1 每步的 batch 切片改成连续(先把 M/N 转成 [G, B*HV*NG, K, K]),
    Mj@Q+N 用 baddbmm 一条 kernel,P_in/Q_in 预分配、bmm 直接 out= 写入,不再 torch.stack 复制。"""
    B, HV, NT, K, _ = M.shape
    V = N.shape[-1]
    NG = NT // G
    R = B * HV * NG
    Mg = M.reshape(R, G, K, K).transpose(0, 1)               # [G, R, K, K](view)
    Ng = N.reshape(R, G, K, V).transpose(0, 1)
    if work is None:
        work = (torch.empty(G, R, K, K, dtype=M.dtype, device=M.device),
                torch.empty(G, R, K, V, dtype=M.dtype, device=M.device))
    P_in, Q_in = work
    Mg_c = Mg.contiguous(); Ng_c = Ng.contiguous()           # 一次性整理成连续(生产实现里 build_MN 直接按这个布局写)
    P_in[0].copy_(Mg_c[0]); Q_in[0].copy_(Ng_c[0])
    for j in range(1, G):
        torch.bmm(Mg_c[j], P_in[j - 1], out=P_in[j])
        torch.baddbmm(Ng_c[j], Mg_c[j], Q_in[j - 1], out=Q_in[j])
    Pt = P_in[G - 1].reshape(B * HV, NG, K, K)
    Qt = Q_in[G - 1].reshape(B * HV, NG, K, V)
    Sg = h0.reshape(B * HV, K, V).to(M.dtype)
    Sg_start = torch.empty(NG, B * HV, K, V, dtype=M.dtype, device=M.device)
    Sg_start[0].copy_(Sg)
    for gi in range(NG - 1):
        torch.baddbmm(Qt[:, gi], Pt[:, gi], Sg_start[gi], out=Sg_start[gi + 1])
    Sg_b = Sg_start.transpose(0, 1).reshape(R, 1, K, V)      # [R,1,K,V]
    S_after = torch.baddbmm(Q_in.transpose(0, 1).reshape(R * G, K, V),
                            P_in.transpose(0, 1).reshape(R * G, K, K),
                            Sg_b.expand(R, G, K, V).reshape(R * G, K, V))
    return S_after.reshape(B, HV, NT, K, V)

def scan_states_serial(M, N, h0):
    """对照:同样用 M_i/N_i,但 NT 步串行。"""
    B, HV, NT, K, _ = M.shape
    S = h0.to(M.dtype)
    out = []
    for i in range(NT):
        S = M[:, :, i] @ S + N[:, :, i]
        out.append(S)
    return torch.stack(out, 2)

# ----------------------------------------------------------------------------- 输出阶段(= K2 的 P1/P3/P4 那部分工作,现在全 chunk batched)
def outputs(q, g, w, u, Aqk, S_before, out_dtype):
    """S_before:[B,HV,NT,K,V] 每 chunk 之前的状态"""
    S_before = S_before.to(q.dtype)
    v_new = u - w @ S_before                                 # [B,HV,NT,BT,V]
    o = (q * g.exp()) @ S_before + Aqk @ v_new
    return rearrange(o, 'b h n c d -> b (n c) h d').to(out_dtype)

def scan_kda(q, k, v, g, beta, scale=None, initial_state=None, chunk_size=16, G=32,
             mn_dtype=None, serial=False):
    """与 naive_chunk_kda 同接口(返回 (o, S_final)),S_final 形状 [B,HV,K,V]。"""
    dtype = v.dtype
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    q, k, v, g, beta = map(lambda x: x.to(torch.float) if x.dtype != torch.float64 else x, [q, k, v, g, beta])
    qc, kc, vc, gc, w, u, Aqk = precompute(q, k, v, g, beta, scale, chunk_size)
    M, N = build_MN(kc, gc, w, u, dtype=mn_dtype)
    h0 = torch.zeros(B, HV, K, V, dtype=M.dtype, device=q.device) if initial_state is None else initial_state.to(M.dtype)
    S_after = scan_states_serial(M, N, h0) if serial else scan_states(M, N, h0, G)
    S_before = torch.cat([h0[:, :, None], S_after[:, :, :-1]], 2)
    o = outputs(qc, gc, w, u, Aqk, S_before, dtype)
    return o, S_after[:, :, -1].to(torch.float)

# ----------------------------------------------------------------------------- 工具
def err(name, got, ref):
    got = got.double(); ref = ref.double()
    d = (got - ref).abs()
    rel = (d.flatten().square().mean().sqrt() / (ref.flatten().square().mean().sqrt() + 1e-30)).item()
    print(f"  {name:44s} max_abs={d.max().item():.3e} rel_rms={rel:.3e}")
    return rel

def make_raw(B, T, H, D, device, seed=0):
    import torch.nn.functional as F
    torch.manual_seed(seed)
    q = F.normalize(torch.randn(B, T, H, D, device=device), dim=-1)
    k = F.normalize(torch.randn(B, T, H, D, device=device), dim=-1)
    v = torch.randn(B, T, H, D, device=device)
    g = torch.randn(B, T, H, D, device=device)
    beta = torch.randn(B, T, H, device=device)
    A_log = torch.rand(H, device=device); dt_bias = torch.rand(H, D, device=device)
    h0 = torch.randn(B, H, D, D, device=device) * 0.1
    return q, k, v, g, beta, A_log, dt_bias, h0

def activate(g, beta, A_log, dt_bias, lb, H, D):
    g_act = lb * torch.sigmoid(torch.exp(A_log).view(1, 1, H, 1) * (g + dt_bias.view(1, 1, H, D)))
    return g_act, torch.sigmoid(beta)

# ----------------------------------------------------------------------------- check
def check(device):
    ok = True
    for (T, H, G) in [(64, 2, 4), (256, 2, 8), (512, 3, 16)]:
        for lb in (-5.0, -1.0):
            D = 128; B = 1
            q, k, v, g, beta, A_log, dt_bias, h0 = make_raw(B, T, H, D, device)
            f = lambda x: x.double()
            g_act, b_act = activate(f(g), f(beta), f(A_log), f(dt_bias), lb, H, D)
            h0d = f(h0)   # [B,H,K,V]
            o_rec, S_rec = naive_recurrent_kda(f(q), f(k), f(v), g_act, b_act, initial_state=h0d, output_final_state=True)
            o_chk, S_chk = naive_chunk_kda(f(q), f(k), f(v), g_act, b_act, initial_state=h0d, output_final_state=True, chunk_size=16)
            print(f"[T={T} H={H} lb={lb} G={G}] fp64")
            err("naive_chunk vs naive_recurrent (o)", o_chk, o_rec)
            # naive.py 内部把输入 .to(torch.float),参照实际是 fp32 精度(~1e-7);扫描版全程 fp64。
            # 所以:扫描 vs 自己的串行 M/N 版要求 1e-12(纯重排序),vs naive 参照要求 1e-5。
            o_ser, S_ser = scan_kda(f(q), f(k), f(v), g_act, b_act, initial_state=h0d, chunk_size=16, G=G, serial=True)
            err("serial-MN vs naive_chunk (o)", o_ser, o_chk)
            err("serial-MN vs naive_recurrent (o)", o_ser, o_rec)
            o_s, S_s = scan_kda(f(q), f(k), f(v), g_act, b_act, initial_state=h0d, chunk_size=16, G=G, serial=False)
            r0 = err(f"scan(G={G}) vs serial-MN (o)", o_s, o_ser)
            r0b = err(f"scan(G={G}) vs serial-MN (S_final)", S_s, S_ser)
            r1 = err(f"scan(G={G}) vs naive_chunk (o)", o_s, o_chk)
            r2 = err(f"scan(G={G}) vs naive_chunk (S_final)", S_s, S_chk)
            r3 = err(f"scan(G={G}) vs naive_recurrent (o)", o_s, o_rec)
            ok &= r0 < 1e-12 and r0b < 1e-12 and r1 < 1e-5 and r2 < 1e-5 and r3 < 1e-5
    print("E8 CHECK:", "PASS" if ok else "FAIL", "(scan==serial-MN to 1e-12 fp64; vs naive(fp32 internals) to 1e-5)")
    return ok

# ----------------------------------------------------------------------------- bench(GPU)
def bench(T, H, G, iters=20, lb=-5.0):
    import flash_kda
    from torch.profiler import profile, ProfilerActivity
    device = "cuda"; D = 128; B = 1
    q, k, v, g, beta, A_log, dt_bias, h0 = make_raw(B, T, H, D, device)
    qb, kb, vb, gb, bb = [x.bfloat16() for x in (q, k, v, g, beta)]
    q, k, v, g, beta = [x.float() for x in (qb, kb, vb, gb, bb)]   # 扫描版吃同样的 bf16 舍入后输入
    h0b = h0.transpose(-1, -2).contiguous().bfloat16()              # flash_kda 状态布局 [N,H,V,K]
    h0 = h0b.float().transpose(-1, -2).contiguous()                 # 扫描版 [K,V],同一份数值
    scale = 1 / math.sqrt(D)
    # --- flash_kda 本体:K2 kernel 时间
    out = torch.zeros_like(vb); hT = torch.zeros_like(h0b)
    fn = lambda: flash_kda.fwd(qb, kb, vb, gb, bb, scale, out, A_log=A_log, dt_bias=dt_bias, lower_bound=lb,
                               initial_state=h0b, final_state=hT)
    for _ in range(5): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters): fn()
        torch.cuda.synchronize()
    agg = {}
    for e in prof.events():
        if e.device_type.name == "CUDA":
            nm = e.name.split("<")[0][:50]
            a = agg.setdefault(nm, 0.0); agg[nm] = a + e.time_range.elapsed_us()
    k2 = sum(us for nm, us in agg.items() if "recurrence" in nm) / iters
    k1 = sum(us for nm, us in agg.items() if "prepare" in nm) / iters
    tot = sum(agg.values()) / iters
    print(f"\n===== T={T} H={H} NT={T//16} G={G} NG={T//16//G} =====")
    print(f"flash_kda: K1 {k1:.1f} us, K2 {k2:.1f} us, total {tot:.1f} us")

    # --- 扫描版:各阶段计时(fp32 / tf32 / bf16 的 M,N,S)
    g_act, b_act = activate(g, beta, A_log, dt_bias, lb, H, D)
    qc, kc, vc, gc, w, u, Aqk = precompute(q, k, v, g_act, b_act, scale, 16)
    torch.cuda.synchronize()

    def timeit(f, n=iters):
        for _ in range(3): f()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n): f()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) * 1000 / n

    results = {}
    for mode in ("fp32", "tf32", "bf16"):
        torch.backends.cuda.matmul.allow_tf32 = (mode == "tf32")
        mn_dtype = torch.bfloat16 if mode == "bf16" else torch.float32
        h0m = h0.to(mn_dtype)
        t_mn = timeit(lambda: build_MN(kc, gc, w, u, dtype=mn_dtype))
        M, N = build_MN(kc, gc, w, u, dtype=mn_dtype)
        t_scan = timeit(lambda: scan_states(M, N, h0m, G))
        S_after = scan_states(M, N, h0m, G)
        S_before = torch.cat([h0m[:, :, None], S_after[:, :, :-1]], 2)
        t_out = timeit(lambda: outputs(qc, gc, w, u, Aqk, S_before, torch.bfloat16))
        # CUDA graph:整条 K2 替代链(MN + scan + out)
        static_o = None
        def whole():
            M_, N_ = build_MN(kc, gc, w, u, dtype=mn_dtype)
            S_ = scan_states(M_, N_, h0m, G)
            Sb_ = torch.cat([h0m[:, :, None], S_[:, :, :-1]], 2)
            return outputs(qc, gc, w, u, Aqk, Sb_, torch.bfloat16), S_[:, :, -1]
        t_eager = timeit(lambda: whole())
        try:
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2): whole()
            torch.cuda.current_stream().wait_stream(s)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                o_g, S_g = whole()
            t_graph = timeit(lambda: gr.replay())
        except Exception as ex:
            print("  graph capture failed:", ex); t_graph = float('nan'); o_g, S_g = whole()
        # 精度:对 fp64 naive_recurrent(小 T 才算得动,用前 2048 个 token 的子问题)
        results[mode] = (t_mn, t_scan, t_out, t_eager, t_graph)
        print(f"  [{mode:4s}] build_MN {t_mn:7.1f}  scan {t_scan:7.1f}  out {t_out:7.1f}  | eager {t_eager:7.1f}  graph {t_graph:7.1f} us"
              f"   vs flash_kda K2 {k2:.1f} us  -> graph/K2 = {t_graph/k2:.2f}x, K2/graph = {k2/t_graph:.2f}x")
        # 精度 vs flash_kda 输出(同 bf16 输入)
        err(f"  [{mode}] o   vs flash_kda", o_g, out)
        err(f"  [{mode}] S_T vs flash_kda", S_g.float().transpose(-1, -2), hT)   # flash_kda 状态是 [N,H,V,K]
    torch.backends.cuda.matmul.allow_tf32 = False
    return k2, results

def bench2(iters=20, lb=-5.0):
    """同一会话内:bmm 128^3 天花板 / flash_kda K2 / 扫描 v1 / 扫描 v2(bf16),并按每步有效 TFLOP/s 归因。"""
    import flash_kda
    from torch.profiler import profile, ProfilerActivity
    device = "cuda"; D = 128; B = 1
    def timeit(f, n=iters):
        for _ in range(3): f()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n): f()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) * 1000 / n
    for (T, H, G) in [(8192, 12, 32), (8192, 96, 32), (32768, 12, 32)]:
        NT = T // 16; NG = NT // G; R = B * H * NG
        q, k, v, g, beta, A_log, dt_bias, h0 = make_raw(B, T, H, D, device)
        qb, kb, vb, gb, bb = [x.bfloat16() for x in (q, k, v, g, beta)]
        q, k, v, g, beta = [x.float() for x in (qb, kb, vb, gb, bb)]
        h0b = h0.transpose(-1, -2).contiguous().bfloat16(); h0 = h0b.float().transpose(-1, -2).contiguous()
        scale = 1 / math.sqrt(D)
        out = torch.zeros_like(vb); hT = torch.zeros_like(h0b)
        fn = lambda: flash_kda.fwd(qb, kb, vb, gb, bb, scale, out, A_log=A_log, dt_bias=dt_bias, lower_bound=lb,
                                   initial_state=h0b, final_state=hT)
        for _ in range(5): fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iters): fn()
            torch.cuda.synchronize()
        k2 = sum(e.time_range.elapsed_us() for e in prof.events() if e.device_type.name == "CUDA" and "recurrence" in e.name) / iters
        # 天花板:和 level-1 同 batch 的连续 bmm
        a = torch.randn(R, D, D, device=device, dtype=torch.bfloat16); b_ = torch.randn(R, D, D, device=device, dtype=torch.bfloat16)
        t_bmm = timeit(lambda: torch.bmm(a, b_))
        ceil_tflops = 2.0 * R * D ** 3 / t_bmm / 1e6
        g_act, b_act = activate(g, beta, A_log, dt_bias, lb, H, D)
        qc, kc, vc, gc, w, u, Aqk = precompute(q, k, v, g_act, b_act, scale, 16)
        M, N = build_MN(kc, gc, w, u, dtype=torch.bfloat16)
        h0m = h0.bfloat16()
        t_v1 = timeit(lambda: scan_states(M, N, h0m, G))
        work = (torch.empty(G, R, D, D, dtype=torch.bfloat16, device=device), torch.empty(G, R, D, D, dtype=torch.bfloat16, device=device))
        t_v2 = timeit(lambda: scan_states_v2(M, N, h0m, G, work))
        S1 = scan_states(M, N, h0m, G); S2 = scan_states_v2(M, N, h0m, G, work)
        # 扫描总 FLOP:level-1 2*(G-1)*R, level-2 (NG-1)*B*H, level-3 R*G 条 128^3
        n_gemm = 2 * (G - 1) * R + (NG - 1) * B * H + R * G
        flop = 2.0 * n_gemm * D ** 3
        t_mn = timeit(lambda: build_MN(kc, gc, w, u, dtype=torch.bfloat16))
        S_before = torch.cat([h0m[:, :, None], S2[:, :, :-1]], 2)
        t_out = timeit(lambda: outputs(qc, gc, w, u, Aqk, S_before, torch.bfloat16))
        print(f"\n===== T={T} H={H} NT={NT} G={G} NG={NG} level-1 batch R={R} =====")
        print(f"  flash_kda K2                  {k2:9.1f} us")
        print(f"  bmm 128^3 x{R} ceiling         {t_bmm:9.1f} us  {ceil_tflops:7.1f} TFLOP/s")
        print(f"  scan v1                       {t_v1:9.1f} us  {flop / t_v1 / 1e6:7.1f} TFLOP/s effective ({n_gemm} GEMMs, {flop/1e9:.1f} GFLOP)")
        print(f"  scan v2                       {t_v2:9.1f} us  {flop / t_v2 / 1e6:7.1f} TFLOP/s effective")
        print(f"  scan at ceiling (lower bound) {flop / ceil_tflops / 1e6:9.1f} us")
        print(f"  build_MN {t_mn:.1f} us, out {t_out:.1f} us; v2 total {t_mn + t_v2 + t_out:.1f} us  -> K2/v2total = {k2 / (t_mn + t_v2 + t_out):.2f}x")
        err("  scan v2 vs v1 (S_final)", S2[:, :, -1], S1[:, :, -1])

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "check":
        check("cuda" if torch.cuda.is_available() else "cpu")
    elif mode == "bench2":
        assert torch.cuda.is_available()
        bench2()
    elif mode == "bench":
        assert torch.cuda.is_available()
        # 精度参照:小形状对 fp64 递推(和 REPORT §2.4 的口径一致)
        check("cuda")
        for (T, H, G) in [(8192, 12, 32), (8192, 12, 16), (8192, 96, 32), (32768, 12, 64), (32768, 12, 32)]:
            bench(T, H, G)
