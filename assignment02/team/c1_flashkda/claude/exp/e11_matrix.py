"""E11:系统多轮测试——分层并行 K2(E9 v3:merged + bf16 phase 2 + CUDA graph)vs 原版 flash_kda.fwd,不用任何 SM100 特性。
负载矩阵 B∈{1,2,4} × H∈{12,16,32,64} × T∈{4K,16K,64K,128K};每形状 R 轮,baseline/hier 交替,每轮 warmup+iters 次 CUDA event 计时。
输出 mean±std、最优 G、门控后收益。"""
import sys, math, time, statistics, torch
import e9_hier as e

R = int(sys.argv[1]) if len(sys.argv) > 1 else 5
lb = -1.0
Gs = (32, 64)
free_b = torch.cuda.mem_get_info()[0]

def iters_for(B, T, H): return 50 if B * T * H <= 1 * 65536 * 16 else 30

rows = []
print(f"rounds={R} lb={lb} G in {Gs}; per-round warmup 5 + timed iters (50 / 30 for big);  free mem {free_b/1e9:.0f} GB")
print(f"{'B':>2} {'H':>3} {'T':>6} {'B*H':>4} | {'baseline us (mean±std)':>24} | " + " | ".join(f"{'G='+str(G)+' us (mean±std)':>22} {'x':>5}" for G in Gs) + " | best  gated")
for B in (1, 2, 4):
    for H in (12, 16, 32, 64):
        for T in (4096, 16384, 65536, 131072):
            need = B * T * H * 128 * 2 * (5 + 6)   # q,k,v,g,out + merged(2x q,k,v,g,out) + scratch, bytes
            if need > 0.7 * free_b:
                print(f"{B:>2} {H:>3} {T:>6} {B*H:>4} | skipped (needs ~{need/1e9:.0f} GB)"); continue
            q, k, v, g, beta, A_log, dt_bias, h0 = e.make_inputs(B, T, H)
            scale = 1 / math.sqrt(e.D); it = iters_for(B, T, H)
            out_ref = torch.zeros_like(v); hT_ref = torch.zeros_like(h0)
            base_fn = lambda: e.baseline(q, k, v, g, beta, A_log, dt_bias, h0, scale, lb, out_ref, hT_ref)
            hiers = {}
            for G in Gs:
                hh = e.Hier(B, T, H, G, lb); hh.merge_p1 = True; hh.p2_mode = "seq"; hh.p2_dtype = torch.bfloat16
                o = torch.zeros_like(v); hh.run(q, k, v, g, beta, A_log, dt_bias, h0, scale, o); hh.capture_p2()
                hiers[G] = (hh, o)
            tb, th = [], {G: [] for G in Gs}
            for r in range(R):
                tb.append(e.timeit(base_fn, warm=5, iters=it))
                for G in Gs:
                    hh, o = hiers[G]
                    th[G].append(e.timeit(lambda: hh.run(q, k, v, g, beta, A_log, dt_bias, h0, scale, o), warm=5, iters=it))
            mb, sb = statistics.mean(tb), statistics.pstdev(tb)
            stats = {G: (statistics.mean(th[G]), statistics.pstdev(th[G])) for G in Gs}
            bestG = min(Gs, key=lambda G: stats[G][0]); sp = mb / stats[bestG][0]
            gated = max(sp, 1.0)   # 门控:预测不赢就走原版
            rows.append((B, H, T, mb, sb, stats, bestG, sp))
            print(f"{B:>2} {H:>3} {T:>6} {B*H:>4} | {mb:>12.1f} ± {sb:>7.1f}  | " +
                  " | ".join(f"{stats[G][0]:>12.1f} ± {stats[G][1]:>6.1f} {mb/stats[G][0]:>5.2f}" for G in Gs) +
                  f" | G={bestG:<3d} {sp:5.2f}x  gated {gated:4.2f}x", flush=True)
            del q, k, v, g, beta, out_ref, hT_ref, hiers; torch.cuda.empty_cache()

print("\n== 汇总:按 B*H 分组的最优加速(几何均值 over T) ==")
from collections import defaultdict
grp = defaultdict(list)
for (B, H, T, mb, sb, st, bg, sp) in rows: grp[B * H].append(sp)
for bh in sorted(grp):
    v = grp[bh]; gm = math.exp(sum(map(math.log, v)) / len(v))
    print(f"  B*H={bh:>4}: n={len(v)} speedup geo-mean {gm:4.2f}x  min {min(v):4.2f}x  max {max(v):4.2f}x")
