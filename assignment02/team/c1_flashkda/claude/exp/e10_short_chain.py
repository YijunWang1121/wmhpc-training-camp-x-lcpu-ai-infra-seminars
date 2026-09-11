"""E10(brief 第 9 步):算法层(E9 分层扫描)确认有收益后,才问 SM100 特性能不能再加分。
新 regime = phase 1/3:很多条 G=32 chunk 的短链同时跑(每 SM 2 个 CTA 共驻)。用现成的 tcgen05 版 K2(k2_tc2)
和 FlashKDA 的 mma.sync K2 在"短序列 × 很多 head"上直接对比:T=512(=32 chunk,一个组),H 从 148(1 CTA/SM)
扫到 1184(8 CTA/SM 排队),看两种指令的链在共驻/排队条件下的整卡吞吐。"""
import sys, os, torch
sys.argv = [sys.argv[0]]
import e7_k2tc as e
from torch.profiler import profile, ProfilerActivity

def prof(fn, key, iters=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(iters): fn()
        torch.cuda.synchronize()
    return sum(ev.time_range.elapsed_us() for ev in pr.events() if ev.device_type.name == "CUDA" and key in ev.name) / iters

print(f"{'T':>6} {'H':>5} {'CTA/SM':>7} | {'mma.sync K2 us':>15} {'tcgen05 K2 us':>14} {'ratio':>6} | {'cyc/chunk/CTA mma':>18} {'tcgen05':>8} | chunks/us mma tcgen05")
for (T, H) in [(8192, 12), (512, 148), (512, 296), (512, 592), (512, 1184), (1024, 296), (1024, 592), (2048, 296)]:
    p = e.setup(T, H)
    e.run_ref(p); torch.cuda.synchronize()
    t_ref = prof(lambda: e.run_ref(p), "recurrence")
    t_tc = prof(lambda: e.run_tc(p), "k2_tc")
    o1, h1 = e.run_ref(p); o2, h2, _ = e.run_tc(p); torch.cuda.synchronize()
    mhz = e.sm_mhz()
    nchunk = T // 16
    waves = H / 148
    print(f"{T:>6} {H:>5} {waves:>7.2f} | {t_ref:>15.1f} {t_tc:>14.1f} {t_ref/t_tc:>6.2f} | {t_ref/nchunk*mhz/max(waves,1):>18.0f} {t_tc/nchunk*mhz/max(waves,1):>8.0f} | {H*nchunk/t_ref:8.1f} {H*nchunk/t_tc:8.1f}   (mhz={mhz}, tc-vs-ref out rel {e.rel(o2,o1):.1e})")
