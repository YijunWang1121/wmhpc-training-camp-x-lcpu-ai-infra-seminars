# 实验日志(时间顺序,原始发现)

## 2026-09-09 S0 — 环境(login 节点,无 GPU)
- 工具链:CUDA 13.0 (V13.0.88),torch 2.14.0+cu130,triton 3.8.0,ncu/nsys 在 /usr/local/cuda/bin
- Slurm:partition `gpu` MaxTime=01:00:00,节点 dev-slurm gpu:8;`~/bin/gpu` = srun/sbatch -G 1 --time 30min
- 已读:harness/*.py、vllm_msa_ref/sparse_attn.py(全文)、msa_cutlass_sparse_decode.py、sparse_attention_msa.py、
  test_sparse_attn_fp8_scale.py、index_topk.py(头部与 block_table 用法)

### 基线 kernel 结构(读代码得出,待实验验证)
- decode kernel grid = (total_q × NUM_TOPK_CHUNKS, num_kv_heads);
  NUM_TOPK_CHUNKS = 2^floor(log2(min(16, 256 // (total_q×4))))
  → b=1:16 chunks(每 CTA 1 块);b=4:16;b=8:8(每 CTA 2 块);b=16:4(4 块);b=32:2;b=64:1
  → grid CTA 数 = b×chunks×4:b=1:64,b=4:256,b=8:256,b=16:256,b=32:256,b=64:256
- 每 CTA:q tile [16 heads, 128] bf16;每块 K [128d,128pos] + V [128pos,128d] 各 32 KB;两次 tl.dot M=16,N=128,K=128
- 每 CTA 写 partial o [16,128] bf16 + lse [16] fp32;merge kernel grid=(total_q, 64 heads),每 CTA 读 chunks×128 bf16
- 一步 decode 总 KV 读量 = b × 4 kv_heads × 16 blocks × 64 KB = b × 4 MiB(b=1:4 MiB;b=16:64 MiB)

## S1 — job 24073:E1/E2/E5 基线测量(`exp/e1_baseline.py`,原始输出 `logs/e1_final.out`)
GPU:**NVIDIA B300 SXM6 AC**,148 SM,cc 10.3,HBM 268 GB,L2 126.5 MB,SM 2032 MHz,mem 3996 MHz。
harness 三组 check 全 PASS(err_ratio 3.2e-3 / 2.8e-3 / 3.2e-3)。

### 原始数据(seq=8192, topk=16, kvh=4, gqa=16, dql=1;us)
| b | chunks | CTA 数 | eager | graph | decode | merge | KV MiB | 有效 GB/s |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 16 | 64 | 32.5 | 8.2 | 5.9 | 2.7 | 4 | 510 |
| 2 | 16 | 128 | 32.4 | 9.2 | 6.4 | 2.8 | 8 | 910 |
| 4 | 16 | 256 | 32.4 | 11.2 | 8.0 | 3.1 | 16 | 1495 |
| 8 | 8 | 256 | 32.7 | 14.9 | 11.4 | 3.4 | 32 | 2252 |
| 16 | 4 | 256 | 32.5 | 21.9 | 18.5 | 3.2 | 64 | 3068 |
| 32 | 2 | 256 | 47.3 | 42.9 | 39.0 | 3.7 | 128 | 3127 |
| 64 | 1 | 256 | 78.7 | 74.4 | 70.1 | 4.0 | 256 | 3607 |

### 发现
1. **eager 时间 b≤16 恒 ≈32.5 us**,而 CUDA graph 下 b=1 只要 8.2 us → 非 graph 时 ~24 us 是 Python/Triton launcher 开销。
   vLLM 生产用 CUDA graph,所以后续以 graph 数字为准;但 eager 数字说明 host 侧才是最大的那块。
2. graph 下 b=1..16 时间 8.2→21.9 us,而数据量 4→64 MiB。有效带宽 0.5→3.1 TB/s,远低于 B300 HBM3e 峰值(标称 8 TB/s)。
   即使 b=64(256 MiB)也只有 3.6 TB/s ≈ 45% 峰值 → 这个 kernel 在大 batch 也不是带宽饱和的。
3. **E5 chunks 扫描揭示每 CTA 逐块串行代价**:b=1, chunks=1(1 CTA 处理 16 块)= 40.3 us → ≈2.4 us/块;
   chunks=16(每 CTA 1 块)= 5.9 us。时间几乎正比于「每 CTA 块数」:16→40, 8→22, 4→12.7, 2→8.2, 1→5.9。
   线性拟合:t ≈ 3.6 us + 2.3 us × (块/CTA)。→ 3.6 us 是 kernel 固定开销(launch + 序言 + 收尾),2.3 us 是**每块的串行延迟链**
   (topk_idx 标量 load → block_table 标量 load → K 载入 → dot → softmax → V 载入 → dot,块间无重叠)。
   一块 64 KB 若按 CTA 级带宽算不该要 2.3 us,这是延迟不是带宽。
4. **merge kernel 固定 ~2.7–4 us**,b=1 时占总 8.2 us 的 1/3;它只读 16×64×16×128×2 B = 4 MiB… 不对,b=1 时 partial 为 16 chunks×64 heads×128×2 B = 256 KB,纯固定开销。
5. b=8 chunks=16(512 CTA)比 chunks=8 慢(14.0 vs 11.3),b=16 chunks=16(1024 CTA)26 us:CTA 数超过一波后
   反而变慢 → 单 CTA 资源占用大(smem),每 SM 驻留 CTA 数少,多波排队。待 ncu 确认 occupancy。
6. seq 长度 256→32768 对时间无影响(10.2→11.2)→ 数据只来自 top-k 块,与 seq 无关(符合预期)。

### 由此得到的瓶颈判断(待 E3 ncu/nsys 佐证)
小 batch:**延迟链暴露**(每块 2.3 us 串行 × 块/CTA)+ **固定开销**(decode kernel ~3.6 us + merge ~2.7 us),
不是算力,也不是带宽(0.5 TB/s ≪ 8 TB/s)。

## S2 — job 24074:E3 nsys + ncu(`exp/job_e3.sh`,原始文件 `profiles/`)
### nsys(eager,20 步/batch,中位数,us)
| b | decode grid | decode | merge | gap dec→mrg | gap mrg→next dec | 周期 |
|--:|--|--:|--:|--:|--:|--:|
| 1 | 16×4 | 6.11 | 2.98 | 5.01 | 16.13 | 30.2 |
| 4 | 64×4 | 8.40 | 3.41 | 3.30 | 15.68 | 30.8 |
| 8 | 64×4 | 11.68 | 3.74 | 2.34 | 13.12 | 30.9 |
| 16 | 64×4 | 18.98 | 3.65 | 1.94 | 5.98 | 30.5 |
- eager 周期恒 ≈30.5 us,与 E1 的 32.5 一致:GPU 空闲(两个 gap 之和)b=1 时 21 us = 70%。
  gap 来自 Triton launcher(Python 侧参数打包 + 驱动 launch)。**graph 是前提**,否则 kernel 优化毫无意义。
- decode kernel:128 线程 / CTA(4 warps),195 reg/thread,动态 smem 73.7 KB。

### ncu(单次 launch,`--set full`)decode kernel
| 指标 | b=1 | b=4 | b=8 | b=16 | b=1 ch=1 | b=1 ch=4 |
|--|--:|--:|--:|--:|--:|--:|
| Grid | 64 | 256 | 256 | 256 | 4 | 16 |
| Duration (us, ncu 时钟锁定下) | 9.92 | 13.25 | 18.37 | 26.66 | 51.23 | 17.92 |
| DRAM read | 4.23 MB | — | — | 67.4 MB | 4.23 MB | — |
| DRAM Throughput % | 5.6 | 16.7 | 24.0 | 33.4 | 1.1 | 3.1 |
| Compute(SM) % | 4.8 | 16.1 | 16.9 | 19.3 | 0.5 | 1.6 |
| tensor pipe (hmma) % of active | 7.8 | — | — | 16.5 | 16.0 | — |
| Theoretical occupancy | 12.5% (8 warps/SM) | ← | ← | ← | ← | ← |
| Achieved occupancy | 6.4% | 10.9% | 11.1% | 10.8% | 6.3% | 6.4% |
| Block limit: registers / smem | 2 / 3 | ← | ← | ← | ← | ← |
| Waves per SM | 0.22 | 0.86 | 0.86 | 0.86 | 0.01 | 0.05 |
| No Eligible % | 81 | 75 | 75 | 74 | 79 | 79 |
| Warp cycles / issued inst | 5.25 | 6.89 | 6.99 | 6.73 | 4.80 | 4.86 |
| 主要 stall(cycles/inst) | long_sb 1.32, wait 1.22, short_sb 0.75 | | | long_sb 1.97, wait 1.45, short_sb 1.20 | wait 1.56, long_sb 1.07 | |
- DRAM 读 4.23 MB = 4 MiB KV + 索引,**数据量与手算完全一致**(没有多读)。L2 命中 ~1–3%:纯流式。
- 所有 batch 都是:占用率 ≤ 11%(每 SM 最多 2 CTA = 8 warps,寄存器 195 限制),Waves ≤ 0.86 → **一波都填不满 148 SM**;
  每调度器只有 ~1–1.7 个活跃 warp,80% 周期无可发射 warp。DRAM/SM/tensor 三条 pipe 都 < 35%。
  → 典型的**延迟受限**(latency-bound),不是吞吐受限。
- ch=1(1 CTA 做 16 块)51 us、DRAM 1%:一条 warp 链的串行延迟暴露到极致。

### Triton 生成代码结构(~/.triton/cache/.../_gqa_sparse_decode_kernel.{ptx,ttgir})
- `.target sm_103a`,`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` ×64,`cp.async.cg.shared.global` ×64,
  `ldmatrix` ×48,3× `cp.async.wait_group`。→ 在 B300 上 Triton 3.8 对这个 kernel 用的仍是 **Ampere 风格 mma.sync + cp.async**,
  不是 tcgen05/TMA。M=16 正好等于 GQA 组 16 = m16n8k16 的 M,tensor core 一行都不浪费。
- TTGIR:num_stages=3 但 K/V smem 只有 **1 级缓冲**(`memdesc<1x128x128xbf16>`),循环体是
  `async_wait(k,v,page) → 计算 → 发起下一块 k/v/page 的 cp.async`;下一轮开头立刻 wait。
  即**下一块的载入与本块计算不重叠**,只有 page 索引提前一轮。每块 = 完整一次 DRAM 往返 + 计算,串行。
  这解释了 E5 的 2.3 us/块。

### 瓶颈结论(任务 1 / 讨论点 5)
小 batch(b≤16)decode 的时间构成(graph 模式,b=1 共 8.2 us):
1. decode kernel 固定开销 ~3.6 us(launch + 序言 + 索引两级 load + 收尾)
2. 每块串行延迟链 ~2.3 us × (块/CTA)(b=1: 1 块)
3. merge kernel ~2.7 us(几乎全是固定开销:b=1 只读 272 KB)
4. eager 模式再加 ~22 us host launch gap
不是算力(tensor pipe <17%),不是带宽(DRAM <34%,b=1 仅 5.6%)。是 **latency + launch**。
