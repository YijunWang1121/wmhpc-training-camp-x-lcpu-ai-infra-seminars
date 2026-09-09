# C2:MiniMax M3 MSA decode,小 batch 这一半 — 报告(Claude 独立版本,分支 `c2-claude`)

所有数字来自本机 **NVIDIA B300 SXM6 AC**(148 SM,cc 10.3,HBM3e 268 GB)实测;CUDA 13.0,torch 2.14,Triton 3.8。
原始日志在 `logs/`,nsys/ncu 导出在 `profiles/`,逐 session 记录在 `LOG.md`,文档原文摘录在 `DOCS.md`,
验收方案在 `ACCEPTANCE.md`。复现命令见 §9。

## 0. 结论速览

| 问题 | 结论 |
|--|--|
| 小 batch Triton 基线瓶颈(任务 1 / 讨论点 5) | **延迟 + 固定开销**,不是算力也不是带宽。graph 模式 b=1 共 8.2 us = decode kernel 5.9(其中 ~3.6 固定 + 每块串行链 2.3)+ merge 2.7;eager 再加 ~22 us host launch gap。DRAM 利用率 5.6%(b=1)–33%(b=16),tensor pipe ≤17%,占用率 ≤11%,80% 周期无可发射 warp。 |
| 讨论点 1 AI | bf16 KV:**16 FLOP/B**(fp8:32),B300 ridge ≈ 280 FLOP/B → 内存侧问题;但 CUDA core fp32 ridge ≈ 10 FLOP/B < 16,**没有 tensor core 连带宽都跟不上**。M=16 = GQA 组 = `mma.m16n8k16` 的 M,一行不浪费;Triton 在 B300 上也确实用 mma.sync。 |
| 讨论点 2 融合 | 值得:merge kernel 在 b=1 占 1/3。cluster + DSMEM 合并可行且已实现(`exp/msa_decode.cu`),文档依据见 DOCS.md;cluster=16 会因 GPC 内凑不齐 SM 而串行化,只可用 ≤4。 |
| 讨论点 3 TMA | **能表达**:tensormap 坐标是运行期 `.s32` 寄存器,两级间接寻址算出 page 后作为最高维坐标即可(实测校验和一致,单块 0.8 us);无需 gather 模式;PTX 9.4 的 `.override::global_address` 更直接但 CUDA 13.0 不可用;1-D `cp.async.bulk` 也可(块内 64 KiB 连续)。 |
| 讨论点 4 FP8 scale | 标量 scale 折进 Q(host)+ 输出(epilogue),kernel 零成本、与上游口径逐位一致;per-token scale 只能在 kernel 内,但应加在 S/P tile 上而不是反量化 K/V。上游 Triton 的 fp8 路径比 bf16 **更慢**(转换开销 > 省下的带宽)。 |
| 挑战 | 选 **(a)**:CUDA 融合 kernel(TMA + mma.sync + cluster/DSMEM 合并)。数字见 §7。 |
| crossover=16 | 见 §8:CUTLASS 路径每 (req, kvh) 一个 CTA 流式吃 16 块,固定开销大但每块几乎无串行延迟;Triton 靠 split-K 把 16 块摊到 16 个 CTA 抢并行度。b×4 个 CTA 在 b≥16 时才 ≥ 64 → 用得上足够 SM,同时 Triton 的每 CTA 块数从 1 涨到 4、串行链变长。两条线在 b≈16 交叉。 |

## 1. 对象与形状

`vllm_msa_ref/sparse_attn.py` 的 decode 路径:split-K over top-k blocks + LSE merge。
- 形状:topk=16,page=块=128,head_dim=128,64 q heads / 4 kv heads(GQA 组 16),dql=1(投机 dql=2)。
- 每步每 (req, kv_head) 读 16 × 128 × 256 × 2 B = **1 MiB**;每 req 4 MiB;b=16 → 64 MiB。
- 基线 grid = (total_q × NUM_TOPK_CHUNKS, 4),chunks = 2^⌊log2 min(16, 256/(4·total_q))⌋:
  b=1/4:16 chunks(每 CTA 1 块);b=8:8;b=16:4;b=32:2;b=64:1。CTA 数 b=1 为 64,b≥4 恒 256。
- 每 CTA 128 线程(4 warps),195 reg/thread,smem 73.7 KB → 每 SM ≤2 CTA(寄存器限制),
  理论占用率 12.5%。生成代码是 `mma.sync.m16n8k16.bf16` + `cp.async.cg` + `ldmatrix`(不是 tcgen05/TMA)。

## 2. 测量(任务 1)

### 2.1 端到端(`exp/e1_baseline.py`,`logs/e1_final.out`;seq=8192;us)
| b | chunks | CTA 数 | eager | **graph** | decode | merge | KV MiB | 有效 GB/s |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 16 | 64 | 32.5 | **8.2** | 5.9 | 2.7 | 4 | 510 |
| 2 | 16 | 128 | 32.4 | 9.2 | 6.4 | 2.8 | 8 | 910 |
| 4 | 16 | 256 | 32.4 | **11.2** | 8.0 | 3.1 | 16 | 1495 |
| 8 | 8 | 256 | 32.7 | **14.9** | 11.4 | 3.4 | 32 | 2252 |
| 16 | 4 | 256 | 32.5 | **21.9** | 18.5 | 3.2 | 64 | 3068 |
| 32 | 2 | 256 | 47.3 | 42.9 | 39.0 | 3.7 | 128 | 3127 |
| 64 | 1 | 256 | 78.7 | 74.4 | 70.1 | 4.0 | 256 | 3607 |

- eager 恒 ≈32.5 us(b≤16):nsys 看到两 kernel 之间 gap 5 us + 16 us(`profiles/nsys_b*`),即 **GPU 70% 时间空转等 host**。
  vLLM 生产用 CUDA graph,下文以 graph 数字为准,但这条说明 kernel 级优化的前提是 graph。
- 有效带宽最高 3.6 TB/s(b=64)= 45% 峰值;小 batch 0.5–3 TB/s。数据量与手算完全一致(ncu DRAM read 4.23 MB @ b=1)。
- seq 长度 256→32768 不影响时间(10.2→11.2 us @ b=4):数据只来自 top-k 块。

### 2.2 每 CTA 块数扫描(E5,graph,decode kernel 单独计时,us)
| 块/CTA | 16 | 8 | 4 | 2 | 1 |
|--|--:|--:|--:|--:|--:|
| b=1 | 40.3 | 21.9 | 12.7 | 8.2 | 5.9 |
| b=4 | 39.9 | 21.9 | 12.9 | 8.6 | 8.1 |

线性拟合 **t ≈ 3.6 us + 2.3 us × (块/CTA)**:3.6 us 是 kernel 固定开销,2.3 us 是每块的串行延迟链。
TTGIR 证实(`LOG.md` S2):K/V smem 只有 1 级缓冲,循环体 = `async_wait → 计算 → 发起下一块载入`,载入与计算不重叠。

### 2.3 ncu(`profiles/ncu_dec_b*.txt`)
| 指标 | b=1 | b=4 | b=8 | b=16 |
|--|--:|--:|--:|--:|
| DRAM Throughput % | 5.6 | 16.7 | 24.0 | 33.4 |
| Compute (SM) % | 4.8 | 16.1 | 16.9 | 19.3 |
| tensor pipe (hmma) % of active | 7.8 | – | – | 16.5 |
| Achieved occupancy % | 6.4 | 10.9 | 11.1 | 10.8 |
| Waves per SM | 0.22 | 0.86 | 0.86 | 0.86 |
| No Eligible % | 81 | 75 | 75 | 74 |
| 主要 stall | long_scoreboard 1.32, wait 1.22 | | | long_scoreboard 1.97, wait 1.45 |

三条 pipe 都远未饱和、一波都填不满 148 SM、每调度器 1–1.7 个 warp、80% 周期无 warp 可发射 → **latency-bound**。

### 2.4 瓶颈结论(讨论点 5)
b=1 的 8.2 us = decode 固定 3.6 + 每块链 2.3 × 1 + merge 2.7(merge 只读 272 KB,纯固定开销)。
到 b=16 变成 3.6 + 2.3 × 4 + 3.2 ≈ 16(实测 21.9,多出的是 256 CTA 在 148 SM 上的两波排队)。
**该优化什么**:(i) 去掉第二个 kernel;(ii) 让载入与计算重叠、把 16 块的延迟链压成 1–2 个;(iii) 提高每 SM 的 warp 数以掩盖发射延迟。
**不该优化什么**:tensor core 利用率、带宽——它们在小 batch 根本没被用满。

## 3. 讨论点 1:arithmetic intensity 与 tensor core

每 (req, kv_head):FLOP = 2·(2·16·2048·128) = 16.8 M;bytes ≈ 1.008 MiB(KV 1 MiB + Q/O/索引)→ **AI = 15.9 FLOP/B**(fp8 KV:31.5)。
与 batch 无关(每个 req 读自己的块,无复用)。
- B300 ridge(bf16 dense ≈ 2.25 PFLOPS / 8 TB/s)≈ 281 FLOP/B,fp8 ≈ 562 → 离 ridge 18–35×,**纯内存侧**,与 4.2 的结论一致。
- 但 CUDA core fp32 峰值 ≈ 80 TFLOPS → ridge ≈ 10 FLOP/B < 16:**不用 tensor core,算力反而先于带宽成为瓶颈**(b=16:CUDA core 需 13.4 us > HBM 8.4 us)。
  所以 tensor core 有用武之地,但只需其峰值的 ~6%(16/281)。
- 与 4.5 的联动:瘦 GEMM M=16 在 tcgen05 上达成率塌方,是因为 128 行 tile 只填 16 行;这里 `mma.sync.m16n8k16` 的 M 正好 16,
  没有浪费——ncu 显示 Triton 的 tensor pipe 活跃率 8–17%,和 "只需 6%" 的量级一致,tensor core 不是瓶颈。
- 推论:在 B300 上用 tcgen05(M≥64)做 decode 会浪费 3/4 的 MMA 行,但 MMA 不是瓶颈,tcgen05 的真正价值是**单线程异步发射**
  (免去每块 ~460 条 ldmatrix/mma/exp 指令的串行发射),这是 CUTLASS 路径在大 batch 更快的原因之一(§8)。

（§4–§9 见下,随实验推进补齐）
