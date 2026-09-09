# C2:MiniMax M3 MSA decode,小 batch 这一半 — 报告(Claude 独立版本,分支 `c2-claude`)

所有数字来自本机 **NVIDIA B300 SXM6 AC**(148 SM,cc 10.3,HBM3e 268 GB)实测;CUDA 13.0,torch 2.14,Triton 3.8。
**注意本机 SM 时钟被固定在 1095 MHz(标称 2032),不随负载提升(§7.0 实测);所有 us 都是这个时钟下的,memory 时钟正常(3996 MHz)。**
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
| 挑战 | 先按 (a) 做了完整一轮(CUDA 融合 kernel:TMA + mma.sync + cluster/DSMEM 合并,4 个版本,全部过验收),最好配置 b=1 14.7 us vs Triton 8.2 us,**没赢**;把这一轮的测量变成 (b) 的证据链:小 batch 收益上限 ≈ 1.5×(硬地板 3.3 us / 8.2 us),工程成本 = 一条 Blackwell 专有 fmha 路径。**结论 (b)**,数字见 §7。 |
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


## 4. 讨论点 2:两个 kernel 融不融?merge 放 cluster / mbarrier 里做?

**结论:融。** 证据:
- merge kernel 在 b=1 时 2.7 us / 8.2 us = 33%,b=16 时 3.2 / 21.9 = 15%;它读的数据 b=1 只有 272 KB(ncu),纯固定开销。
- split-K 本身在小 batch 是必要的(E5:1 CTA 串行吃 16 块要 40 us),所以"不 split"不是选项;要去掉的是**第二个 launch**。

**怎么融(两条路,都已实现并实测):**
1. **cluster + DSMEM**(`exp/msa_decode.cu`):一个 (token, kv_head) 的 16 块分给一个 cluster 的 CL 个 CTA;各 CTA 把
   (m[16], l[16], acc[16×128] fp32) 留在自己 smem;`cluster.sync()` 后 rank 0 通过 `map_shared_rank`(PTX `mapa`)读其余 CTA 的
   partial 做 LSE 合并并写输出;再 `cluster.sync()` 保证被读的 CTA 不提前退出——这是 Programming Guide 的硬性要求
   (DOCS.md 引文)。partial 不落 global、不占 L2、没有第二个 launch。mbarrier 在这里只负责 TMA 完成计数
   (`cp.async.bulk.tensor … mbarrier::complete_tx::bytes`),CTA 间同步用 cluster barrier 更直接。
2. **"最后到达的 CTA 做 merge"**(`exp/triton_v2.py`,改良 Triton):partial 照旧写 global,每个 (token, kv_head) 一个原子计数器
   (`atomic_add(sem="acq_rel")`),数到 NUM_CHUNKS−1 的 CTA 用 volatile load 读全部 partial 合并,并把计数器归零以便 graph 重放。
   不需要 cluster、任何架构可用、Triton 就能写。

**cluster 大小的限制(实测 + 文档):** 可移植上限 8,B200/B300 opt-in 16(Blackwell Tuning Guide)。实测 cluster=16
(每 CTA 1 块)b=1 40 us、b=16 242 us——随 CTA 数线性变慢:16 个 CTA 必须同时落在同一 GPC 的 16 个空 SM 上,
只要 GPC 里有 SM 被占,整个 cluster 排队。cluster ≤ 4 才是可用区间(§7 数据)。

## 5. 讨论点 3:top-k 两级间接寻址,TMA tensor map 能不能表达?

**结论:能,而且不需要任何 gather 模式。** 文档依据(DOCS.md A–D,PTX ISA 原文):
- `cp.async.bulk.tensor` 的 `tensorCoords` 是 `.s32` 寄存器向量——坐标是运行期值。把 kv_cache 整体建成 4D tensor map
  `[num_pages][num_kv_heads][128][256]`,kernel 里做 `topk_idx[slot] → block_table[req][blk] → page` 两次标量 load,
  然后以 `{0, 0, kh, page}` 为坐标发 TMA。TMA 只负责"规则 box",间接寻址由 SM 线程算坐标完成。
- 唯一带 gather 语义的 `.tile::gather4` 只做 4 行、仅 2D(PTX 8.6,sm_100a);本题一块是页内连续 128 行,用不上。
- PTX 9.4 新增 `.override::global_address`(运行期覆盖 tensormap 基址,"一个 map 描述一页、按页换基址"),更贴合
  paged KV,但需要 CUDA 13.4;本机 13.0(PTX 9.0)不可用。
- 不用 tensormap 也行:vLLM 布局下一块 (page, kv_head) 的 K|V 是连续 64 KiB,1-D `cp.async.bulk` 给全局地址即可;
  代价是没有 swizzle(ldmatrix 会 8-way bank conflict)且 K/V 行内交错无法只取一半。

**实验验证**(`exp/tma_indirect.cu`,job 24076/24077;每 CTA 先做两级间接寻址,再用三种方式搬块,与 host 按同样寻址算的校验和比对):
| 搬法 | 校验 | 单块延迟 | 16 块流式(3 级 ring)/CTA |
|--|--|--:|--:|
| (A) TMA 4D map,page 作运行期坐标 | OK | 0.79–0.86 us | 4.6–5.0 us(≈ 220 GB/s / CTA) |
| (B) 1-D `cp.async.bulk`,运行期地址 | OK | 0.80–0.95 us | 4.7–5.1 us |
| (C) 128 线程 `ld.global.v4` | OK | 3.0–5.2 us | 75–80 us |
→ 间接寻址 + TMA 正确;单块 0.8 us 就是"一次 DRAM 往返 + 64 KiB 进 smem"的下限,Triton 每块 2.3 us 里载入只占 ~0.9 us。
融合 kernel(§7)就是用 (A) 加 128B swizzle 实现的。

## 6. 讨论点 4:FP8 KV cache 的 scale 放哪一层?

上游口径(`test_sparse_attn_fp8_scale.py`):K = fp8 × k_scale,V = fp8 × v_scale;标量或 `[kv_head, token]` 两种;
带 scale 的输出要与"反量化成 bf16 再跑同一 kernel"在 atol=rtol=2e-2 内一致,不带 scale 必须明显不同。

**结论:标量 scale 不该进 kernel;per-token scale 进 kernel但加在 S/P 上,不要反量化 K/V。**
- 数学上 softmax(Q·(k_s K)ᵀ)·(v_s V) = v_s · softmax((k_s Q)·Kᵀ)·V:标量 k_scale 折进 Q(host 一个乘法,或直接折进
  `sm_scale`),v_scale 乘在输出上。kernel 完全不感知 scale。实测(E8,scale 0.3/0.7,`logs/e10_*.out`):
  与反量化参照 err_ratio 4e-3、max|Δ| 2e-4,远在 2e-2 内。
- per-token/head scale 不能折进 Q,但 S[:, j] *= k_scale[j] 是对 16×128 的 S tile 做一次列缩放,
  P[:, j] *= v_scale[j] 同理——比对 128×128 的 K/V 每个元素做 fp8→bf16→乘→bf16 便宜 8 倍,且不在载入路径上。
- 上游 Triton 的做法(load 后逐元素 `.to(bf16) * scale`)让 fp8 路径**比 bf16 还慢 30–60%**
  (b=1:8.3 → 11.0/11.8/12.5 us;b=8:14.9 → 19.8/24.2/21.7 us):小 batch 不缺带宽,多出来的转换指令却直接加在串行链上。
  这与讨论点 1 一致:fp8 把 AI 翻倍(16→32 FLOP/B)对一个 latency-bound 的 kernel 毫无帮助。
- CUTLASS 路径(`msa_cutlass_sparse_decode.py`)正是走"折进标量"这条:`q_scale/k_scale/v_scale/o_scale` 作为 float 传入
  `fmha_sm100`,且要求 `query_fp8`——用 fp8 MMA,scale 全在 epilogue。

## 8. crossover 为什么在 16(挑战 (a)/(b) 共同要求)

上游注释:"Kernel benchmarks put the CUTLASS crossover at 16 requests for TP1 and TP4"(`_MIN_CUTLASS_BATCH_SIZE = 16`)。
本机没有 fmha_sm100 的构建(需 vLLM third_party + FP8 KV),不能直接测 CUTLASS;但两条路径的**结构**可以在本机复现并标定:

**两种结构。**
- Triton split-K:每 (token, kv_head) 起 `chunks` 个 CTA,每个 CTA 串行吃 16/chunks 块;`chunks = 2^⌊log2 min(16, 256/(4b))⌋`。
  b≤4 每 CTA 1 块,b=8 两块,b=16 四块,b=32 八块,b=64 十六块——**batch 越大,每 CTA 的串行链越长**,
  时间 ≈ 3.6 + 2.3 × (块/CTA) + merge(§2.2),所以 b 从 16 到 64 时间从 22 → 75 us(每块 2.3 us 是它的斜率)。
- CUTLASS `fmha_sm100` decode(`sparse_kernel_mode="decode"`):plan 一次,每 (req, kv_head) 一个 CTA/warp-group 用 TMA 流式
  吃完 16 块,tcgen05 异步 MMA,不 split、无 merge。特点:**固定开销大**(plan、tensormap、64/128 行 tile 只填 16 行、
  epilogue),但每块的边际成本很低(MMA 由单线程异步发射,SM 只做 softmax),且 b 增大时时间几乎不涨——直到 4b 个 CTA 的
  聚合带宽撞到 HBM。

**本机标定。** 我的 CUDA kernel cl1s3(一个 CTA 流 16 块、3 级 TMA ring)就是"流式结构"用 legacy mma.sync 的实现:
b=1..16 恒 27–28 us(平),b=32 34 us,b=64 65 us(带宽项接管);Triton 8.3 → 22 → 43 → 75。**两条线在 b≈24–32 交叉**——
crossover 现象被复现,只是位置偏右,因为我的每块成本 c_blk ≈ 1.4 us(mma.sync + ldmatrix + softmax 串行发射,§7 消融)。

**模型**(`logs/crossover_model.txt`):`T_stream(b) = T_fix + max(16·c_blk, b·4 MiB / BW_eff(b))`,T_fix = launch 2.2 + 索引链 1.8 +
epilogue 1.0(§7 的实测下限),BW_eff = min(4b × 220 GB/s, 0.85 × 8 TB/s)(TMA 流式实验的单 CTA 带宽);Triton 用实测表。
| c_blk(每块边际成本) | 含义 | crossover |
|--:|--|--:|
| 1.4 us | mma.sync,本机实测(我的 kernel / Triton 的每块成本) | b≈32 |
| 0.8 us | 只剩 TMA 载入延迟(MMA 完全隐藏) | **b≈16** |
| 0.5 us | tcgen05:每块 16 条异步 MMA + softmax | b≈8 |
| 0.3 us | tcgen05 + 2 GHz 时钟 | b≈4 |
上游的 16 落在 "MMA 基本被异步化、每块成本 ≈ 载入延迟" 这一档,与 fmha_sm100 的实现方式一致;它的 T_fix 比我的模型更大
(plan + fp8 转换 + 64 行 tile),又把 crossover 往右推回 16 附近。

**一句话:** crossover 不是"tensor core 在 b<16 时不划算",而是 **固定开销大、边际成本小的流式 kernel** 对
**固定开销小、边际成本随 b 线性涨的 split-K kernel**——前者的平线在 b·(4 MiB/BW) 超过 16·c_blk 之前一直平,
后者从 b=8 起每翻倍 batch 每 CTA 多串一倍的块。交点位置只取决于 c_blk 与 T_fix,上游测出来是 16。
此外上游 CUTLASS 路径只支持 FP8 KV,而 Triton 的 FP8 路径比 bf16 慢 30–60%(§6),这也把交点往左拉。

## 7. 挑战:先做 (a),再用它的数据论证 (b)

### 7.0 两个影响一切数字的环境事实(先说)
- **SM 时钟固定 1095 MHz**(`exp/clock_probe.cu`:globaltimer 对 clock64,冷/热/重载后都是 1094–1095 MHz;`nvidia-smi` 报
  Applications Clocks 2032 但当前 1095,无降频事件)。计算/延迟受限部分若在 2.03 GHz 会快 ~1.85×,DRAM 部分不变。
- **B300 上 legacy `mma.sync.m16n8k16` 只有 ~8 cycle/条/SMSP ≈ 320 TFLOPS 全卡**(`exp/mma_rate.cu`;fp8 mma.sync 1200,
  FFMA 38);Triton 3.8 在 sm_103 上就是用它。tcgen05 峰值是它的 ~7×,这是 CUTLASS 路径和 Triton 路径的根本差别。

### 7.1 做了什么(`exp/msa_decode.cu`,约 400 行 CUDA)
- 一个 (token, kv_head) = 一个工作单元;`CL` 个 CTA 组成 cluster 分 16 块;CTA 内 `STAGES` 个 warp-group(各 4 warps)各自独占
  一个 64 KiB smem stage、轮流吃块;每 warp 32 个 key 的在线 softmax;warp 状态经 smem 合并,CTA 状态经 **DSMEM** 合并
  (rank 0 写输出,两次 `cluster.sync`)。
- 载入:4-D tensormap + `cp.async.bulk.tensor.4d`,page 是运行期坐标(§5),128B swizzle,mbarrier `complete_tx` 计数。
- MMA:`mma.sync.m16n8k16 bf16`,Q 常驻寄存器(32 reg),K 用 `ldmatrix`、V 用 `ldmatrix.trans`,P 从 C 布局直接打包成 A 布局。
- 另有两个对照实现:(i) **无 cluster 的 split-K**(每 CTA 1/2/4 块,partial 用 Triton 的格式 + 上游 merge kernel);
  (ii) **改良 Triton**(`exp/triton_v2.py`:last-CTA 原子合并去掉 merge kernel,num_warps 8)。
- 版本史(LOG.md S4–S9):v1 单 group;v2 双 group 共享 ring → **mbarrier 相位竞态**(sanitizer 定位,cl4s1 崩溃);
  v3 每 group 独占 stage;v3.1 合并 scratch 加 padding 消 8-way bank conflict;每一版都过 ACCEPTANCE 全矩阵。

### 7.2 验收(ACCEPTANCE.md,`logs/e12_final.out`,`logs/e15_final.out`)
- 形状矩阵 b∈{1,2,3,4,8,16} × seq∈{50–300, 1k–8k, 32k} × dql∈{1,2} × 3 seed × 8 配置 = 864 次比对,**全部 PASS**:
  对 fp32 参照 err_ratio 2.4e-3–2.6e-3(Triton 2.5e-3–3.3e-3,**新 kernel 更准**,因为 P 只做一次 bf16 舍入、acc 全程 fp32);
  逐 head 最大 3.6e-3;对 Triton 逐元素 max|Δ| ≤ 9.8e-4(阈 1.0e-2);无 NaN/Inf(含 real_topk<16、尾块、dql=2)。
- 无 cluster split-K 版本、改良 Triton 版本同样全 PASS,且 split 版的 last-CTA 计数器在 graph 重放下自复位正确。

### 7.3 性能(graph,us;`logs/e15_final.out`、`logs/e14_split_final.out`、`logs/e12_final.out`)
| b | Triton | cl4s2(最好的融合配置) | cl2s2 | cl1s3(流式) | 我的 CTA + Triton merge(split16) | 改良 Triton(fuse, nw8) |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | **8.2** | 14.7 | 17.9 | 26.0 | 10.6 | 10.2 |
| 4 | **11.2** | 15.0 | 18.2 | 26.7 | 13.9 | 17.1 |
| 8 | **14.8** | 16.2 | 18.3 | 26.9 | 22.5 | 16.1 |
| 16 | **22.1** | 32.0 | 20.7 | 27.8 | 38.8 | 24.6 |
| 32 | 42.9 | 66.3 | 44.3 | **32.9** | 67.6 | 45.3 |
| 64 | 74.4 | 128.2 | 85.4 | **64.0** | – | 81.1 |
- **小 batch 全线输给 Triton**;流式配置 cl1s3 在 b≥32 反超(§8 的 crossover 现象)。
- eager 下融合 kernel 22 us vs Triton 32 us(少一次 Python launch),但生产用 graph,这不算数。

### 7.4 为什么输——每一项都量化了(LOG.md S8,`logs/e13_ablation_final.out`,kernel 内 clock64 分相)
| 项 | cycles(≈ns @1.095 GHz) | 说明 |
|--|--:|--|
| launch(空 kernel,graph) | 0.9–1.6 us | 与 Triton 相同 |
| 索引链 seq_lens→topk→block_table | +0.9 us | 三次依赖 L2 往返,与 Triton 相同 |
| 一块 TMA 载入 | +1.5 us | 单块 0.8 us + mbarrier;`index+TMA` 探针 3.3 us @b=1 |
| **每块 compute(4 warps)** | **1.4k** | 消融:MMA 0.2k、ldmatrix 0.35k、exp 0.1k、其余(掩码/max/shuffle/重缩放/地址)0.7k;~400 条指令串行发射 |
| 4-warp smem 合并 + 写 partial | 1.0–2.4k | 8-way bank conflict 修掉后仍 ~1k |
| cluster 合并(2× cluster.sync + DSMEM) | **3.7k** | 比 Triton 整个 merge kernel(2.7 us)还贵 |
| 多 warp-group | 每组每块 1.4k → 2.1k | 共享 SMSP,总 compute 只降 2×,而 epilogue 涨到 7.5k |
- 同一"1 块/CTA"结构下:我的 decode CTA 7.8 us vs Triton 5.9 us(`split16` 列):Triton 的 codegen 已经很接近这个结构在
  这块卡上的下限;我多出的 ~2 us 是 Q 经 smem 中转 + 4 warp 合并。
- cluster=16(1 块/CTA + DSMEM 合并,理论最优形态)因 GPC 内凑 16 个空 SM 而串行化(b=1 40 us),cluster ≤4 又把每 CTA
  串行块数抬到 ≥4。**"1 块/CTA" 与 "cluster 合并" 在本卡上互斥。**

### 7.5 (b) 的证据链
**收益上限(roofline + 实测地板)。** b=1 一步的硬地板 = launch 0.9 + 索引链 0.9 + 一块载入 1.5 = **3.3 us**(探针实测,任何
1 块/CTA 的 kernel 都逃不掉),再加至少一块 compute 1.3 us、一次合并 ≥1 us → **≥5.6 us**;Triton 8.2 us → 上限 **≈1.5×**。
HBM roofline(4 MiB / 8 TB/s = 0.5 us)在这里没有意义:它假设 148 个 SM 一起拉带宽,而 b=1 只有 64 个块任务。
b=4:地板 4.1 + 1.3 + 1 = 6.4 vs 11.2 → ≈1.7×;b=16:Triton 22 us 里 4 块/CTA 的串行链 9 us 是可压的,但 CUTLASS 已接管。
换算到端到端:MiniMax M3 每个 decode step 有 ~80 层 attention(以 MoE 主体 ~15–25 ms/step 计),
每层省 2–3 us → **每 step 省 0.2 ms ≈ 1%**。这是收益的天花板,还没扣掉 graph 里其它 kernel 的重叠。

**工程成本(从 CUTLASS 路径复杂度估 + 本轮实测)。**
- 要越过"1.5×"这条线只有一条路:让每块 compute 从 1.4k cycle 降到 ~0.3k——即 **tcgen05 异步 MMA + TMEM 累加 + 单线程发射**,
  这正是 `fmha_sm100` 的结构(`msa_cutlass_sparse_decode.py` 316 行只是它的 Python 胶水:plan cache、65536 行预分配、
  graph-stable 地址、`_update_runtime_metadata_kernel`,还要求 fp8 KV、page=128、topk=16、dql≤32、head 几何固定)。
- 本轮 400 行 mma.sync 版本已经踩到:mbarrier 相位语义(竞态 + 硬件错误)、cluster 调度粒度、smem bank conflict、
  DSMEM 合并开销;换成 tcgen05 要再加 TMEM 分配/布局、smem 描述符 + swizzle 匹配、P 回写 TMEM 作 A 操作数、
  warp 专化。上游把这条路径做成 opt-in 且 b≥16 才开,说明他们也没在小 batch 拿到正收益。
- **上限 ~1.5× 的 kernel(端到端 ~1%)vs 一条 Blackwell 专有 fmha 路径的维护成本:不值得。** 小 batch 继续走 Triton split-K,
  是对的。真要在小 batch 榨一点,低风险的是:(1) 让 Triton 的 K/V 载入两级缓冲、与计算重叠(每块 2.3 → ~1.5 us,b≥8 受益);
  (2) 用 PDL(`launch_pdl`,上游代码里已有 `USE_PDL` 路径)把 merge 的 launch 与 decode 的尾巴重叠,省 ~1 us。

## 9. 复现
```
cd team/c2_msa_decode/claude
# 编译(登录节点即可)
nvcc -O3 -std=c++17 -gencode arch=compute_100f,code=sm_100f --shared -Xcompiler -fPIC -o exp/libmsa_decode.so exp/msa_decode.cu -lcuda
nvcc -O3 -std=c++17 -gencode arch=compute_100f,code=sm_100f -o exp/tma_indirect exp/tma_indirect.cu -lcuda
nvcc -O3 -std=c++17 -gencode arch=compute_100f,code=sm_100f -o exp/mma_rate exp/mma_rate.cu
nvcc -O3 -gencode arch=compute_100f,code=sm_100f -o exp/clock_probe exp/clock_probe.cu
# GPU(每个 ≤25 min,单卡):
sbatch -G 1 --time=00:20:00 exp/job_e1.sh     # 基线测量 E1/E2/E5
sbatch -G 1 --time=00:45:00 exp/job_e3.sh     # nsys + ncu
sbatch -G 1 --time=00:10:00 exp/job_tma.sh    # 讨论点 3 TMA 间接寻址
sbatch -G 1 --time=00:25:00 exp/job_e15.sh    # 融合 kernel 验收 + 性能;split 版;地板探针
sbatch -G 1 --time=00:15:00 exp/job_e13.sh    # 消融(需先按 job 脚本里的宏编译各变体)
sbatch -G 1 --time=00:15:00 exp/job_e12.sh    # 改良 Triton
sbatch -G 1 --time=00:05:00 exp/job_mma.sh    # mma.sync 吞吐
sbatch -G 1 --time=00:05:00 exp/job_clock.sh  # SM 时钟
python3 exp/e8_fp8scale.py                    # (GPU) 讨论点 4;logs/crossover_model.txt 由 REPORT §8 的模型脚本生成
```
注意:B300 是 cc 10.3,cubin 必须用 `sm_100f`(family)而不是 `sm_100a`(LOG.md S3 的教训)。
