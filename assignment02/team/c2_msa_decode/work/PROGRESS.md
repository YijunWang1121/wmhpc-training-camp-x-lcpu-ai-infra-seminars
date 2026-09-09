# C2 MSA decode — 进度日志

时间线倒序不强制,追加为主。日期基准:2026-09-09。

## GPU sessions
| # | 目的 | 结果 | 备注 |
|---|------|------|------|
| 1 | smoke test `p1_measure.py check` | 3/3 PASS,lib.py 拆分 kernel 复现正确 | srun 8min,job 23408 |
| 2 | `p1_run.sh`:check+sweep+nsys(b1/4/8/16)+ncu | check PASS,sweep+nsys 拿到,ncu 全部失败 | srun 1h |
| 3 | `p1_run2.sh`:ncu debug + CUDA graph sweep + nsys gpu-metrics | graph sweep + gpu-metrics 拿到;ncu 仍失败(`-s` 被当成 `--launch-skip`) | srun 1h |
| 4 | `ncu_probe.sh` | **ncu 修好**:必须用 `--section` 长名 或 `--set`,不能用 `-s` | srun 9min,job 23415 |
| 5 | `ncu_full.sh`:decode+merge × b{1,4,8,16} 全 section | 进行中 | srun 30min |

**ncu 用法教训**:`-s` = `--launch-skip`,不是 section。用 `--section SpeedOfLight` 或 `--set detailed`。
`--set launch` 对 Triton kernel 报 "No metrics"(用 `--section LaunchStats`)。

---

## 2026-09-09 — P0 准备

### 已读材料 & 关键事实
- **harness/run.py**:`check` 用 SDPA fp32 参考对拍(err_ratio < 2e-2 判 PASS),3 组形状
  (常规 batch=4 / 短序列尾块 / 投机 decode dql=2)。`bench` 扫 batch∈{1,4,8,16,32,64},
  seq 固定 8192,用 cuda event 计时 100 次取平均,单位 us/call。计的是
  `minimax_m3_sparse_attn_decode` 整个 wrapper(decode kernel + merge kernel 两次 launch)。
- **synth.py**:KV cache `[num_pages, num_kv_heads, 128, 2*head_dim]`(前半 K 后半 V)。
  block_table `[num_reqs, max_blocks]` 逻辑块→物理页,物理页 `randperm` 打乱(两级间接寻址真实)。
  topk_idx `[num_kv_heads, total_q, topk]` 存逻辑块号,恒含当前块,其余槽填 0。
- **sparse_attn.py**(基线,重点):
  - decode 走 split-K flash-decoding:`_gqa_sparse_decode_kernel` 产 partial [chunk, q, h, d] + lse,
    再 `_merge_topk_attn_out_kernel` 按 lse 加权合并。
  - decode kernel grid = `(total_q * num_topk_chunks, num_kv_heads)`。
    `num_topk_chunks` = 把 `TARGET_GRID=256` 除以 `total_q*num_kv_heads` 再向下取 2 的幂,
    clamp 到 [1, max_topk=16]。 **小 batch 关键**:
    - batch=1, dql=1: total_q=1, num_kv_heads=4 → target = 256/4 = 64 → clamp 16 → chunks=16.
      grid = (1*16, 4) = (16,4) = **64 CTA**.
    - batch=4: total_q=4 → target = 256/16 = 16 → chunks=16. grid=(64,4)=**256 CTA**.
    - batch=8: target = 256/32 = 8 → chunks=8. grid=(64,4)=**256 CTA**.
    - batch=16: target = 256/64 = 4 → chunks=4. grid=(64,4)=**256 CTA**.
    - batch=32: target = 256/128 = 2 → chunks=2. grid=(64,4)=256 CTA.
    - batch=64: target = 1 → chunks=1. grid=(64,4)=256 CTA.
    → B300 有 148 SM。batch=1 只有 64 CTA,**SM 占用率 < 50%**,且每 CTA 只处理
      1 个 topk 块(16 chunks / 16 blocks)。这就是小 batch regime 的核心矛盾之一。
  - 每个 decode CTA 的 tile:`q [BLOCK_SIZE_H=16, BLOCK_SIZE_D=128]`,K/V `[128,128]`。
    `tl.dot(q,k)`:16×128×128 GEMM —— M=16 极瘦,tensor core 几乎空转(联动 4.5)。
  - merge kernel grid = `(total_q, num_heads=64)`。batch=1 → (1,64) = 64 CTA,每个只 reduce
    `[NUM_TOPK_CHUNKS, 128]`,访存极小 → 纯 launch/延迟 bound。
- **msa_cutlass_sparse_decode.py**:`_MIN_CUTLASS_BATCH_SIZE = 16`。CUTLASS 路径要求
  fp8 kvcache + sm100 + head 几何 + page=128 + topk=16。用 `fmha_sm100` plan/planner,
  per-query-token page indices。crossover=16 是官方 kernel benchmark 的结论(TP1/TP4)。
- **形状 arithmetic intensity 预估**(待 GPU 验证):
  - 一步 decode,一个 (req, kv_head):Q [16,128] bf16。选 16 块 × [128,128] K + V。
  - K+V bytes = 16 * 128 * 128 * 2(KV) * 2(bytes bf16) = 1,048,576 B ≈ 1 MiB / (req,kv_head).
  - 全 batch=1: 4 kv_heads → 4 MiB KV 读。
  - FLOPs: QK = 16*128*128*2 = 524,288; PV 同 = 524,288; ×16 块 ×4 heads
    = 2 * 524288 * 16 * 4 ≈ 67 MFLOP.
  - AI ≈ 67e6 / 4.19e6 ≈ **16 FLOP/byte**(bf16)。B300 HBM ~8 TB/s, bf16 tensor
    ~2.5 PFLOP/s → roofline ridge point ~300+ FLOP/byte。AI=16 ⇒ **强 memory bound**,
    tensor core 无用武之地(结论待 ncu 验证 achieved occupancy / DRAM %)。
  - 但 batch=1 KV 只有 4 MiB,8 TB/s 下 ~0.5 us 就能读完 → 实际耗时若 >> 0.5us,
    说明瓶颈在 launch overhead / 低并行度 / 尾延迟,不在带宽。← 这是要 profile 确认的核心假设。

---

## 2026-09-09 — P1 第一轮结果(GPU session 2)

硬件确认:**NVIDIA B300 SXM6**,148 SM,268 GiB,cc 10.3,CUDA 13.0,torch 2.14+cu130,triton 3.8.0。

### 正确性
lib.py 拆两个 kernel 单独 launch,对 SDPA fp32:err_ratio 3.2e-3 / 2.8e-3 / 3.2e-3,全 PASS。

### End-to-end sweep(`p1_measure.py`,cuda event,eager 模式,200 iters)
| batch | chunks | decode_grid | merge_grid | e2e us | decode us | merge us | launch_gap us |
|---|---|---|---|---|---|---|---|
| 1  | 16 | (16,4) | (1,64)  | 32.3 | 17.2 | 9.9  | 5.2 |
| 2  | 16 | (32,4) | (2,64)  | 32.1 | 17.3 | 9.9  | 5.0 |
| 4  | 16 | (64,4) | (4,64)  | 32.3 | 17.6 | 10.1 | 4.6 |
| 8  | 8  | (64,4) | (8,64)  | 32.4 | 17.2 | 10.1 | 5.1 |
| 16 | 4  | (64,4) | (16,64) | 32.3 | 17.3 | 10.7 | 4.3 |
| 32 | 2  | (64,4) | (32,64) | 33.1 | 29.3 | 10.1 | -6.3 |
| 64 | 1  | (64,4) | (64,64) | 56.9 | 50.7 | 9.9  | -6.3 |

→ **batch 1..16 端到端延迟完全持平 ~32 us**。工作量 ×16 而时间不动。
   这些 us/call 数(decode 17us 等)是 **CPU 侧 Triton launch 开销主导**,不是 GPU 时间。

### GPU 侧真实 kernel 时间(nsys cuda_gpu_kern_sum,稳态)
| batch | decode kernel (ns) | merge kernel (ns) | GPU 合计 |
|---|---|---|---|
| 1  | 3893 | 2034 | 5.9 us |
| 4  | 5253 | 2264 | 7.5 us |
| 8  | 7352 | 2470 | 9.8 us |
| 16 | 11671| 2451 | 14.1 us |

- decode kernel:b1→b16 工作量 ×16,GPU 时间只 ×3 → **强烈 sub-linear,低占用/延迟 bound**。
- merge kernel:**恒定 ~2.0–2.5 us**,与 batch 无关。b1 时占 GPU 时间 34%,纯固定开销。
- b1 decode:4 MiB KV / 8 TB/s ≈ 0.5 us 的带宽下界,实测 3.9 us → **约 8× 高于 roofline**,
  不是带宽 bound。

### nsys GPU trace(b1 稳态,`cuda_gpu_trace`)
- decode kernel:grid (16,4,1),128 thr/CTA = **64 CTA / 148 SM(56% SM 空闲)**,
  195 reg/thr,74 KB dynamic smem。
- merge kernel:grid (1,64,1) = 64 CTA,128 thr,32 reg,~1 KB smem。
- eager loop 里 kernel 之间 GPU 空隙 ~20–27 us(CPU 喂不上)→ **eager 模式 GPU 占空比 <25%**。

### 初步瓶颈判断(待 ncu occupancy/stall 确认)
小 batch decode **不是** compute-bound(AI≈16,tensor core 空转)、**不是** HBM-BW-bound
(实测远低于带宽上限),而是三重固定开销叠加:
  (1) CPU launch 开销(eager)—— CUDA graph 可消除,但 kernel 结构本身还有:
  (2) decode kernel 低网格占用(b1 只 64 CTA)+ split-K 每 CTA 只 1 块 → 延迟链无法被并行掩盖;
  (3) merge kernel 固定 ~2 us —— 两次 launch + 中间 partial 往返 HBM。

---

## 2026-09-09 — P1 第二轮结果(GPU sessions 3-4)

⚠️ 注意:`dev-slurm` 节点是**共享**的(squeue 显示 mix 态),GPU 时钟不锁,不同 run 之间
kernel 绝对时间有 ~2× 抖动(b1 decode 见过 3.9 / 6.3 / 8.2 / 10.8 us)。**只比同一次测量内的
相对 scaling**,绝对值给区间。

### CUDA graph sweep(`p1_graph.py`,去掉 CPU launch 开销,500 replay 取平均)
| batch | chunks | e2e us | decode us | merge us |
|---|---|---|---|---|
| 1  | 16 | 10.3 | 8.2  | 4.1 |
| 2  | 16 | 10.8 | 8.2  | 4.1 |
| 4  | 16 | 14.3 | 10.2 | 4.2 |
| 8  | 8  | 18.4 | 14.3 | 6.2 |
| 16 | 4  | 24.6 | 20.5 | 6.1 |
| 24 | 2  | 41.0 | 36.9 | 6.2 |
| 32 | 2  | 45.6 | 41.2 | 6.2 |
| 48 | 1  | 73.5 | 69.4 | 6.2 |
| 64 | 1  | 78.0 | 72.6 | 6.2 |

- **knee 在 batch ≈ 2–4**:b1=b2 完全持平(固定开销 bound),b4 起每加 batch 线性涨。
- merge:batch 无关,~4–6 us 固定(graph replay 下)。
- chunks 掉档处(b16→b24 chunks 4→2)有台阶:每 CTA 串行块数翻倍 → 时间跳。
- graph 下 e2e ≈ decode + merge,两 kernel 之间几乎没 gap(graph node 依赖直连)。

### nsys GPU-metrics(`gpu_metrics.py` 按 kernel 窗口聚合;采样 50 kHz)
| 指标 | b1 decode | b1 merge | b16 decode | b16 merge |
|---|---|---|---|---|
| SMs Active [%]        | **9.2**  | 5.9  | 51.3 | 58.4 |
| SM Issue [%]          | 2.0  | 1.2  | 18.4 | 19.4 |
| Tensor Active [%]     | **0.8**  | 0.2  | **9.1**  | 11.0 |
| DRAM Read BW [%]      | ~0   | ~0   | ~1   | ~1   |
→ b1:**SM 9% / Tensor 0.8% / DRAM ~0%** —— 没有任何资源吃满,纯延迟暴露。
→ b16:SM 51% / Tensor 9% / DRAM ~1% —— 仍远未 memory/compute bound。

### ncu — b1 `_gqa_sparse_decode_kernel` (grid (16,4,1), 128 thr)  [session 4, `--set`]
| Metric | Value | 解读 |
|---|---|---|
| Duration | 10.78 us (此次时钟偏低) | |
| Elapsed / SM-Active cycles | 11659 / **2910** | SM 平均只在 25% 时间有活干 |
| Compute (SM) Throughput | **4.89 %** | 不是 compute bound |
| DRAM Throughput | **5.15 %** | 不是 HBM 带宽 bound |
| L1/TEX Cache Throughput | 25.7 % | KV 走 L1/TEX,是最高的一项 |
| L2 Cache Throughput | 4.33 % | |
| Full waves | **0.2** | grid 64 CTA / 148 SM,半台机器没分到活 |
| Theoretical Occupancy | 12.5 % | 被 **寄存器**(195 reg/thr → 2 block/SM)限死 |
| Achieved Occupancy | **6.22 %** | 比理论还低(grid 太小 + 尾部不均) |
| Achieved Active Warps/SM | 3.98 / 64 | |

**b1 decode 结论(证据充分)**:没有任何硬件资源接近打满。瓶颈 = 三件事叠加
  (1) grid 64 CTA « 148 SM(0.2 wave),
  (2) 寄存器压力把每 SM 占用压到 12.5%(实测 6%),
  (3) split-K 让每 CTA 只处理 1 个块 → load K→QK→softmax→load V→PV 的串行依赖链
      靠 4 warp/SM 完全掩盖不住 HBM 延迟。
→ 联动 assignment 4.5(瘦 GEMM 达成率塌方):问题形状太小,喂不满机器。

### ncu full sweep(session 5,`ncu_full.sh`,`--clock-control none`,共享节点)

**`_gqa_sparse_decode_kernel`**(grid 恒 = (total_q·chunks, 4),但因 chunks 反比 batch,
grid CTA 数 b1=64、b≥4=256;每 CTA 处理块数 = 16/chunks):
| batch | Dur us | DRAM % | Mem BW (TB/s) | Compute SM % | L2 % | L1/TEX hit | Achieved Occ % | Waves/SM | No-Eligible % |
|---|---|---|---|---|---|---|---|---|---|
| 1  | 10.5 | 5.3  | 0.40 | 4.5  | 4.4  | 0.3% | 6.3  | 0.22 | 81 |
| 4  | 14.0 | 15.8 | 1.21 | 15.5 | 13.4 | –    | 11.1 | 0.86 | – |
| 8  | 18.5 | 24.0 | 1.83 | 17.1 | 20.0 | –    | 11.1 | 0.86 | – |
| 16 | 27.3 | 33.0 | 2.50 | 19.3 | 27.4 | 0.2% | 10.8 | 0.86 | – |

- 寄存器 195/thr → Block Limit Registers = 2 → **理论占用天花板 12.5%**,batch 再大也不动。
- Waves/SM 永远 ≤ 0.86(< 1 wave)。
- L1/L2 命中率 ≈ 0:每个 KV 块只读一次,必打 HBM。
- b1:Compute 4.5% / DRAM 5.3% / **No-Eligible 81%**,4 warp/SM,issue 只 19% 周期
  → 纯 HBM 延迟暴露,占用太低盖不住。
- 随 batch 增大逐渐往 memory-bound 靠(b16 DRAM 33%),但 compute 始终 <20%,
  **tensor core 全程 <10% 活跃**。

**`_merge_topk_attn_out_kernel`**(grid (total_q, 64)):
| batch | Dur us | Compute % | Mem % | L1/TEX % | Achieved Occ % |
|---|---|---|---|---|---|
| 1  | 5.6 | 1.6  | 2.6  | 19 | 5.7 |
| 4  | 6.1 | 6.0  | 10   | 30 | 11 |
| 8  | 6.5 | 9.5  | 16   | 45 | 21 |
| 16 | 6.4 | 14   | 17   | 46 | 39 |

- **Duration 恒 ~6 us**,工作量 b1→b16 涨 16×,时间不动 → 小 batch 下是**纯固定开销**。
- b1 时 merge 占 decode+merge 的 34%,且 100% 是"低占用 + 第二次 kernel launch + partial
  从 HBM 往返"的开销,没有一点是必要计算。

### P1 一句话结论(讨论点 5)
小 batch(1–8)decode **既不是 compute-bound、也不是 HBM-带宽-bound**,而是
**低并行度导致的延迟暴露 + 两次 launch 的固定开销**。三个可量化的根因:
(1) decode kernel 占用被寄存器锁在 12.5%(实测 6–11%),grid < 1 wave;
(2) split-K 在小 batch 反而有害——chunks=16 让每 CTA 只算 1 块,依赖链最长、复用最差;
(3) merge kernel 固定 ~6 us 全是开销。
tensor core 全程 <10%,对这个问题**没有用武之地**(见讨论点 1 手算 AI≈16)。

---
## 状态:P1 完成。已用 5 个 GPU session。

## 2026-09-09 — P2 完成 + P3 决策

- `work/REPORT.md` 写完 §0–§8:形状梳理、讨论点 1(AI≈16,tensor core 无用,双重原因)、
  讨论点 5(瓶颈 = 延迟暴露 + launch 固定开销,非 compute/BW bound,证据齐)、
  讨论点 2(小 batch 应融合不 split;cluster-merge 理论可行但性价比低)、
  讨论点 3(TMA 不能表达两级 gather,但可一页一条 TMA;实验待跑)、
  讨论点 4(fp8 scale 在 load 后点积前片上 dequant,方案 A 正确)、
  讨论点 6(验收方案:3 档参照 + 形状覆盖 + 计时口径 + 自曝弱点)、
  §7(CUTLASS crossover=16 的 I/O cost model 解释:nsb/N<2.85)。

### P3 决策:选 (a) 改良 Triton(scope 受控)
理由:roofline 显示 b1 decode 有 8–20× headroom;merge ~6us 纯开销可删。
不碰 CUTLASS TMEM/planner。四个动作,按预期收益排序:
1. **删 merge**:小 batch 不 split-K,单 kernel 在线 softmax 直接写出。
2. **软件流水**:16 块循环加 `num_stages` 预取 K/V,盖 HBM 延迟。
3. **提占用**:降寄存器(BLOCK_SIZE 调整 / 拆 D 维);多 warp/CTA。
4. **自适应启发式**:batch 小→融合路径,batch 大→保留 baseline split-K。

验收:先按 REPORT §6 的方案(已贴),再动工。

### P3 attempt 1 — `v1_fused.py`(不 split + 融合 merge)结果:**失败,慢 4×**

正确性:全 11 形状 × 5 seed × 2 档 PASS(vs SDPA 2.6e-3,vs 基线 3.4e-3)。

性能(CUDA graph e2e,tuning sweep):
| batch | 基线 | v1 best | 加速 |
|---|---|---|---|
| 1  | 10.2 | 39.2 (4w2s) | **0.26×** |
| 4  | 12.3 | 41.0 | 0.30× |
| 8  | 16.4 | 41.0 | 0.40× |
| 16 | 24.6 | 44.9 | 0.55× |
| 32 | 45.9 | 54.3 | 0.85× |
| 64 | 77.2 | 71.3 | 1.08× |

**根因**:b1 grid = (total_q=1, kvh=4) = **只 4 CTA**。去掉 split-K 就去掉了基线在
小 batch 唯一的并行来源(基线 16 chunks → 64 CTA)。且 `tl.range(0, real_topk,
num_stages=N)` 用**运行期** trip count,Triton 不做软件流水 → 16 块的 load→dot→
softmax→load→dot 完全串行,~2.4 us/块 × 16 = ~39 us。

**教训**:
1. 小 batch 下 split-K 的 grid 不是可有可无 —— 它是唯一的填充手段,不能删。
2. Triton 的 loop pipelining 需要 constexpr trip count。
3. "单 CTA 延迟优化 > 占用" 的假设在这里不成立(4 CTA 太少 + 框架不给流水)。

### P3 attempt 2 方向
- v2:**保留 split-K 的 grid**,只把 merge 干掉 —— chunk CTA 用 `tl.atomic` 或
  cluster/DSMEM 就地合并;或用 constexpr 展开的 16 次循环让流水生效。
- v3:干脆**只调基线**(NUM_TOPK_CHUNKS 启发式 + num_warps + 降寄存器),看能不能
  把 merge 的固定 6us 压下去 / decode 占用提上去。
- 若都不行 → 转 **challenge (b)**:Triton 框架内小 batch 已近前沿,真正的 win 需要
  CUDA(TMEM + warp-specialized TMA pipeline + cluster merge)= CUTLASS 路径的工程成本,
  这本身解释了 crossover=16。

### P3 attempt 3 — 只调基线(`p3_retune.py`,num_warps × NUM_TOPK_CHUNKS sweep)
结果:**基本无收益**。b1/b2/b16 = 1.00×,b8 偶尔 1.12×,b4 反而 0.86×。
基线默认 config 已在其结构的前沿。而且 CUDA-graph e2e 有 ~2us/kernel 的 replay
latency 地板(多组 config 结果精确相同到小数点后 2 位)。

### 讨论点 3 实验(`exp_tma.py`,GPU session)—— **完成**
两个 kernel 从 paged cache 取同一 [128,128] tile,`page` 来自运行期 tensor:
- `load_plain`(`tl.load` 算指针):PTX = `ld.global.v4.b32 ×N`,**同步**,无 TMA。
- `load_tma`(host `TensorDescriptor.from_tensor(kv2,[128,128])` + `desc.load([page*128,0])`):
  **正确**,PTX =
  `mbarrier.init` / `mbarrier.arrive.expect_tx ... 32768`(=128·128·2B 整块)/
  `cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [smem],[desc,{r4,r5}],[bar]` /
  `mbarrier.try_wait.parity`。
→ **运行期 page 索引 + TMA 描述符可以工作**,gather(topk)与 block_table 查表仍是
  普通标量 load(TMA 无 gather 模式,与文档一致),但最后那段 [128,128] tile 搬运
  可以一块一条 `cp.async.bulk.tensor`,硬件算地址 + mbarrier 完成。FA3/FlashInfer 同法。

### §7 crossover 实验(`exp_crossover.py`,CPU)—— **完成**
KV-stationary decode 对每个去重 (kv_head, page) 收集选它的 query token 跑一个 MMA,
`M = avg_reuse × GQA(16)`。实测(两种选块模型,seq=8192):
| batch | avg_reuse | M | MMA fill(/128) |
|---|---|---|---|
| 4  | 1.5–2.0 | 25–32 | 20–25% |
| 8  | 2.4–2.8 | 38–45 | 30–35% |
| **16** | **4.2–4.3** | **68–69** | **53–54%** |
| 24 | 6.3 | 100 | 78% |
| 32 | 8.1–8.4 | 129–135 | 100% |
→ **M 在 batch≈16 越过 64(半个 tcgen05 tile)**。这就是 `_MIN_CUTLASS_BATCH_SIZE=16`
  的机制:batch<16 时 KV-stationary 的 MMA <50% 满,喂不饱 tensor core,CUTLASS 那套
  固定 prologue(148 SM 持久网格、planner metadata、TMA/TMEM init)摊不平;
  batch≥16 才划算。与讨论点 1 / assignment 4.5 同一现象(M 维决定一切)。
  注:Fireworks 的 nsb/N<2.85 阈值是 prefill 的 Q/KV-outer 选择(L2-BW vs HBM-BW),
  seq~8k decode 下 nsb/N 在此 batch 区间始终 >>2.85,不是 M3 decode 的驱动因素。

### 2026-09-09 — 合并确认 run(`final_bench.sh`,单一时间片,`profiles/FINAL.txt`)
一次 srun 内跑完:正确性 + CUDA-graph sweep + nsys 纯 GPU 时间。与前面分次结果一致。
- 正确性:11 形状 × 5 seed 全 PASS(vs SDPA 2.6e-3,vs 基线 3.3–3.5e-3)。
- CUDA-graph e2e:基线 b1=10.2 / b4=12.5 / b8=18.4 / b16=24.6 / b64=78.2 us;
  fused-v1 b1=41.0(0.25×)… b64=72.0(1.09×)。
- nsys 纯 GPU(此时间片,节点较忙,比 session-2 高但 scaling 一致):
  decode b1=6.3 / b4=8.6 / b8=12.0 / b16=19.5 us;merge 恒 3.0–3.8 us。
  → decode:b1→b8 工作 8×,时间 1.9×(sub-linear,再次印证低并行度延迟 bound)。

---
## 状态:P1+P2+P3 全部完成。已用 12 个 GPU session(均单卡 ≤1h)。
## 交付:work/REPORT.md(主报告)、work/DEFENSE.md(答辩)、work/PLAN.md(索引)、
##       本文件(日志)、work/profiles/(原始 profile)、work/*.py(复现脚本)。
## 挑战结论:选 (b)。小 batch(<16)decode 收益上限(Triton ~1.3–1.7×,CUDA 至多 ~2–3×)
##       < 工程成本(重写 Blackwell 专有 warp-specialized fmha = 上游 CUTLASS 路径),
##       与上游「CUTLASS opt-in 且 batch≥16」的实际选择一致;补上了「为什么是 16」的
##       量化依据(M = reuse·16 在 batch≈16 越过半个 MMA tile)。
## 结论:challenge **(b)** —— Triton 框架内小 batch 已近前沿,收益上限 ~1.3–1.7×
##       < CUDA 重写(= CUTLASS 路径)的工程成本。三次 (a) 尝试的数据是 (b) 证据链的一部分。
## 下一步:把 REPORT.md §8 换成完整的 P3 章节 + 收益上限/成本对照;更新 report.md 主文件引用。
