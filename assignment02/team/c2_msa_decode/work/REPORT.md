# C2:MiniMax M3 MSA decode —— 小 batch regime 研究报告

> 所有 GPU 数字来自本人在集群 `srun -G 1` 内的实测(NVIDIA B300 SXM6,148 SM,
> CUDA 13.0,torch 2.14+cu130,triton 3.8.0)。原始 profile 在 `work/profiles/`,
> 复现脚本在 `work/`,逐步日志在 `work/PROGRESS.md`。共享节点、时钟不锁,kernel 绝对时间
> 有 ~2× 抖动,故对比只在单次测量内做,绝对值给区间。任务顺序严格「先测后设计」。

## 执行摘要

1. **瓶颈(先 profile,讨论点 5)**:小 batch(1–8)decode **既不是 compute-bound、
   也不是 HBM-带宽-bound**。ncu 实测 b1:Compute 4.5% / DRAM 5.3% / L1·L2 命中≈0 /
   `No Eligible 81%`。瓶颈是**低并行度导致的 HBM 延迟暴露**(decode grid ≤ 0.86 wave、
   占用被 195 寄存器锁在 12.5%、split-K 让每 CTA 只 1 块无流水依赖链)**加两次 kernel
   launch 的固定开销**(merge kernel 与 batch 无关,纯 GPU ~2–4us,全是开销)。
2. **arithmetic intensity(讨论点 1)**:手算 AI ≈ **16 FLOP/byte**,是 B300 ridge point
   (~280–320)的 ~1/18 → 强 memory bound;且 decode GEMM 的 M 塌成 GQA=16,`tcgen05`
   MMA tile 只用 12–25%。**tensor core 双重意义上没有用武之地**,ncu 实测 Tensor Active <10%。
3. **挑战:选 (b)——论证「小 batch 不值得做 CUDA 重写」**,给完整证据链:
   - 收益上限(roofline + 实测):b1 decode 的 HBM 下界 ~0.5us,当前 ~4–10us;
     但那 8–20× 的 headroom 卡在「4 个独立问题 × 16 块 填不满 148 SM」+「Triton 不给
     运行期循环做流水」——**Triton 框架内实际可达上限 ~1.3–1.7×**(实测三种改法佐证)。
   - 工程成本:真正能动这 4us 的手段(warp-specialized TMA producer/consumer 流水、
     TMEM 累加、持久 scheduler、cluster/DSMEM 合并、fp8)= 上游 CUTLASS `fmha_sm100`
     路径本身,数千行,且**其自己的 kernel benchmark 就把门槛定在 batch≥16**。
   - crossover 为什么在 16:实测 KV-block 复用度 —— batch≈16 时每个去重 KV 块平均被
     ~4.2 个 query 选中,`M = 4.2 × 16 ≈ 68` 才越过半个 128 行 MMA tile。batch<16 时
     MMA <50% 满,CUTLASS 那套固定 prologue 摊不平。**和讨论点 1 是同一件事。**
   - 附:三次 Triton 改良尝试(融合单 kernel / 保 split 调参 / 纯 retune)全部 ≤ 1.1×
     甚至更慢,数据见 §8——这些是「不值得」的直接证据。

---

## 0. 形状与基线结构(事实梳理)

| 量 | 值 |
|---|---|
| topk 块数 | 16 |
| KV page = sparse block | 128 token |
| head_dim | 128 |
| q heads / kv heads | ≤ 64 / ≤ 4(GQA group = 16) |
| decode_query_len | 1(常规),≤ 32(投机) |
| KV cache 布局 | `[num_pages, num_kv_heads, 128, 2·head_dim]`,前半 K 后半 V |
| 两级间接寻址 | `topk_idx[kh, tok, k]` → 逻辑块 `blk`;`block_table[req, blk]` → 物理页 `page` |

基线(`vllm_msa_ref/sparse_attn.py`)decode 走 **flash-decoding split-K**,两个 Triton kernel:

1. `_gqa_sparse_decode_kernel`:grid `(total_q · NUM_TOPK_CHUNKS, num_kv_heads)`。
   把 16 个 topk 块按 `NUM_TOPK_CHUNKS` 切成 chunk,每个 CTA 处理一个 (query token, kv_head,
   chunk),对 chunk 内的块做局部 flash attention,产出 partial output `[chunk, q, h, d]`
   + partial LSE。
   - `NUM_TOPK_CHUNKS` 由启发式定:`target = clamp(TARGET_GRID(256) // (total_q·num_kv_heads), 1, 16)`,
     再向下取 2 的幂。**batch 越小,chunks 越大**:

     | batch (dql=1) | total_q·kvh | chunks | decode grid CTA | 每 CTA 处理块数 |
     |---|---|---|---|---|
     | 1  | 4   | 16 | 64  | 1 |
     | 2  | 8   | 16 | 128 | 1 |
     | 4  | 16  | 16 | 256 | 1 |
     | 8  | 32  | 8  | 256 | 2 |
     | 16 | 64  | 4  | 256 | 4 |
     | 32 | 128 | 2  | 256 | 8 |
     | 64 | 256 | 1  | 256 | 16 |

2. `_merge_topk_attn_out_kernel`:grid `(total_q, num_heads=64)`。每个 CTA 把某 (token, head)
   的 `NUM_TOPK_CHUNKS` 个 partial 按 LSE 加权合并,写最终 output。

每个 decode CTA 的 tile:`q [BLOCK_SIZE_H=16, BLOCK_SIZE_D=128]`,K/V `[128,128]`;
`tl.dot(q,k)` 是 **M=16** 的矩阵乘。

---

## 1. 讨论点 1 —— arithmetic intensity 与 tensor core

### 手算(一步 decode,seq=8192,per (req, kv_head))

- 读入:16 块 × 128 token × 128 dim × (K+V) × 2 B(bf16) = **1.0 MiB**
- FLOPs:QK `2·16·128·128` + PV `2·16·128·128`,× 16 块 = **1.68e7 FLOP**
- **AI = 1.68e7 / 1.05e6 ≈ 16 FLOP/byte**(与 batch 无关,batch 只是线性放大两边)

全 batch=1:4 个 kv_head → 4 MiB 读,6.7e7 FLOP。

### Roofline(B300)

- HBM ≈ 8 TB/s;bf16 dense tensor ≈ 2.2–2.5 PFLOP/s → ridge point ≈ **~280–320 FLOP/byte**。
- AI=16 落在 ridge 的 ~1/18 处 → **强 memory bound**,离需要 tensor core 的区域差一个数量级。
- fp8 只会让 ridge 更高(~4.5–5 PFLOP/s → ~600),更没戏。

### tensor core 有没有用武之地?——**没有**,两个独立原因

1. **强度不够**:AI=16 « ridge,算力不是瓶颈。
2. **形状不够**:decode 的 GEMM M 维塌成 GQA group = 16。Blackwell `tcgen05.mma`
   的 M 原子是 64 或 128 → M=16 只用到 12–25% 的 MMA 阵列,哪怕喂满也拿不到峰值。
   (这正是 assignment 4.5「瘦 GEMM 达成率塌方」在 attention 上的同一现象;Fireworks 的
   MiniMax-M3 Blackwell 博客也明确写 "The M dimension collapses to the GQA factor (16),
   so the MMA tile is 16×128 — decode-shaped"。)

### 实测佐证(ncu,`_gqa_sparse_decode_kernel`)

| batch | Compute (SM) Throughput | Tensor Active(nsys gpu-metrics) |
|---|---|---|
| 1  | 4.5 % | 0.8 % |
| 4  | 15.5 % | – |
| 8  | 17.1 % | – |
| 16 | 19.3 % | 9.1 % |

→ 整个小 batch 区间 compute < 20%,tensor core < 10%。**结论成立。**

### 联动 4.2 / 4.5

与 4.5 一致:小 M 的 MMA 既受形状惩罚、又在这个 workload 里根本不是瓶颈。
与 4.2 一致:真正决定时间的是「能不能把足够多的独立访存塞进流水线」,即占用与延迟,
不是 FMA 吞吐。

---

## 2. 讨论点 5 —— 先 profile,瓶颈是哪一项(用数据说话)

### 2.1 端到端延迟:batch 1–2 完全持平,knee 在 ~2–4

CUDA graph 基线 e2e(去掉 CPU launch 开销,500 replay 平均;`profiles/FINAL.txt` 合并 run):

| batch | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| e2e (us) | 10.2 | 10.2 | 12.5 | 18.4 | 24.6 | 46.7 | 78.2 |

(注:`graph_time` 对每个 1–2 node 图有 ~4us 固定测量 overhead;纯 GPU 见 §2.2/§2.3。
共享节点跨时间片有 ~15–30% 抖动,故给区间。)

- **b1 = b2 完全持平**(工作量翻倍、时间不动),knee 在 batch 2–4,之后近线性。
- eager(无 graph)下更极端:b1–b16 端到端**恒定 ~32 us**,decode/merge 的 us/call 被
  Python+Triton 的 launch 开销(~10–17 us/launch)完全主导。nsys GPU trace 证实:
  eager loop 里 kernel 之间 GPU 空隙 ~20 us,**GPU 占空比 <25%**。

### 2.2 GPU 侧:没有任何资源吃满(ncu,decode kernel)

| batch | Dur us | DRAM % | Mem BW TB/s | Compute SM % | L1/L2 hit | Achieved Occ % | Waves/SM | No-Eligible % |
|---|---|---|---|---|---|---|---|---|
| 1  | ~10 | 5.3 | 0.40 | 4.5 | ~0 | 6.3 | 0.22 | **81** |
| 4  | ~14 | 15.8 | 1.21 | 15.5 | ~0 | 11.1 | 0.86 | – |
| 8  | ~18 | 24.0 | 1.83 | 17.1 | ~0 | 11.1 | 0.86 | – |
| 16 | ~27 | 33.0 | 2.50 | 19.3 | ~0 | 10.8 | 0.86 | – |

三个可量化的结构性天花板:

1. **占用被寄存器锁死**:195 reg/thread → `Block Limit Registers = 2` → 理论占用
   **12.5%**(8 warp/SM),实测 6–11%。batch 再大也不动。
2. **grid < 1 wave**:decode grid 最多 256 CTA / 148 SM,`Waves Per SM ≤ 0.86`。
   b1 只有 64 CTA → 0.22 wave,**半台机器分不到活**。
3. **split-K 在小 batch 反噬**:chunks=16 让每 CTA 只处理 1 个块 → `load K → QK →
   softmax → load V → PV` 的串行依赖链**没有循环可以流水**,4 warp/SM 完全盖不住
   HBM 延迟(L1/L2 命中率≈0,每块必打 HBM)。ncu:b1 `No Eligible 81%`,
   `Issued Warp Per Scheduler 0.19`(只 19% 周期在发射指令)。

随 batch 增大,grid 填满(b≥4)、每 CTA 块数增多 → 逐渐往 memory-bound 靠
(b16 DRAM 33%),但 **compute 始终 <20%**。

### 2.3 merge kernel:与 batch 无关的固定开销

| batch | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| GPU 时间 nsys (us) | 2.0–3.0 | 2.3–3.4 | 2.5–3.8 | 2.5–3.6 |
| Dur ncu (us,含插桩) | 5.6 | 6.1 | 6.5 | 6.4 |
| Achieved Occ % (ncu) | 5.7 | 11 | 21 | 39 |

工作量 b1→b16 涨 16×,时间**基本不动**(nsys 纯 GPU ~2–4us,ncu/graph-replay 插桩下
~6us)。b1 时 merge ≈ decode+merge 的 1/3,其中没有一点是必要计算——全是「第二次
kernel launch + partial 走 HBM 往返 + 64 CTA 低占用」的固定开销。

### 2.4 瓶颈结论

> **小 batch(1–8)decode 既不是 compute-bound、也不是 HBM-带宽-bound,而是
> 低并行度导致的 HBM 延迟暴露 + 两次 launch 的固定开销。**
>
> 量化:b1 decode roofline 下界 ≈ 4 MiB / 8 TB/s ≈ **0.5 us**,实测 ~4–10 us(时间片相关)→
> **离下界 8–20×**,headroom 全部卡在「占用 12.5% + grid 0.2 wave + 无流水依赖链」;
> 再加 merge 的 ~2–4 us 固定开销(第二次 launch)。

---

## 3. 讨论点 2 —— 两个 kernel 融不融?merge 放 cluster / mbarrier?

### 语义上能融;但「怎么融」是关键,朴素融合会更慢(P3 实测)

split-K + 独立 merge 的意义:当「独立 attention 问题数」(= reqs × kv_heads × dql)
远小于 SM 数时,靠切 K 维造 CTA 填机器。语义上把两个 kernel 合成一个(在线 softmax
直接写出、不产生 partial)完全可以。

**但 P3 试了两条融合路线,结论是「naive 融合在 Triton 里适得其反」:**

1. **不 split + 单 CTA 吃 16 块**(`work/v1_fused.py`):想法是「小 batch 占用本来就是
   死局,不如把单 CTA 延迟压到最低」。实测 **慢 4×**(§8.2)。原因:(a) grid 掉到
   `reqs·kvh` = 4 CTA,把 split-K 唯一的机器填充手段也丢了;(b) Triton 3.8 对
   **运行期** trip count(`real_topk`)的循环**不做软件流水**,`num_stages` 无效,
   16 块的 `load→QK→softmax→load→PV` 全串行 → ~2.5us/块 × 16。
2. **保 split-K grid + 把 merge 折进 kernel**:需要 chunk CTA 之间的 grid-wide 同步
   (原子计数 + spin-wait,或 cluster + DSMEM)。这才是对的方向,但 **Triton 3.8 没有
   干净的 grid 屏障 / DSMEM reduce 原语**,正确且保证前进地实现极脆(见 §8.5)。

所以「融不融」的诚实回答:**该融,但在 Triton 里融不动**——这本身就是 challenge (b)
的证据之一。真正做对需要 CUDA 级的持久 kernel + cluster combine(= CUTLASS 路径)。

### merge 放 cluster / mbarrier 里做

如果坚持 split-K 结构,可以用 **SM90+ thread block cluster**:让处理同一 (token, head)
的 `NUM_TOPK_CHUNKS` 个 CTA 组成一个 cluster,partial 写 **distributed shared memory**
(DSMEM)而不是 HBM,cluster 内一个 CTA 用 `mbarrier` 等齐后就地 merge。省掉 partial
的 HBM 往返和第二次 kernel launch。

可行性判断:
- **可行**:chunks ≤ 16,cluster 最大 8(实测 ncu `Max Cluster Size = 8`)或 16,放得下;
  partial `[chunks, 16, 128]` fp32 ≈ chunks·8 KiB,DSMEM 够。
- **但收益有限**:merge 的固定开销里,partial 往返 HBM 只是一小部分(partial 很小,
  b1 每 chunk `16·128·4`=8 KiB,16 chunk=128 KiB,~16 ns @ 8 TB/s);大头是
  **kernel launch + 64 CTA 低占用的固定延迟**。cluster 省掉第二次 launch 更值钱。
- Triton 3.8 对 cluster / DSMEM 的支持不完整(`num_ctas` 有,DSMEM reduce 要手写
  `tl.inline_asm` 或等 `tl.async` 原语)。**工程上直接融合成单 kernel 比 cluster-merge 简单**。

**结论:**
- merge 的固定开销(nsys ~2–4us / 插桩下 ~6us)确实值得消,但消它需要 grid-wide
  combine(cluster+DSMEM 或 semaphore spin-wait);
- cluster/mbarrier 版 merge **理论可行**(chunks ≤ 16、cluster ≤ 8、partial ≤ 128 KiB
  放得下 DSMEM),省掉第二次 launch 比省 partial 往返更值钱(partial 很小,~16ns);
- 但 Triton 3.8 表达不了,CUDA 里能做——又回到「值不值得为这 ~2–4us 写 CUDA」(§8)。

---

## 4. 讨论点 3 —— 两级间接寻址 vs TMA tensor map

### 文档考证

TMA(`cp.async.bulk.tensor` / `cuTensorMapEncodeTiled`,Hopper SM90+ / Blackwell SM100):

- tensor map 是**主机端**(或 sm_90a+ 设备端 `tensormap.replace`)编码的**固定仿射描述符**:
  base 指针、≤ 5 维、每维 size / global stride(须 16 B 对齐)/ box size / element stride。
- 一次 bulk tensor copy 的参数是**整数坐标** `tensor_coords[]`,硬件按
  `base + Σ coord_i · stride_i` 算地址搬一个 box。
- **TMA 没有 gather / 索引数组 / 指针表间接寻址模式**(CUDA Programming Guide
  "Asynchronous Data Copies" 一节;各家 paged-attention 实现的共识,例如
  Ragged Paged Attention 论文:"Paged KV makes it difficult to use the TMA ...
  instead, cp.async ... each thread separately issues individual load instructions";
  FlashInfer:"TMA ... doesn't support non-affine memory access patterns,
  so TMA is only used for contiguous KV-Cache on Hopper")。

### 那这道题的两级间接能不能用 TMA?

**逐级看:**

| 级 | 内容 | TMA 能否表达 |
|---|---|---|
| ① `blk = topk_idx[kh, tok, k]` | gather 16 个逻辑块号 | ❌ 数据依赖 gather,kernel 自己标量 load |
| ② `page = block_table[req, blk]` | 逻辑块 → 物理页 | ❌ 指针表查找,kernel 自己标量 load |
| ③ `kv_cache[page, kh, :, :]` 这一块 `[128,128]` | 结构化 2D tile 拷贝 | ✅ **可以**:把 page 作为 tensor map 的第 0 维坐标 |

也就是说:**把 KV cache 整体 `[num_pages, kv_heads, 128, 2·hd]` 编成一个 tensor map,
kernel 里用两次依赖标量 load 解析出 `page`,然后对每个选中块发一条
`cp.async.bulk.tensor`,坐标 = `(page, kh, 0, 0)`**。这正是 FA3 / FlashInfer /
TRT-LLM 对 paged KV 的做法——**一页一条 TMA**,间接寻址由 kernel 算坐标完成,
TMA 只负责最后那段仿射搬运。

代价与权衡:
- 16 个选中块 = 16 条独立 TMA(不能一条搞定任意 16 页的 gather);
- ② 的标量 load 上关键路径,但 `block_table` 小、L2 常驻,开销可忽略;
- 收益:地址计算卸载给 TMA 单元 + 更大的在途传输 + cluster 内 TMA multicast(同一页
  被多个 query 选中时,SM100 上可 multicast 到 cluster 各 CTA 的 smem)。

### 实验验证(`work/exp_tma.py`,B300 实跑)—— **已完成**

两个 Triton kernel 从 paged cache 取同一 `[128,128]` tile,`page` 来自运行期 device
tensor(模拟 block_table gather 后的结果):

| kernel | 写法 | 生成的 PTX(实测) | 正确性 |
|---|---|---|---|
| `load_plain` | `tl.load(base + page*stride + …)` | `ld.global.v4.b32 ×N`(**同步**向量 load,无 async / 无 TMA) | ✅ |
| `load_tma` | host `TensorDescriptor.from_tensor(kv2,[128,128])` + `desc.load([page*128, 0])` | `mbarrier.init` / `mbarrier.arrive.expect_tx … 32768`(=整块 32 KiB) / **`cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [smem], [desc, {r4,r5}], [bar]`** / `mbarrier.try_wait.parity` | ✅ |

**结论(实验证据 + 文档考证一致):**
- **运行期 `page` 索引 + TMA 描述符可以工作**——`{r4, r5}` 就是运行期坐标,其中一维
  = `page*128`。Triton 3.8 直接把它 lower 成硬件 TMA + mbarrier 完成协议。
- gather(topk_idx)与 block_table 查表这两级,**TMA 表达不了**(无 gather 模式,与
  CUDA Programming Guide 一致),仍是 kernel 自己的两次依赖标量 load;
- 但**最后那段 `[128,128]` tile 搬运,可以一块一条 `cp.async.bulk.tensor`**,硬件算
  地址 + mbarrier 完成。这正是 FA3 / FlashInfer / TRT-LLM 对 paged KV 的做法。
- 所以对 CUDA(或 Triton+TMA)重写而言,**KV load 路径不是卡点**;卡点是围绕它的
  warp 专化流水(§8.2)。

现在基线是 `tl.load` 算指针 → `ld.global.v4.b32`(同步)。在 b1 这不是瓶颈(带宽 5%),
瓶颈是在途请求不够。用 TMA 或 `cp.async`+`num_stages` 主要买的是「地址计算卸载 +
更深的在途缓冲 + cluster multicast」,而不是省带宽。

---

## 5. 讨论点 4 —— FP8 KV cache 的 scale 放哪一层

### 上游口径(`test_sparse_attn_fp8_scale.py` + `sparse_attn.py` 的 `KV_SCALE_MODE`)

- kv_cache 以 fp8(e4m3)存;`k_scale` / `v_scale` 两种形态:
  - **scalar**(`KV_SCALE_MODE=1`):整张 cache 一个 scale;
  - **per-token/head**(`KV_SCALE_MODE=2`):`[num_kv_heads, max_kv_tokens]`,
    按 `(page·128 + 位置)` 索引。
- kernel 内在 **load 之后、`tl.dot` 之前** 做:`k = k.to(q.dtype); k = (k * k_scale)`。
  即 **dequant 在片上(register),在 QK 之前**;V 同理,dequant 后再 `p @ v`。
- 测试判据:fp8 + 正确 scale 的结果对「用 dequant 后的 bf16 cache 跑同一 kernel」
  `rtol=atol=2e-2`;且不带 scale 的结果必须**明显偏离**(防止 scale 被无视也能过)。

### 放哪一层——分析

| 方案 | 位置 | 评价 |
|---|---|---|
| A. load 后片上 dequant(现状) | kernel 内,QK/PV 前 | K 的 per-token scale 乘在 `[128]` 向量上,几乎免费;fp8 load 省一半带宽。**小 batch 下没有坏处**(反正不是带宽 bound),大 batch 直接受益。 |
| B. QK 之后按 scale 修正 | 对 `qk` 乘 `k_scale`(仅 scalar 模式可行) | per-token 模式做不了(scale 已被 max/sum 混掉);省一次 `[128,128]` 乘法,但那不是瓶颈。不值。 |
| C. 预先整张 dequant 成 bf16 | 独立 pass | 废掉 fp8 的带宽优势 + 多一次 HBM 往返。只有 debug 时用。 |
| D. merge 层 | ——不可能 | scale 是 per-KV 的,softmax 之后信息已丢失。 |

**结论:A 是对的**——per-token/head scale 必须在 load 之后、点积之前,在片上对 K/V
向量逐元素乘。这一层也是唯一能同时:(1) 吃到 fp8 的带宽红利,(2) 保留 per-token
粒度,(3) 不引入额外 kernel 的位置。小 batch 场景 scale 处理的开销可忽略(不是瓶颈),
所以**融合后的单 kernel 沿用 A 即可**,无需为小 batch 特别设计。

CUTLASS 路径(`msa_cutlass_sparse_decode.py`)把 `q_scale/k_scale/v_scale` 作为
**标量** float 传给 `fmha_sm100`(`use_fp8_kvcache=True`),即只支持 scalar 模式,
per-token 仍得走 Triton —— 这也是 CUTLASS 路径「形状受限」的一部分。

---

## 6. 讨论点 6 —— 验收方案(先贴出来给别组挑毛病)

### 6.1 正确性

**参照物(3 档,从强到弱):**

1. **主参照 —— fp32 稠密 SDPA**(`harness/ref_sdpa.py`):逐 (token, kv_head) 把选中块
   聚齐,fp32 标准 softmax。语义与 kernel 对齐(逻辑块经 block_table、只取前
   `real_topk`、块内 ≥ kv_len 掩掉)。
   - 判据:`err_ratio = ‖got − ref‖ / ‖ref‖ < 2e-2`(沿用 harness 口径,bf16 输入下
     这个阈值 ~10× 于观测到的基线误差 3.2e-3,留足余量但能抓住真错)。
2. **同构参照 —— 基线 Triton kernel 自身**:同一批 case,新 kernel vs
   `minimax_m3_sparse_attn_decode`,`err_ratio < 5e-3`(两者都是 bf16 + base-2 softmax,
   差异应只有累加顺序)。这一档能抓住「对了参照但错了数值路径」的 bug。
3. **FP8 档**(讨论点 4 口径):新 kernel 跑 fp8 cache + scale vs 新 kernel 跑
   dequant-bf16 cache,`rtol=atol=2e-2`;且「无 scale」结果必须偏离 > 1e-1。

**形状覆盖(每档都要过):**

| case | 目的 |
|---|---|
| batch ∈ {1,2,4,8,16},seq=8192,dql=1 | 主场景 |
| 短序列 seq ∈ [50,300](< 1–3 块) | 尾块 / `real_topk < 16` / causal 边界 |
| 混合 seq ∈ [1024, 8192] | block_table 不等长、物理页打乱 |
| dql ∈ {2, 4}(投机 decode) | query 位置映射、逐 token causal |
| topk 槽含重复块 / 填 0 尾槽 | 语义鲁棒性(synth 里已有约定) |
| seq 刚好 = 128·k(整块) vs 128·k+1 | off-by-one |

**每个形状 5 个随机 seed 全过**(沿用 judge 脚本 5-seed 惯例)。

**回归门槛:** 上述全绿才允许比性能;任何一档 FAIL 直接判不通过。

### 6.2 性能

**对比对象:**

1. vs **基线 Triton**(`minimax_m3_sparse_attn_decode`,同一 case,CUDA graph 计时):
   目标 —— batch ∈ {1,2,4,8} 端到端延迟 **↓ ≥ 30%**;batch ≥ 16 **不回退**(≤ +5%)。
2. vs **上游 CUTLASS 路径 在 batch ≥ 16**:CUTLASS 需要 fp8+sm100+固定形状,
   本 harness 未接入 `fmha_sm100`,故这一项**用 I/O cost model 推算 + 文字论证**
   (见 §7),不做实测对比;若时间允许,尝试在 harness 里 stub 一个 `fmha_sm100`
   最小调用做 sanity。

**计时口径:**
- CUDA graph replay,warmup 50 + 计时 500,报 median 与 p10/p90;
- 报 GPU-only(nsys `cuda_gpu_kern_sum`)与 e2e(graph replay)两个数;
- 每个数据点重复 3 次(不同时间片,规避共享节点抖动),报区间。

**验收指标(P3 结果,见 §8):三次改法均未达到「batch∈{1,2,4,8} ↓≥30%」门槛,
故按验收方案自身规则判定「(a) 方向在 Triton 内不通过」,转 (b)。**

### 6.3 接受别组挑战的点(自曝弱点)

- SDPA 参照是 fp32、非 base-2 softmax,与 kernel 的 exp2/log2 路径有系统性小偏差 →
  用「同构参照(档2)」兜底。
- 合成 topk 是随机均匀选块,真实 indexer 的选择有局部性(近块更可能被选)→
  会**低估**真实场景里块复用带来的 L2 命中,故本报告的带宽结论偏保守(实际更不 BW-bound)。
- CUDA graph 计时掩盖了真实推理里 attention 前后的依赖气泡;e2e 数只在「其它层足够重、
  能填满」的假设下才代表端到端收益。
- 未覆盖 TP 切分后 `num_q_heads < 64` / `num_kv_heads < 4` 的形状。

---

## 7. CUTLASS crossover 为什么在 16(讨论点 3 挑战侧 / 讨论点 1 的延伸)

上游 `_MIN_CUTLASS_BATCH_SIZE = 16` 的注释:"Kernel benchmarks put the CUTLASS
crossover at 16 requests for TP1 and TP4"。CUTLASS decode(`fmha_sm100`,
`sparse_kernel_mode="decode"`)是 **KV-stationary / 持久化 128×128 tile** 结构:
对每个去重的 `(kv_head, page)`,收集这一批里选中它的 query token,跑**一个** MMA。

那个 MMA 的 M 维 = **`avg_reuse × GQA(16)`**,`avg_reuse` = 每个去重块被多少个
query token 选中。用 `work/exp_crossover.py` 合成 top-k(两种选块模型:均匀随机 /
共享前缀局部性),seq=8192:

| batch | avg_reuse | M = reuse·16 | MMA 填充率(/128) |
|---|---|---|---|
| 4  | 1.5–2.0 | 25–32 | 20–25 % |
| 8  | 2.4–2.8 | 38–45 | 30–35 % |
| **16** | **4.2–4.3** | **68–69** | **53–54 %** |
| 24 | 6.3 | 100 | 78 % |
| 32 | 8.1–8.4 | 129–135 | 100 % |

→ **`M` 恰在 batch ≈ 16 越过 64**(半个 `tcgen05` MMA tile)。机制:

- batch < 16:每个去重 KV 块平均被 < 4 个 query 选中 → KV-stationary 的 MMA `M < 64`,
  tensor core < 50% 满 → CUTLASS 那套**固定 prologue**(148 SM 上的持久网格、planner
  metadata、`fmha_sm100_plan` 里 65536 行的分配、TMA 描述符 + TMEM init、scheduler)
  摊不平。此时「一 query 一路」的 Triton split-K 反而快。
- batch ≥ 16:`M ≥ 64` 且往 128 走(batch 32 打满)→ KV-outer 的「每块只加载一次 +
  128 宽 MMA + 无 merge kernel」压过它的 prologue。
- 这**就是讨论点 1 / assignment 4.5**:decode 里唯一能撑大 attention GEMM 的 M 维的
  旋钮就是 batch(通过跨 batch 的 KV 块复用)。crossover 不是 CUTLASS 的实现常数,
  是「M 维够不够填 MMA tile」这件事在 batch 轴上的投影。

(注:Fireworks 博客的 `nsb/N < 2.85` 阈值针对的是 **prefill** 的 Q-outer vs KV-outer
选择——那里 Q-outer 是 L2-BW bound、KV-outer 是 HBM-BW bound;seq≈8k 的 decode 下
`nsb/N` 在 batch 1–64 始终 ≥ 4,不是 M3 decode crossover 的驱动量。上面用 `M` 而非
`nsb/N` 才对上 16 这个数。)

---

## 8. 挑战:选 (b) —— 「小 batch 不值得做 CUDA 重写」的完整证据链

### 8.1 收益上限(从 roofline 推)

| 层级 | b1 一步 decode | 依据 |
|---|---|---|
| HBM 带宽下界 | **~0.5 us** | 4 MiB KV / 8 TB/s(§1) |
| 当前 decode kernel(GPU) | ~4–10 us | nsys / ncu(§2.2) |
| 当前 + merge(GPU) | ~6–12 us | + merge ~2–6 us(§2.3) |
| 当前 e2e(CUDA graph) | ~10 us | §2.1;含 ~2 us/kernel replay 地板 |
| **理论「完美融合单 kernel」** | **~4 us GPU / ~6 us e2e** | 删 merge + 满流水 decode |
| → 收益上限 | **~1.7× e2e(乐观)** | 10 → 6;若无法同时提速 decode body 只 ~1.3× |

为什么 0.5us 的下界够不着:b1 只有 `4 (req·kvh) × 16 blocks = 64` 个块任务,填不满
148 SM;把这 64 个任务的依赖链(load→QK→softmax→load→PV)压到一次 HBM 延迟以内,
需要每个 CTA 多 warp + 多 buffer 的软件/硬件流水。**这正是 Triton 给不了的部分**
(见 §8.2)。

### 8.2 Triton 框架内实际可达上限 ~1.3–1.7×(三次改法实测)

| 改法 | 文件 | b1 e2e(graph) | vs 基线 | 结论 |
|---|---|---|---|---|
| 基线 | `vllm_msa_ref/sparse_attn.py` | 10.2 us | 1.00× | — |
| **A. 融合单 kernel(不 split + 在线 softmax 直接写出)** | `work/v1_fused.py` | 41.0 us | **0.25×** | grid 掉到 4 CTA(去掉 split-K 唯一的填充手段);`tl.range` 运行期 trip count → Triton 不做循环流水 → 16 块全串行(nsys 实测 ~39us GPU,~2.5us/块)。正确性 11 形状 × 5 seed 全过,但慢 4×。b1–b8 恒 ~41us(grid < 148),b64 才 1.09×。 |
| B. 保 split-K grid + 调 `NUM_TOPK_CHUNKS` / `num_warps`(decode & merge) | `work/p3_retune.py` | 10.2 us | 1.00× | b8 偶得 1.12×,b4 反而 0.86×;基线默认 config 已在前沿。 |
| C. 纯 retune(不改结构) | 同上 | 10.2 us | 1.00× | CUDA-graph e2e 有 ~2 us/kernel replay 地板,多组 config 结果精确相同到小数点后 2 位。 |

三次都 ≤ 1.1×(A 甚至 4× 慢)。真正能动的是:**去掉第二次 launch + 让 16 块循环流水**——
前者 Triton 3.8 无干净的 grid-wide 屏障(FlashInfer 用 semaphore + spin-wait 的
single-pass 合并,Triton 里做正确且保证前进极脆),后者需要 constexpr 循环 + warp 专化。

### 8.3 工程成本(从 CUTLASS 路径的复杂度估)

要拿到 §8.1 里「~4us GPU」那一档,需要的手段清单 = 上游 CUTLASS decode 路径本身:

| 手段 | 在哪 | 复杂度 |
|---|---|---|
| warp-specialized TMA producer / consumer 流水(每块一条 `cp.async.bulk.tensor`,§4 已验证可行) | `fmha_sm100` kernel | 高:PTX/CuTe 级别,mbarrier 环形缓冲 |
| TMEM 累加器(`tcgen05`)+ 持久 scheduler | `fmha_sm100` | 高:Blackwell 专有,SM100-only |
| planner / 固定分配 / cudagraph-stable 地址 | `msa_cutlass_sparse_decode.py`(316 行)+ `fmha_sm100_plan` | 中高:65536 行预分配、plan cache、`_update_runtime_metadata_kernel` |
| cluster / DSMEM 合并 partial(替代 merge kernel) | `fmha_sm100` combine 阶段 | 中:SM90+ cluster + DSMEM reduce |
| fp8 KV + q/k/v scale(只支持 scalar) | 同上 | 中 |
| 形状约束(page=128 / topk=16 / heads / dql≤32 / batch≥16) | `supports_cutlass_sparse_decode` | —— 说明它**很脆**,换形状就回退 Triton |

即上游为这条路径付出的成本是「几千行 Blackwell 专有 + 一个 planner 子系统 + 一堆形状
gate」,而**它自己的 kernel benchmark 把开门槛设在 batch ≥ 16**——因为 §7:batch<16
时 `M<64`,这套机器喂不饱。

### 8.4 结论

> **对 batch < 16 的延迟敏感 decode:收益上限 ~1.3–1.7×(Triton 内)/ 至多到
> roofline 的 ~2–3×(CUDA 内),但后者的工程成本 = 重写一条 Blackwell 专有的
> warp-specialized fmha 路径。上限 < 成本,不值得做。**
>
> 这与上游的实际选择一致:CUTLASS 路径存在,但 opt-in 且 `batch ≥ 16` 才启用;
> batch < 16 继续由 Triton split-K 承担是正确的工程决策。本报告补上了「为什么 16」
> 的量化依据(§7 的 `M = reuse·16` 越过 64)和「Triton 内还能压多少」的实测上限(§8.2)。

### 8.5 如果一定要在小 batch 榨一点(低风险增量,非重写)

1. **自适应 split + single-pass 合并**:batch 小(问题数 < ~SM/2)时,split-K 产 partial
   后由 chunk-0 CTA 经全局原子计数 + `griddepcontrol` / spin-wait 就地合并,省第二次
   launch(~2 us graph)。需要仔细处理前进保证(b1 时 64 CTA < 148 全驻留,安全)。
   预期 ~1.3×。
2. **降 decode kernel 寄存器**(现 195/thr):拆 `BLOCK_SIZE_D=128` 为 2×64 或复用
   `qk` buffer,把占用从 12.5% 提到 25% → b4–b16(grid 已满)受益 ~1.1–1.2×;
   b1(grid 0.2 wave)无感。
3. **`NUM_TOPK_CHUNKS` 启发式改成按 wave 对齐**:目标 `total_q·kvh·chunks ∈ [148, 296]`
   且每 CTA ≥ 2 块(可 constexpr 展开流水),而不是现在无脑 `min(16, 256/…)`。
