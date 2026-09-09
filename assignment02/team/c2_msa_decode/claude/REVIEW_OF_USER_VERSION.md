# 对 `../work/`(用户版本)的点评

先说总体:这一版**方法论是对的、证据是真的、结论(b)我也同意**。测量先于设计、每个讨论点都有"结论 + 证据"、
验收方案先贴、三次 (a) 尝试的失败数据都保留并写进证据链、GPU session 日志完整(含 ncu `-s` 这种踩坑)——
这些都是这道题最看重的部分。下面的点评按"会被别组挑战的地方"排序,先重后轻;每条都附我这边的对照数据(分支 `c2-claude`,
`claude/` 目录)。

## A. 会被挑战的硬伤(建议答辩前修)

### A1. crossover=16 的解释建立在一个未经核实的前提上(§7,`exp_crossover.py`)
- 报告说 CUTLASS decode 是 "KV-stationary:对每个去重 (kv_head, page) 收集选中它的 query 跑一个 MMA,M = avg_reuse × 16,
  b≈16 时 M 越过 64"。但 `msa_cutlass_sparse_decode.py` 的 decode 调用是 `fmha_sm100(..., kv_block_indexes=topk, ...)`,
  文件头写的是 "per-query-token page indices",plan 是按 request 的 `qo_lens` 建的;**K2Q CSR(`build_k2q_csr`,真正的
  KV-stationary 结构)只在 prefill 路径出现**(`sparse_attention_msa.py` 第 266–308 行)。decode 路径从代码看是 Q-outer:
  每个 token 吃自己的 16 块。KV-stationary 的前提没有代码证据。
- `exp_crossover.py` 的 "reuse" 把**不同 request 的同一逻辑块号当成同一物理页**(注释里也写了 "page shared across reqs")。
  harness 的 `synth.py` 每个 request 有自己打乱的物理页;真实 vLLM 里不同 request 也只有 prefix caching 时才共享页。
  所以 b=16 时 "每块被 4.2 个 query 选中" 是合成数据的假象:dql=1 时每个 request 只有 1 个 token,跨 request 无复用,reuse 恒 = 1。
  脚本里还有一行 `per_head_blocks[kh] |= {...} if False else set()` 的死代码,和 docstring 里残留的 nsb/N<2.85 口径,说明这段
  改过几轮但没清理。
- 对照:我用本机可标定的模型解释 16(REPORT §8):流式 kernel `T = T_fix + max(16·c_blk, b·4MiB/BW)` 对 split-K 的实测曲线,
  交点只由每块边际成本 c_blk 决定——c_blk=1.4 us(mma.sync,实测)交点 b≈32,c_blk≈0.8(MMA 全异步)交点 b≈16。
  我的 cl1s3 kernel 实测在 b≈24–32 反超 Triton,现象可复现。建议把 §7 改成这个口径,或至少把 KV-stationary 标成假设。

### A2. 讨论点 1 "tensor core 双重失效" 的第二重不成立于基线
- "tcgen05 的 M 原子是 64/128,M=16 只用 12–25%" 对 CUTLASS 路径成立,**但 Triton 基线在 sm_103 上生成的是
  `mma.sync.aligned.m16n8k16.bf16`(我 dump 了 `~/.triton/cache/.../*.ptx`),M 恰好 = 16,一行不浪费**。这一条应改为:
  "Triton 用 legacy mma.sync,形状刚好;CUTLASS 用 tcgen05,M 塌成 16 才浪费 3/4"。
- 更重要的反面:我实测 B300 上 FFMA 只有 38 TFLOPS(1.1 GHz),要跟上 8 TB/s 需 ≥130 TFLOPS → **不用 tensor core 连带宽都喂不饱**
  (REPORT §3)。"tensor core 没有用武之地" 这句会被这样反问。建议改成"算力不是瓶颈,但 tensor core 是达到带宽的必要条件"。
- 顺带:B300 上 `mma.sync` bf16 只有 ~320 TFLOPS(tcgen05 峰值的 1/7,`claude/exp/mma_rate.cu`),这是 Triton 路径与 CUTLASS 路径
  真正的差别所在,报告里没有这一层。

### A3. "共享节点、时钟不锁、~2× 抖动" 的归因不对
- 我用 globaltimer 对 clock64 实测:**SM 时钟固定 1095 MHz**,冷/热/重载都不变(`claude/logs/clock_final.out`),
  `nvidia-smi` 报 Applications Clocks 2032 但当前 1095。所以不是"不锁频抖动",而是**恒定跑在 54% 标称时钟**。
- 你看到的 b1 decode 3.9 / 6.3 / 8.2 / 10.8 us 更可能来自**测量方法不同**:nsys 纯 kernel(3.9–6.3)、graph 内单 kernel
  (8.2)、ncu 默认 `--clock-control base` + 插桩(10.8)。ncu 的 Duration 在报告里多处被当成 GPU 时间(merge "~6 us"),
  应统一口径。我的 graph 数字与你的 session-3 graph 数字一致(b1 decode 5.9–8.2,merge 2.7–4.1)。

### A4. "Triton 对运行期 trip count 不做软件流水" 是断言不是证据
- 我看了基线的 TTGIR:循环是 `scf.for`(运行期上下界)且**有** `async_commit_group/async_wait` 结构、`num_stages=3`,
  只是 K/V 的 smem 缓冲深度为 1,效果是"下一块的载入在本块计算之后才发起"。结论(每块串行 ~2.3 us)对,机制描述不准。
  `v1_fused.py` 的 `tl.range(0, real_topk, num_stages=N)` 慢 4× 的根因与我的 cl1 版本一致(16 块串行 × 1.4–2.3 us),
  但要说 Triton "不流水",应贴 TTGIR/PTX 证据。

## B. 需要补强的地方

### B1. 收益上限只有推断,没有"地板"实测
- §8.1 的 "理论完美融合 ~4 us GPU / ~6 us e2e → 1.3–1.7×" 方向正确,但缺硬证据。我用三个探针 kernel 给出任何 1 块/CTA 结构
  逃不掉的地板:空 kernel 0.9、索引链 1.8、索引 + 一块 TMA 3.3 us(b=1,graph),再加一块 compute 1.3 与合并 ≥1 → ≥5.6 us,
  上限 ≈1.5×,与你的区间一致——可以直接引用来把 §8.1 变成实测。
- 少了端到端换算:每层省 2–3 us × ~80 层 / ~20 ms per step ≈ 1%。这一句对 (b) 最有说服力。

### B2. 讨论点 4 的结论与实测相反
- 报告说方案 A(kernel 内逐元素反量化)"小 batch 下没有坏处"。我实测上游 fp8 路径 **比 bf16 慢 30–60%**(b=1:8.3 → 11.0/11.8/12.5 us;
  `claude/logs/e10_*.out`),因为转换指令直接加在串行链上而带宽根本不缺。方案 B(标量 scale 折进 Q + 输出)被你判为"不值",
  但它 kernel 零成本且与上游口径在 2e-2 内(err 4e-3)——CUTLASS 路径正是这么做的(q/k/v_scale 标量传给 fmha_sm100)。
  建议:标量 → host 折叠;per-token → 加在 S/P tile 上(对 16×128 做列缩放)而不是反量化 128×128 的 K/V。
- 你写 "K 的 per-token scale 乘在 [128] 向量上几乎免费" 不对:基线代码是 `k * k_scale[None, :]` 作用在 [128,128] 的 K tile 上。

### B3. 讨论点 2 只讨论了 cluster 的可行性,没有做/没有成本数字
- 我做了两种融合:cluster+DSMEM(两次 cluster.sync + 合并 = 3.7k cycle ≈ 3.4 us,**比 merge kernel 2.7 us 还贵**)和
  last-CTA 原子合并(Triton 就能写,b=1 反而 10.8 vs 8.3 us)。"该融但融不动"的判断是对的,但可以用数字说:融合的代价在本卡上
  ≥ merge kernel 本身。另外 cluster=16 会因 GPC 内凑不齐 16 个空 SM 而串行化(b=1 40 us),你 §3 写 "cluster 最大 8 或 16,放得下"
  应补这一条。

### B4. 讨论点 3 少了 PTX 原文与两个细节
- 文档考证引的是 CUDA Guide 概述、论文和博客;可直接引 PTX ISA:坐标 "are of type .s32"(运行期);`.tile::gather4`
  (PTX 8.6,sm_100)确实存在,"TMA 没有 gather 模式" 不严格——只是 4 行、2D、本题用不上;PTX 9.4 的 `.override::global_address`
  就是给 paged 场景设计的(CUDA 13.4 才有)。摘录见 `claude/DOCS.md`。
- `exp_tma.py` 用 Triton 的 TensorDescriptor 验证了 2D 描述符 + 运行期坐标,好;但 box 是 [128,128] bf16 = 行 256 B,
  没有 swizzle(Triton 默认会选),值得写明 swizzle 的限制(128B swizzle 要求 box 内维 ≤128 B,所以 CUDA 版要拆成 4 个 16 KiB box)。

### B5. 验收方案
- 三档参照 + 形状矩阵 + 自曝弱点写得好。两点可补:(1) 逐 head/逐 token 的局部误差阈(全局范数比会掩盖单 head 错位);
  (2) "不支持的形状必须显式报错"。性能门槛 "b≤8 ↓≥30%" 是好设计——并且按它判 (a) 不通过、转 (b),逻辑自洽。

## C. 小问题
- PLAN.md 说"全部完成",PROGRESS.md 末尾说"下一步:把 REPORT §8 换成完整 P3 章节",两处不一致。
- REPORT §2.1 "b1 = b2 完全持平,knee 在 2–4" 与 §2.2 的 nsys 表(b1 3.9 → b4 5.3)和我的数据(8.2 → 9.2 → 11.2)略有出入:
  graph 下 b2 比 b1 多 ~1 us,不是"完全持平";持平的是 eager(host 开销主导)。
- `p1_measure.py` 的 eager per-kernel us(decode 17、merge 10)是 CPU launch 时间,表里应标明,否则会被当成 GPU 时间读。
- 报告多处引用 Fireworks 博客的 "nsb/N < 2.85" 又说它不适用,可以删掉减少干扰。
- `v1_fused.py` 正确性覆盖(11 形状 × 5 seed)比我的多 fp8 模式,这点比我好;但性能表里 "b64 1.09×" 应说明是 grid 已满的情形。

## D. 我这边与你一致的结论(可互相引用)
- 瓶颈定性(延迟 + 固定开销,非算力/带宽)、AI≈16、merge 是纯固定开销、eager 被 host 主导、split-K 是小 batch 唯一的填充手段、
  不 split 的单 CTA 流式在 b≤16 慢 3–4×、(b) 不值得——全部一致,且数字在同一区间。
- 我额外有的、可以直接用的材料:`claude/exp/msa_decode.cu`(CUDA 融合 kernel,全过验收)、`claude/logs/e13_ablation_final.out`
  (分相 + 消融)、`claude/logs/mma_rate_final.out`(mma.sync 吞吐)、`claude/logs/clock_final.out`(时钟)、`claude/DOCS.md`(PTX 原文)。
