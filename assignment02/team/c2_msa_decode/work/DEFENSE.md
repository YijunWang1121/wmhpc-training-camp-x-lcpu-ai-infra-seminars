# C2 答辩要点

## 一句话
小 batch decode 的瓶颈是**低并行度下的 HBM 延迟暴露 + 两次 launch 的固定开销**,
不是算力也不是带宽;tensor core 因 AI≈16 且 M≈16 双重失效。挑战选 **(b)**:
Triton 内收益上限 ~1.3–1.7×(实测三改法佐证),达到 roofline 需重写 Blackwell 专有
fmha 路径 = 上游 CUTLASS 路径的成本,而它自己也只在 batch≥16 才启用。

## 预期质疑 & 回应

**Q: 你 profile 的绝对时间不稳(b1 decode 见过 3.9/6.3/10.8us),结论可信吗?**
A: 共享节点 + 不锁频。但结论不依赖绝对值:(1) 每次测量内的 scaling 一致(decode
sub-linear、merge 恒定);(2) ncu 的比率指标(Compute 4.5% / DRAM 5.3% / No-Eligible
81% / 占用 6%)对时钟不敏感;(3) 三个方法(cuda event / nsys / ncu)定性一致。

**Q: batch 1–16 端到端持平,会不会是你的 harness/计时有问题?**
A: 分三层验证过。eager(Python 循环)持平是 Triton launch 开销主导(nsys 看到
kernel 间 20us GPU 空隙)。CUDA graph 去掉 launch 后 b1≈b2 仍持平、b4 才起涨——
因为 decode grid 在 b1 只有 64 CTA(0.22 wave),b4 才 256 CTA(0.86 wave)填满,
之前加 batch 只是让更多 SM 有活干、不加时间。nsys 纯 GPU kernel 时间也印证:
b1 3.9us → b4 5.3us(工作 4×,时间 1.35×)。

**Q: 为什么不干脆做 (a)?融合 + 提占用听起来不难。**
A: 做了,三种改法都在 `work/`,数据在 §8.2。融合单 kernel(v1_fused.py)正确性
11 形状 × 5 seed 全过但**慢 4×**——去掉 split-K 就去掉了小 batch 唯一的填充手段
(grid 掉到 4 CTA),而 Triton 3.8 对运行期 trip count 的循环不做软件流水,16 块
全串行。retune 基线(num_warps × chunks sweep)= 1.0×。真正要动的是「消掉第二次
launch + 让块循环流水」——前者 Triton 无干净 grid 屏障,后者要 warp 专化。

**Q: crossover=16 你归因于 M 维,证据?**
A: `work/exp_crossover.py`:合成 top-k,数每个去重 KV 块被多少 query 选中。
batch 16 时 avg_reuse≈4.2 → KV-stationary MMA 的 M = 4.2×16 ≈ 68,刚过半个
128 行 tcgen05 tile;batch 32 时 M≈130 打满。batch<16 时 MMA<50% 满,CUTLASS
的固定 prologue(148 SM 持久网格 + planner + TMA/TMEM init)摊不平。Fireworks 的
nsb/N<2.85 是 prefill 口径,seq~8k decode 下 nsb/N 全程 ≥4,对不上 16。

**Q: TMA 到底能不能用于两级间接?**
A: `work/exp_tma.py` 实测:运行期 page 索引 + Triton TMA 描述符**能工作**,PTX 出
`cp.async.bulk.tensor.2d ... {r4,r5}`(r 是运行期坐标)。gather(topk)和 block_table
查表 TMA 表达不了(无 gather 模式,文档一致),但最后 [128,128] tile 一块一条 TMA
可以。所以 KV load 不是重写的卡点。

**Q: 验收方案里 SDPA 是 fp32 非 base-2,不会误判?**
A: 所以设了「同构参照」档(vs 基线 Triton kernel 自身,err_ratio<5e-3),专抓
「对了物理参照但错了数值路径」。实测新 kernel vs 基线 3.4e-3,vs SDPA 2.6e-3,
两档都过。

## 数字速查(B300,共享节点,区间)
- b1 decode kernel:GPU 3.9–10us,grid 64 CTA,占用 6–12%,No-Eligible 81%,
  Compute 4.5%,DRAM 5.3%,Tensor 0.8%。
- merge kernel:恒定 ~2–6us,与 batch 无关,b1 占 decode+merge 的 34%。
- e2e(CUDA graph):b1 10us,b2 11us,b4 14us,b16 25us。
- HBM roofline 下界(b1):~0.5us。可达上限(Triton):~1.3–1.7×。
- AI ≈ 16 FLOP/byte;B300 bf16 ridge ≈ 275。
