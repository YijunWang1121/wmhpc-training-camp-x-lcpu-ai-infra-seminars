# 答辩要点(Claude 版本)

## 一句话
小 batch MSA decode 的 Triton 基线是 **launch + 索引链 + 单块延迟 + merge kernel** 四段固定开销叠加(b=1:0.9 + 0.9 + 1.5 + 2.7 us,
加一块 compute 1.3 us ≈ 8.2 us),不是算力也不是带宽。我用 CUDA(TMA + mma.sync + cluster/DSMEM)完整做了一版融合 kernel,
全部通过验收但小 batch 没赢(14.7 vs 8.2 us);它的分相计时给出了任何同结构 kernel 的硬地板 3.3 us → 收益上限 ~1.5×,端到端 ~1%,
而越过这条线需要 tcgen05 级别的重写 = 上游 CUTLASS 路径。**结论 (b):不值得**;crossover=16 是"固定开销大/边际成本小的流式
kernel" 对 "固定开销小/边际成本随 b 线性涨的 split-K" 的交点,位置由每块边际成本决定。

## 预期质疑 & 回应
**Q:你的 CUDA kernel 输了,是不是写得不好而不是方法不行?**
A:分相数据(LOG S8)说明每一段都到了它的物理下限:launch/索引链/单块 TMA 与 Triton 完全一样(探针实测),每块 compute 1.4k cycle
里 MMA 只占 0.2k,其余是 ldmatrix/softmax/掩码 ~400 条指令的串行发射——这是 mma.sync 编程模型本身的开销,Triton 的同结构 CTA 也是
5.9 us。能再降的只有 cluster 合并(3.7k)和 4-warp 合并(1k),就算全部归零也只到 ~9 us 级别,与 Triton 持平。

**Q:为什么不上 tcgen05?**
A:那就是 CUTLASS fmha_sm100 路径。tcgen05 的价值不是 MMA 吞吐(AI=16 用不上),是把每块 ~400 条 warp 指令换成 16 条单线程异步指令,
每块成本 1.4k → ~0.3k cycle。代价:TMEM/描述符/swizzle/P 回写/warp 专化,以及 fp8-only、形状固定的 gate——上游做了,也只敢在 b≥16 开。

**Q:时钟 1095 MHz 是不是让你的结论失真?**
A:所有对比在同一时钟下做;计算/延迟部分在 2 GHz 会快 ~1.85×,但 launch 与 DRAM 延迟不变,Triton 与我的 kernel 同比例受益。
mma_rate / 消融 / 地板探针都在同一时钟,比例结论不变。

**Q:讨论点 1 你说 tensor core 有用,别的组说没用?**
A:两个层面。AI=16 ≪ ridge 280 → 算力不是瓶颈,这一点一致。但 CUDA core fp32 实测 38 TFLOPS(1.1 GHz)→ 要跟上 8 TB/s 需 ~130 TFLOPS,
不用 tensor core 连带宽都跟不上;而 Triton 的 mma.sync.m16n8k16 的 M 恰好 = 16,没有浪费——"M 太瘦"只对 tcgen05(M≥64)成立。

**Q:TMA 能不能表达间接寻址?**
A:能。PTX 原文:坐标是 .s32 寄存器。实验:4-D tensormap + 运行期 page 坐标,校验和一致,单块 0.8 us。TMA 不做 gather,但本题
不需要 gather(块 = 页内连续 tile);.tile::gather4 只 4 行,.override::global_address 要 CUDA 13.4。

**Q:FP8 scale 放哪?**
A:标量折进 Q/输出(host,零 kernel 成本,与上游口径逐位一致,err 4e-3);per-token 加在 S/P tile。实测上游 Triton fp8 路径比 bf16 慢 30–60%。

**Q:验收方案的弱点?**
A:合成 top-k 是随机块,没有 prefix 复用;fp32 参照的"真值"只到 bf16 输入精度;未覆盖 per-token fp8、TP 切分形状。见 ACCEPTANCE.md §5。

## 数字速查(B300,SM 1095 MHz,graph)
- Triton:b=1 8.2 / b=4 11.2 / b=8 14.8 / b=16 22.1 / b=64 74.4 us;decode 5.9 + merge 2.7 @ b=1。
- 地板探针 @b=1:空 kernel 0.9,索引链 1.8,索引+一块 TMA 3.3 us。
- 每块:TMA 0.8 us;compute 1.4k cycle(4 warps);Triton 每块 2.3 us。
- 我的 kernel:cl4s2 14.7,cl1s3 26.0(b≤16 平),split16+Triton merge 10.6;改良 Triton 10.2。
- mma.sync bf16 320 TFLOPS(1/7 tcgen05);fp8 mma.sync 1200;FFMA 38。
- crossover 模型:c_blk 1.4 → b≈32;0.8 → 16;0.5 → 8。
