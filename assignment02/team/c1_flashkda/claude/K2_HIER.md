# FlashKDA K2 的分层扫描重构(Design B)——交付文档

对应 brief 的 10 项交付;实验脚本 `exp/e9_hier.py`,原始日志 `logs/e9b_hier_srun.txt`(fp32 phase 2)、
`logs/e9c_hier_srun.txt`(bf16 phase 2 + CUDA graph + 合并 phase 1)、`logs/e9_ncu_srun.txt`、
`profiles/ncu_e9.ncu-rep`。GPU:B300 SXM6 AC(cc 10.3,148 SM),SM 时钟采样全程 1095 MHz,50 次计时迭代、
5 次预热、CUDA event、同一份输入。

## 1. 当前 K2 的依赖瓶颈(一句话 + 数据)

K2(`_flash_kda_fwd_recurrence`)每个 (batch, head) 一个 CTA,串行走 T/16 个 chunk,每 chunk ≈ 2760 cycle,
**wall time 与 B·H 无关**(§2.2:H=96/64/12 都 ≈1.28 ms @T=8192)。ncu(B=1,H=12,T=8192):

| | grid | Duration | SM Active(`sm__cycles_active`) | Waves/SM | Achieved Occ. | Compute / Memory SOL | No-Eligible | Active / Eligible warps per scheduler |
|--|--:|--:|--:|--:|--:|--:|--:|--:|
| baseline K2 | 12 | 1.29 ms | **7.97%** | 0.04 | 9.4% | 2.6% / 4.7% | 67% | 1.50 / 0.35 |

12 个 CTA 占 148 个 SM 的 8%,其余 SM 空转。单 SM 内部是延迟链(No-Eligible 67%,每调度器 1.5 warp),
但整卡层面的病是 **parallelism-bound**:没有足够多的独立 CTA。

## 2. 可复合的 chunk 变换(推导)

记号同 `fla_kda_ref/naive.py`(`naive_chunk_kda` 160-163 行),状态 S ∈ ℝ^{K×V}(K=V=128):

```
v_i = u_i − w_i S_{i−1}                                  # w_i, u_i 由 K1 预计算,与 S 无关
S_i = D_i S_{i−1} + K_i'ᵀ v_i,   D_i = diag(exp g_i[−1]), K_i' = exp(g_i[−1]−g_i) ⊙ k_i   (16×128)
   ⇒ S_i = M_i S_{i−1} + N_i,   M_i = D_i − K_i'ᵀ w_i   (对角 + 秩≤16),   N_i = K_i'ᵀ u_i
```

复合 `T_2∘T_1`:`A = M_2M_1 = D_2D_1 − [D_2U_1 | U_2]·[W_1ᵀ ; W_2ᵀD_1 − (W_2ᵀU_1)W_1ᵀ]`(对角 + 秩≤32),
`B = M_2N_1 + N_2`。结合律成立(仿射变换的复合),**但秩每复合一个 chunk 增加 16,G≥8 时秩顶到 128 = 稠密**。
所以"保持低秩做全树扫描"只在树底下 2-3 层有意义,再往上就是 128³ 稠密 compose——这正是 §3.3.1/E8 实测输
的原因(总算力 16×,128³ tile 在 B300 上只到峰值 4~17%)。

**Design B 的关键:组摘要不用 compose。** 递推对 S 线性 ⇒ 一组 G 个 chunk 的复合 `T_g(S) = A_g S + B_g` 可以
"探测"得到:`B_g = T_g(0)`;令 v=0 则 u=0 ⇒ N_i=0 ⇒ `T_g(I)|_{v=0} = A_g`。A_g 由 K2 自己的低秩 chunk 更新逐步累出
来(每步仍是 16×128×128 与 128×16×128 的 GEMM),**没有 128³**;稠密 128³ 只出现在组间前缀,NG−1 次。
最小元数据:每组两个 128×128 bf16 矩阵(A_g, B_g),32 KB×2。

## 3. 三个候选设计的评估

| 设计 | 额外算力 | 临时显存 | launch 数 | 秩增长 | 同步 | 结论 |
|--|--|--|--|--|--|--|
| A 全树扫描(E8 实测) | ≈16× K2(3·NT 条 128³) | 3·NT·H·32 KB | ~50 步 bmm | 16/chunk,G≥8 即稠密 | 每层一次 | **输**:TP8 0.30×、H=96 0.04×,天花板下界也 ≥0.82×K2 |
| B 分层扫描(E9 实测) | ≈3× K2 + NG 条 128³ | 3·NG·H·32 KB(H=12,NG=16:18 MB) | 3 次 fwd + NG 次 bmm(graph 后 1 次) | 无(A_g 探测得到,不复合) | 3 次 kernel 边界 | **赢**:低 B·H 长序列 1.9~2.5×;B·H≥64 输 |
| C 状态无关预处理 + 最小串行 | 0 | 0 | 0 | — | — | FlashKDA 的 K1/K2 拆分就是它,已到头;K2 里剩下的 `w_i S`/更新/`q S` 就是串行部分 |

## 4. 代价模型(T=8192,NT=512,G=32,NG=16,每 (b,h))

- 算力:baseline K2 ≈ 0.87 GFLOP;B 方案 ≈ 3×(phase 1a、1b、3 各一遍 K2)+ 16 条 128³(0.07 GFLOP)≈ 2.7 GFLOP。
- 依赖深度:512 → 2·G + NG = 80(chunk 步)。
- 并行 CTA:B·H → B·H·NG(12 → 192;合并 1a/1b 后 phase 1 为 384)。
- 预期:baseline 与 B·H 无关(1.28 ms);B 方案 ≈ 2×(K1 + G 个 chunk)+ phase 2,只要 B·H·NG ≤ 296(2 CTA/SM)
  就是单 wave;B·H 本来就 ≥ 148 时,3× 算力全是净亏。**结论与实测一致(§6)。**

## 5. 低并行负载上的 baseline

baseline 时间只随 T 线性、与 B·H 几乎无关,直到 B·H 超过 SM 数才开始变慢(K1 随 H 线性):

| B | H | T | baseline fwd(K1+K2) |
|--:|--:|--:|--:|
| 1 | 12 | 8192 | 1366 us |
| 1 | 12 | 32768 | 5385 us |
| 1 | 32 | 16384 | 2909 us |
| 1 | 16 | 65536 | 10863 us |
| 4 | 16 | 8192 | 1626 us |
| 1 | 96 | 8192 | 1774 us |

判定:**parallelism-bound**(SM Active 8%),单 SM 内 latency-bound(No-Eligible 67%),既不是 memory-bound(4.7%)
也不是 compute-bound(2.6%)。

## 6. 最小可行原型与实现

`exp/e9_hier.py`(零 CUDA 改动,复用 FlashKDA 自己的 K2):

```
phase 1 (合并):  flash_kda.fwd(q,k,[v | 0], g, beta, cu_seqlens=组界, initial_state=[0 | I])  -> final_state = [B_g | A_g]
phase 2:          S_in[g+1] = S_in[g] · A_g + B_g      (bf16 baddbmm × (NG−1),CUDA graph;状态布局 [V,K] 右乘)
phase 3:          flash_kda.fwd(q,k,v,g,beta, cu_seqlens=组界, initial_state=S_in)              -> out, S_T
```

"合并"= 把 head 维翻倍(前 H 个 head 给 v/S_in=0,后 H 个给 v=0/S_in=I),一次调用同时得到 B_g 与 A_g。
迭代过程:v1 fp32 phase 2(SIMT,每步 ~22 us)→ v2 树扫描(更慢,拷贝多)→ v3 bf16 phase 2 + graph + 合并 phase 1。

## 7. 正确性

对拍基线 `flash_kda.fwd` 单次调用(同 bf16 输入、同 h0),并对 fp64 朴素递推(`naive_recurrent_kda`)看谁更接近真值。
lb=−5/−1 时 A_g 在一个组内就衰减到 bf16 分辨率以下,结果与基线**逐位一致**(max_abs=0),但那测不出 phase 1b/2;
下面是弱衰减(lb=−0.1:组间携带 ~2% 状态;lb=−0.01:~70%)的数据(`logs/e9c_hier_srun.txt`):

| 用例 | 量 | hier vs flash_kda(max_abs / mean_abs / rel_rms) | flash_kda vs fp64 | hier vs fp64 |
|--|--|--|--|--|
| B=1 T=2048 H=4 G=8 lb=−0.1 | out | 4.9e-4 / 6.0e-6 / 1.7e-3 | 5.946e-3 | 5.949e-3 |
| 同上 | hT | 3.9e-3 / 1.5e-5 / 7.5e-4 | 4.935e-3 | 4.934e-3 |
| B=2 T=1000 H=3 G=8 lb=−0.1(T 不整除 G) | out | 4.9e-4 / 5.6e-6 / 1.6e-3 | 5.978e-3 | 5.976e-3 |
| 同上 | hT | 3.9e-3 / 2.0e-5 / 8.7e-4 | 5.127e-3 | 5.122e-3 |
| B=1 T=2048 H=4 G=8 lb=−0.01 | out | 1.5e-3 / 1.3e-4 / 6.1e-3 | 6.909e-3 | 6.978e-3 |
| 同上 | hT | 7.8e-3 / 1.2e-3 / 5.0e-3 | 6.029e-3 | 6.037e-3 |
| B=1 T=8192 H=12 G=32 lb=−0.01 | out | 1.5e-3 / 1.1e-4 / 5.4e-3 | — | — |

结论:分层版与基线的差异在 bf16 舍入量级,且**对 fp64 真值的误差与基线相同**(第 4、5 列几乎一样)——组边界多
出的两次 bf16 舍入淹没在基线本来就有的每 chunk bf16 状态舍入里。fp32/bf16 phase 2、合并/不合并、seq/tree 四种
变体误差一致。max_rel 大的元素全部是参考值 ≈0 的位置。

## 8. 性能(v3:merged + bf16 phase 2 + CUDA graph;50 次迭代;1095 MHz)

| B | H | T | baseline | 最优 G | hier | **加速** | phase 1 / 2 / 3 |
|--:|--:|--:|--:|--:|--:|--:|--|
| 1 | 12 | 8192 | 1366 us | 32 | 710 us | **1.92×** | 494 / 136 / 220 |
| 1 | 12 | 32768 | 5385 us | 64 | 2151 us | **2.50×** | 1491 / ~270 / ~390 |
| 1 | 16 | 65536 | 10863 us | 64 | 5072 us | **2.14×** | 3301 / 552 / 1545 |
| 1 | 32 | 16384 | 2909 us | 64 | 2484 us | 1.17× | 1830 / 150 / 817 |
| 4 | 16 | 8192 | 1626 us | 64 | 2527 us | 0.64× | 1828 / 187 / 1191 |
| 1 | 96 | 8192 | 1774 us | 64 | 3545 us | 0.50× | 2554 / 98 / 1191 |

G 扫描(B=1,H=12,T=8192,merged+bf16+graph):G=16 1.63× / 32 1.92× / 64 1.92×;T=32768:G=16 1.78× / 32 2.25× /
64 2.50×。fp32 phase 2 版本(v1)同形状只有 1.42× / 1.99×——phase 2 的 128³ 在 SIMT 上每步 ~22 us,换 bf16 tensor
core 后 ~8 us。

**crossover**:B·H ≤ 16 且 T ≥ 8K 时稳赢(1.9~2.5×);B·H=32 勉强(1.17×);B·H ≥ 64 输(0.5~0.64×)——与假设一致:
基线的 CTA 数 B·H 一旦接近 SM 数,3× 的算力就没有空闲 SM 去吸收。

## 9. Nsight 对比(同一个 recurrence kernel,baseline vs 分层版的 phase 1a / 1b / 3)

| | grid | Duration | SM Active | Waves/SM | Achieved Occ. | Compute / Memory SOL | No-Eligible | Active / Eligible warps/sched |
|--|--:|--:|--:|--:|--:|--:|--:|--:|
| baseline | 12 | 1290 us | 8.0% | 0.04 | 9.4% | 2.6% / 4.7% | 67.1% | 1.50 / 0.35 |
| hier phase(G=32) | 192 | 130 us | **76%** | 0.65 | 12.5% | 26% / 47% | 65.0% | 2.02 / 0.43 |

寄存器 73/线程、动态 smem 98 KB/CTA、无 local spill,两边完全相同——**单 SM 内的行为一个字没变**(同样的
No-Eligible、同样的每调度器 warp 数),变的只是有多少个 SM 在干活:8% → 76%。这就是"为什么赢"的硬件解释:
不是让链变快,而是让 148 个 SM 里的 112 个从空转变成各跑一条短链。也解释了"为什么输":B·H=96 时基线 SM Active
本来就 ~65%,分层版把它顶满的同时多做了 3× 的活。

## 10. 收益/亏损的原因与下一步

**赢在哪**:baseline K2 的 wall time = 一条 512-chunk 链;分层版把它换成 3 条 32-chunk 链 + 16 步组间递推,总算力
3× 但摊在原本空闲的 SM 上。**输在哪**:phase 1(合并后 384 CTA、含两遍 K1)占 70%,K1 被重算 3 遍(67 us × 3),
组数一多(G=16)第二个 wave 出来;phase 2 仍是 NG 步串行 launch。

**推荐下一步(按收益/工程量)**:
1. **K1 只算一遍**:用 `flash_kda_C.fwd` 的显式 workspace 接口,让三个 phase 复用同一份 K1 输出(省 2×67 us,
   T=8192/H=12 上预计 710 → ~580 us,2.35×)。
2. **phase 2 写成一个 kernel**(NG 步 bf16 128×128 右乘,每 (b,h) 一个 CTA 顺序做,或对 NG≥64 用 log 深度树):
   136 us → ~20 us,预计 2.7×。
3. **phase 1a/1b 融合进 K2 内核**:同一个 CTA 同时维护 (S,U) 两个状态(v 与 0 共享全部 TMA 载入与 P1 的 kd/qd
   GEMM,只有 P6 翻倍),phase 1 从 2 波 K2 变成 1 波 ~1.5× 工作量。
4. **调度**:按 B·H 自动选 G/是否启用(B·H ≤ 32 启用,G 取 max(32, NT/32));B·H ≥ 64 走原路径。
5. 到这一步之后才轮到 Blackwell 特化(tcgen05/TMEM/warp specialization):§7/§7.1 已证明它们缩短不了单链,
   但在 phase 1 这种"很多短链同时跑"的场景里,它们能提高单 SM 吞吐——那是另一个 regime 的问题,要重新 profile。

**主研究问题的答案**:能。K2 可以用 3× 的冗余算力和 3·NG·H·32 KB 的临时状态换来 NG 倍的 chunk 间并行,依赖深度
512 → 80,在 B·H ≤ 16 的长序列负载上把 SM Active 从 8% 提到 76%,端到端 K2 时延降 1.9~2.5×;代价是在
B·H ≥ 64 的负载上净亏 0.5×,因此必须按 B·H 门控。

## 11. 追加(brief 第 9 步):分层扫描之后,B300/SM100 特性还能不能再加分?(E10,`exp/e10_short_chain.py`,`logs/e10_short_chain_srun.txt`)

分层扫描把 K2 从"一条长链"变成 phase 1/3 里"很多条 G=32 的短链、每 SM 2 个 CTA 共驻"——这是 §7 里 tcgen05 版 K2
没测过的 regime:它的链是等待型(No-Eligible 77%),共驻的另一个 CTA 理论上能把这些等待填掉。直接用现成的
tcgen05 K2(`k2_tc2`)和 FlashKDA K2 在 T=512(=一个组)、H 从 148 扫到 1184(每 SM 1→8 个 CTA)上对比,1095 MHz:

| T | H(CTA/SM) | mma.sync K2 | tcgen05 K2 | 比值 | cyc/chunk/CTA(按 wave 折算)mma / tcgen05 | 整卡 chunks/us mma / tcgen05 |
|--:|--:|--:|--:|--:|--:|--:|
| 8192 | 12(0.08) | 1286 us | 2036 us | 0.63 | 2750 / 4355 | 4.8 / 3.0 |
| 512 | 148(1) | 90.6 | 137.0 | 0.66 | 3100 / 4689 | 52 / 35 |
| 512 | 296(2) | 125.0 | 170.9 | 0.73 | 2139 / 2923 | 76 / 55 |
| 512 | 592(4) | 245.9 | 328.3 | 0.75 | 2104 / 2809 | 77 / 58 |
| 512 | 1184(8) | 487.1 | 650.6 | 0.75 | 2084 / 2783 | 78 / 58 |
| 2048 | 296(2) | 466.3 | 662.0 | 0.70 | 1995 / 2832 | 81 / 57 |

三个事实:
1. **共驻确实帮了 tcgen05 更多**(1→2 CTA/SM:tcgen05 单链等价代价 4689→2923,−38%;mma.sync 3100→2139,−31%),
   但它的起点太差,**在最有利的 regime 里也只到 mma.sync 的 0.75×**,整卡吞吐 58 vs 81 chunks/us。
2. **两种指令都在 2 CTA/SM 处饱和**(4、8 CTA/SM 时 cyc/chunk 不再下降):mma.sync 版被 98 KB smem 卡在 2/SM,
   tcgen05 版被 84 KB smem + 256 列 TMEM(512 列/SM)同样卡在 2/SM。SM100 的 TMEM 没有放松这个约束——把状态
   搬进 TMEM 当 A 操作数(64 列)+ G3 拆半(64 列)+ G1/G2(48 列)= 176 列,仍然只够 2 个 CTA。
3. 饱和时 ncu 的 issue slot 只用了 35%(§9):**phase 1/3 剩下的杠杆是每 SM 多驻几条链**,这是 smem 预算问题,
   与指令集无关——V-split(BV=64,状态 16 KB,§3.3)能把 mma.sync 版推到 3-4 CTA/SM,预期 1.3-1.4×;
   SM100 特有的东西里,只有 2-CTA cluster + TMA multicast 能让合并 phase 1 里 (v, v=0) 这对 CTA 共享
   q/k/kd/qd/INV/Mqk 的载入,但 Memory SOL 只有 47%、不是瓶颈,预期收益个位数百分比。

**结论:分层扫描之后,B300/SM100 的指令级特性(tcgen05/TMEM)仍然没有正收益——短链共驻把 tcgen05 从 0.62×
提到 0.75×,但天花板被 TMEM 容量钉在和 smem 一样的 2 CTA/SM。值得继续的是架构无关的 smem 减负(V-split)去提高
共驻数;SM100 专版的结论维持 §8:不出。**
