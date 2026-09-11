# C1:FlashKDA 停在 SM80 MMA,值不值得上 SM100?——报告(Claude 版本)

所有数字来自本机 **NVIDIA B300 SXM6 AC**(cc 10.3,148 SM,HBM3e 275 GB);CUDA 13.0,torch 2.14,fla 0.5.2,
FlashKDA pin 1ce47ea(cutlass 5c149f5)。**本机 SM 时钟因分配而异:job 24796/24844 全程 1095 MHz,
job 24819/24904 全程 2032 MHz**(`logs/clock_<job>.csv`),因此跨作业比较一律用 cycle/chunk。
方法与决策链见 `PROFILING.md`,逐作业记录见 `LOG.md`,复现命令见 §9。

## 0. 结论速览

| 问题 | 结论 |
|--|--|
| 复现 | 装起来了(`.venv`,sm_100a+sm_103a);官方 bench 全部形状复现,H=96 fixed 对 fla `chunk_kda` 加速 2.24×(GB200 表 2.31×);对 fp64 朴素递推 rel_rms 5e-3(Triton 3.4e-3)。SASS:K2 只有 `HMMA.16816.F32.BF16`,K1 另有 12 条 `HMMA.16816.F16`,全库 0 条 HGMMA/UTC*MMA。 |
| K2 本质 | 每 (seq, head) 一个 CTA 串行走 T/16 个 chunk,**每 chunk ≈ 2760 cycle,与 H 无关**;T=8192 时 K2 ≈ 1.28 ms @1095 MHz,占端到端 73%;TP8(每卡 12 头)只用 12/148 个 SM 却要同样时间。 |
| 讨论点 4 | K2 既非算力(SM 20%,tensor pipe 30%)也非带宽(DRAM 19%)bound,是**单 CTA 串行延迟链**:No-Eligible 68%,每调度器 1.5 warp,smem 是最忙部件(61%)。消融:P1 610 + P6 770 + P3/4 347 + 流水线/同步 800 + 其余 230。K1 相反,是吞吐型(L2 72%,DRAM 59%,issue 71%)。 |
| 讨论点 1 | CHUNK=32 时**先破的是数值范围**:lb=−5 下 benchmark 分布 15.7% 的 k_d 下溢、14.6% 的 k_inv 上溢,L 直接 nan;C=64 时 57%。Neumann 与 MMA 形状在 C=32 都不破(fp16 求逆误差 2e-6 不随 C 变;寄存器 ~112)。但范围问题只需把 cumsum 的参考点移到 chunk 中点(零成本),见 §3.1。 |
| 讨论点 2 | tcgen05 M 最小 64,CHUNK=16 当 M 不合法;**把 D=128 当 M、CHUNK 当 N/K** 则 5 个 GEMM 全部合法。但 microbench:N=16 的指令只有 653 MAC/cycle/SM(比 mma.sync 峰值 1020 还低),N≥128 才 4088;一次 issue→commit→wait 固定 ≈265 cycle,TMEM 回读 128×16 要 139 cycle。纸面 ≈2900 cycle/chunk;实测(§7)只换指令 = **0.62×**,把线程侧工作全删掉的地板也只有 1.34×。 |
| 讨论点 3 | 候选:V-split(fla 就是 BV=32/64)、多 head/CTA、persistent、2-CTA。**只有缩短单 chunk 链才能帮 TP8 长序列**;V-split 只压缩吞吐型的 ~40%,多 head/CTA 只提高 SM 利用率不减时延,persistent 只对 varlen 负载均衡有用。追问(§3.3.1):链本身能不能拆(不是绕开)?数学上**能**——状态递推代入化简后是仿射矩阵递推 `S_i=M_i S_{i-1}+N_i`,可结合律重组成分块前缀扫描,深度 512→~50;**做出来并实测了(E8)**:正确(对 flash_kda 5.5e-3,同 §2.4 量级),但性能 TP8 **0.30×**、H=96 **0.04×**、T=32768 **0.33×**,全部大幅落败;归因是每次 compose 是 128³ 稠密 GEMM、总算力 16×,而 128³ tile 在 B300 上 cuBLAS 只能到峰值的 4%(batch 192)~17%,按天花板计费的下界(H=12 仅 compose 就 1062 us ≈ K2 的 0.82×)也赢不了。FLA 自己的生产 kernel 也是纯串行,现在知道为什么了。**但换一种拆法就赢了(§3.3.2,E9):不 compose,用 K2 自己的 varlen + initial_state 探测每组的 (A_g, B_g),组间 NG 步前缀,再并行回放——B·H≤16 的长序列上 1.9~2.5×(SM Active 8%→76%),误差与基线同量级;B·H≥64 输 0.5×,需按 B·H 门控。** |
| 讨论点 5 | 用 fp64 递推做金标准、fla fp32 状态做对照:flash_kda 的 hT 误差是 Triton 的 1.9–2.3×(4.5e-3 vs 2.4e-3),**不随 T 增长**(512→16384 平),弱衰减(gate→0)时 7.4e-3 vs 4.2e-3;bf16/fp32 状态 I/O 分段调用逐位相同。 |
| 挑战 | 写了 tcgen05 版 K2(`exp/k2tc/k2_tc2.cu`,消费原 workspace,TMA + SW128 + 双缓冲 TMEM 累加器),**与 FlashKDA 逐位一致**(out/hT rel 0.0),三轮 profile 驱动的迭代从 0.31× 到 **0.62×**;消融:去掉状态更新 = 1.02×,去掉全部线程侧 epilogue = 1.34×(异步 MMA 链的地板 2030 cycle)。**没有正收益**,原因量化在 §7。追问"换 CUTLASS 而不是手写会不会翻盘"——从 CUTLASS 4.7.0 源码验证不会(§7.1):单线程发射是 tcgen05 的 ISA 约束、K2 的 chunk 间递推是真数据依赖,两者都不是"代码写得不够专业"能解释的,换库无效。 |
| 讨论点 6 | **v2 不出 sm100a 专版**:算术峰值不是瓶颈(tensor pipe 30%),换指令后每 chunk 3 次 MMA 往返 + 4 次 TMEM 回读的固定延迟 ≈ 1800 cycle 抵掉了 4× 的峰值;真正的杠杆是架构无关的——CHUNK=32 + 中点重标定(纸面 K2 −30%)、去掉 800 cycle 的流水线同步开销、V-split 提高 TP8 下的 SM 利用率。 |

## 1. 对象与形状

- K3 KDA:96 头 × D=128,T=8192 fixed 是主形状;TP8 每卡 12 头。CHUNK=16 → 512 个 chunk/序列。
- K1(`_flash_kda_fwd_prepare`,grid = tiles×H = 49152 CTA × 256 线程):L2 norm、gate、cumsum、k_d/q_d/k_r、
  L/Mqk(16×16×128 两个小 GEMM)、Neumann 求逆,写 13.8 KB/tile 的 workspace。
- K2(`_flash_kda_fwd_recurrence`,grid = N×H,192 线程 = 4 MMA warp + load warp + store warp):
  每 chunk 5 个 GEMM:P1 k_d@S、q_d@S(各 16×128×128);P3 INV@u、P4 Mqk@U(16×16×128);P6 k_r^T@U(128×16×128);
  状态 128×128 bf16 常驻 smem(32 KB),每 chunk 851,968 MAC。

## 2. 复现(任务 1)

### 2.1 官方 benchmark(`logs/e1_24796.out`,1095 MHz,ms)

| 形状 | flash_kda | fla chunk_kda | 加速 | GB200 表 flash / 加速 |
|--|--:|--:|--:|--|
| H=96 fixed 8192 | 1.755 | 3.930 | 2.24× | 1.009 / 2.31× |
| H=96 varlen [1300,547,2048,963,271,3063] | 1.475 | 4.057 | 2.75× | 0.860 / 2.71× |
| H=96 varlen 1024×8 | 1.186 | 3.955 | 3.33× | 0.706 / 3.27× |
| H=64 fixed | 1.611 | 2.673 | 1.66× | 0.925 / 1.70× |
| H=64 varlen 6 段 | 1.114 | 2.838 | 2.55× | 0.655 / 2.42× |
| H=64 varlen 1024×8 | 0.803 | 2.621 | 3.26× | 0.481 / 3.21× |

绝对时间是 GB200 表的 1.7×,与时钟比(1095 vs ~1.9 GHz)一致;加速比全部复现。

### 2.2 分 kernel(`exp/e1_kernels.py`,1095 MHz,us)

| 形状 | K2 | K1 | 其它 |
|--|--:|--:|--:|
| fixed H=96 | 1285 | 458 | 9 |
| fixed H=64 | 1287 | 309 | 7 |
| fixed H=12(TP8) | 1276 | 67 | 5 |
| T=32768 H=12 | 5078 | 235 | 5 |
| varlen 1024×8 H=96 | 689 | 480 | 12 |

K2 与 H 无关、随 T 线性:2.5 us/chunk ≈ 2760 cycle。varlen 1024×8 时 768 个 CTA 各 64 chunk,K2 只有 689 us,
说明 2 个 CTA 同驻一个 SM 几乎不互相拖慢——单 CTA 远没有用满 SM。

### 2.3 SASS(`profiles/sm103a_hist.txt`)与 ncu 核对

见 §0;K2 的 `sm__inst_executed_pipe_tensor.sum` = 20,447,232 = 96 × 512 × 4 warp × 104 条,与源码逐条对应。

### 2.4 正确性(`exp/e3_correct.py`)

T∈{64,500,1024,2048}、N∈{1,2}、lb∈{−5,−1}:flash_kda 对 fp64 朴素递推 out rel_rms 5.2–5.5e-3、hT 4.4–4.7e-3;
fla Triton 3.4e-3 / 2.3e-3;flash_kda 与 Triton 互差 5.6e-3。PASS。

## 3. 讨论点

### 3.1 CHUNK=16 的三个理由,C=32/64 哪个先破(`logs/e5_chunk_range_24819.txt`)

**(a) 数值范围——先破。** kernel 算 k_d = k·2^{cs},k_inv = k·2^{−cs},cs = cumsum(lb·log2e·sigmoid(·)),
fp32/bf16 正规数范围 2^−126..2^128,|cs| 上界 = C·|lb|·log2e:C=16 → 115(刚好),32 → 231,64 → 462。

| 分布 | lb | C=16 | C=32 | C=64 |
|--|--|--|--|--|
| benchmark(g~N(0,1)) | −5 | 0% 溢出,L 误差 3e-3 | k_d 下溢 15.7%,k_inv 上溢 14.6%,**L=nan** | 57%,nan |
| benchmark | −3 | 0% | 0%(2e-3) | 27.8%,nan |
| benchmark | −1 | 0% | 0% | 0% |
| 最坏(gate≡lb) | −5 | 0% | 46.9%,nan | 73%,nan |

代价:不能再用 K1 现在的分解形式,要么像 fla 那样按 (i,j,d) 三元组重算 exp(C²D 次 MUFU,C=64 时是 32×),
要么——**把 cs 的参考点从 chunk 起点改成中点**(k_d = k·2^{cs−c_mid},k_inv = k·2^{c_mid−cs},L 不变):
指数范围减半,C=32 等价于现在的 C=16,K1 只多一次减法;q_d、k_r 用真实 2^{cs} 的地方下溢到 0 是正确舍入。
这是"大 CHUNK + rescale"最便宜的实现。

**(b) Neumann 求逆——不破,只是变贵。** L 严格下三角幂零,(I−L)^{−1} = Π(I+L^{2^i}) 精确;fp16 误差
C=16/32/64 都是 2–3e-6(|inv| 上界 1.00 实测成立),bf16 1.6–2.3e-5,与 C 无关。代价按 m16n8k16 条数:
C=16 每 chunk 12(kernel 实际 14)= 0.75/token;C=32 128 = 4/token;C=64 1280 = 20/token。
对比 K2 每 warp 每 token 6.5 条 HMMA,C=32 的求逆开销可接受,C=64 不行(且 32×32 求逆需要 fp16 4 个 16×16 块的分块乘)。

**(c) MMA 形状/寄存器——C=32 不破。** m16n8k16 的 M=16 只是 CHUNK 的下界;C=32 时 K2 每线程累加器 64 + U 16 + 环 32
≈ 112 寄存器(现在 73),C=64 ≈ 192 + 寻址 → 溢出。K1 的 smem 翻倍会把 8 CTA/SM 的占用压到 4。

**结论:** C=32 只有 (a) 破,而 (a) 有零成本修法;C=64 三条都有问题。每 token 的 HMMA 数几乎不随 C 变
(6.5 → 7.0 → 8.0),大 CHUNK 的收益不在 MMA 而在**摊薄每 chunk 的固定开销**(§3.4 的 800 cycle 同步 + P6 的
状态 smem 往返 500 cycle,都按 chunk 计费)。

### 3.2 tcgen05 最小 tile 与 CHUNK=16(`logs/mb_mma_24819.txt`,cycle)

纸面:`tcgen05.mma.kind::f16` cta_group::1 的 M ∈ {64,128},N ∈ [8,256] 步长 8(M=128 时步长 16),K=16。
CHUNK=16 当 M 不合法(M=64 也只填 25%);把 D 当 M:P1 → M128 N16(或叠成 N32)K128,P3/4 → M128 N16 K16,
P6 → M128 N128 K16(B 用 MN-major),全部合法,workspace 布局(GMMA INTER)就是 tcgen05 描述符的 no-swizzle 布局。

microbench(每个配置先对 CPU 参考 PASS):

| 指令/形状 | 吞吐 cycle/条 | MAC/cycle/SM | 说明 |
|--|--:|--:|--|
| mma.sync m16n8k16,1 warp 4 累加器 | 8.4 | 244/SMSP | 依赖延迟 20.8 |
| mma.sync,4 warp × 4 累加器(K2 配置) | 8.4 | **977** | ≈ SM 峰值 1020(=331 TFLOPS@1095) |
| tcgen05 M128 N16 K16 | 50 | 653 | 按条计费,小 N 吃亏 |
| tcgen05 M128 N32 K128(8 条) | 372 | 1410 | k_d、q_d 叠一起 |
| tcgen05 M128 N128 K16 | 64 | **4088** | = 4.0× mma.sync 峰值 |
| tcgen05 M128 N128 K128(8 条) | 512 | 4095 | |
| issue→commit→mbarrier 往返 | 265(K16)/752(K128 链) | | 固定延迟 |
| tcgen05.ld 128×16 / 128×128 fp32 | 139 / 446 | | 回读延迟 |

结论:只换指令时 P1 的 8 条 N32 指令 ≈ 372 cycle(mma.sync 538),但加上往返 752;P6 一条 64 cycle 但回读 446;
每 chunk 3 次真依赖往返 ≈ 800 cycle 固定。**纸面 ≈2900 cycle/chunk,不比 2760 快**——实测见 §7。

### 3.3 并行度从哪来

| 方案 | 机制 | 反例/限制 |
|--|--|--|
| V-split(每 CTA 状态 128×BV) | u、U、S 的 V 列彼此独立(delta rule 只在 K 维收缩),fla 已如此(BV=32/64,grid = V/BV × N×HV);K1 输出被多个 CTA 重复读(L2 流量 ×128/BV) | 每 chunk 的固定延迟链(barrier、LDSM 依赖、MMA 延迟)不随 BV 缩小,只有吞吐型部分(P1 发射、P6 smem 流量 ≈ 40%)按比例缩;BV=64 上限 ≈1.25×,BV=32 ≈1.4× |
| 多 head 进一个 CTA | 两条独立链交错,掩盖彼此延迟,SM 利用率 ×2 | **不缩短任何一条链**,固定 H 时 CTA 数减半、总时间不变;只对 head×seq ≫ 148 的吞吐场景有意义(等价于 2 CTA/SM,现在 smem 92 KB 已允许,varlen 1024×8 就是证据) |
| persistent kernel | varlen 长短不一时做负载均衡,避免长序列拖尾 | fixed 单序列 96 CTA < 148 SM,无东西可调度;对 TP8 的 12 CTA 无效 |
| 2-CTA(cluster/multicast) | 与 V-split 组合:两 CTA 各持一半 V,k_d/q_d/INV 用 TMA multicast 只读一次 | 只省 L2 流量,不省时延;cta_group::2 的 tcgen05 也没改变链的依赖结构 |

对 TP8 长序列(12 CTA,K2 1.28 ms)唯一有效的是**缩短单 chunk 链**(去同步开销、大 CHUNK)加 V-split。

#### 3.3.1 追问:512 步的串行依赖链,能不能把链本身拆开(而不是绕开)?

上面四个方案都是"绕开"(换更多独立 CTA 去掩盖延迟),没有一个动"chunk 间必须严格串行"这个前提本身。
从 `fla_kda_ref/naive.py`(`naive_chunk_kda`,160-163 行)把递推代入化简一遍,发现**这个前提其实不成立
——chunk 间状态递推本质是一个可以结合律重组的仿射矩阵递推,原则上能从 O(NT) 深度压到 O(log NT)。**

**推导(记号照抄 naive.py 第 160-163 行)。** 每个 chunk 的状态更新是:

```
v_i = u_i - w_i @ S              # S: [K,V] 是上一 chunk 传入的状态
S'  = diag(exp(g_i[-1])) @ S + K_i'^T @ v_i     # K_i' = exp(g_i[-1]-g_i) * k_i, 形状 [BT,K]
```

把 `v_i` 代进去展开:

```
S' = diag(exp(g_i[-1])) @ S + K_i'^T @ (u_i - w_i @ S)
   = (diag(exp(g_i[-1])) - K_i'^T @ w_i) @ S + K_i'^T @ u_i
   =                 M_i                @ S +      N_i
```

即 `S_i = M_i @ S_{i-1} + N_i`,一个标准的一阶仿射矩阵递推,其中 **`M_i`(K×K = 128×128)和 `N_i`(K×V =
128×128)只用本 chunk 自己的 `k_i, w_i, u_i, g_i`就能算出来,完全不需要 `S`**——`w_i`、`u_i` 本来就是 K1
里已经算好的、chunk 之间互相独立的量(fla 论文里的 WY 表示,FlashKDA 对应 §7 表格里的 G2/G2' 之前那部分)。
额外要付的代价只是显式形成 `M_i`:一次 `K_i'^T @ w_i`([128,16]@[16,128]),形状和现有 P6 的
128×16×128 GEMM 完全同型,单价不贵,而且和其余 chunk 的 `M_i`、`N_i` 计算**互相独立、可以对全部 512 个
chunk 一次性并行算完**(这一步可以直接塞进现在的 K1)。注意 `K_i'^T @ w_i` 的秩 ≤ BT=16 ≪ K=128,
`M_i` 是"对角 + 秩 16"结构(DPLR,和 S4/SSM 文献里的参数化是同一类结构),这一点在下面估算代价时有用。

**递推可以结合律重组。** 定义算子 `(M_a,N_a) ⊕ (M_b,N_b) = (M_b@M_a, M_b@N_a+N_b)`(表示"先套用 a 再套用
b"这个仿射变换的复合),这个算子满足结合律(仿射变换的复合本来就满足结合律)。于是给定全部 512 个
`(M_i,N_i)`,可以用标准的 Blelloch(work-efficient)前缀扫描,在 `2×log2(512)=18` 轮**依赖**运算内,
并行算出全部 512 个 `S_i`——而不是现在这样必须严格排队 512 步。§7.1 反驳"CUTLASS 能不能救 K2"时说的
"K2 没有可以流水的独立工作项"这句话,准确地说应该是"**在当前算法写法下**没有独立工作项";换一种数学上
等价的写法(仿射递推 ⇒ 结合律 ⇒ 扫描),独立工作项是能造出来的。

**但代价不是零,且赢面强依赖并发场景,以下都是纸面估算,没有上机验证:**

- **总算力代价**:现在的串行写法从不显式生成 `M_i`(直接用 `w_i@S` 这种 [16,128]@[128,128] 的矮阵去乘 S,
  代价 ∝ K×V,不是 K×K×K);扫描版必须先把每个 `M_i` 显式材料化成 128×128 矩阵,扫描本身的每次
  `⊕` 又是两条 128×128×128 的 GEMM(`M_b@M_a` 和 `M_b@N_a`)。Blelloch 上扫+下扫总共 ≈2×(512−1)≈1022
  次 `⊕`,每次 ≈8.4 MFLOP,扫描阶段总计 ≈8.6 GFLOP/(seq,head)——是现在这部分工作(≈537 MFLOP,按 §1
  851,968 MAC/chunk 里状态相关那一半估)的 **≈16×**。深度换总量:这笔账在"算力有富余、缺的是延迟隐藏"
  的场景才划算(K2 恰好是,tensor pipe 只 30% 忙,§3.4),在算力已经吃紧的场景(高并发 serving,SM 本来
  就被其他请求占满)反而是纯负担。
- **实际能拿到多少深度收益,取决于有多少空闲 SM 可以横向摊开扫描的每一轮。** 扫描第 r 轮理论上有多达
  NT/2 个互相独立的 `⊕` 可以同时做,但一次只能塞进"当前空闲的 SM 数"个。TP8 长序列(报告优先场景)只有
  12 个 CTA 在跑,其余 136 个 SM 空闲,每个 head 平均能借到 ≈136/12≈11 个空闲 SM——扫描 1022 次
  `⊕` 除以 11 路并行 ≈93 "波",每波按现有 microbench 的 M128N128K128 issue→commit→wait 往返
  (§3.2,≈752 cycle)估,93×752≈70,000 cycle ≈ 64 us(1095 MHz)——vs 现在整条链 1.28 ms,**理论上限
  ≈20×**。但如果 GPU 上同时有更多请求在跑(高并发 serving,SM 本来就被占满,没有 136 个空闲 SM 可借),
  这个数字会跌回接近 1×(扫描的深度优势没有硬件去兑现,总算力代价的 16× 反而净亏)。
- **工程量不小,且没有先例。** 直接读了 FLA 自己的生产 Triton kernel
  (`fla/ops/common/chunk_delta_h.py:166`,`chunk_kda_fwd` 最终调用的就是它),状态递推同样是一个
  `for i_t in range(NT): ...` 的**纯串行循环**,没有任何扫描结构——写这篇论文("Parallelizing Linear
  Transformers with the Delta Rule over Sequence Length")、维护这个库的团队自己都没有在生产 kernel 里
  用扫描版本。这不代表扫描版本理论上不成立(上面的推导是对的),更可能的原因是:(a) 常见的高并发 serving
  场景本来就有大把独立 CTA 可用,扫描省的那点深度买不起 16× 的总算力代价;(b) 扫描版本要写一个"多轮
  grid-wide 同步 + 128×128 矩阵 combine"的新 kernel,比现在的单 CTA 循环复杂得多,轮次间的同步开销
  (grid sync 或多次 kernel launch)本身也要占用这条本来就短的关键路径,上面的 64us 估算完全没有计入这部分
  开销,实际很可能吃掉相当一部分理论收益。

**实现与实测(E8,`exp/e8_scan.py`,`logs/e8_scan_srun.txt` / `logs/e8_bench2_srun.txt`,job 26190/26199,
全程 1095 MHz,与 flash_kda 同一进程同一会话)。** 按 TASK 第三层"并行度重构"这条路线真的把它做了出来:
PyTorch/cuBLAS 实现,`build_MN`(全部 chunk 一次性批量算 `M_i`、`N_i`)→ 两级分块扫描(level-1 组内 batched
compose、level-2 组间传播、level-3 batched apply)→ 输出阶段(v_i、o_i 全 chunk batched);另写了去掉 torch 层面
低效的 v2(连续 batch、`baddbmm`、预分配 out=)。

- **正确性**:fp64 下扫描版与"同样 M/N 但 NT 步串行"逐位一致(rel 0~2e-17),对 `naive_chunk_kda`/
  `naive_recurrent_kda` 2e-7(参照内部是 fp32);bf16 全链对 flash_kda 的 o/S 误差 5.5e-3/4.4e-3,与 §2.4 里
  flash_kda 自己对 fp64 参照的误差(5e-3/4.5e-3)同量级——**扫描重排本身没有引入额外误差**。
- **性能:全线落败,而且不是实现粗糙的问题。**

| T / H | flash_kda K2 | 扫描 v2(仅 compose) | build_MN + 输出 | 扫描链合计 | 结果 |
|--|--:|--:|--:|--:|--:|
| 8192 / 12(TP8) | 1289 us | 2604 us(29 TFLOP/s 有效) | 555 + 1111 | 4270 us | **0.30×** |
| 8192 / 96 | 1303 us | 17255 us(35 TFLOP/s) | 4202 + 8519 | 29975 us | **0.04×** |
| 32768 / 12 | 5132 us | 9304 us(33 TFLOP/s) | 2145 + 4285 | 15733 us | **0.33×** |

  CUDA graph 与 eager 只差 3-7%(`logs/e8_scan_srun.txt`),排除 launch 开销;v2 比 v1 只快 5%,排除 torch 切片/
  拷贝开销。真正的原因是同一会话里单独测出来的 **128³ batched GEMM 天花板**(`exp/e8_gemm_ceiling.py`,
  `logs/e8_gemm_ceiling_srun.txt`):cuBLAS bf16 bmm 128³ 在 batch=192(H=12 时 level-1 的 batch)只有
  **72 TFLOP/s**,batch≥768 也只到 **226-295 TFLOP/s**——同一进程里 4096³ 单条 GEMM 是 1706 TFLOP/s。
  每条 compose 是 M=N=K=128 的 GEMM,每个 CTA 只做一个 128×128 tile、K 维只有 8 个 k-step,正是 assignment
  4.5 的 `f_b_proj`(K=128)那一行和 §3.2 小 N tcgen05 指令"按条计费"同一种病:tile 太小、K 太短,摊不掉
  发射与流水线建立成本。
- **就算实现到天花板也赢不了。** 把扫描里每一条 GEMM 都按各自 batch 的 cuBLAS 天花板计费,得到的下界是:
  H=12 **1062 us**、H=96 2707 us、T=32768/H=12 1333 us——H=12 时仅 compose 一项就已经 ≈ 整个 K2 的 0.82×,
  而 flash_kda 的 K2 里还包含了 build_MN 对应的 P6 和输出阶段对应的 P1/P3/P4。也就是说纸面估算错在
  "16× 的额外算力能被空闲 SM 以接近峰值吸收"这一步:H=12 时总共只有 12×512 = 6144 个 128×128 矩阵,
  扫描树每一层的 batch 只有 12~192,把 148 个 SM 摊薄到每个 SM 一两个 128³ tile,效率只剩峰值的 4%;
  H=96 时 batch 够大、效率上来了,但总算力 612 GFLOP 又远超 K2 在 1.3 ms 里能做的事。两头都堵死。

**结论:依赖链本身在数学上是可拆的(仿射递推 ⇒ 可结合 ⇒ 可扫描),而且做出来了、正确性也对——这是本报告
之前没发现的一条真实存在的并行度来源,不是"绕开"而是"拆开";但实测在 TP8(0.30×)、满头(0.04×)、长上下文
(0.33×)三个场景全部大幅慢于 flash_kda,天花板下界也证明它在 K=V=128 这个状态规模下不可能赢:每次 compose
是 128³ 的稠密 GEMM,总算力 16×,而这种 tile 在 B300 上最多只能跑到峰值的 4%~17%。要让扫描路线成立,需要
的是**换算法或换形状**——利用 `M_i` 的"对角 + 秩 16"结构做低秩 compose 而不是稠密 128³(但 32 个 chunk
一组的秩会累加到 512,分块深度受限),或者 head_dim=64 把 compose 单价砍 8×——不是再优化实现。这条路线
的结论与 §3.4/§7/§7.1 一致:K2 现在不是算力瓶颈,而所有"用更多算力换更浅依赖"的方案在这个状态规模下都
买不起。

#### 3.3.2 Design B:分层扫描(不显式 compose)——推导、代价模型与原型(E9)

**§3.3.1 的 E8 为什么输,换个角度看:** 它把每个 chunk 的仿射变换 `(M_i, N_i)` 都材料化成稠密 128×128,再用
128³ 的 GEMM 去复合。真正的问题不是"扫描"这个思想,而是 **compose 的形式**。把结构写清楚:

- 单 chunk:`M_i = D_i − U_i Wᵢᵀ`,`D_i = diag(exp(g_i[−1]))`,`U_i = K_i'ᵀ`(128×16),`W_i = w_iᵀ`(128×16)——
  "对角 + 秩 ≤ 16"(DPLR),这是 delta rule / WY 表示直接给的结构;`N_i = U_i u_i`(128×128,但由 128×16 × 16×128 生成)。
- 两个 chunk 复合 `T_2∘T_1`:
  ```
  A = M_2 M_1 = D_2D_1 − D_2U_1W_1ᵀ − U_2W_2ᵀD_1 + U_2(W_2ᵀU_1)W_1ᵀ
              = D_2D_1 − [D_2U_1 | U_2] · [W_1ᵀ ; W_2ᵀD_1 − (W_2ᵀU_1)W_1ᵀ]        (对角 + 秩 ≤ 32)
  B = M_2 N_1 + N_2
  ```
  复合保持"对角 + 低秩"形式,但**秩每复合一个 chunk 增加 16**:G 个 chunk 复合后秩 ≤ 16G,G ≥ 8 时秩顶到 128
  ——等于稠密。所以"保持低秩结构做全树扫描"(Design A 的结构化版本)只在树的最底下 2-3 层有意义,再往上就退化
  成 E8 那种 128³ 稠密 compose;Design A 的代价模型因此不成立(E8 已实测:16× 算力、天花板下界也赢不了)。
- **Design B 的关键:组摘要不用 compose 就能拿到。** 递推对 S 线性,一组 G 个 chunk 的复合 `T_g(S) = A_g S + B_g`
  可以用"探测"得到:`B_g = T_g(0)`(从零状态跑一遍这组 chunk 的末状态),`A_g = T_g(I) − B_g`,而令 `v = 0`
  时 `u = 0 ⇒ N_i = 0 ⇒ T_g(I)|_{v=0} = A_g`。也就是说 **A_g 由 K2 自己的低秩 chunk 更新一步步累出来,每步仍是
  `w_i@S`(16×128×128)和 `K'ᵀv`(128×16×128),没有任何 128³ 的 GEMM**;稠密 128³ 只出现在组间前缀
  `S_in[g+1] = A_g S_in[g] + B_g`,总共 NG−1 次(NG = NT/G,几十次,不是 E8 的 3×NT = 1536 次)。
- **Design C**(状态无关预处理 + 最小串行传播):FlashKDA 的 K1 就是这个拆分——w、u、INV、Mqk、Aqk 全部与 S 无关,
  K2 里剩下的 `w_i@S`、状态更新、`q@S` 才是串行部分;这条线已经做到头,再往下只能靠 B 去拆 K2 本身。

**零 CUDA 改动的最小可行原型(`exp/e9_hier.py`)。** FlashKDA 的 K2 天然支持 varlen(`cu_seqlens`)和每序列
`initial_state`。把一条 T 的序列切成 NG 个组当成 NG 条虚拟序列:

| phase | 做什么 | 调用 | 并行 CTA | 依赖深度 |
|--|--|--|--:|--:|
| 1a | `B_g` = 各组从零状态跑到末尾的 `final_state` | `flash_kda.fwd(cu_seqlens=组界, initial_state=0)` | B·H·NG | G |
| 1b | `A_g` = 各组 v=0、初始状态 = I 的 `final_state` | `flash_kda.fwd(v=0, initial_state=I)` | B·H·NG | G |
| 2 | `S_in[g+1] = S_in[g]·A_g + B_g`(fp32 baddbmm,状态按 [V,K] 布局右乘) | NG−1 次 bmm | B·H | NG |
| 3 | 回放:各组从正确入口状态跑,得到输出与末状态 | `flash_kda.fwd(initial_state=S_in)` | B·H·NG | G |

代价模型(每 (b,h),T=8192,NT=512,G=32,NG=16):算力 ≈ 3× K2(phase 1a/1b/3 各走一遍全部 chunk)+ 16 条 128³
(0.07 GFLOP,可忽略);临时显存 `A_g,B_g,S_in` = 3·NG·H·32 KB(H=12 时 18 MB);kernel 数 3 次 fwd + NG−1 次
bmm;依赖深度从 512 降到 2·32 + 16 = 80。**预期**:baseline K2 时间与 B·H 无关(单 CTA 链 1.28 ms),分层版三个
fwd 的链长各 G/NT = 1/16,只要 B·H·NG 个 CTA 摊得开(≤ 148 SM × 2 CTA/SM),wall ≈ 3 × 1.28/16 + phase 2
≈ 0.24 ms + phase 2;B·H 大到本来就铺满 SM 时,3× 的算力就是净亏。数值口径:与基线完全相同的 kernel、bf16 状态,
只在组边界多两次 bf16 舍入(A_g、S_in),这是唯一的精度差异来源,需要实测。

**实测(E9,`exp/e9_hier.py`,`logs/e9b_hier_srun.txt`、`logs/e9c_hier_srun.txt`,1095 MHz,50 次迭代)——完整交付
见 `K2_HIER.md`。** 正确性:弱衰减(lb=−0.1/−0.01,组间携带 2%~70% 的状态)下与基线差异在 bf16 舍入量级
(out rel_rms 1.6e-3~6e-3),且对 fp64 真值的误差与基线相同(如 5.949e-3 vs 5.946e-3);T 不整除 G、多 batch、
多 head 都测过。性能(v3:1a/1b 合并成一次 2H-head 调用 + bf16 phase 2 + CUDA graph):

| B·H | T | baseline | hier(最优 G) | 加速 | SM Active(ncu) |
|--:|--:|--:|--:|--:|--|
| 12 | 8192 | 1366 us | 710 us(G=32) | **1.92×** | 8% → 76% |
| 12 | 32768 | 5385 us | 2151 us(G=64) | **2.50×** | |
| 16 | 65536 | 10863 us | 5072 us(G=64) | **2.14×** | |
| 32 | 16384 | 2909 us | 2484 us(G=64) | 1.17× | |
| 64 | 8192 | 1626 us | 2527 us | 0.64× | |
| 96 | 8192 | 1774 us | 3545 us | 0.50× | |

ncu 说明了赢和输的同一个原因:分层版的 recurrence kernel 在单 SM 内和基线**完全一样**(73 寄存器、98 KB smem、
No-Eligible 65% vs 67%、每调度器 2.0 vs 1.5 个 warp),变的只是 grid 从 12 到 192、整卡 SM Active 从 8% 到 76%
——基线是 parallelism-bound,不是链本身能被压短;B·H ≥ 64 时没有空闲 SM 可用,3× 算力就是净亏。这是本报告
里**第一条对 TP8 长序列有正收益的路线**,而且和指令集无关(还没碰 tcgen05)。剩余开销:phase 1 含两遍 K1
(可省)、phase 2 的 NG 步 launch(可写成一个 kernel),纸面还能到 ~2.7×,见 `K2_HIER.md` §10。

**分层之后 SM100 特性还能不能加分(E10,`K2_HIER.md` §11):不能。** 在 phase 1/3 的"很多短链共驻"regime 里
直接对比 tcgen05 版 K2 与 mma.sync 版(T=512,H=148..1184):共驻把 tcgen05 从 0.63× 提到 0.75×,但两种指令都在
2 CTA/SM 饱和——mma.sync 被 98 KB smem 卡住,tcgen05 被 84 KB smem + 256 列 TMEM 同样卡住,TMEM 没有放松共驻
约束;整卡吞吐 58 vs 81 chunks/us。剩下的杠杆是 smem 减负(V-split)提高共驻数,与指令集无关。

### 3.4 compute-bound 还是 memory-bound(`profiles/ncu_fixed.txt`,`logs/e4_ablate_24844.txt`)

纸面 AI:K2 每 chunk 851,968 MAC = 1.7 MFLOP,读 workspace 13.8 KB + v 4 KB、写 out 4 KB → 77 FLOP/B;
K1 每 tile 读 q/k/g 12 KB 写 13.8 KB,算 2×16×16×128×2 + Neumann ≈ 0.2 MFLOP → 8 FLOP/B。
B300 bf16 dense ridge ≈ 2.25 PF / 8 TB/s ≈ 280 FLOP/B(mma.sync 峰值口径 331 TF/8 TB/s ≈ 41)。K2 按 mma.sync 口径在 ridge 之上、
K1 在 ridge 之下;但 ncu 说两者都不是那回事:

| 指标 | K2 | K1 |
|--|--:|--:|
| Compute (SM) throughput | 20.5% | 70.0% |
| tensor pipe active | 30.3% | 5.7% |
| DRAM / L2 throughput | 18.7% / 27.1% | 58.9% / 72.2% |
| L1/smem throughput | 61.2% | 68.7% |
| issue-slot busy / No-Eligible | 20.5% / 67.8% | 71.3% / — |
| active warps per scheduler / eligible | 1.50 / 0.34 | — / 2.26 |
| achieved occupancy | 9.4%(6 warp/SM) | 96.7% warps active |
| 主 stall | wait 1.10, sleeping 0.78, short_scoreboard 0.77, long_scoreboard 0.41 | barrier 8.3, long_scoreboard 4.1 |
| smem bank conflicts / wavefronts | 13.6 M / 82 M | 7.5 M / 50 M |

回答用的 metric:`sm__throughput`、`sm__pipe_tensor_cycles_active`、`dram__throughput`、`lts__throughput`、
`l1tex__throughput`、`smsp__warps_eligible.per_cycle_active`、`smsp__average_warps_issue_stalled_*`、
`sm__warps_active`。K2 是 **latency-bound**(单 CTA、6 warp、串行依赖),最忙的部件是 smem;K1 是 **throughput-bound**,
L2/DRAM 与 issue 同时接近上限(混合型,barrier stall 来自 8 次 `__syncthreads`)。

消融(cycle/chunk,1095 MHz):stock 2760 = P1 610 + P6 770 + P3/4 347 + 流水线/同步 800 + 其余 230。
P1 = 64 HMMA × 8.4 ≈ 538 → tensor 发射 bound;P6 只有 32 HMMA(269)却 770 → 状态 64 KB 的 LDSM.T/STSM.T
(FlashKDA 每 chunk 1674 个 smem wavefront,P6 占 ~600);流水线 800 是 load/store warp 的 mbarrier 握手 +
`tma_store_wait` + named barrier。对照 4.5 的瘦 GEMM表(`in_proj_qkvgfab`,M 小时 tensor core 达成率塌掉):
K2 每个 GEMM 的一维都是 16,与 M≤16 的瘦 GEMM 同一类——tensor core 算力根本不是限制。

### 3.5 bf16 状态精度验证方案与数据(`exp/e5_precision.py`,`logs/e5_precision_24819.txt`)

方案见 PROFILING.md §6。数据(rel_rms,flash / Triton-fp32-state,同一 bf16 输入,金标准 fp64 递推):

| 实验 | out | hT | hT max_abs |
|--|--|--|--|
| A. T=512 / 2048 / 8192 / 16384,lb=−5 | 5.3e-3 / 3.4e-3(四组几乎相同) | 4.6e-3 / 2.3e-3 → 4.6e-3 / 2.4e-3 | 4.4e-3 → 5.4e-3 |
| B. T=4096,lb=−5 / −1 / −0.1 / −0.01 | 5.3→6.9e-3 / 3.3→4.5e-3 | 4.5→6.1e-3 / 2.4→3.8e-3 | 4.1e-3→1.3e-2 |
| B'. gate≡0(无衰减) | 8.0e-3 / 4.9e-3 | 7.4e-3 / 4.2e-3 | 3.3e-2 / 1.2e-2 |
| C. 窗口 first/mid/last(T=16384) | 5.33/5.35/5.37e-3 | | 误差不随位置增长 |
| D. 16×512 分段,状态 I/O bf16 vs fp32 | 5.49e-3 = 5.49e-3(逐位同单次调用) | 4.57e-3 = 4.57e-3 | |
| E. v×64 | 相对误差不变 | | 绝对误差 ×64(bf16 是相对精度) |

结论:bf16 状态使误差比 fp32 状态高 1.9–2.3×,但**不随序列长度累积**(衰减把旧误差洗掉;无衰减时也只到 1.8×),
分段调用零额外损失。这是"内部测试通过"的可复现版本;上线前还应加真实权重的 gate 分布(A_log/dt_bias)扫描。

### 3.6 假设我们是作者:v2 出不出 sm100a 专版?

见 §8。

## 7. 挑战:只换指令不动算法——tcgen05 版 K2(`exp/k2tc/k2_tc2.cu`)

**做了什么。** 一个 128 线程的 CUDA kernel(torch 扩展),每 (seq, head) 一个 CTA,消费 FlashKDA K1 写出的
**原封不动的 workspace**(kd/qd/kr/g_total/INV/Mqk 六个分离数组),CHUNK=16、bf16 状态、舍入点都与 K2 相同,
只把 5 个 mma.sync GEMM 换成 tcgen05:

| GEMM | mma.sync 形状 | tcgen05 形状(D 当 M) | 操作数来源 |
|--|--|--|--|
| G1 k_d@S, q_d@S | 2 × (16×128×128),64 HMMA/warp | **一条指令流** M128 N32 K128(B = [kd;qd] 叠成 N=32) | S^T(smem SW128,线程维护)、KQ(TMA SW128) |
| G2 INV@u | 16×16×128 | M128 N16 K16 | uT(线程写 INTER)、INV(bulk + 32 线程重排) |
| G2' Mqk@U | 16×16×128 | M128 N16 K16,独立累加器 | UT、Mqk |
| G3 k_r^T@U | 128×16×128,32 HMMA/warp | M128 N128 K16,B 用 MN-major SW128 | UT、KR(TMA) |

链:TMA 等待 → G1(8 条)→ 读 uT → u=(v−u)β 写 smem → G2 → 读 u2T 写 UT → G2'+G3 → 读 kU^T(8 次 tcgen05.ld)
→ S^T = bf16(S^T·g + kU^T) 写回 smem;上一 chunk 的 out epilogue 与本 chunk 的 G1 重叠(TMEM 累加器双缓冲)。
描述符(INTER / SW128,K-major / MN-major 的 LBO、SBO 语义)全部先在 `exp/mb_mma.cu` 里对 CPU 参考验证再用。

**正确性。** 与 FlashKDA 的 out、final_state **逐位一致**(rel 0.0,T∈{256,1024,2048}),对 fp64 朴素递推与
FlashKDA 同误差(5.2e-3 / 4.5e-3)。逐段 dump(`exp/e7_k2tc.py debug_chunk0`)证明每个 GEMM 与 torch 参考一致。

**性能(1095 MHz,cycle/chunk;`logs/e7d_24944.out`,`logs/e7_ablate_24951.txt`)。**

| 版本 | cyc/chunk | vs mma.sync 2732 | 主要变化 |
|--|--:|--:|--|
| v1(cp.async 逐线程加载,INTER 布局) | 8800 | 0.31× | 结果错(workspace 布局);计时有效 |
| v2(TMA + SW128,布局修正) | 6250 | 0.44× | 逐位正确;线程侧 generic LD/ST、标量 F2F |
| v3(LDS/STS、bf16x2 打包转换、beta bulk、TMA 发射移 warp 1、out epilogue 与 G1 重叠) | **4460** | **0.62×** | |
| v3 − E1(u 计算) | 4171 | 0.66× | E1 ≈ 290 |
| v3 − E3(out tile) | 4238 | 0.64× | E3 ≈ 224 |
| v3 − E4(状态更新 RMW) | 2680 | 1.02× | **E4 ≈ 1780** |
| v3 − E1 − E3 − E4(只剩 tcgen05 链 + TMA + 同步) | **2030** | 1.34× | 异步 MMA 路线的地板 |

v3 分相(线程 0,cycle):TMA 等待 96 | INV 重排+同步+发射 310 | G1 发射(8 条 tcgen05.mma,单线程)525 |
上一 chunk 的 E3+out 与 G1 重叠(G1 等待只剩 75)| E1 434 | G2 往返 193 | E2 125 | G2'/G3 往返 253 | E4 1835 | 合计 ≈ 4960(含计时开销)。
ncu(`profiles/ncu_k2tc2.txt`):tensor 指令 20.4 M → 0.54 M,但线程指令 174 M → **193 M**(反而更多),smem wavefront 79 M → 57 M,
No-Eligible 77%,每调度器 1 warp;stall 前三 wait 0.97、short_scoreboard 0.87、barrier 0.48。

**为什么没有收益(量化)。**
1. 固定往返:每 chunk 3 次真依赖的 issue→commit→mbarrier 往返(G1 629、G2 193、G3 253)+ 4 次 TMEM 回读(139–446)
   ≈ 1800 cycle,与 FlashKDA 整条 mma.sync 链的 MMA 部分(≈1000)同量级;tensor 峰值 4× 完全被这些延迟吃掉。
2. 小 N 指令按条计费:N=16/32 的 tcgen05.mma 只有 653/1410 MAC/cycle,G1 的 8 条 N32 指令 372 cycle,只比 mma.sync 的 538 少 30%。
3. 状态更新不是 GEMM:S^T = bf16(S^T·g[k] + kU^T) 每 chunk 要把 32 KB 状态读一次写一次并做 16K 次 FMA+转换,
   在 mma.sync 版里它藏在 P6 的寄存器 fragment 流水里(P6 全部 770);在 tcgen05 版里它必须经过 TMEM 回读 → 寄存器 → smem,
   4 个 warp、每调度器 1 个 warp 掩盖不了延迟,花 1780。把它做到 FlashKDA 水平(≈500)也只是回到 ≈3200,仍慢于 2732。
4. 单线程发射:8 条 tcgen05.mma 由一个线程发 525 cycle(每条 ~65),CUTLASS 也如此;mma.sync 由 4 个 warp 并行发。

**要真的赢需要什么(不在"只换指令"范围内)。** 状态常驻 TMEM 做 A 操作数(去掉 32 KB/chunk 的 smem 往返,但 g 缩放仍要 ld/st
128 列 fp32 ≈ 900 cycle)、或把 g 折进 GEMM(A=[diag(g) | k_r^T],K=144,bf16 的 g 有 2^-9 相对误差)、或 CHUNK=32 摊薄往返
——都是算法/数值改动,不是指令替换;且地板实验说明就算线程侧归零也只有 1.34×。

### 7.1 追问:换成 CUTLASS(而不是手写 PTX)会不会翻盘?

动机:assignment02 M4 里同一张卡上做过对照——手写 tcgen05 GEMM 只有 cuBLAS 的 29-35%,换成 CUTLASS
的 collective builder(开箱即用、没调参)直接跳到 **≈95%**(`70_blackwell_fp16_gemm`,4096³,992.9
TFLOPS vs cuBLAS ≈1049)。既然 CUTLASS 对"写得不够好"的 GEMM 有这么大的救场能力,值得倒回来问:
K2 换成 CUTLASS 而不是手写 PTX,会不会把 0.62× 翻成正收益?—— **不会,而且这次不是猜,是从 CUTLASS
自己的源码里验证的。**

1. **单线程发射是 tcgen05 的 ISA 约束,不是 CUTLASS 会不会写代码的问题。** 直接读 CUTLASS 4.7.0 的
   `cute/arch/mma_sm100_umma.hpp`(本机 `/tmp/cutlass`,B300 4090 台 SM100 GEMM 全靠它):文件里全部
   62 处 `tcgen05.mma` 内联汇编,逐一核对,**每一处都被 `if (cute::elect_one_sync()) { ... }` 包着**
   (50 处 `elect_one_sync` 调用,一一对应,无例外),和我们 `k2_tc2.cu`/`03_pipeline.cu` 里手写的
   `elect.sync` 门控写法完全一样。也就是说"8 条 tcgen05.mma 由一个线程发 525 cycle"(§7 原因 4)是
   硬件指令本身的限制——`tcgen05.mma` 就是设计成由每个 warp 里恰好一个被选中的线程发射,CUTLASS 作为
   软件库无法绕开指令集的这条规则,用不用 CUTLASS,这笔发射延迟都躲不掉。
2. **CUTLASS 的深流水线机制吃的是"独立工作项",K2 恰好一个独立工作项都没有。** M4 的经验(4.3 pipeline
   实验)已经证明:软件流水线要有收益,前提是存在多个**互不依赖、可以乱序完成、只在最后累加**的工作单元
   ——GEMM 里是 K 维的多个 tile,grid 里是多个互相独立的 block。CUTLASS 的 warp specialization、
   多级 TMA 预取全部构建在这个前提上。K2 恰恰相反:同一个 (seq, head) 里第 chunk N+1 个 chunk 的 G1
   要读的是**第 N 个 chunk 更新完的状态 S**(见 §7 链路图,"上一 chunk 的 out epilogue 与本 chunk 的
   G1 重叠"这句话本身就说明重叠的只有不碰状态的 out 部分,状态更新完成前 G1 无法开始)——这是一条真数据
   依赖的递推链,不是可结合的归约。512 个 chunk 严格串行,CUTLASS 的深流水线在这种"零独立工作项"的负载
   里没有任何东西可以拿来流水,和"换不换库"无关。
3. **CUTLASS 的示例库里找不到任何递推/scan 类核**(`examples/` 搜 recurrent/scan/ssm/mamba/linear_attn
   零命中)——这不是 CUTLASS 还没来得及支持,而是它整套 collective builder 的抽象(mainloop = 对 K 维
   可结合地归约,tile scheduler = 把独立输出 tile 分给 CTA/cluster)从设计上就是给"大 GEMM/conv"这类
   问题定义域用的,K2 这种"每 chunk 一次状态 RMW + 强制串行"的负载根本不在这个域里,不存在"CUTLASS 版
   K2"这种东西能直接套用现成 mainloop——真要用 CUTLASS 也只是借它的 `tcgen05.mma`/TMA atom 当更方便
   的封装去手写同一条链,不会改变链本身的延迟结构,等价于我们已经做的 `k2_tc2.cu`。

**结论:这次"重新想一遍"没有推翻讨论点 6,反而把它补上了一条本来缺的论据**——即便换成生产级库,K2 的
瓶颈依然成立,因为瓶颈来自 tcgen05 的 ISA 硬约束(证据 1)和算法本身的串行递推(证据 2、3),不是"我们
代码写得不够专业"。这和 M4 里 CUTLASS 能救场的情况(hand-written GEMM 写得不够好,存在大量独立工作项
没利用上)是两类不同的问题——M4 的差距是**工程质量差距**,CUTLASS 能补;K2 的差距是**算法结构性的**,
换任何库都补不了,能救的只有 §7 末尾"要真的赢需要什么"里列的那几条算法/数值改动。

### 7.2 profile 归因:tcgen05 版 K2 到底慢在哪,哪些能改,改完能到哪

数据来源:`profiles/ncu_k2tc2.{txt,csv}`(T=8192,H=96,两个 kernel 同一作业)、§7 的 v3 分相 stamp(1095 MHz,
cycle/chunk)、E10 的共驻扫描。

**(1) 两个 kernel 的 ncu 对照——问题不在 tensor pipe,在线程侧和占用率。**

| 指标 | mma.sync K2(FlashKDA) | tcgen05 K2(k2_tc2 v3) |
|--|--:|--:|
| Duration(T=8192,H=96) | 1.29 ms | 2.09 ms |
| tensor 指令数 | 20.4 M | **0.54 M**(−97%) |
| 线程指令数(`smsp__inst_executed`) | 174 M | **193 M**(+11%) |
| smem wavefronts / bank conflicts | 79 M / 10.6 M | 57 M / 5.2 M |
| 寄存器/线程 | 73 | **242** |
| 线程数/CTA → warp/调度器 | 192 → 1.5 | 128 → **1.0** |
| Block limit(寄存器 / smem) | 4 / 2 | **2** / 2 |
| No-Eligible / Issued warp per scheduler | 67% / 0.33 | **77% / 0.23** |
| stall 构成(cycle/issue,占比) | wait 24%,selected 22%,short_scoreboard 16%,sleeping 15%,long_scoreboard 10% | selected 23%,wait 22%,short_scoreboard 20%,**barrier 11%**,**no_instruction 10%**,long_scoreboard 10% |

换指令把 tensor 指令砍掉 97%,但**线程指令反而多了 11%**——被砍掉的 HMMA 是"免费"藏在 4 个 warp 的 fragment 流水
里的,替换它们的是 TMEM 回读、bf16 打包、swizzle 地址计算、smem RMW 这些标量指令;而执行这些指令的只有 4 个 warp
(每调度器 1 个),任何固定延迟都裸露(No-Eligible 77%)。

**(2) 每 chunk 4460 cycle 的去向(v3 分相,§7)与对应的 stall 机制:**

| 相 | cycle | 占比 | 机制(profile 证据) | 能不能改 |
|--|--:|--:|--|--|
| E4 状态更新 `S^T = bf16(S^T·g + kU^T)` | **1835** | 41% | 每线程一整行:`r[128]` 常驻 128 个寄存器(→242 regs),8 次 tcgen05.ld(128×128 回读 446)+ 16 次 LDS.128 + 32 次 LDS.128 读 g + 128 FFMA + 64 F2FP + 16 STS.128,全部在 1 warp/调度器下串行暴露(wait + short_scoreboard 共 42%);指令数 ≈350/线程/chunk,按 IPC=1 只要 ~350 cycle,实际 5× | **能**:状态常驻 TMEM 当 G1 的 A 操作数(tcgen05.mma 允许 A 来自 TMEM),更新变成 tcgen05.ld×2 → FFMA → tcgen05.st,去掉 32 KB smem RMW、swizzle 地址计算和 `fence.proxy.async`;线程数 128→256 让每线程只管半行(r[64],寄存器 ~130,warp/调度器 2)。预计 1835 → ~500 |
| G1 发射(8 条 K16,单线程) | 525 | 12% | 每条 ~65 cycle 的发射代价,ISA 决定(§7.1) | **不能**:K=128 至少 8 条;只能靠与上一 chunk 的 E3 重叠(已做) |
| E1 `u=(v−u)β` | 434 | 10% | TMEM 回读 139 + 16 次 2 B 标量 LDS 读 v + 16 次 MUFU(sigmoid)+ INTER 布局写 + fence + sync | **部分能**:sigmoid(β) 在 K1 算好(−60);v 用 LDS.128;uT 若也放 TMEM(G2 的 A 可来自 TMEM)省掉 fence+sync(−100)。预计 → ~250 |
| INV/Mqk 重排 + sync + 发射 | 310 | 7% | K1 写的是 row-major,kernel 里 32 线程重排成 INTER 再 sync | **能**:让 K1 直接写 INTER(或 TMA 3-D box 直接落 INTER),整段消失(−250) |
| G2'/G3 往返、G2 往返 | 253 + 193 | 10% | issue→commit→mbarrier 固定延迟(§3.2 microbench 265/条) | **不能**:真依赖 |
| E3 out tile(与 G1 重叠后剩余) | 224 | 5% | TMEM 回读 + 打包 + 写 staging | 部分能(重叠更多) |
| 4 次 `__syncthreads` + TMA 等待 | ~300 | 7% | barrier stall 11%:4 个 warp 每相都要对齐;no_instruction 10%:全展开的 r[128] 循环把代码撑大,I-cache miss(FlashKDA 这一项为 0) | 部分能:减少展开、合并相位(INV 重排消失后少一次 sync) |

**(2b) SASS 级 stall 采样(`profiles/ncu_k2tc2_src.ncu-rep`,`--section SourceCounters`,T=8192 H=12,6477 个采样):**

| stall 落在 | 占比 | 对应 |
|--|--:|--|
| FFMA 20.4% + F2FP 5.5% + IMAD 8.2% + LOP3 3.2% + LDS 4.8% + STS 3.1% | **≈45%** | E4 的 `S·g + kU` 链:最热的单条指令是 `FFMA R144, R208, R237, R116`(5.3%),它等的是 S(LDS)、g(LDS)、kU(LDTM)三路输入;IMAD/LOP3 是 swizzle 地址计算 |
| BRA / @P BRA / BRA.U(`mbarrier.try_wait` 轮询循环) | **≈30%** | 等 MMA commit / TMA 落地:4 个 warp 一起在 `mbar_wait` 里转,这就是 §7 说的 3 次往返 + TMA 等待,ISA 决定的地板 |
| BAR + BSYNC | ≈4% | 每 chunk 4 次 `__syncthreads` |
| LDTM(tcgen05.ld 本身) | 1.2% | TMEM 回读不是问题,问题在等它的消费者 |

两块合起来 75%:一块(45%)是线程侧算术在 1 warp/调度器下裸露延迟——**可改**;另一块(30%)是异步 MMA 的往返等待——
**不可改**。这和 stamp 分相、以及 §7 "删光线程侧只剩 2030" 的地板实验三方互相印证。

**(3) 改完能到哪(纸面,按上表逐项扣):** 4460 − E4 1335 − INV 250 − E1 180 − E3/sync ~150 ≈ **2550 cycle/chunk**,
对 mma.sync 的 2732 是 **≈1.07×**——也就是**全部可改的项都改到位,tcgen05 版最多和 mma.sync 打平**。原因是 §7 的
地板实验:把线程侧工作全部删掉,异步 MMA 链 + TMA + 同步本身就是 2030 cycle,占 mma.sync 全部时间的 74%;线程侧
无论怎么优化都只能在剩下的 26% 里做文章。这个上限估计还没算 TMEM 预算的代价:状态常驻 TMEM(64 列)+ G1 32 +
G2/G2' 32 + G3 128 + 双缓冲 64 = 320 列 > 256,会把共驻从 2 CTA/SM 压到 1(E10:共驻 1→2 值 1.6×);要保 2 个
CTA 就得把 G3 拆成两个 N=64 半程(多一次往返 ≈ +250)。

**(4) E10 补充的一层:** 在分层扫描造出来的"多短链共驻"regime 里,tcgen05 版从 0.63× 升到 0.75×,说明它的
等待型 stall 确实可以被共驻 CTA 填,但两种指令都被 smem/TMEM 卡在 2 CTA/SM,填不满(issue slot 35%)。

**但有一个例外(E13,`K2_HIER.md` §13):分层扫描的组摘要是稠密 128³ 链,不是原始 K2 链。** 用 tcgen05 做这种
compose(A=[Sᵀ|uᵀ] 常驻 TMEM、TS 形式取 A、每步 9 条 K16 + 重打包)microbench 实测 1054 cycle/步(1 CTA/SM)、
705 cycle/步(2 CTA/SM),比 FlashKDA K2 每 chunk 快 2.6~3.0×,对 CPU 参考 PASS;投影到整条分层流水线对现版
再 ~2.2×、对原版 ~4.2×。这是 SM100 特性在本报告里唯一的正收益,前提是先做算法重构。

**结论:** tcgen05 版慢的直接原因是**线程侧工作变多(+11% 指令)且只有 1 warp/调度器来执行**——E4 的 smem RMW
(41%)是最大的单项,可以用"状态常驻 TMEM + 256 线程"砍掉约 3/4;INV 重排、E1 的标量路径、I-cache 也各有一两百
cycle 可省。但把这些全做完,上限也只是和 mma.sync 打平(≈1.07×),因为 tcgen05 链本身的固定往返(2030)已经占了
mma.sync 总时间的 74%。这与 §8 "v2 不出 sm100a 专版"一致;能在 K2 上真正赢的仍然是 §3.3.2 的分层扫描和 smem 减负。

## 8. 综合结论(讨论点 6):v2 不出 sm100a 专版

**峰值侧论据(支持出专版)**:tcgen05 在 N=128 时 4088 MAC/cycle/SM,是 mma.sync 的 4.0×;K2 的 P1 是 tensor 发射 bound(610 cycle);
TMEM 累加器省寄存器,TMA + SW128 与现有 GMMA 布局兼容,workspace 零改动即可接入(本工作证明)。

**可移植性/收益侧论据(反对)**:
- K2 的 2760 cycle/chunk 里 MMA 发射只占 ≈1000(36%),其余是流水线/同步 800、状态 smem 往返 500、小 GEMM 与转置 350;
  tensor pipe 只有 30% 忙——不是算力问题(§3.4)。
- 换指令后引入的固定延迟(3 次往返 + 4 次回读 ≈ 1800)≥ 省下的发射时间;实测 0.62×,地板 1.34×(§7)。
- CHUNK=16 使每个 GEMM 一边只有 16,tcgen05 在这个形状上每条指令只有 1/6 效率;要用满需要 CHUNK ≥ 128 当 N/K,而 CHUNK=32 就已破数值范围(§3.1)。
- SM80 mma.sync 版一份源码跑 SM80/90/100/120,而 sm100a 专版 = 另一套 TMEM/描述符/同步语义的代码(本目录 ~400 行,3 个只在硬件上才暴露的 bug:
  分离布局、1024 对齐的 swizzle 相位、generic vs shared 指针)。
- 精度上 bf16 状态已是 fp32 状态误差的 2×(§3.5),sm100a 路线不改善精度。

**该做什么(架构无关,按预期收益排序)**:
1. **CHUNK=32 + cumsum 中点重标定**:零成本修范围(§3.1),把每 chunk 800 的流水线同步和 500 的状态往返摊到 32 个 token,
   Neumann 从 14 条涨到 128 条 HMMA/chunk 但每 token 仍只 4 条;纸面 K2 ≈ 3800 cycle/32 token vs 5520 → −30%。
2. **压掉 800 cycle 的 load/store 流水线开销**(29%):ncu stall `wait`/`sleeping` 最高,是 mbarrier 握手与 `tma_store_wait` 的固定延迟,
   与 HBM 带宽无关(DRAM 19%)。
3. **V-split(BV=64/32)**:对 TP8(12 CTA / 148 SM)把 CTA 数 ×2–4,压缩 P1 发射与 P6 流量(≈40% 的吞吐型部分),预期 1.25–1.4×;
   fla 的 Triton 就这么做。
4. **分层扫描(§3.3.2,E9 已实现并实测):做,且排在 V-split 前面——它是唯一实测对 TP8 长序列有正收益的路线**
   (1.92× @T=8192、2.50× @T=32768,零 CUDA 改动即可拿到;K1 只算一遍 + phase 2 单 kernel 后纸面 ~2.7×),
   条件是按 B·H 门控(≤32 启用)。详见 `K2_HIER.md`。
5. **全树扫描(§3.3.1,E8 已实现并实测):不做。** 数学成立、实现正确,但 TP8 0.30×、H=96 0.04×、
   长上下文 0.33×,天花板下界也赢不了(H=12 仅 compose 就 ≈ K2 的 0.82×)。根因是 128³ 稠密 compose 的
   16× 总算力在 B300 上只能以峰值 4%~17% 的效率执行;只有换成利用"对角+秩 16"结构的低秩 compose 或
   head_dim=64 才有翻盘可能,那是算法改动,不在 K2 优化范围内。
6. 若一定要上 SM100:**只在分层扫描的组摘要/组间前缀上用**(E13:稠密 128³ compose 链 2.6~3.0×,纸面整体再 2.2×);"只换指令"做原始 K2 链本身是负收益。

## 附加观察(K1、varlen、fp32 状态 I/O)

- K1:tensor pipe 5.7%,L2 72%、DRAM 59%、issue 71%,barrier stall 8.3——吞吐型、接近多个上限,随 H 线性,不是难点。
- varlen:1024×8 的 K2 只有 689 us(768 CTA × 64 chunk),同一 SM 上 2 个 CTA 几乎不互相拖慢,再次说明单 CTA 远未用满 SM。
- fp32 状态 I/O 只是 K2 前后各一次 32 KB 转换,与 bf16 版时间相差 <4%。

## 9. 复现

```
# 环境(登录节点):
git clone --recurse-submodules https://github.com/MoonshotAI/FlashKDA ~/flashkda-build/FlashKDA && cd $_ && git checkout 1ce47ea
cd assignment02 && uv pip install einops "flash-linear-attention>=0.5.0" && FLASH_KDA_CUDA_ARCHS=100a,103a uv pip install --no-build-isolation -v ~/flashkda-build/FlashKDA
# 变体与挑战 kernel(登录节点编译):
python3 team/c1_flashkda/claude/exp/build_variants.py ; (cd team/c1_flashkda/claude/exp/k2tc && TORCH_CUDA_ARCH_LIST=10.3a ../../../../../.venv/bin/python setup.py build_ext --inplace)
nvcc -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a -o team/c1_flashkda/claude/exp/mb_mma team/c1_flashkda/claude/exp/mb_mma.cu
# GPU 作业(一次一张卡):
cd team/c1_flashkda/claude && sbatch -G 1 --time=00:30:00 -o logs/e1_%j.out exp/job_e1.sh     # E1/E3
sbatch -G 1 --time=00:55:00 -o logs/e2_%j.out exp/job_e2.sh                                    # E6/E5/E2/E4
sbatch -G 1 --time=00:20:00 -o logs/e7d_%j.out exp/job_e7d.sh                                  # E7 tcgen05 K2 正确性/计时/ncu
sbatch -G 1 --time=00:15:00 -o logs/e7e_%j.out exp/job_e7e.sh                                  # E7 消融(地板)

# E8(§3.3.1 扫描版,直接 srun 占卡,不走 sbatch):
# PYTHONPATH=$C1 $PY exp/e8_scan.py check                 # 登录节点 CPU 也能跑:fp64 对拍 naive
# srun -G 1 --time=00:30:00 bash -c 'PYTHONPATH=$C1 $PY exp/e8_scan.py bench'    # -> logs/e8_scan_srun.txt
# srun -G 1 --time=00:20:00 bash -c 'PYTHONPATH=$C1 $PY exp/e8_scan.py bench2'   # -> logs/e8_bench2_srun.txt(含天花板/v1/v2)
# srun -G 1 --time=00:10:00 bash -c '$PY exp/e8_gemm_ceiling.py'                  # -> logs/e8_gemm_ceiling_srun.txt
```
