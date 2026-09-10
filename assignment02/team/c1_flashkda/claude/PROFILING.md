# 我是怎么 profile 的,以及每一步 profile 结果怎么决定下一步

这份文档只讲方法和决策链;结论与完整数据表在 `REPORT.md`。所有数字来自本机 B300(cc 10.3,148 SM),
原始日志 `logs/`,ncu 报告 `profiles/`。

## 0. 先定"计量单位":cycle,不是 us

第一个作业(24796)的 SM 时钟全程 1095 MHz,第二个作业(24819)全程 2032 MHz——同一台机器不同的卡/分配,
时钟策略不同(`logs/clock_<job>.csv` 是每 0.5 s 一次的 `nvidia-smi` 采样)。第一次对比时 K2 从 1285 us
"变成" 748 us,差 1.72×,正好是时钟比。所以:

- 每个作业启动一个后台时钟采样器(`exp/env.sh`),报告里每个 us 数字都标注时钟;
- 跨作业比较一律换算成 **cycle/chunk**(us × MHz / chunk 数),或直接用 ncu 的 `sm__cycles_elapsed`;
- microbench 一律用 `clock64()` 计 cycle,天然与时钟无关。

换算后两次测得的 K2 都是 ≈2760 cycle/chunk,数据一致,才敢往下走。

## 1. 复现层:先确认"主路径是 SM80 MMA"是事实而不是文档说法

- **工具**:`cuobjdump -sass` 对已安装的 `flash_kda_C*.so`,按 kernel 拆分做指令直方图
  (`profiles/sm103a_hist.txt`,脚本在 LOG 里)。
- **看到**:K2 每个实例 104 条 `HMMA.16816.F32.BF16`,K1 32 条 bf16 + 12 条 `HMMA.16816.F16`(Neumann 求逆),
  两份 cubin(sm_100a / sm_103a)完全一样;全库 0 条 HGMMA / UTC*MMA(tcgen05)。TMA(UTMALDG/UTMASTG)、
  LDSM/STSM/MOVM 都在,说明数据通路是 SM90 的,只有算术是 SM80 的。
- **104 这个数怎么核对**:按源码数 Phase 1 = 8 k-step × 2 块 × (k,q) × 2 = 64,Phase 3/4 各 4,Phase 6 = 8 × 2 × 2 = 32,
  合计 104——源码到 SASS 一一对应,后面消融时才能按 phase 拆。
- **ncu 佐证**(`profiles/ncu_pipes.txt`):K2 `sm__inst_executed_pipe_tensor.sum` = 20,447,232 =
  96 CTA × 512 chunk × 4 warp × 104 ✓。`*_op_hmma` 系列指标在 Blackwell 上 n/a,所以用 `pipe_tensor` 总量核对。

## 2. 第一次分解:kernel 级(torch.profiler)

- **工具**:`exp/e1_kernels.py`,`torch.profiler` 的 CUDA 事件,按 kernel 名汇总。比 nsys 轻,够用。
- **看到**(1095 MHz):T=8192 时 K2 ≈ 1285 us、K1 = 458 us(H=96)/309(H=64)/67(H=12);
  **K2 与 H 完全无关**(H=96/64/12 都是 1280 ± 10 us),T=32768 时 K2 线性到 5078 us。
- **推断**:K2 = 每个 head 一个 CTA 串行走 T/16 个 chunk;时间 = chunk 数 × 每 chunk 固定时长(≈2.5 us ≈ 2760 cycle),
  与并行度无关。TP8 部署(每卡 12 头)只用 12/148 个 SM 却要同样的 1.28 ms。
- **决定**:所有后续分析集中在"K2 的每 chunk 2760 cycle 里是什么";K1 是 token 并行的、随 H 缩放,不是难点。

## 3. 第二次分解:ncu 看 K2 是什么 bound

- **工具**:官方 `benchmarks/ncu.sh` 模板(`--set full`,`--clock-control none`)+ 自选 metrics。
- **看到**(`profiles/ncu_fixed.txt/.csv`,K2):Compute(SM) 20.5%,DRAM 18.7%,L2 27%,**L1/smem 61%**(最高);
  issue slot busy 20%,No-Eligible 68%,每调度器 1.5 个 active warp、0.34 个 eligible;
  达成占用率 9.4%(6 warp/SM),Waves/SM 0.32;tensor pipe active 30%;
  stall 前三:`wait`(固定延迟)1.10、`sleeping`(mbarrier try_wait)0.78、`short_scoreboard`(smem/LDSM 依赖)0.77,
  `long_scoreboard`(global)只有 0.41。smem bank conflict 13.6 M(其中 st 10.7 M)对 82 M wavefront。
- **推断**:既不是算力 bound 也不是带宽 bound,是**单 CTA 内的串行延迟链**:4 个 MMA warp 大部分时间在等
  LDSM/HMMA 的短依赖和 barrier;smem 是最忙的部件(状态 32 KB 每 chunk 读两次写一次)。
- **决定**:要知道 2760 cycle 在 phase 之间怎么分,ncu 的整体指标不够——做消融构建。

## 4. 第三次分解:消融构建(profile 做不到的,就改代码量)

- **方法**:`exp/build_variants.py` 从上游源码打补丁,每个变体单独编成 `flash_kda_C_<name>.so`(sm_103a),
  `exp/e4_ablate.py` 直接加载 .so 计时(结果是错的,只要时间)。上游自带 `TMA_DISABLE_ALL` 宏(去掉 load/store warp
  与流水线等待),我在它之上再逐个删 phase。
- **看到**(cycle/chunk,job 24844 @1095 MHz;24819 @2032 MHz 数字一致):

  | 变体 | cycle/chunk | 差值 → 归因 |
  |--|--:|--|
  | stock | 2760 | |
  | notma(无 TMA/流水线) | 1959 | 流水线 + 同步 + store warp ≈ **800(29%)** |
  | notma − Phase 1 | 1349 | Phase 1(k@S, q@S,64 HMMA/warp)≈ **610** |
  | notma − Phase 6 | 1189 | Phase 6(状态更新 32 HMMA + 全状态 LDSM.T/STSM.T)≈ **770** |
  | notma − Phase 3/4 GEMM | 1612 | INV@u, Mqk@U(8 HMMA + MOVM 转置)≈ **347** |
  | stock − Phase 6 | 1770 | 带流水线时删 Phase 6 省 990:Phase 6 的 smem 流量与 TMA 抢 smem |
  | 余下 | ≈230 | cast、beta、STSM out、named barrier |

- **对照 microbench**(`exp/mb_mma.cu`,T1/T2):HMMA.16816 依赖延迟 20.8 cycle,单 warp 独立发射 8.4 cycle/条;
  K2 的配置(4 warp × 4 个独立累加器)能打到 977 MAC/cycle/SM ≈ SM 峰值(1020)。Phase 1 = 64 HMMA × 8.4 = 538 ≈ 实测 610,
  **Phase 1 是 tensor pipe 发射 bound**;Phase 6 只有 32 HMMA(269 cycle)却花 770,**多出的 500 是状态的 smem 往返**
  (LDSM.T 32 KB + STSM.T 32 KB + FMA 缩放)。
- **决定**:tcgen05 只能加速"HMMA 发射"那部分(610 + 269 + ~100 ≈ 1000 cycle,36%),而且 K2 每个 GEMM 的一边都只有 16
  ——先用 microbench 量 tcgen05 在这些形状上到底多快,再决定值不值得写 kernel。

## 5. microbench 决定 tcgen05 方案的取向

- **工具**:`exp/mb_mma.cu`(纯 PTX,不用 cute),每个配置先对 CPU 参考做正确性,再测 latency/throughput。
  正确性检查抓到了两个 bug:K=128 时 k-step 偏移应是 2×LBO(第一次全错);MN-major 描述符的 LBO/SBO 语义与我假设的相反
  (两个假设各编一个模板,PASS 的那个就是对的)。M=64 的 TMEM lane 映射也不是 0..63 直排,直接放弃 M=64 取向。
- **看到**:
  - tcgen05 一条 K=16 指令在 N ≤ 64 时都 ≈46 cycle(按条计费),N=128 才 64 cycle 到峰值 4088 MAC/cycle/SM;
    N=16 只有 653 MAC/cycle——**比 mma.sync 的 SM 峰值 1020 还低**。
  - issue → commit → mbarrier wait 的固定往返 ≈ 265 cycle;8 步 K=128 链 752 cycle。
  - `tcgen05.ld` 128×16 fp32 回读 139 cycle,128×128 回读 446 cycle。
- **决定**:(1) 把 k_d、q_d 叠成 N=32 的一个 B tile(同一条指令流,几乎不加时间);(2) 每 chunk 至少 3 次
  MMA 往返(u → U → S 是真依赖)≈ 800 cycle 固定开销 + 4 次 TMEM 回读 ≈ 500,纸面预估 ≈ 2900 cycle/chunk,
  **预测不会比 mma.sync(2760)快**。仍然把 kernel 写出来(`exp/k2tc/k2_tc.cu`)验证预测——挑战层"做不出正收益
  也算完成",但要用实测而不是纸面说话。kernel 内置 `clock64` 分相计时(TIMING 模板参数),以便实测后再归因。

## 6. 精度实验的设计逻辑(讨论点 5)

官方只说"内部测试通过"。要回答"bf16 状态到底丢多少",必须把 bf16 **状态**的误差从 bf16 **输入**的误差里分离出来:
fp64 逐 token 递推(`fla_kda_ref/naive.py`)做金标准,fla Triton(fp32 状态、同样的 bf16 输入)做对照,
flash_kda 的误差 / Triton 的误差 就是"bf16 状态"的代价。再沿三个轴扫:序列长度(误差随 T 累积吗)、衰减强度
(gate→0 时状态是 T 个 rank-1 更新的和,是 bf16 累加器最坏情况)、多次调用状态传递(bf16 vs fp32 状态 I/O)。

## 7. 讨论点 1 的数值实验为什么这样做

"CHUNK=32 时哪个先破"不能只看 |cumsum| 上界(C·|lb|·log2e = 231 > 126),要看真实分布下有多少元素真的溢出:
用 benchmark 的 g/A_log/dt_bias 分布和最坏情况(gate ≡ lb)各算一遍,统计 k·2^cumsum 下溢、k·2^-cumsum 上溢的比例,
并算 L = k_d k_inv^T 与 fp64 的相对误差(溢出后是 nan)。Neumann 部分用真实构造的 L 测 fp16/bf16/fp32 三种精度,
并按 m16n8k16 条数折算成每 token 代价。

## 8. 挑战 kernel 的"实测 → 归因 → 改"循环(`exp/k2tc/`,日志 `logs/e7*_*.txt`)

kernel 内置 `clock64` 分相计时(TIMING 模板参数,线程 0 记录每相累计 cycle),每一版都先看分相,再决定改什么。
全部 cycle 数在 1095 MHz 作业上测得(2032 MHz 作业的 cycle 数一致)。

| 版本 | cyc/chunk | 相对 mma.sync | 分相看到什么 | 据此改什么 |
|--|--:|--:|--|--|
| v1 | 8800 | 0.31× | 结果 NaN;dump 逐段对比发现 chunk 0 各 GEMM 全对、但 out 与 FlashKDA 不符 | **workspace 布局读错**:不是每 tile 13.8 KB 连续,而是 kd/qd/kr/gt/INV/Mqk 六个分离数组——kernel 与我的 torch 参考都按错的布局读,所以互相一致 |
| v1 | 同上 | | "prefetch issue" 1800、E1 1200、E3+E4 3800 cycle,远超纸面 | cp.async 逐线程发 1122 个 16 B 请求太贵;改 TMA(2-D/3-D box + 1-D bulk),布局改 SW128 让 TMA 原生落盘 |
| v2 | —(chunk 1 起数据错) | | chunk 0 全对,chunk 1 的 KQ/V tile rel 1.4(像另一个 tile) | **stage 1 基址 18560 不是 1024 对齐**,SW128 的 XOR 相位取自地址位,TMA 写入与 MMA 读取相位不一致;stage 大小圆整到 1024 |
| v2 | 8800 → 6250 | 0.33 → 0.44× | 逐位正确。"relayout+sync+issue" 945:tid<16 同步读全局 beta(~800 cycle 延迟)挡住了发射;E1 1150、E4 2240 仍高 | beta 改成 host 转置 + 32 B bulk copy;**SASS 里 smem 访问是 generic LD/ST 而不是 LDS/STS**(对齐指针的 uintptr 运算丢了地址空间),改回 __shared__ 指针算术 |
| v2' | 6250 | 0.44× | SASS 有 512 条 `F2F.BF16.F32`(慢管线标量转换),FlashKDA 用的是 `F2FP...PACK_AB` | 所有转换改 `cvt.rn.bf16x2.f32`;g 用 float4 读;TMA 发射搬到 warp 1;G1 累加器双缓冲、把上一 chunk 的 out epilogue 挪到本 chunk G1 之后与其重叠;kU 的 8 次 TMEM 读一次发完 |
| v3 | 4340 | 0.62× | E4 1750、E3 820、out store 300、单线程发 8 条 MMA 525;ncu:线程指令数 193 M 比 FlashKDA K2(174 M)还多,No-Eligible 77%,每调度器 1 个 warp | 到此为止,转为消融量化"tcgen05 链的地板"(去掉 E1/E3/E4 的线程工作) |

方法上的两条经验:(1) 每一版先 dump 中间量对 torch 参考,再看时间——v1 的 3 个 bug 里有 2 个是"和参考一致地错";
(2) 看 SASS 直方图比猜快:generic LD/ST 和标量 F2F 各占了一半以上的线程侧时间,分相计时只能告诉你"哪相慢",指令直方图才告诉你"为什么"。
