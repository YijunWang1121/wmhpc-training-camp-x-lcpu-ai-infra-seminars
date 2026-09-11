# C1 工作日志(按时间)

2026-09-10

- 08:5x 读 TASK / deep-dive / K1 K2 源码 / bench 脚本;clone 上游 1ce47ea + cutlass 5c149f5(`~/flashkda-build`),
  与本目录快照 diff 为空;装 fla 0.5.2、flash_kda(`FLASH_KDA_CUDA_ARCHS=100a,103a`)进 `.venv`。
- job 24793 probe:B300 SXM6 AC,cc 10.3,148 SM。SASS 统计(`profiles/sm103a_hist.txt`):K2 每实例 104 条
  `HMMA.16816.F32.BF16`,K1 32 条 bf16 + 12 条 `HMMA.16816.F16`;没有 HGMMA/UTC*MMA。
- job 24796(E1/E3,SM 时钟全程 1095 MHz):E3 对拍 PASS;官方 bench H=96 fixed 1.755 ms(GB200 表 1.009),
  加速 2.24×(表 2.31×)。分 kernel:K2 1285 us,K1 458 us;**K2 与 H 无关**(H=96/64/12 都 ≈1280 us)。
- job 24809:脚本 bug(env.sh 语法错、$PY 空),只跑出 microbench;但发现 tcgen05 K=128 的 k-step 偏移写错
  (每步应前进 2×LBO),M=64 的 TMEM lane 映射不同;MN-major 描述符假设 (b) 正确。
- job 24819(E6/E5/E2/E4,**SM 时钟 2032 MHz**,不是 1095!):所有 us 数字与 job 1 不可直接比,改用 cycle。
  发现:tcgen05 单条 K=16 指令 ~46-64 cyc(N≤64 时按指令数计费,N=16 只有 653 MAC/cyc,N=128 才 4088);
  issue→commit→wait 固定往返 ~265 cyc;tcgen05.ld 128×16 fp32 ~139 cyc,128×128 ~446 cyc。
  消融:K2 每 chunk ≈ 2777 cyc = P1 598 + P6 758 + P3/4 337 + 流水线/同步 836 + 其余 ~250。
  ncu K2:SM 20.5%,DRAM 18.7%,L1/smem 61%,No-Eligible 68%,每调度器 1.5 warp,占用率 9.4%。
- 写 tcgen05 版 K2(`exp/k2tc/k2_tc.cu`,消费原 workspace),纸面预估 ≈ 2900 cyc/chunk(不比 mma.sync 快),
  job 24844 实测中。
- job 24844(1095 MHz):消融重跑,cycle 数与 24819 一致(stock 2760)。
- job 24904/24909/24920/24922:tcgen05 K2 v1 —— NaN;分段 dump 定位到 workspace 是六个分离数组(不是每 tile 连续)。
- job 24933/24934:microbench 验证 SW128 描述符(MN-major:LBO = MN 块跨度 2048,SBO = 1024);v2 chunk 0 全对、chunk 1 错 → stage 基址未 1024 对齐。
- job 24936:v2 逐位正确,6250 cyc/chunk(0.44×)。job 24942:beta bulk + LDS/STS 修正,仍 6250;SASS 发现 512 条标量 F2F。
- job 24944:v3(bf16x2 打包转换、TMA 发射移 warp 1、双缓冲累加器 + out epilogue 重叠)4460 cyc/chunk(0.62×),逐位正确;ncu 对比。
- job 24951:tcgen05 K2 消融:−E4 = 2680(1.02×),−E1−E3−E4 = 2030(1.34×,异步链地板)。写 REPORT §7/§8、PROFILING §8。

2026-09-11

- 追问(源自 assignment02 M4 的 CUTLASS 对照:手写 tcgen05 GEMM 只有 cuBLAS 29-35%,CUTLASS collective
  builder 开箱即用 ≈95%):K2 换成 CUTLASS 而不是手写 PTX 会不会翻盘?不跑 GPU,直接读本机
  `/tmp/cutlass`(4.7.0)的 `cute/arch/mma_sm100_umma.hpp` 源码验证:全部 62 处 `tcgen05.mma` 内联汇编
  无一例外被 `elect_one_sync()` 门控(50 处调用一一对应)——单线程发射是 ISA 约束,CUTLASS 也躲不开;
  `examples/` 里搜 recurrent/scan/ssm/mamba/linear_attn 零命中,CUTLASS 的 mainloop 抽象是给"K 维可
  结合归约"设计的,K2 的 chunk 间递推(第 N+1 个 chunk 的 G1 依赖第 N 个 chunk 更新完的状态)是真数据
  依赖,没有 CUTLASS 深流水线能利用的独立工作项。结论写进 REPORT §7.1:不推翻讨论点 6,补一条论据——
  K2 的瓶颈是 ISA 约束+算法递推,不是代码工程质量,换库救不了(M4 的 GEMM 差距是工程质量差距,K2 不是)。
- 追问(用户直接指定):§3.3 说"链本身有没有办法拆,不只是绕开"——读 `fla_kda_ref/naive.py` 第 160-163
  行的 `naive_chunk_kda` 状态更新,把 `v_i = u_i - w_i@S` 代入 `S' = diag(exp(g_i[-1]))@S + K_i'^T@v_i`
  化简,发现是标准仿射矩阵递推 `S_i = M_i@S_{i-1} + N_i`,`M_i`(K×K,对角+秩≤BT=16)、`N_i`(K×V)只用
  本 chunk 自己的量就能算,不需要 `S`——可以结合律重组成 Blelloch 前缀扫描,深度 O(NT)→O(log NT)=18。
  纸面代价:扫描要多付 ≈16× 总算力(≈8.6 GFLOP vs ≈537 MFLOP/(seq,head)),深度收益上限 ≈20× 只在
  "空闲 SM≫并发 head 数"(TP8 长序列)的场景成立;查了 `/tmp/flashkda-b300-venv` 里 fla 0.5.2 自己的
  生产 Triton kernel(`fla/ops/common/chunk_delta_h.py:166`,`chunk_kda_fwd` 实际调的就是它)确认状态
  递推也是纯 `for i_t in range(NT)` 串行循环,没人在生产里做过扫描版本。写进 REPORT §3.3.1,是没有实现
  /上机验证过的纸面推导,下一步该做的是先 microbench "多 CTA 矩阵 combine + grid sync" 往返延迟。
- E8(用户指定"实现并验证"§3.3.1;按用户要求全程 `srun` 占卡跑,不用 sbatch):`exp/e8_scan.py` 把递推改成
  两级分块扫描(build_MN → level-1 组内 batched compose → level-2 组间传播 → level-3 batched apply → 输出)。
  CPU fp64 对拍:扫描 vs 自己的串行 M/N 版逐位一致(rel 0~2e-17),vs naive(内部 fp32)2e-7。
  GPU(job 26190,`logs/e8_scan_srun.txt`,flash_kda K2 = 1288 us 与 1095 MHz 的历史数字一致):扫描版对
  flash_kda 的 o/S 误差 5.5e-3/4.4e-3(与 §2.4 flash_kda 对 fp64 的误差同量级,正确);但计时全线落败:
  T=8192 H=12 bf16+CUDA graph 4566 us = K2 的 3.5×,H=96 32969 us = 25×,T=32768 H=12 17195 us = 3.4×;
  graph ≈ eager,说明不是 47 次 launch 的问题,是 128^3 batched GEMM 本身效率极低。`exp/e8_gemm_ceiling.py`
  单独测 cuBLAS bmm 128^3 的吞吐上限做归因(`logs/e8_gemm_ceiling_srun.txt`)。
- E8b(job 26196/26199,`logs/e8_gemm_ceiling_srun.txt`、`logs/e8_bench2_srun.txt`,采样时钟全程 1095 MHz):
  同会话测 cuBLAS bmm 128³ 天花板(bf16:batch 192 → 72 TFLOP/s,batch 768/1536 → 230/226,batch 12288 → 295;
  对照 4096³ 单条 1706),扫描 v2(连续 batch + baddbmm + 预分配)只比 v1 快 5%,有效吞吐 29-35 TFLOP/s;
  按天花板计费的下界 H=12 1062 us(≈K2 0.82×,仅 compose)。结论:扫描路线在 K=V=128 下不可能赢,写进
  REPORT §3.3.1 实测段、速览表、§8 第 4 条。挑战按 TASK "做不出正收益也算完成"口径关闭。
- E9(用户 brief:算法层拆依赖链优先于 Blackwell 特化;Design B 分层扫描):`exp/e9_hier.py`,零 CUDA 改动——
  用 FlashKDA K2 的 varlen + 每序列 initial_state,把长序列切成 NG 组:phase 1 探测组摘要 (B_g = 从 0 跑的末状态,
  A_g = v=0、S_in=I 跑的末状态),phase 2 组间 NG 步前缀(bf16 baddbmm,graph),phase 3 并行回放。
  job 26226/26232/26239(1095 MHz):正确性在弱衰减(lb=-0.1/-0.01)下与基线差 bf16 量级,对 fp64 误差与基线相同;
  性能 v1(fp32 phase 2)1.42×/1.99×,v3(bf16 phase 2 + graph + 1a/1b 合并成 2H-head 调用)B·H=12:1.92×(T=8K)、
  2.50×(T=32K);B·H=16/T=64K 2.14×;B·H=32 1.17×;B·H=64/96 0.5~0.64×。ncu(job 26240,`profiles/ncu_e9`):
  同一 kernel,grid 12→192,SM Active 8%→76%,单 SM 指标不变。交付文档 `K2_HIER.md`;REPORT §3.3.2、§0、§8 更新。
  CuTe 地板 kernel `exp/k2cute/k2_cute_floor.cu` 已写未编译,按 brief 顺序搁置到算法层收益确认之后。
- E10(brief 第 9 步,分层扫描之后的 SM100 问题):`exp/e10_short_chain.py`,job 26271,1095 MHz。tcgen05 K2 vs
  mma.sync K2 在短链多 CTA regime(T=512,H=148/296/592/1184):比值 0.66→0.73→0.75→0.75,两者都在 2 CTA/SM 饱和
  (smem 98 KB vs smem 84 KB + TMEM 256 列),整卡 58 vs 81 chunks/us。结论:SM100 指令级特性在新 regime 仍无正收益,
  杠杆是 smem 减负提高共驻数。写进 K2_HIER.md §11、REPORT §3.3.2。期间 /home/lcpu 共享盘写满(3.1T 100%),清了
  自己的 ~/.cache/{uv,pip,vscode-cpptools} 才能落盘;E10 日志先写 /tmp 再拷回。
- E11(用户要求"系统多轮测试,不上 SM100 特性"):`exp/e11_matrix.py`,job 26278,2032 MHz。48 形状(B∈{1,2,4}×
  H∈{12,16,32,64}×T∈{4K..128K}),5 轮交替,std≤1.5%。B·H=12 几何均值 2.16×(4K 1.48× → 128K 2.66×),B·H=16 1.86×,
  B·H=24/32 1.20×/1.07×(短序列持平、长序列略赢),B·H≥48 输(0.33~0.79×,门控后 1.0)。crossover B·H≈24~32。
  最后一格 B=4/H=64/128K OOM(脚本显存保护估计偏松),其余 47 格有效。写进 K2_HIER.md §12。
- §7.2(用户问"sm100 为什么慢、能不能改"):用 `profiles/ncu_k2tc2.csv` 的 stall 构成 + 寄存器/指令数 + §7 分相,
  加一次源码级采样(`exp/e12_ncu_target.py`,`profiles/ncu_k2tc2_src.ncu-rep`;/tmp 是节点本地盘,目标脚本和输出必须
  放共享目录才能在计算节点跑)。结论:tcgen05 版慢在线程侧指令反而 +11%、242 寄存器、1 warp/调度器;SASS 采样 45%
  在 E4 的 FFMA/F2FP/LDS/STS/地址计算,30% 在 mbarrier 轮询;可改项(状态常驻 TMEM + 256 线程、INV 直接 INTER、
  E1 精简)加起来 4460 → ≈2550,对 mma.sync 2732 只是打平(1.07×),因为 2030 的异步链地板占 mma.sync 的 74%。
- E13(用户澄清:问的是分层扫描 workload 上 tcgen05 行不行):`exp/mb_scan.cu`,稠密 compose 链步
  S_new^T=[S^T|u^T]@[M_i^T|K'_i](M128 N128 K144),A 常驻 TMEM(TS 形式),fp32→bf16 TMEM 重打包,TMEM 200 列。
  一次编译通过、正确性 PASS(TMEM A 打包:lane=行,每列两个相邻 k)。1054 cycle/步 @1 CTA/SM,705/步 @2 CTA/SM,
  比 K2 每 chunk 快 2.6~3.0×。投影分层流水线 710 → ~325 us(对原版 ~4.2×)。写进 K2_HIER §13、REPORT §7.2/§8。
- 收口(用户:以有收益为最终结论重整全部材料 + 加强实验 + 讨论点 + 精度):E13b `mb_scan` mode 2(B 每步从 HBM
  cp.async.bulk 双缓冲流式载入)1460/919 cycle/步,仍 1.9×/2.3× 快于 K2 chunk,带宽 4.1~6.5 TB/s;E14 精度随 T
  (1K~16K,lb=-0.01)分层版对 fp64 误差与原版逐项相同、不增长,分段调用无损失,计时与 lb 无关(1.98×/1.91×)。
  REPORT.md 重写为"收益条件表优先"的结构:§0 速览+收益表,§3 主结果(诊断/推导/E11 收益条件/精度/Nsight),
  §4 SM100 在新 workload 上的收益(E13/E13b),§5 TASK 六个讨论点逐条,§6 负结果证据链(E7/§7.1/§7.2/E8/E10),
  §7 建议,§9 复现;旧版在 git 历史(11eb86a)。
- E15(用户:"不搞树,树还是有依赖"):在 `exp/e9_hier.py` 加衰减窗口回看(lookback W=1/2/3):判据 max_dim exp(Σ_组 g) < 2⁻⁹
  时 S_in[g] = B_{g−1}(W=1),不算 A_g、无 phase 2,2 次 K2 调用,零跨组依赖;W≥2 为窗口内 batched GEMM。
  job(1095 MHz,`logs/e15_lookback_srun.txt`):lb=−5/−1 逐位一致;lb=−0.1 与树版同;lb=−0.01、G=8(衰减 0.58)W=1 错
  0.18,误差按 0.58^W 收敛(W=2 0.057,W=3 0.019),G=32 时 W=2 即同树版。性能 B·H=12 8K 3.16×、32K 4.14×;16/64K 3.51×;
  32/16K 1.80×。W≥2 切片索引改过一次("size 14 vs 13")。写进 REPORT §3.6、K2_HIER §15。
- E16:`exp/e11_matrix.py 3 lb1`(加 MODE 参数),job 1095 MHz(`logs/e16_clock.txt`),47 形状 × 3 轮,std ≤ 0.8%。
  W=1 几何均值:B·H=12 3.63×(4K 2.33 → 128K 4.58),16 3.08×,24 2.12×,32 1.79×,48 1.23×,64 1.03×,128 0.63×,256 0.50×。
  crossover 24~32 → ≈64。写进 REPORT §0 收益表(树版/W=1 两列)、§3.3 末尾、K2_HIER §15。
- E17(用户再问"精度对不对",针对 W=1):`exp/e9_hier.py precision_lb`,job 26318,1095 MHz。原版/树版/W=1/W=2 各自对 fp64
  朴素递推,lb∈{−5,−1,−0.1,−0.01}×G∈{32,64}×T∈{1K,4K,8K,16K}(24 格)+ 分段调用 + lb=−5 大形状逐位比对。判据成立
  的 18 格 W=1 与原版误差逐格相同;lb=−5/−1 逐位一致;lb=−0.01/G=32 W=1 out 误差 +9~13%,W=2 恢复;G=64 时判据保守。
  写进 REPORT §3.6 末尾、§0、§9,K2_HIER §15。
