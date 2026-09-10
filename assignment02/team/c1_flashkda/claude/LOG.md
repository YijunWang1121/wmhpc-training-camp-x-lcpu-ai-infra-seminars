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
