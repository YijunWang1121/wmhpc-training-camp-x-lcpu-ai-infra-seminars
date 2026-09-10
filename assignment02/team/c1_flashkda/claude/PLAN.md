# C1 FlashKDA — 工作计划(Claude 版本)

目标:严格按 `../TASK.md` 三层(复现 → 分析 6 个讨论点 → 挑战)交付,每个结论都有可复现的证据
(脚本在 `exp/`,原始日志在 `logs/`,ncu/nsys/SASS 在 `profiles/`)。另写 `PROFILING.md`
记录"怎么 profile、profile 看到什么、据此下一步做什么"的决策链。

约束:GPU 只有一张 B300(cc 10.3,148 SM,SM 时钟固定 1095 MHz),`sbatch -G 1`,每作业 ≤ 1 h,
同一时刻只挂一个作业。编译全部在登录节点做(不占卡)。

## 层 1 复现(E0–E3)
- E0 `exp/job_probe.sh`:设备/时钟/工具链。
- 构建:`~/flashkda-build/FlashKDA`(pin 1ce47ea,cutlass 5c149f5,与本目录快照 diff 为空),
  `FLASH_KDA_CUDA_ARCHS=100a,103a` 装进 `assignment02/.venv`;fla 0.5.2 装进同一 venv
  (`fla/ops/kda` 与快照 a3edffc 仅差 int32/int64 索引 cast,共 33 行)。
- E1 `exp/job_e1.sh`:官方 `benchmarks/bench_fwd.py`(H=96/64,fixed + 两组 varlen)对照
  `BENCHMARK_GB200.md`;`exp/e1_kernels.py` 用 torch.profiler 拆 K1/K2/beta-transpose 各自耗时。
- E2 `exp/job_e2_ncu.sh`:官方 `benchmarks/ncu.sh` 模板(去掉 pip install)+ pipe metrics;
  SASS 用 `cuobjdump -sass` 按 kernel 统计 HMMA/HGMMA/UTC*MMA(`profiles/sm103a_hist.txt`)。
- E3 `exp/e3_correct.py`:对拍 `fla_kda_ref/naive.py`(fp64)与 fla Triton `chunk_kda`。

## 层 2 分析(讨论点 1–6)
1. CHUNK 三理由量化:范围(纸面 + `exp/e5_chunk_range.py` 数值扫描 CHUNK=16/32/64 的
   2^cumsum 下溢/上溢比例)、Neumann 代价(MMA 条数与 fp16 误差实验)、MMA 形状/寄存器。
2. tcgen05 最小 tile vs CHUNK=16:纸面(orientation 交换后 5 个 GEMM 全部合法)+ `exp/mb_mma.cu`
   microbench(mma.sync vs tcgen05 在 K2 五个形状上的 latency/throughput)。
3. 并行度候选:V-split(fla 已如此,BV=32/64)、多 head/CTA、persistent、2-CTA/cluster multicast;
   用 E1 的 varlen-vs-fixed 数据和 TP8(H=12)数据互相找反例。
4. bound 判定:E2 的 SOL/pipe/stall 指标 + 手算 AI;对照 4.5 的 `in_proj_qkvgfab`。
   K2 归因用消融构建(`-DTMA_DISABLE_ALL`,再逐相删减)。
5. bf16 状态精度验证方案 + 数据:`exp/e5_precision.py`(T 扫描、decay 强度扫描、多次调用状态
   传递、对 fp64 递推 / fla fp32 状态)。
6. v2 sm100a 结论:综合 2/4 的数据。

## 层 3 挑战:"只换指令不动算法"
把 K2 的 5 个 mma.sync GEMM 换成 tcgen05(D 作为 M 维,CHUNK=16 作为 N/K 维),workspace 格式、
K1、算法、CHUNK 全部不动;先做 microbench(层 2 第 2 点),再做 K2 替换 kernel,正确性对
`fla_kda_ref`,性能对 FlashKDA K2。做不出正收益就把原因量化成讨论点 6 的论据。
