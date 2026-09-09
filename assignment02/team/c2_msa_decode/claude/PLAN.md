# C2 MSA decode — Claude 独立工作计划(分支 `c2-claude`)

目标:按 TASK.md 三层(测量 → 分析 → 挑战)独立完成,每一步给出文档/实验证据。
独立于用户 `../work/` 的版本工作;完成后再对该版本做点评(见 REVIEW_OF_USER_VERSION.md)。

约束:单卡 srun/sbatch,单次 ≤ 1h(`~/bin/gpu` 默认 30min)。所有数字来自本机 GPU 实测。

## 目录
- `PLAN.md`     本文件:计划 + 状态勾选
- `LOG.md`      按时间的实验日志(每个 GPU session:目的 / 命令 / 原始发现 / 结论)
- `exp/`        实验脚本(可复现)
- `logs/`       每次 GPU 运行的原始 stdout
- `profiles/`   nsys / ncu 原始文件与导出表
- `REPORT.md`   最终报告(讨论点 1–6 逐条「结论 + 证据」+ 挑战答案)
- `ACCEPTANCE.md` 验收方案(讨论点 6,先贴出来)

## 阶段
### P0 读材料 + 环境 — [ ]
- [x] harness 四文件、sparse_attn.py 全文、cutlass 路径、dispatch、fp8 test
- [ ] GPU 型号 / SM 数 / 带宽 / triton 版本确认(session 1)
- [ ] `run.py check` 通过(基线正确)

### P1 测量(任务 1 + 讨论点 5)— 测完再设计 — [ ]
- [ ] E1 batch ∈ {1,4,8,16,32,64} 端到端 us/call(eager)+ CUDA graph 下的纯 GPU 时间
- [ ] E2 nsys:decode kernel vs merge kernel 各自时长 + 两 kernel 之间空隙(launch gap)
- [ ] E3 ncu:decode kernel 在 b=1/4/8/16 的 SOL(DRAM%、SM%、tensor pipe%)、achieved occupancy、
      warp stall 原因(long scoreboard vs 其他)、grid 大小 vs SM 数
- [ ] E4 roofline 定位:实测 bytes/时间 vs 峰值带宽;实测 FLOP/时间 vs 峰值 → 判定瓶颈类别
- [ ] E5 grid 依赖:固定 batch=1,改 NUM_TOPK_CHUNKS(1/2/4/8/16)看时间 — 分辨 "并行度不足" vs "延迟链"

### P2 分析(任务 2 + 讨论点 1–4,6)— [ ]
- [ ] D1 arithmetic intensity 手算 + 与 4.2/4.5 的联动 + ncu tensor pipe 佐证
- [ ] D2 融合可行性:cluster/DSMEM/mbarrier 文档考证 + 成本估算
- [ ] D3 TMA 两级间接寻址:PTX ISA 文档考证(tensormap 坐标语义)+ 小实验验证
- [ ] D4 FP8 scale 放哪层:上游 test 口径 + 数值分析
- [ ] D6 验收方案先贴(ACCEPTANCE.md)

### P3 挑战 — [ ]
- [ ] 基于 P1 数据决定 (a)/(b);不论选哪个都给 crossover=16 解释(含 CUTLASS 路径 M 维/plan 结构分析)
- [ ] 若 (a):自写 kernel(CUDA 或 Triton 改良),过 ACCEPTANCE,与基线对比 b∈{1,4,8,16}

### P4 交付 — [ ]
- [ ] REPORT.md、答辩要点、复现命令
- [ ] REVIEW_OF_USER_VERSION.md:对 `../work/` 的点评
