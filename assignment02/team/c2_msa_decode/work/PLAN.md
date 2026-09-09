# C2 MSA decode — 工作计划 & 状态

目标:系统性完成 `TASK.md` 三层任务(测量 → 分析 → 挑战),每步 solid 证据。
**当前状态:全部完成。** 交付见下「交付物」。

## 交付物
| 文件 | 内容 |
|---|---|
| `work/REPORT.md` | 主报告:执行摘要 + 讨论点 1–6 逐条「结论+证据」+ §7 crossover + §8 挑战 (b) 证据链 |
| `work/DEFENSE.md` | 答辩要点 + 预期质疑回应 + 数字速查 |
| `work/PROGRESS.md` | 逐步日志(11 个 GPU session 的原始发现、失败尝试、教训) |
| `work/profiles/` | 所有 nsys / ncu 原始输出 + 提取的 CSV/txt |
| `work/v1_fused.py` | P3 (a) 尝试:融合单 kernel(正确但慢 4×,是 (b) 的证据) |
| `work/lib.py` `p1_*.py` `p3_*.py` `exp_*.py` `gpu_metrics.py` | 复现脚本 |

## 复现
```
cd team/c2_msa_decode/work
# 无 GPU:
uv run --no-sync --project ../../.. python exp_crossover.py      # §7 crossover 模型
# GPU(srun -G 1):
srun -G 1 --time=00:15:00 uv run --no-sync --project ../../.. python p1_measure.py   # 瓶颈测量
srun -G 1 --time=00:15:00 uv run --no-sync --project ../../.. python p3_test.py check # 正确性
bash p1_run2.sh        # CUDA graph sweep + gpu-metrics
bash ncu_full.sh       # ncu 全 section 扫 b{1,4,8,16}
python exp_tma.py      # 讨论点 3 TMA PTX 实验
```

## 阶段回顾

### P0 准备 — done
- [x] 读 harness + vllm_msa_ref 全部文件;uv sync(torch 2.14 / triton 3.8 / CUDA 13)

### P1 测量(任务1 + 讨论点5)— done
- [x] `run.py check` 等价物通过(lib.py 拆 kernel 复现,对 SDPA 3.2e-3)
- [x] batch 扫描:eager + CUDA graph;decode vs merge 拆分计时
- [x] nsys:kernel timeline / GPU 占空比 / gpu-metrics(SM/Tensor/DRAM %)
- [x] ncu b{1,4,8,16}:SoL / Occupancy / Scheduler / Memory —— 结构性天花板量化

### P2 分析(任务2 + 讨论点1–4,6)— done
- [x] 讨论点1:AI≈16 手算 + roofline + M=16 双重原因 + ncu 佐证
- [x] 讨论点2:小 batch 应融合不 split;cluster/mbarrier merge 理论可行但性价比低
- [x] 讨论点3:TMA 文档考证 + `exp_tma.py` PTX 实测(运行期 page + TMA 可行)
- [x] 讨论点4:fp8 scale 在 load 后点积前片上 dequant(方案 A)
- [x] 讨论点5:瓶颈 = 延迟暴露 + launch 固定开销(非 compute/BW)
- [x] 讨论点6:验收方案(3 档参照 + 形状矩阵 + 计时口径 + 自曝弱点)—— 已先贴出

### P3 挑战 — done,选 (b)
- [x] (a) 尝试 ×3:融合单 kernel / 保 split retune / 纯 retune —— 全 ≤ 1.1×
- [x] §7:crossover=16 的 `M = reuse·16` 量化解释(`exp_crossover.py`)
- [x] §8:收益上限(roofline + 实测)< 工程成本(CUTLASS 复杂度)证据链
- [x] 结论:小 batch 继续用 Triton split-K 是正确工程决策

### P4 交付 — done
- [x] REPORT.md / DEFENSE.md / PROGRESS.md / 脚本 / profile 原始数据

## GPU session 预算
共 ~11 次 srun(多为 8–30min,3 次 1h 后台)。均单卡、≤1h。明细见 PROGRESS.md。
