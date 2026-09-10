#!/bin/bash
source /home/lcpu/00737767/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/claude/exp/env.sh
echo "===== E3 correctness ====="
$PY exp/e3_correct.py 2>&1 | grep -v Warning
echo "===== E1a official benchmark H=96 ====="
$PY $C1/FlashKDA/benchmarks/bench_fwd.py --H 96 2>&1 | grep -v Warning
echo "===== E1a official benchmark H=64 ====="
$PY $C1/FlashKDA/benchmarks/bench_fwd.py --H 64 2>&1 | grep -v Warning
echo "===== E1b per-kernel breakdown ====="
$PY exp/e1_kernels.py 2>&1 | grep -v Warning
echo "DONE"
