#!/bin/bash
source /home/lcpu/00737767/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/claude/exp/env.sh
export PYTHONPATH=$CL/exp:$PYTHONPATH
echo "===== E7 ablation of tcgen05 K2 ====="
$PY exp/e7_ablate.py 2>&1 | grep -v "Warning\|USDT" | tee logs/e7_ablate_${SLURM_JOB_ID}.txt
echo DONE
