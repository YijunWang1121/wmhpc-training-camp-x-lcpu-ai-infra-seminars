#!/bin/bash
source /home/lcpu/00737767/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/claude/exp/env.sh
export PYTHONPATH=$CL/exp:$PYTHONPATH
echo "===== E6b microbench SW128 validation ====="
./exp/mb_mma 2>&1 | sed -n '/T7/,$p' | tee logs/mb_mma_sw128_${SLURM_JOB_ID}.txt
for V in k2_tc2; do
  echo "===== E7 $V ====="
  K2TC=$V $PY exp/e7_k2tc.py 2>&1 | grep -v "Warning\|USDT" | tee logs/e7_${V}_${SLURM_JOB_ID}.txt
done
echo DONE
