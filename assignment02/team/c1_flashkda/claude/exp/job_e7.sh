#!/bin/bash
# Job 3: tcgen05 K2 (challenge) correctness + timing, plus ablation re-run with clock readout.
source /home/lcpu/00737767/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/claude/exp/env.sh
export PYTHONPATH=$CL/exp:$PYTHONPATH
echo "===== E7 tcgen05 K2 ====="
$PY exp/e7_k2tc.py 2>&1 | grep -v "Warning\|USDT" | tee logs/e7_${SLURM_JOB_ID}.txt
echo "===== E4 ablation (with clock) ====="
$PY exp/e4_ablate.py stock notma notma_noP1 notma_noP6 notma_noP34 noP6 2>&1 | grep -v "Warning\|USDT" | tee logs/e4_ablate_${SLURM_JOB_ID}.txt
echo DONE
