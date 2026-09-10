#!/bin/bash
# Job 2: E6 microbench, discussion-point 1/5 numerics, E2 ncu (official template), E4 ablation timing.
source /home/lcpu/00737767/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/claude/exp/env.sh
echo "===== E6 mb_mma (mma.sync vs tcgen05 at K2 shapes) ====="
./exp/mb_mma | tee logs/mb_mma_${SLURM_JOB_ID}.txt
echo "===== E5 chunk range / Neumann / registers (GPU) ====="
$PY exp/e5_chunk_range.py 2>&1 | grep -v Warning | tee logs/e5_chunk_range_${SLURM_JOB_ID}.txt
echo "===== E5 bf16-state precision ====="
$PY exp/e5_precision.py 2>&1 | grep -v Warning | tee logs/e5_precision_${SLURM_JOB_ID}.txt
echo "===== E2 ncu ====="
bash exp/job_e2_ncu.sh 2>&1 | grep -v "^host=\|^NVIDIA"
echo "===== E4 ablation ====="
ls exp/variants/
$PY exp/e4_ablate.py 2>&1 | grep -v "Warning\|USDT" | tee logs/e4_ablate_${SLURM_JOB_ID}.txt
echo DONE
