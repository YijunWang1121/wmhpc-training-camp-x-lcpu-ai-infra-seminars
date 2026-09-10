#!/bin/bash
source /home/lcpu/00737767/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/claude/exp/env.sh
export PYTHONPATH=$CL/exp:$PYTHONPATH
echo "===== E7 k2_tc2 ====="
K2TC=k2_tc2 $PY exp/e7_k2tc.py 2>&1 | grep -v "Warning\|USDT" | tee logs/e7_k2tc2_${SLURM_JOB_ID}.txt
echo "===== ncu k2_tc2 vs recurrence (T=8192 H=96) ====="
cat > /tmp/claude-20007/ncu_target.py <<'PY'
import sys; sys.argv=[sys.argv[0]]
import e7_k2tc as e
p = e.setup(8192, 96); e.run_ref(p); e.run_tc(p)
import torch; torch.cuda.synchronize()
PY
ncu --kernel-name-base function -k "regex:k2_tc2_kernel|_flash_kda_fwd_recurrence" --clock-control none -c 2 \
    --section SpeedOfLight --section WarpStateStats --section SchedulerStats --section Occupancy --section MemoryWorkloadAnalysis \
    --metrics sm__inst_executed_pipe_tensor.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,smsp__inst_executed.sum,sm__cycles_elapsed.avg,gpu__time_duration.sum \
    -f -o profiles/ncu_k2tc2 $PY /tmp/claude-20007/ncu_target.py 2>&1 | grep -v "Warning\|USDT\|==PROF" | tail -3
ncu -i profiles/ncu_k2tc2.ncu-rep --page details 2>/dev/null > profiles/ncu_k2tc2.txt; ncu -i profiles/ncu_k2tc2.ncu-rep --page raw --csv 2>/dev/null > profiles/ncu_k2tc2.csv
grep -E "k2_tc2_kernel|_flash_kda_fwd_recurrence|Duration|SM Frequency|Compute \(SM\)|Memory Throughput|L1/TEX|No Eligible|Active Warps Per|Eligible Warps|Warp Cycles Per Issued|Achieved Occupancy|Issue Slots" profiles/ncu_k2tc2.txt | cut -c1-110
echo DONE
