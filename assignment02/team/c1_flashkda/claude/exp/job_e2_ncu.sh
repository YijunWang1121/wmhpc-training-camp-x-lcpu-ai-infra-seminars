#!/bin/bash
# E2: ncu on K1 (_flash_kda_fwd_prepare) and K2 (_flash_kda_fwd_recurrence), official template
# (benchmarks/ncu.sh) minus the pip install, plus text/csv exports.  --clock-control none as upstream.
source /home/lcpu/00737767/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/claude/exp/env.sh
cd $C1/FlashKDA
for MODE in fixed varlen; do
  OUT=$CL/profiles/ncu_${MODE}
  rm -f $OUT.ncu-rep
  ncu --set full --kernel-name-base function -k "regex:_flash_kda_fwd_(prepare|recurrence)" \
      --clock-control none --import-source yes --source-folders . -c 4 \
      --export $OUT.ncu-rep $PY benchmarks/bench_fwd.py --mode $MODE --warmup 0 --iters 1 --repeats 1 2>&1 | grep -v Warning | tail -5
  ncu -i $OUT.ncu-rep --page details > $OUT.txt 2>/dev/null
  ncu -i $OUT.ncu-rep --page raw --csv > $OUT.csv 2>/dev/null
  echo "wrote $OUT.{txt,csv}  lines=$(wc -l < $OUT.txt)"
done
# extra: instruction-mix + pipe metrics (sanity for 'SM80 MMA main path')
OUT=$CL/profiles/ncu_pipes
ncu --kernel-name-base function -k "regex:_flash_kda_fwd_(prepare|recurrence)" --clock-control none -c 2 \
    --metrics sm__inst_executed_pipe_tensor_op_hmma.sum,sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_tensor.sum,smsp__inst_executed.sum,sm__cycles_active.avg,sm__cycles_elapsed.avg,gpu__time_duration.sum,dram__bytes.sum,lts__t_bytes.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,sm__warps_active.avg.pct_of_peak_sustained_active,launch__grid_size,launch__occupancy_limit_shared_mem,launch__occupancy_limit_registers,launch__registers_per_thread,launch__shared_mem_per_block_dynamic \
    $PY benchmarks/bench_fwd.py --mode fixed --warmup 0 --iters 1 --repeats 1 2>&1 | grep -v Warning > $OUT.txt
tail -60 $OUT.txt
echo DONE
