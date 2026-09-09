#!/bin/bash
# P1 measurement + profiling. Run inside: ~/bin/gpu  (srun -G1 --pty bash), then
#   cd .../team/c2_msa_decode/work && bash p1_run.sh
set -e
cd "$(dirname "$0")"
ROOT=../../..
PY="uv run --no-sync --project $ROOT python"
OUT=profiles
mkdir -p $OUT

echo "==================== 1. correctness + sweep ===================="
$PY p1_measure.py all 2>&1 | tee $OUT/measure.txt

echo "==================== 2. nsys timeline (batch 1,4,8,16) ===================="
for N in 1 4 8 16; do
  nsys profile --force-overwrite true -o $OUT/nsys_b$N \
    --trace=cuda --sample=none --cpuctxsw=none \
    $PY p1_target.py $N 20 >/dev/null 2>&1
  nsys stats --report cuda_gpu_kern_sum --format csv $OUT/nsys_b$N.nsys-rep \
    2>/dev/null | tee $OUT/nsys_b${N}_kern.csv
done

echo "==================== 3. ncu decode+merge (batch 1,4,8,16) ===================="
SECTIONS="-s SpeedOfLight -s Occupancy -s LaunchStats -s MemoryWorkloadAnalysis -s WarpStateStats -s ComputeWorkloadAnalysis -s SchedulerStats"
for N in 1 4 8 16; do
  ncu --target-processes all -f -o $OUT/ncu_b$N \
    -k "regex:_gqa_sparse_decode_kernel|_merge_topk_attn_out_kernel" \
    -c 4 $SECTIONS \
    $PY p1_target.py $N 3 >/dev/null 2>&1 || true
  ncu -i $OUT/ncu_b$N.ncu-rep --page raw --csv 2>/dev/null > $OUT/ncu_b${N}.csv || true
  ncu -i $OUT/ncu_b$N.ncu-rep 2>/dev/null > $OUT/ncu_b${N}.txt || true
  echo "  wrote $OUT/ncu_b$N.*"
done
echo "DONE"
