#!/bin/bash
# Full ncu sweep — decode + merge kernels, batch 1/4/8/16.
set -e
cd "$(dirname "$0")"
ROOT=../../..
PY="uv run --no-sync --project $ROOT python"
OUT=profiles
mkdir -p $OUT
SEC="--section SpeedOfLight --section Occupancy --section LaunchStats \
--section WarpStateStats --section SchedulerStats \
--section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis \
--section SpeedOfLight_RooflineChart"

for N in 1 4 8 16; do
  ncu --target-processes all -f -o $OUT/ncu_b$N --clock-control none \
    -k "regex:_gqa_sparse_decode_kernel|_merge_topk_attn_out_kernel" -c 6 \
    $SEC $PY p1_target.py $N 3 2>&1 | tail -4
  ncu -i $OUT/ncu_b$N.ncu-rep 2>/dev/null > $OUT/ncu_b${N}.txt
  ncu -i $OUT/ncu_b$N.ncu-rep --csv --page raw 2>/dev/null > $OUT/ncu_b${N}_raw.csv
  echo "  -> ncu_b$N.txt ($(wc -l < $OUT/ncu_b${N}.txt) lines)"
done
echo DONE
