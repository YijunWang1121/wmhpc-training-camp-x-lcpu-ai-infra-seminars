#!/bin/bash
# P1 round 2: CUDA-graph timing + ncu debug + nsys gpu-metrics timeline.
set -e
cd "$(dirname "$0")"
ROOT=../../..
PY="uv run --no-sync --project $ROOT python"
OUT=profiles
mkdir -p $OUT

echo "==================== ncu diagnostics ===================="
ncu --version 2>&1 | head -3
echo "--- minimal ncu run (b1, decode only, 1 pass) ---"
ncu --target-processes all -f -o $OUT/ncu_probe \
  -k "regex:_gqa_sparse_decode_kernel" -c 1 -s SpeedOfLight \
  $PY p1_target.py 1 2 2>&1 | tail -30
ls -la $OUT/ncu_probe.ncu-rep 2>&1 || echo "NO NCU REPORT"

echo "==================== CUDA-graph sweep ===================="
$PY p1_graph.py 2>&1 | tee $OUT/graph.txt

echo "==================== nsys gpu-metrics timeline (b1,b16) ===================="
for N in 1 16; do
  nsys profile --force-overwrite true -o $OUT/nsysm_b$N \
    --trace=cuda --gpu-metrics-devices=all --gpu-metrics-frequency=50000 \
    $PY p1_target.py $N 40 >/dev/null 2>&1 || echo "gpu-metrics failed b$N"
done

echo "==================== ncu full (if probe worked) ===================="
if [ -f $OUT/ncu_probe.ncu-rep ]; then
  SEC="-s SpeedOfLight -s Occupancy -s LaunchStats -s MemoryWorkloadAnalysis -s WarpStateStats -s SchedulerStats -s ComputeWorkloadAnalysis"
  for N in 1 4 8 16; do
    ncu --target-processes all -f -o $OUT/ncu_b$N \
      -k "regex:_gqa_sparse_decode_kernel|_merge_topk_attn_out_kernel" \
      -c 4 $SEC $PY p1_target.py $N 3 >/dev/null 2>&1 || true
    ncu -i $OUT/ncu_b$N.ncu-rep --csv --page raw 2>/dev/null > $OUT/ncu_b${N}.csv || true
    echo "  ncu_b$N: $(wc -l < $OUT/ncu_b${N}.csv 2>/dev/null || echo 0) csv lines"
  done
fi
echo "DONE"
