#!/bin/bash
cd "$(dirname "$0")"
ROOT=../../..
PY="uv run --no-sync --project $ROOT python"
OUT=profiles
set -x
ncu --list-sets 2>&1 | head -20
# 1: bare launch metrics, no sections
ncu --target-processes all -f -o $OUT/ncu_p1 -c 1 --set launch \
  -k "regex:_gqa_sparse_decode_kernel" $PY p1_target.py 1 2 2>&1 | tail -25
ls -la $OUT/ncu_p1.ncu-rep 2>&1
# 2: try --section long form
ncu --target-processes all -f -o $OUT/ncu_p2 -c 1 \
  --section SpeedOfLight --section Occupancy \
  -k "regex:_gqa_sparse_decode_kernel" $PY p1_target.py 1 2 2>&1 | tail -25
ls -la $OUT/ncu_p2.ncu-rep 2>&1
