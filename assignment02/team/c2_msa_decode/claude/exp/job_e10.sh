#!/bin/bash
cd "${SLURM_SUBMIT_DIR}"; PROJ=$(cd ../../.. && pwd); PY="uv run --no-sync --project $PROJ python"
for cfg in "1 3" "1 2" "2 3" "2 2" "4 3" "4 2" "4 1"; do set -- $cfg
  echo "--- cl=$1 st=$2"; CUDA_LAUNCH_BLOCKING=1 $PY exp/e10_debug.py $1 $2 2>&1 | grep -v "make_block_ptr\|warn(" | tail -2
done
echo "--- sanitizer on cl=1 st=2"; compute-sanitizer --tool memcheck --kernel-name-exclude regex:merge $PY exp/e10_debug.py 1 2 2>&1 | grep -v "make_block_ptr\|warn(" | grep -A6 "=========" | head -40
echo "--- sanitizer on cl=4 st=1"; compute-sanitizer --tool memcheck $PY exp/e10_debug.py 4 1 2>&1 | grep -v "make_block_ptr\|warn(" | grep -A6 "=========" | head -40
echo "== fp8 (non-pow2 scales) =="; $PY exp/e8_fp8scale.py 2>&1 | grep -v "make_block_ptr\|warn("
echo E10_DONE
