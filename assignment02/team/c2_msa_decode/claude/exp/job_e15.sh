#!/bin/bash
cd "${SLURM_SUBMIT_DIR}"; PROJ=$(cd ../../.. && pwd); PY="uv run --no-sync --project $PROJ python"
echo "== split (padded scratch) =="; $PY exp/e14_split.py 2>&1 | grep -v "make_block_ptr\|warn("
echo "== fused check+bench (padded scratch) =="; $PY exp/e6_fused.py all 2>&1 | grep -v "make_block_ptr\|warn(" | grep -v "PASS$"
echo E15_DONE
