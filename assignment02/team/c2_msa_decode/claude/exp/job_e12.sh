#!/bin/bash
cd "${SLURM_SUBMIT_DIR}"; PROJ=$(cd ../../.. && pwd); PY="uv run --no-sync --project $PROJ python"
echo "== fused v3 =="; $PY exp/e6_fused.py all 2>&1 | grep -v "make_block_ptr\|warn("
echo "== triton v2 =="; $PY exp/e11_triton_v2.py 2>&1 | grep -v "make_block_ptr\|warn("
echo E12_DONE
