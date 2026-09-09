#!/bin/bash
cd "${SLURM_SUBMIT_DIR}"
PROJ=$(cd ../../.. && pwd)
echo "== TMA streaming experiment =="
for b in 1 4 16; do for n in 1 4 16; do ./exp/tma_indirect $b $n; done; done
echo "== fused kernel =="
uv run --no-sync --project "$PROJ" python exp/e6_fused.py all
echo E6_DONE
