#!/bin/bash
cd "${SLURM_SUBMIT_DIR}"; PROJ=$(cd ../../.. && pwd); PY="uv run --no-sync --project $PROJ python"
echo "== fused v2 =="; $PY exp/e6_fused.py all
echo "== fp8 scale placement (E8) =="; $PY exp/e8_fp8scale.py
echo E9_DONE
