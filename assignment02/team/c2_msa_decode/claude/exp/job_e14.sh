#!/bin/bash
cd "${SLURM_SUBMIT_DIR}"; PROJ=$(cd ../../.. && pwd); PY="uv run --no-sync --project $PROJ python"
$PY exp/e14_split.py 2>&1 | grep -v "make_block_ptr\|warn("
echo E14_DONE
