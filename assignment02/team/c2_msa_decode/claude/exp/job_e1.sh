#!/bin/bash
# sbatch -G 1 --time=00:20:00 exp/job_e1.sh   (run from claude/ dir)
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJ=$(cd ../../.. && pwd)
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.max.mem,memory.total --format=csv
uv run --no-sync --project "$PROJ" python exp/e1_baseline.py logs/e1_rows.json
