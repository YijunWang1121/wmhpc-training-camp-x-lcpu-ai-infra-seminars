#!/bin/bash
cd "${SLURM_SUBMIT_DIR}"; PROJ=$(cd ../../.. && pwd); PY="uv run --no-sync --project $PROJ python"
nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv
for l in lib_abl.so lib_abldabl_no_load.so lib_abldabl_no_mma.so lib_abldabl_no_ldsm.so lib_abldabl_no_exp.so lib_abldabl_no_mmadabl_no_ldsm.so lib_abldabl_no_mmadabl_no_ldsmdabl_no_expdabl_no_load.so; do
  MSA_LIB=$l $PY exp/e13_ablate.py 2>&1 | grep -v "make_block_ptr\|warn("
done
echo E13_DONE
