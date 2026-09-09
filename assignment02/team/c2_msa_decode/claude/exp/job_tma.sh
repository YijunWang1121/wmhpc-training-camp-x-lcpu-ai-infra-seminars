#!/bin/bash
cd "${SLURM_SUBMIT_DIR}"
for b in 1 4 16; do for n in 1 4 16; do ./exp/tma_indirect $b $n; done; done
echo TMA_DONE
