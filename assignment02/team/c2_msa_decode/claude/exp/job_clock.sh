#!/bin/bash
cd "${SLURM_SUBMIT_DIR}"
nvidia-smi -q -d CLOCK | grep -A3 "Applications Clocks\|Max Clocks\|SM  \|Default Applications" | head -30
nvidia-smi -q -d PERFORMANCE | grep -i "throttle\|idle\|cap\|slowdown\|Performance State" | head -20
./exp/clock_probe
echo CLOCK_DONE
