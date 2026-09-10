# shared by all job scripts (sbatch copies the script, so no $0-relative paths)
if [ -n "$C1_ENV_LOADED" ]; then cd $CL; return 0 2>/dev/null; fi
export C1_ENV_LOADED=1
export ROOT=/home/lcpu/00737767/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02
export C1=$ROOT/team/c1_flashkda
export CL=$C1/claude
export PY=$ROOT/.venv/bin/python
export PYTHONPATH=$C1:$PYTHONPATH   # for `import fla_kda_ref`
cd $CL
echo "host=$(hostname) job=$SLURM_JOB_ID date=$(date -Is) gpu=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,compute_cap,clocks.sm,clocks.max.sm --format=csv,noheader
# background SM-clock sampler -> logs/clock_<job>.csv
( while true; do echo "$(date +%s.%N),$(nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader,nounits)"; sleep 0.5; done ) > $CL/logs/clock_${SLURM_JOB_ID}.csv 2>/dev/null &
CLOCK_PID=$!
trap "kill $CLOCK_PID 2>/dev/null" EXIT
