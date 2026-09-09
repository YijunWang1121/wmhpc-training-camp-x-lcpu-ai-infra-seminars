#!/bin/bash
# E3: nsys kernel timeline (gaps) + ncu full sections for the decode kernel at b in {1,4,8,16}
# sbatch -G 1 --time=00:40:00 exp/job_e3.sh
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJ=$(cd ../../.. && pwd)
PY="uv run --no-sync --project $PROJ python"
set -x
# --- nsys: timeline of 20 steps per batch; export kernel trace as CSV ---
for b in 1 4 8 16; do
  nsys profile -c cudaProfilerApi -t cuda --cuda-graph-trace=node -f true -o profiles/nsys_b$b $PY exp/e3_target.py $b 20
  nsys stats -r cuda_gpu_trace -f csv -o profiles/nsys_b${b} profiles/nsys_b$b.nsys-rep >/dev/null
  nsys stats -r cuda_gpu_kern_sum -f csv -o profiles/nsys_b${b} profiles/nsys_b$b.nsys-rep >/dev/null
done
# --- ncu: decode kernel + merge kernel, full section set, 1 launch each ---
for b in 1 4 8 16; do
  ncu --set full --import-source no -k regex:_gqa_sparse_decode_kernel -c 1 -s 2 \
      -f -o profiles/ncu_dec_b$b $PY exp/e3_target.py $b 3
  ncu --set full -k regex:_merge_topk_attn_out_kernel -c 1 -s 2 \
      -f -o profiles/ncu_mrg_b$b $PY exp/e3_target.py $b 3
done
# chunks variants at b=1 to see per-CTA behaviour with more blocks per CTA
for ch in 1 4; do
  ncu --set full -k regex:_gqa_sparse_decode_kernel -c 1 -s 2 \
      -f -o profiles/ncu_dec_b1_ch$ch $PY exp/e3_target.py 1 3 $ch
done
# text exports
for f in profiles/ncu_*.ncu-rep; do ncu -i $f --page details --print-units base > ${f%.ncu-rep}.txt; done
for f in profiles/ncu_*.ncu-rep; do ncu -i $f --page raw --csv > ${f%.ncu-rep}_raw.csv; done
echo E3_DONE
