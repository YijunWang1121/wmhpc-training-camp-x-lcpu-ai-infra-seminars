#!/bin/bash
cd "${SLURM_SUBMIT_DIR}"; PROJ=$(cd ../../.. && pwd); PY="uv run --no-sync --project $PROJ python"
for cfg in "1 1 3" "1 4 2" "1 16 1" "16 1 3"; do set -- $cfg
  nsys profile -c cudaProfilerApi -t cuda -f true -o profiles/nsys_fused_b$1_cl$2_s$3 $PY exp/e7_diag.py $1 $2 $3 20 2>&1 | grep "graph us"
  nsys stats -r cuda_gpu_trace -f csv -o profiles/nsys_fused_b$1_cl$2_s$3 profiles/nsys_fused_b$1_cl$2_s$3.nsys-rep >/dev/null
  python3 - profiles/nsys_fused_b$1_cl$2_s$3_cuda_gpu_trace.csv <<'PY'
import csv,sys,statistics as st
rows=[r for r in csv.DictReader(open(sys.argv[1])) if 'decode' in r['Name']]
d=[int(r['Duration (ns)'])/1e3 for r in rows]; s=[int(r['Start (ns)']) for r in rows]
gaps=[(s[i+1]-s[i]-int(rows[i]['Duration (ns)']))/1e3 for i in range(len(rows)-1)]
print(f"  {sys.argv[1].split('/')[-1]}: n={len(rows)} kernel median {st.median(d):.2f} us (min {min(d):.2f}) gap median {st.median(gaps):.2f} us grid={rows[0]['GrdX']} smem={rows[0]['DymSMem (MB)']}MB reg={rows[0]['Reg/Trd']}")
PY
done
ncu --set full -k regex:decode_kernel -c 1 -s 3 -f -o profiles/ncu_fused_b1_cl1s3 $PY exp/e7_diag.py 1 1 3 5 >/dev/null 2>&1
ncu -i profiles/ncu_fused_b1_cl1s3.ncu-rep --page details --print-units base > profiles/ncu_fused_b1_cl1s3.txt
ncu -i profiles/ncu_fused_b1_cl1s3.ncu-rep --page raw --csv > profiles/ncu_fused_b1_cl1s3_raw.csv
grep -E "^\s+(Duration|Elapsed Cycles|SM Active Cycles|DRAM Throughput|Achieved Occupancy|Grid Size|Registers Per Thread|Dynamic Shared Memory Per Block|No Eligible|Warp Cycles Per Issued)" profiles/ncu_fused_b1_cl1s3.txt | sed 's/  */ /g'
echo E7_DONE
