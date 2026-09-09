#!/bin/bash
set -e
cd "$(dirname "$0")"
ROOT=../../..
PY="uv run --no-sync --project $ROOT python"
OUT=profiles
mkdir -p $OUT

echo "==================== 1. correctness ===================="
$PY p3_test.py check 2>&1 | tee $OUT/p3_check.txt

echo "==================== 2. tuning sweep ===================="
$PY p3_tune.py 2>&1 | tee $OUT/p3_tune.txt

echo "==================== 3. nsys fused vs baseline (b1,b4,b8,b16) ===================="
cat > p3_target.py <<'EOF'
import sys, torch
from lib import make_case
from v1_fused import fused_sparse_decode
n=int(sys.argv[1]); w=int(sys.argv[2]); s=int(sys.argv[3])
case=make_case(num_reqs=n, seq_range=(8192,8192), seed=0)
out=torch.empty_like(case["q"])
for _ in range(10):
    fused_sparse_decode(case["q"],case["kv_cache"],case["topk_idx"],case["block_table"],
        case["seq_lens"],case["num_kv_heads"],case["sm_scale"],out,case["decode_query_len"],
        num_warps=w,num_stages=s)
torch.cuda.synchronize()
for _ in range(30):
    fused_sparse_decode(case["q"],case["kv_cache"],case["topk_idx"],case["block_table"],
        case["seq_lens"],case["num_kv_heads"],case["sm_scale"],out,case["decode_query_len"],
        num_warps=w,num_stages=s)
torch.cuda.synchronize()
EOF
for N in 1 4 8 16; do
  nsys profile --force-overwrite true -o $OUT/nsys_fused_b$N --trace=cuda \
    $PY p3_target.py $N 4 3 >/dev/null 2>&1
  nsys stats --force-export=true --report cuda_gpu_kern_sum $OUT/nsys_fused_b$N.nsys-rep 2>/dev/null \
    | grep -E "decode_kernel|merge|_fused" | tee -a $OUT/p3_nsys.txt
done

echo "==================== 4. ncu fused kernel (b1,b16) ===================="
SEC="--section SpeedOfLight --section Occupancy --section LaunchStats --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis"
for N in 1 16; do
  ncu --target-processes all -f -o $OUT/ncu_fused_b$N --clock-control none \
    -k "regex:_fused_sparse_decode_kernel" -c 3 $SEC \
    $PY p3_target.py $N 4 3 >/dev/null 2>&1 || true
  ncu -i $OUT/ncu_fused_b$N.ncu-rep 2>/dev/null > $OUT/ncu_fused_b${N}.txt || true
  echo "  ncu_fused_b$N.txt: $(wc -l < $OUT/ncu_fused_b${N}.txt 2>/dev/null || echo 0) lines"
done
echo DONE
