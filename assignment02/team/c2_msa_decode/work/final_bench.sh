#!/bin/bash
# One consolidated run for the report's headline numbers (single time-slice).
set -e
cd "$(dirname "$0")"
PY="uv run --no-sync --project ../../.. python"
OUT=profiles; mkdir -p $OUT
{
  echo "# ===== C2 final consolidated run ====="
  date
  echo
  echo "## correctness (fused v1: 11 shapes x 5 seeds, vs SDPA <2e-2, vs baseline <5e-3)"
  $PY p3_test.py check
  echo
  echo "## CUDA-graph e2e sweep: baseline vs fused-v1(4w,3s)"
  $PY - <<'EOF'
import torch
from lib import make_case, sa
from v1_fused import fused_sparse_decode
from p1_graph import graph_time
print(f"{'batch':>6} {'base_e2e_us':>12} {'fused_e2e_us':>13} {'speedup':>8}")
for n in (1,2,4,8,16,32,64):
    c = make_case(num_reqs=n, seq_range=(8192,8192), seed=0)
    o = torch.empty_like(c["q"])
    b = graph_time([lambda: sa.minimax_m3_sparse_attn_decode(
        c["q"],c["kv_cache"],c["topk_idx"],c["block_table"],c["seq_lens"],
        c["num_kv_heads"],c["sm_scale"],o,c["decode_query_len"])])
    f = graph_time([lambda: fused_sparse_decode(
        c["q"],c["kv_cache"],c["topk_idx"],c["block_table"],c["seq_lens"],
        c["num_kv_heads"],c["sm_scale"],o,c["decode_query_len"],num_warps=4,num_stages=3)])
    print(f"{n:>6} {b:>12.2f} {f:>13.2f} {b/f:>7.2f}x")
EOF
  echo
  echo "## nsys pure-GPU kernel time (baseline decode + merge), eager"
  for N in 1 4 8 16; do
    nsys profile --force-overwrite true -o $OUT/fin_b$N --trace=cuda \
      $PY p1_target.py $N 20 >/dev/null 2>&1
    echo "-- batch $N --"
    nsys stats --force-export=true --report cuda_gpu_kern_sum $OUT/fin_b$N.nsys-rep 2>/dev/null \
      | grep -E "decode_kernel|merge"
  done
} 2>&1 | tee $OUT/FINAL.txt
echo DONE
