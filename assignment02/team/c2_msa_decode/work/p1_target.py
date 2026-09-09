"""Minimal profiling target: run decode+merge for one batch size N times.
    python p1_target.py <batch> [<iters>]
Used under ncu / nsys. Kernel names to filter:
  _gqa_sparse_decode_kernel   _merge_topk_attn_out_kernel
"""
import sys
import torch
from lib import make_case, plan

n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 5

case = make_case(num_reqs=n, seq_range=(8192, 8192), seed=0)
pl = plan(case)
# warmup / JIT compile
for _ in range(10):
    pl["run_decode"](); pl["run_merge"]()
torch.cuda.synchronize()
for _ in range(iters):
    pl["run_decode"](); pl["run_merge"]()
torch.cuda.synchronize()
