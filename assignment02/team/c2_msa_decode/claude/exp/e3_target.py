"""Profiling target: run N decode steps for batch B (eager), used under nsys / ncu.
usage: python e3_target.py B [N] [chunks]"""
import sys
import torch
from common import *
b = int(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
chunks = int(sys.argv[3]) if len(sys.argv) > 3 else None
case = make_case(num_reqs=b, seq_range=(8192, 8192), seed=0)
sd = SplitDecode(case, chunks=chunks)
for _ in range(5):
    sd()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
for _ in range(n):
    sd()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print("ok", b, sd.chunks)
