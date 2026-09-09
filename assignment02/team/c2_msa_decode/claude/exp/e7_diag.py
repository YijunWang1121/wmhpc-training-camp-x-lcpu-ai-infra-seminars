"""E7: where do the fused kernel's ~20us go? nsys-able target + occupancy query."""
import sys, torch
from common import *
from fused import fused_decode
b = int(sys.argv[1]); cl = int(sys.argv[2]); st = int(sys.argv[3]); n = int(sys.argv[4]) if len(sys.argv) > 4 else 20
case = make_case(num_reqs=b, seq_range=(8192, 8192), seed=0)
out = torch.empty_like(case["q"])
for _ in range(5): fused_decode(case, out=out, cluster=cl, stages=st)
torch.cuda.synchronize()
print("graph us:", time_graph(lambda: fused_decode(case, out=out, cluster=cl, stages=st)))
torch.cuda.cudart().cudaProfilerStart()
for _ in range(n): fused_decode(case, out=out, cluster=cl, stages=st)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
