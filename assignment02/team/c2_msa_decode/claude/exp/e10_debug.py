import sys, torch
from common import *
from fused import fused_decode
cl, st = int(sys.argv[1]), int(sys.argv[2])
case = make_case(num_reqs=1, seq_range=(1024, 8192), seed=0)
ref = sdpa_ref(case)
out = fused_decode(case, cluster=cl, stages=st); torch.cuda.synchronize()
print(f"cl={cl} st={st} err={err_ratio(out, ref):.2e}")
