"""ncu 目标(§7.2 源码级 stall 归因):一次 tcgen05 K2(k2_tc2)调用,T=8192 H=12。"""
import sys; sys.argv=[sys.argv[0]]
import e7_k2tc as e, torch
p = e.setup(8192, 12); e.run_ref(p); e.run_tc(p); torch.cuda.synchronize()
