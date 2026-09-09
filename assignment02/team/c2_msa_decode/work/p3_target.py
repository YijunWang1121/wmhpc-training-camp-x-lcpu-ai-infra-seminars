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
