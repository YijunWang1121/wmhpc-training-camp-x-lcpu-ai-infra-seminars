"""ctypes wrapper for exp/libmsa_decode.so (fused cluster/TMA decode kernel)."""
import ctypes, os
import torch
_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libmsa_decode.so"))
_lib.msa_decode_launch.restype = ctypes.c_int
_lib.msa_decode_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_float, ctypes.c_long, ctypes.c_long, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]


def fused_decode(case, out=None, cluster=4, stages=2):
    q, kv, t, bt, sl = case["q"], case["kv_cache"], case["topk_idx"], case["block_table"], case["seq_lens"]
    assert q.dtype == torch.bfloat16 and kv.dtype == torch.bfloat16, "bf16 only"
    assert q.shape[2] == 128 and kv.shape[2] == 128 and kv.shape[3] == 256 and t.shape[-1] == 16, "shape unsupported"
    assert q.shape[1] // case["num_kv_heads"] == 16, "gqa group must be 16"
    assert kv.is_contiguous() and t.is_contiguous() and bt.is_contiguous() and q.stride(2) == 1
    if out is None:
        out = torch.empty_like(q)
    total_q = q.shape[0]
    rc = _lib.msa_decode_launch(q.data_ptr(), kv.data_ptr(), out.data_ptr(), t.data_ptr(), bt.data_ptr(), bt.stride(0),
                                sl.data_ptr(), total_q, case["num_kv_heads"], kv.shape[0], case["decode_query_len"],
                                case["sm_scale"], q.stride(0), q.stride(1), cluster, stages,
                                torch.cuda.current_stream().cuda_stream)
    if rc != 0:
        raise RuntimeError(f"msa_decode_launch failed rc={rc}")
    return out
