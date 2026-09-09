"""ctypes wrapper for exp/libmsa_decode.so (fused cluster/TMA decode kernel)."""
import ctypes, os
import torch
_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("MSA_LIB", "libmsa_decode.so")))
_lib.msa_decode_launch.restype = ctypes.c_int
_lib.msa_decode_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_float, ctypes.c_long, ctypes.c_long, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]


def fused_decode(case, out=None, cluster=4, stages=2, dbg=None):
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
                                torch.cuda.current_stream().cuda_stream, dbg.data_ptr() if dbg is not None else None)
    if rc != 0:
        raise RuntimeError(f"msa_decode_launch failed rc={rc}")
    return out

_lib.msa_decode_split_launch.restype = ctypes.c_int
_lib.msa_decode_split_launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                         ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_long,
                                         ctypes.c_long, ctypes.c_int, ctypes.c_void_p]
_lib.msa_probe_launch.restype = ctypes.c_int
_lib.msa_probe_launch.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                  ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]


class FusedSplit:
    """Our TMA/mma.sync decode CTA with the upstream Triton merge kernel (same two-launch structure as the baseline)."""
    def __init__(self, case, split=16):
        from common import sa
        self.sa, self.case, self.split = sa, case, split
        q = case["q"]; self.total_q, self.nh, self.hd = q.shape
        self.o_part = torch.empty(split, self.total_q, self.nh, self.hd, dtype=q.dtype, device=q.device)
        self.lse_part = torch.empty(split, self.total_q, self.nh, dtype=torch.float32, device=q.device)
        self.out = torch.empty_like(q)

    def decode(self):
        c = self.case; q, kv, t, bt = c["q"], c["kv_cache"], c["topk_idx"], c["block_table"]
        rc = _lib.msa_decode_split_launch(q.data_ptr(), kv.data_ptr(), self.o_part.data_ptr(), self.lse_part.data_ptr(), t.data_ptr(),
                                          bt.data_ptr(), bt.stride(0), c["seq_lens"].data_ptr(), self.total_q, c["num_kv_heads"], kv.shape[0],
                                          c["decode_query_len"], c["sm_scale"], q.stride(0), q.stride(1), self.split,
                                          torch.cuda.current_stream().cuda_stream)
        if rc != 0: raise RuntimeError(f"split launch rc={rc}")

    def merge(self):
        self.sa._merge_topk_attn_out_kernel[(self.total_q, self.nh)](
            self.o_part, self.lse_part, self.out, self.hd, *self.o_part.stride(), *self.lse_part.stride(), *self.out.stride(),
            NUM_TOPK_CHUNKS=self.split, USE_PDL=False)

    def __call__(self):
        self.decode(); self.merge(); return self.out


def probe(case, mode, lse):
    c = case; kv = c["kv_cache"]
    rc = _lib.msa_probe_launch(mode, kv.data_ptr(), lse.data_ptr(), c["topk_idx"].data_ptr(), c["block_table"].data_ptr(),
                               c["block_table"].stride(0), c["seq_lens"].data_ptr(), c["q"].shape[0], c["num_kv_heads"], kv.shape[0],
                               c["decode_query_len"], torch.cuda.current_stream().cuda_stream)
    if rc != 0: raise RuntimeError(f"probe rc={rc}")
