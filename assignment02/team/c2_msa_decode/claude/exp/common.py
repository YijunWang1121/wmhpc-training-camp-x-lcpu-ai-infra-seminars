"""Shared helpers: load harness + vendored Triton baseline, timing (eager / CUDA graph),
and a re-implementation of the wrapper that lets us force NUM_TOPK_CHUNKS."""
import os, sys, math, time
HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, "..", "..", "harness")
sys.path.insert(0, HARNESS)
import torch
import vllm_shim  # noqa  (must precede sparse_attn import)
from synth import make_case  # noqa
from ref_sdpa import sdpa_ref  # noqa
sa = vllm_shim.load_sparse_attn()


def device_info():
    p = torch.cuda.get_device_properties(0)
    d = dict(name=p.name, sms=p.multi_processor_count, cc=f"{p.major}.{p.minor}",
             mem_GB=round(p.total_memory / 2**30, 1),
             l2_MB=round(getattr(p, "L2_cache_size", 0) / 2**20, 1),
             torch=torch.__version__, triton=__import__("triton").__version__)
    return d


def run_decode(case, out=None):
    if out is None:
        out = torch.empty_like(case["q"])
    sa.minimax_m3_sparse_attn_decode(
        case["q"], case["kv_cache"], case["topk_idx"], case["block_table"],
        case["seq_lens"], case["num_kv_heads"], case["sm_scale"], out,
        case["decode_query_len"])
    return out


def default_chunks(total_q, num_kv_heads=4, max_topk=16):
    target = max(1, min(max_topk, 256 // max(1, total_q * num_kv_heads)))
    return 1 << (target.bit_length() - 1)


class SplitDecode:
    """Same two kernels as the upstream wrapper, but chunks is a parameter and the
    partial buffers are preallocated so each kernel can be timed on its own."""

    def __init__(self, case, chunks=None, use_fp8=False, k_scale=None, v_scale=None):
        q = case["q"]
        self.case = case
        total_q, num_heads, head_dim = q.shape
        self.total_q, self.num_heads, self.head_dim = total_q, num_heads, head_dim
        self.nkv = case["num_kv_heads"]
        self.gqa = num_heads // self.nkv
        self.max_topk = case["topk_idx"].shape[-1]
        self.chunks = chunks or default_chunks(total_q, self.nkv, self.max_topk)
        self.o_partial = torch.empty(self.chunks, total_q, num_heads, head_dim,
                                     dtype=q.dtype, device=q.device)
        self.lse_partial = torch.empty(self.chunks, total_q, num_heads,
                                       dtype=torch.float32, device=q.device)
        self.out = torch.empty_like(q)
        kv = case["kv_cache"]
        self.use_fp8 = kv.dtype in sa._FP8_DTYPES
        if self.use_fp8:
            (self.k_scale_arg, self.v_scale_arg, self.s_ks_h, self.s_ks_t,
             self.s_vs_h, self.s_vs_t, self.scale_mode) = sa._kv_scale_args(
                 self.out, self.nkv, k_scale, v_scale)
        else:
            self.k_scale_arg = self.v_scale_arg = self.out
            self.s_ks_h = self.s_ks_t = self.s_vs_h = self.s_vs_t = 0
            self.scale_mode = 0

    def decode(self):
        c = self.case
        q, kv, t, bt = c["q"], c["kv_cache"], c["topk_idx"], c["block_table"]
        grid = (self.total_q * self.chunks, self.nkv)
        sa._gqa_sparse_decode_kernel[grid](
            q, kv, self.k_scale_arg, self.v_scale_arg, t, self.o_partial,
            self.lse_partial, bt, c["seq_lens"], self.total_q, self.gqa,
            self.head_dim, self.max_topk, c["sm_scale"], c["decode_query_len"],
            q.stride(0), q.stride(1), q.stride(2),
            kv.stride(0), kv.stride(1), kv.stride(2), kv.stride(3),
            self.s_ks_h, self.s_ks_t, self.s_vs_h, self.s_vs_t,
            t.stride(0), t.stride(1), t.stride(2),
            *self.o_partial.stride(), *self.lse_partial.stride(), bt.stride(0),
            BLOCK_SIZE_K=128, NUM_TOPK_CHUNKS=self.chunks, USE_FP8=self.use_fp8,
            KV_SCALE_MODE=self.scale_mode, USE_PDL=False)

    def merge(self):
        sa._merge_topk_attn_out_kernel[(self.total_q, self.num_heads)](
            self.o_partial, self.lse_partial, self.out, self.head_dim,
            *self.o_partial.stride(), *self.lse_partial.stride(),
            *self.out.stride(), NUM_TOPK_CHUNKS=self.chunks, USE_PDL=False)

    def __call__(self):
        self.decode()
        self.merge()
        return self.out


def time_eager(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # us


def time_graph(fn, iters=200, warmup=20, n_in_graph=20):
    """Capture n_in_graph calls in one CUDA graph; return us per call. Removes
    Python/launch overhead so only device time (incl. inter-kernel gaps) remains."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            for _ in range(n_in_graph):
                fn()
    torch.cuda.synchronize()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / (iters * n_in_graph) * 1e3


def err_ratio(x, ref):
    x, ref = x.double().flatten(), ref.double().flatten()
    return ((x - ref).norm() / ref.norm().clamp_min(1e-30)).item()


def kv_bytes(case):
    """Bytes of K/V actually touched by one decode step (unique blocks per kv head)."""
    c = case
    b = c["seq_lens"].shape[0]
    return b * c["num_kv_heads"] * c["topk"] * 128 * 2 * c["head_dim"] * c["kv_cache"].element_size()
