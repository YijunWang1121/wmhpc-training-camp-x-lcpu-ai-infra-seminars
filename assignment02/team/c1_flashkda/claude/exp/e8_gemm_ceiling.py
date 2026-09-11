"""E8b: 扫描版慢的归因——128x128x128 批量 GEMM 在 B300 上到底能跑多快(cuBLAS bmm,同一进程内对照大 GEMM 峰值)。
扫描每次 compose = 2 条 128^3 GEMM;T=8192 时扫描总量 ≈ 3*NT*HV 条(H=12: 18432 条 ≈ 77 GFLOP;H=96: ≈ 620 GFLOP)。
要打平 flash_kda K2(1.29 ms)需要的持续吞吐:H=12 ≈ 60 TFLOP/s,H=96 ≈ 470 TFLOP/s。"""
import torch

def bench(fn, iters=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # us

def run(n, batch, dtype, tf32=False):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    a = torch.randn(batch, n, n, device="cuda", dtype=dtype)
    b = torch.randn(batch, n, n, device="cuda", dtype=dtype)
    us = bench(lambda: a @ b)
    flop = 2.0 * batch * n ** 3
    tag = "tf32" if tf32 else str(dtype).replace("torch.", "")
    print(f"  bmm {n}^3 x{batch:<6d} {tag:8s} {us:9.1f} us  {flop / us / 1e6:8.1f} TFLOP/s")

if __name__ == "__main__":
    print(torch.cuda.get_device_name())
    print("== 128^3 batched(扫描的 compose 形状)==")
    for batch in (192, 384, 1536, 3072, 12288):
        run(128, batch, torch.bfloat16)
    run(128, 1536, torch.float32); run(128, 1536, torch.float32, tf32=True)
    print("== 同样总 FLOP,更大的 tile ==")
    run(256, 384, torch.bfloat16); run(512, 96, torch.bfloat16); run(1024, 12, torch.bfloat16)
    print("== 参照:单条大 GEMM 峰值 ==")
    run(4096, 1, torch.bfloat16); run(8192, 1, torch.bfloat16)
