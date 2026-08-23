"""问题 7.7（压轴）：softmax in TileLang（FROM-SCRATCH）。

contract：
- softmax(x) 接收形状 (M, N) 的 float32 CUDA tensor，返回同形状结果，
  对每一行独立做 softmax；
- kernel 用 TileLang 自己写，一个 block 处理一行（或一小批行）；
- 为了确保数值稳定，要求行内先减最大值，再做 exp 与求和。测试里有一行
  数值巨大的输入，不稳定的实现会得到 inf/nan；
- 行宽 N 任意，可以假设 N <= 4096。TileLang 的 kernel 按形状编译，
  用 make_xxx(M, N) 针对形状生成、在 wrapper 里按形状缓存编译结果
  是常见做法（结构可以参考 7.3、7.4）；
- 归约用 T.reduce_max / T.reduce_sum，逐元素部分用 T.Parallel 加 T.exp；
- fragment 的宽度建议取不小于 N 的 2 的幂（类比 Triton 的
  next_power_of_2），不足的位置补 -inf（T.if_then_else 加 T.infinity），
  否则布局推断可能报 no available layout；
- 通过 pytest tests/test_tilelang_softmax.py 即为完成。

(Optional) 将你的实现和 torch.softmax 比较一下性能（行宽取 256/1024/4096），
Tip: elementwise + 行内归约的 kernel 大概率是带宽瓶颈，可以想想理论上限是多少。
"""

import torch
import tilelang
import tilelang.language as T

def make_softmax(M, N, threads=128, dtype="float32"):
    BLOCK_N = 1 << (N - 1).bit_length()
    @T.prim_func
    def main(
        X: T.Tensor((M, N), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        
        with T.Kernel(M, threads=threads) as by:
            X_local = T.alloc_fragment((N,), dtype)
            max_local = T.alloc_fragment((1,), dtype)
            sum_local = T.alloc_fragment((1,), dtype)
            for j in T.Parallel(N):
                X_local[j]=T.if_then_else(j<N, X[by, j], -T.infinity(dtype))
            T.reduce_max(X_local, max_local)
            for j in T.Parallel(N):
                X_local[j] = T.exp(X_local[j]-max_local[0])
            T.reduce_sum(X_local, sum_local)
            for j in T.Parallel(N):
                Y[by, j] = X_local[j] / sum_local[0]
    return main


def softmax(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    y = torch.zeros_like(x)
    func = make_softmax(M, N)
    kernel = tilelang.compile(func, out_idx=[1])
    return kernel(x)


def benchmark(M, N, warmup=20, repeat=100):
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)

    # 先编译，不能把 TileLang compilation 算进 kernel latency
    func = make_softmax(M, N)
    kernel = tilelang.compile(func, out_idx=[1])

    # warmup
    for _ in range(warmup):
        y = kernel(x)
    torch.cuda.synchronize()

    # TileLang
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeat):
        y = kernel(x)
    end.record()

    torch.cuda.synchronize()

    tilelang_ms = start.elapsed_time(end) / repeat

    # PyTorch warmup
    for _ in range(warmup):
        y_ref = torch.softmax(x, dim=-1)
    torch.cuda.synchronize()

    # PyTorch
    start.record()
    for _ in range(repeat):
        y_ref = torch.softmax(x, dim=-1)
    end.record()

    torch.cuda.synchronize()

    torch_ms = start.elapsed_time(end) / repeat

    print(
        f"M={M:5d}, N={N:4d} | "
        f"TileLang {tilelang_ms:.4f} ms | "
        f"Torch {torch_ms:.4f} ms | "
        f"speedup {torch_ms / tilelang_ms:.2f}x"
    )

for N in [256, 1024, 4096]:
    benchmark(M=4096, N=N)