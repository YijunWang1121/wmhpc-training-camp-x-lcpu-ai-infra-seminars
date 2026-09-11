"""问题 5.1:per-tensor scale 与 outlier。

构造一个张量:一万个元素均匀分布在 [-1, 1],外加一个 3000 的
outlier。按 per-tensor 方式量化到 E4M3(scale = amax / 448,cast 用
torch.float8_e4m3fn),反量化后测逐点相对误差,填题面的表并回答三问。

需要动手的是下面两个 TODO;跑法:
    uv run python kernels/quant_outlier.py
输出直接用于报告,没有自动判测。
"""

import torch

E4M3_MAX = 448.0


def build_tensor(n: int = 10000, outlier: float = 3000.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    x = torch.rand(n, generator=g) * 2 - 1
    return torch.cat([x, torch.tensor([outlier])])


def quant_dequant_per_tensor(x: torch.Tensor) -> torch.Tensor:
    """per-tensor E4M3 量化再反量化。

    scale = amax / 448;除 scale 后 cast 到 torch.float8_e4m3fn;
    cast 回 float 再乘 scale。
    """
    scale = x.abs().max() / E4M3_MAX
    xq = (x / scale).to(torch.float8_e4m3fn)
    return xq.float() * scale


def rel_err_at(x: torch.Tensor, y: torch.Tensor, value: float) -> float:
    """取 x 中最接近 value 的元素,返回该点的相对误差。"""
    idx = torch.argmin((x - value).abs())
    xi, yi = x[idx].item(), y[idx].item()
    return abs(yi - xi) / abs(xi)


def zero_threshold(scale: float) -> float:
    """二分搜出量化后变成 0 的正数阈值(相对给定 scale)。"""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        xq = torch.tensor([mid / scale], dtype=torch.float32).to(
            torch.float8_e4m3fn)
        if xq.item() == 0.0:
            lo = mid
        else:
            hi = mid
    return hi


def quant_dequant_per_block(x: torch.Tensor, block: int = 128) -> torch.Tensor:
    """1x128 per-block E4M3 量化再反量化,最后一块不足 128 单独成块。"""
    out = torch.empty_like(x)
    n = x.numel()
    for s in range(0, n, block):
        e = min(s + block, n)
        chunk = x[s:e]
        out[s:e] = quant_dequant_per_tensor(chunk)
    return out


def main() -> None:
    x = build_tensor()
    y = quant_dequant_per_tensor(x)
    scale = (x.abs().max() / E4M3_MAX).item()
    print(f"含 outlier: scale = amax/448 = {x.abs().max().item():.1f}/448 "
          f"= {scale:.6f}")
    for v in (0.5, 0.1, 0.01, 0.005, 3000.0):
        print(f"  x≈{v:<8} rel_err={rel_err_at(x, y, v):.3e}")

    # (a) 去掉 outlier 重新量化,对比 0.5 处的误差
    x_no_outlier = x[:-1]
    y_no_outlier = quant_dequant_per_tensor(x_no_outlier)
    scale_no_outlier = (x_no_outlier.abs().max() / E4M3_MAX).item()
    print(f"\n(a) 去掉 outlier: scale = {scale_no_outlier:.6f}"
          f"(是含 outlier 时的 {scale_no_outlier / scale:.1f} 倍)")
    print(f"  x≈0.5 rel_err(无 outlier)="
          f"{rel_err_at(x_no_outlier, y_no_outlier, 0.5):.3e}"
          f"  vs 含 outlier 时 {rel_err_at(x, y, 0.5):.3e}")

    # (b) 找出被量化成 0 的阈值,写出它与 scale 的关系式
    thr = zero_threshold(scale)
    thr_no_outlier = zero_threshold(scale_no_outlier)
    print(f"\n(b) 含 outlier 时量化到 0 的阈值 ≈ {thr:.6e}"
          f"(= scale × {thr / scale:.4f})")
    print(f"    去掉 outlier 时阈值 ≈ {thr_no_outlier:.6e}"
          f"(= scale × {thr_no_outlier / scale_no_outlier:.4f})")

    # (c) 换 1x128 的 per-block scale,对比含/不含 outlier 的 block
    yb = quant_dequant_per_block(x, block=128)
    outlier_pos = x.numel() - 1
    outlier_block = outlier_pos // 128
    s0, e0 = outlier_block * 128, min(outlier_block * 128 + 128, x.numel())
    other_block = 0 if outlier_block != 0 else 1
    s1, e1 = other_block * 128, other_block * 128 + 128
    print("\n(c) 1x128 per-block:")
    print(f"  outlier 所在 block[{outlier_block}] "
          f"scale={x[s0:e0].abs().max().item() / E4M3_MAX:.6f}  "
          f"x≈0.5 rel_err="
          f"{rel_err_at(x[s0:e0], yb[s0:e0], 0.5):.3e}")
    print(f"  不含 outlier 的 block[{other_block}] "
          f"scale={x[s1:e1].abs().max().item() / E4M3_MAX:.6f}  "
          f"x≈0.5 rel_err="
          f"{rel_err_at(x[s1:e1], yb[s1:e1], 0.5):.3e}")


if __name__ == "__main__":
    main()
