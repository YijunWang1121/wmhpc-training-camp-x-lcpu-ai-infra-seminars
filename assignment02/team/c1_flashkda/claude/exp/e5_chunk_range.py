"""Discussion point 1, quantified numerically (runs on GPU or CPU):
 (a) numerical range: the kernel computes exp2(cumsum(g2)) and exp2(-cumsum(g2)) with g2 = lb*log2(e)*sigmoid(.)
     in fp32 (ex2.approx.ftz) then rounds to bf16.  For CHUNK=C the cumsum is bounded by C*lb*log2e; fp32/bf16
     normal range is 2^-126 .. 2^128.  We measure, on the *actual* gate distribution (A_log, dt_bias, g ~ N(0,1)
     as in the benchmark, and the worst case gate=lb), the fraction of underflowed k_decayed / overflowed k_inv
     entries and the resulting relative error of the intra-chunk matrix L = k_d @ k_inv^T for C = 16/32/64.
 (b) Neumann series cost & fp16 accuracy: for strictly lower-triangular L (CxC, nilpotent) (I-L)^-1 = prod (I+L^(2^i)),
     i < log2(C).  Count MMAs (m16n8k16 equivalents) and measure fp16 vs fp64 error of the inverse for C=16/32/64
     with L drawn from the real KDA construction (beta * k_d k_inv^T, masked).
 (c) register/shape budget of K2 for larger C: accumulators per warp = 2 * (C x 32) fp32 / 32 lanes.
"""
import math, torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
LOG2E = 1.4426950408889634

def ex2_ftz(x):  # fp32 exp2 with denormals flushed to zero (ex2.approx.ftz.f32 semantics)
    y = torch.exp2(x.float())
    return torch.where(y.abs() < 2.0 ** -126, torch.zeros_like(y), y)

def gate_cumsum(C, lb, n=1024, H=4, D=128, worst=False):
    g = torch.randn(n // C, C, H, D, device=dev)
    A_log = torch.rand(H, device=dev); dt = torch.rand(H, D, device=dev)
    if worst:
        g2 = torch.full_like(g, lb * LOG2E)                       # sigmoid -> 1
    else:
        g2 = lb * LOG2E * torch.sigmoid(torch.exp(A_log).view(1, 1, H, 1) * (g + dt.view(1, 1, H, D)))
    return g2.cumsum(1)                                           # [chunks, C, H, D]

def part_a():
    print("(a) range: fraction of k_decayed=k*exp2(cumsum) flushed to 0 in fp32 / bf16, and k_inv=k*exp2(-cumsum) -> inf")
    print(f"    fp32/bf16 normal range 2^-126..2^128; bound of |cumsum| = C*|lb|*log2e: " +
          ", ".join(f"C={C}:{C*5*LOG2E:.0f}" for C in (16, 32, 64)))
    for worst in (False, True):
        for lb in (-5.0, -3.0, -1.0):
            for C in (16, 32, 64):
                cs = gate_cumsum(C, lb, worst=worst)
                k = torch.nn.functional.normalize(torch.randn_like(cs), dim=-1)
                kd = k * ex2_ftz(cs); ki = k * ex2_ftz(-cs)
                kd_bf = kd.bfloat16().float(); ki_bf = ki.bfloat16().float()
                under = ((kd == 0) & (k != 0)).float().mean().item()
                under_bf = ((kd_bf == 0) & (k != 0)).float().mean().item()
                over = torch.isinf(ki).float().mean().item()
                over_bf = torch.isinf(ki_bf).float().mean().item()
                # intra-chunk matrix in the factorised form vs exact per-element form (fp64)
                cs64 = cs.double(); k64 = k.double()
                L_fact = torch.einsum("njhd,nihd->nhji", kd_bf.double(), ki_bf.double())
                L_ref = torch.einsum("njhd,nihd->nhji", k64 * torch.exp2(cs64), k64 * torch.exp2(-cs64))
                err = ((L_fact - L_ref).flatten().square().mean().sqrt() / (L_ref.flatten().square().mean().sqrt() + 1e-300)).item()
                print(f"    {'worst(gate=lb)' if worst else 'bench dist    '} lb={lb:4.1f} C={C:2d}: k_d underflow fp32 {under:6.2%} bf16 {under_bf:6.2%} | k_inv overflow fp32 {over:6.2%} bf16 {over_bf:6.2%} | L rel err (bf16 factorised vs fp64) {err:.2e}")

def neumann(L, dtype):
    C = L.shape[-1]; I = torch.eye(C, dtype=dtype, device=L.device)
    L = L.to(dtype); inv = I + L; P = L; n_mma = 0
    steps = int(math.log2(C))
    for _ in range(steps - 1):
        P = P @ P; n_mma += 1
        inv = inv @ (I + P); n_mma += 1   # kernel does inv += inv@P (same cost)
    return inv, n_mma

def part_b():
    print("(b) Neumann inverse: MMA count and fp16 accuracy vs fp64 (L from real KDA construction, beta~sigmoid)")
    for C in (16, 32, 64):
        cs = gate_cumsum(C, -5.0, n=2048)
        k = torch.nn.functional.normalize(torch.randn_like(cs), dim=-1).double(); cs = cs.double()
        A = torch.einsum("njhd,nihd->nhji", k * torch.exp2(cs), k * torch.exp2(-cs))   # [n, H, C, C]
        beta = torch.sigmoid(torch.randn(A.shape[0], A.shape[1], C, 1, device=dev, dtype=torch.float64))
        L = torch.tril(A, -1) * beta
        inv64 = torch.linalg.inv(torch.eye(C, device=dev, dtype=torch.float64) - L)
        for dt in (torch.float16, torch.bfloat16, torch.float32):
            inv, n_mma = neumann(L, dt)
            err = ((inv.double() - inv64).flatten().square().mean().sqrt() / inv64.flatten().square().mean().sqrt()).item()
            mx = inv64.abs().max().item()
            print(f"    C={C:2d} {str(dt)[6:]:8s}: rel err {err:.2e} (max |inv| {mx:.2f}), matmuls {n_mma} of {C}^3 -> "
                  f"{n_mma * C**3 / 2048:.0f} m16n8k16-equiv per chunk = {n_mma * C**3 / 2048 / C:.2f} per token "
                  f"(C=16 kernel: 7 x 2 = 14 HMMA per chunk)")

def part_c():
    print("(c) K2 register budget per thread for CHUNK=C (4 warps, each warp owns 32 of 128 state columns):")
    for C in (16, 32, 64):
        acc = 2 * (C * 32) // 32          # u_acc + out_acc fp32 per thread
        ufrag = (C * 32) // 32 // 2       # u bf16 (B-operand) per thread
        state_ring = 2 * (16 * 16) // 32 // 2 * 2
        print(f"    C={C:2d}: u_acc+out_acc {acc:3d} regs, U(bf16) {ufrag:3d}, A frags/state ring ~{16+state_ring}, total ~{acc+ufrag+16+state_ring} (+addressing); limit 255 -> {'ok' if acc+ufrag+40 < 255 else 'SPILLS'}")
        print(f"          per-chunk MMA per warp: P1 {2*2*(C//16)*8*2:3d}  P3/4 {2*2*(C//16)**2*2:3d}  P6 {8*2*(C//16)*2:3d}  -> per token {(2*2*(C//16)*8*2 + 2*2*(C//16)**2*2 + 8*2*(C//16)*2)/C:.1f} HMMA/warp")

if __name__ == "__main__":
    torch.manual_seed(0)
    part_a(); part_b(); part_c()
