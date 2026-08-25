// 问题 1.4:把 1.3 的手工装载换成 ldmatrix,两条路径共存、都要 PASS。
//
// 把你 1.3 的 kernel 拆成两个装载函数移植进来:
//   load_manual:1.3 的逐 byte 手工装载(公式来自 1.1)
//   load_ldsm:  用 ldmatrix 装载。A 是 fp8——ldmatrix 的元素是 16 个
//               原始 bit,不关心类型;1.1 附加问的打包方向在这里起作用。
//               变体(.x1/.x2/.x4、是否 .trans)自己从 PTX 文档选。
// 数据已在 smem(main 里先从 global 拷入),两条路径都从 smem 装载。
//
// 都 PASS 之后:make ptx/m1_sm80/04_ldmatrix 或 nvdisasm 反汇编,
// 数两条路径 smem->fragment 段的指令构成(装载条数、地址算术条数),
// 报告里回答:ldmatrix 消掉的是哪部分工作?为什么手工路径绕不开它?
//
//在 m16n8k32 FP8 MMA 中，A fragment 需要 4 个 32-bit 寄存器，B fragment 需要 2 个。手工路径的 smem→fragment 阶段生成了 16+8=24 条 ld.shared.u8，并需要额外的 shift/mul/or 指令将 byte 数据拼装为 MMA 所需的 32-bit register fragments；ldmatrix 路径则分别用一条 ldmatrix.m8n8.x4 和一条 ldmatrix.m8n8.x2 完成 A、B 的 warp-level cooperative load 和 fragment 排布。因而 ldmatrix 主要省去了大量标量 shared-memory load 以及显式的 fragment packing/rearrangement。手工路径无法避免这些工作，因为 mma.sync 的输入必须预先按照规定的 per-lane fragment layout 放入寄存器，而普通 ld.shared 不负责完成 shared-memory layout 到 MMA register fragment layout 的转换。
// 运行:make run/m1_sm80/04_ldmatrix(内部两条路径各跑多 seed)
#include <cuda_fp8.h>
#include <cstdlib>
#include <random>
#include "../common.h"

// smem 布局:sA 按 [16][32] 行主序;B 备了两种布局——sBk 按 [32][8]
// (k-major,1.3 用的就是它),sBn 按 [8][32](n-major,每个 n 的 32
// 个 k 字节连续)。手工路径用哪种都行;ldmatrix 的每个"行地址"要求
// 16 byte 连续,B 的 fragment 需要 k 方向相邻的字节成对进 b16——
// 想清楚哪种布局能满足它。
//
// TODO: 实现两个装载函数。
__device__ __forceinline__

uint32_t smem_u32(const void* ptr) {

    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

}

__host__ __device__
static int a_row_of(int lane, int i) { 
    int groupID = lane >> 2;
    if ((0 <= i && i < 4) || (8 <= i && i < 12)){
        return groupID;
    }
    else{
        return groupID + 8;
    }
}
__host__ __device__
static int a_col_of(int lane, int i) { 
    int threadID_in_group = lane % 4;
    if (i<8){
        return (threadID_in_group * 4) + (i & 0x3);
    }
    else{
        return (threadID_in_group * 4) + (i & 0x3) + 16;
    }
}
__host__ __device__
static int b_row_of(int lane, int i) { 
    int threadID_in_group = lane % 4;
    if (i<4){
        return (threadID_in_group * 4) + (i & 0x3)  ;
    }
    return (threadID_in_group * 4) + (i & 0x3) + 16;
    
}  // k
__host__ __device__
static int b_col_of(int lane, int i) { 
    int groupID = lane >> 2;
    return groupID;
}  // n


__device__ void load_manual(
    const uint8_t* sA,
    const uint8_t* sBk,
    const uint8_t* sBn,
    unsigned (&a)[4],
    unsigned (&b)[2])
{
    int lane = threadIdx.x & 31;
    #pragma unroll
    for (int r = 0; r < 4; ++r)
        a[r] = 0;
    #pragma unroll
    for (int r = 0; r < 2; ++r)
        b[r] = 0;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int row = a_row_of(lane, i);
        int col = a_col_of(lane, i);
        uint8_t x = sA[row * 32 + col];
        int r = i >> 2;
        int j = i & 3;
        a[r] |= uint32_t(x) << (8 * j);
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int k = b_row_of(lane, i);
        int n = b_col_of(lane, i);
        uint8_t x = sBk[k * 8 + n];
        int r = i >> 2;
        int j = i & 3;
        b[r] |= uint32_t(x) << (8 * j);
    }
}
__device__ void load_ldsm(
    const uint8_t* sA,
    const uint8_t* sBk,
    const uint8_t* sBn,
    unsigned (&a)[4],
    unsigned (&b)[2])
{
    (void)sBk;

    int lane = threadIdx.x & 31;

    // ============================================================
    // A: [16][32] FP8
    //
    // 2 FP8 = 1 b16
    // [16][32] FP8 -> [16][16] b16
    //
    //     b16 col
    //        0..7       8..15
    //      +----------+----------+
    // 0..7 | matrix 0 | matrix 2 |
    //      +----------+----------+
    // 8..15| matrix 1 | matrix 3 |
    //      +----------+----------+
    //
    // x4:
    // lane  0..7  -> matrix 0 rows
    // lane  8..15 -> matrix 1 rows
    // lane 16..23 -> matrix 2 rows
    // lane 24..31 -> matrix 3 rows
    // ============================================================

    int rowA = lane & 15;
    int colA = (lane >> 4) * 16;
    uint32_t addrA =
        smem_u32(&sA[rowA * 32 + colA]);

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0,%1,%2,%3}, [%4];\n"
        : "=r"(a[0]),
          "=r"(a[1]),
          "=r"(a[2]),
          "=r"(a[3])
        : "r"(addrA)
    );


    // ============================================================
    // B: sBn = [8][32] FP8
    //
    // 2 FP8 = 1 b16
    // [8][32] FP8 -> [8][16] b16
    //
    //        0..7       8..15
    //      +----------+----------+
    // n    | matrix 0 | matrix 1 |
    // 0..7 |          |          |
    //      +----------+----------+
    //
    // x2:
    // lane 0..7  -> matrix 0 rows
    // lane 8..15 -> matrix 1 rows
    //
    // .trans -> B 的 MMA col fragment
    // ============================================================

    int rowB = lane & 7;
    int colB = (lane >> 3) * 16;

    uint32_t addrB = smem_u32(&sBn[rowB * 32 + colB]);

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
        "{%0,%1}, [%2];\n"
        : "=r"(b[0]),
          "=r"(b[1])
        : "r"(addrB)
    );
}

template <bool USE_LDSM>
__global__ void mma_kernel(const uint8_t* A, const uint8_t* B, float* D) {
    __shared__ uint8_t sA[16 * 32], sBk[32 * 8], sBn[8 * 32];
    for (int i = threadIdx.x; i < 16 * 32; i += 32) sA[i] = A[i];
    for (int i = threadIdx.x; i < 32 * 8; i += 32) {
        sBk[i] = B[i];
        sBn[(i & 7) * 32 + (i >> 3)] = B[i];  // 转成 n-major
    }
    __syncwarp();
    unsigned a[4], b[2];
    if constexpr (USE_LDSM)
        load_ldsm(sA, sBk, sBn, a, b);
    else
        load_manual(sA, sBk, sBn, a, b);
    float c[4] = {0, 0, 0, 0}, d[4];
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    int group = threadIdx.x >> 2, tig = threadIdx.x & 3;
    D[group * 8 + tig * 2] = d[0];
    D[group * 8 + tig * 2 + 1] = d[1];
    D[(group + 8) * 8 + tig * 2] = d[2];
    D[(group + 8) * 8 + tig * 2 + 1] = d[3];
}

static int run_path(bool ldsm, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 15);
    uint8_t hA[16 * 32], hB[32 * 8];
    float fA[16 * 32], fB[32 * 8], ref[16 * 8] = {};
    for (int i = 0; i < 16 * 32; i++) {
        __nv_fp8_e4m3 v = __nv_fp8_e4m3((float)(dist(rng) - 8));
        hA[i] = *(uint8_t*)&v;
        fA[i] = float(v);
    }
    for (int i = 0; i < 32 * 8; i++) {
        __nv_fp8_e4m3 v = __nv_fp8_e4m3((float)(dist(rng) - 8));
        hB[i] = *(uint8_t*)&v;
        fB[i] = float(v);
    }
    for (int r = 0; r < 16; r++)
        for (int n = 0; n < 8; n++)
            for (int k = 0; k < 32; k++)
                ref[r * 8 + n] += fA[r * 32 + k] * fB[k * 8 + n];
    uint8_t *dA, *dB;
    float* dD;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
    CUDA_CHECK(cudaMalloc(&dD, 16 * 8 * 4));
    CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
    if (ldsm)
        mma_kernel<true><<<1, 32>>>(dA, dB, dD);
    else
        mma_kernel<false><<<1, 32>>>(dA, dB, dD);
    CUDA_CHECK_KERNEL();
    float got[16 * 8];
    CUDA_CHECK(cudaMemcpy(got, dD, sizeof(got), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < 16 * 8; i++) bad += got[i] != ref[i];
    cudaFree(dA); cudaFree(dB); cudaFree(dD);
    return bad;
}

int main() {
    long total = 0;
    for (unsigned s : {1u, 7u, 42u}) {
        int bm = run_path(false, s), bl = run_path(true, s);
        printf("seed=%-6u manual %s(%d)  ldsm %s(%d)\n", s,
               bm ? "FAIL" : "PASS", bm, bl ? "FAIL" : "PASS", bl);
        total += bm + bl;
    }
    return total != 0;
}
