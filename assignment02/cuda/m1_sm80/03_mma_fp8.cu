#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

#include "../common.h"

// ============================================================
// 1.1: fragment mapping
// lane: 0..31
//
// A: 16 x 32, 每 lane 16 bytes
// B: 32 x 8,  每 lane  8 bytes
// ============================================================

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


// ============================================================
// GPU kernel
//
// A: [16, 32], row-major
// B: [32,  8], row-major
// D: [16,  8], row-major
// ============================================================

__global__ void mma_fp8_kernel(
    const __nv_fp8_e4m3 *A,
    const __nv_fp8_e4m3 *B,
    float *D)
{
    int lane = threadIdx.x & 31;

    // --------------------------------------------------------
    // Step 1: 手动构造 A fragment
    //
    // 16 个 fp8 = 16 bytes = 4 x b32
    // --------------------------------------------------------

    uint32_t ra[4] = {0, 0, 0, 0};

    for (int i = 0; i < 16; ++i) {
        int row = a_row_of(lane, i);
        int col = a_col_of(lane, i);

        __nv_fp8_e4m3 x = A[row * 32 + col];

        int r = i >> 2;
        int j = i & 3;
        // TODO:
        // 把 x 的 raw 8-bit 数据放进 ra[i / 4] 的
        // 第 (i % 4) 个 byte。
        //
        // 注意 byte 顺序。
        uint8_t byte = *reinterpret_cast<uint8_t*>(&x);

        ra[r] |= uint32_t(byte) << (8 * j);

    }


    // --------------------------------------------------------
    // Step 2: 手动构造 B fragment
    //
    // 8 个 fp8 = 8 bytes = 2 x b32
    // --------------------------------------------------------

    uint32_t rb[2] = {0, 0};

    for (int i = 0; i < 8; ++i) {
        int row = b_row_of(lane, i);
        int col = b_col_of(lane, i);

        __nv_fp8_e4m3 x = B[row * 8 + col];

        // TODO:
        int r = i / 4;

    int j = i % 4;

    uint8_t byte = *reinterpret_cast<uint8_t*>(&x);

    rb[r] |= uint32_t(byte) << (8 * j);

}


    // --------------------------------------------------------
    // Step 3: accumulator
    //
    // m16n8 的 D fragment:
    // 每 lane 4 个 float
    // --------------------------------------------------------

    float c[4] = {
        0.0f, 0.0f, 0.0f, 0.0f
    };

    float d[4];


    // --------------------------------------------------------
    // Step 4: 发一条 FP8 MMA
    // --------------------------------------------------------

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d[0]),
          "=f"(d[1]),
          "=f"(d[2]),
          "=f"(d[3])
        : "r"(ra[0]),
          "r"(ra[1]),
          "r"(ra[2]),
          "r"(ra[3]),
          "r"(rb[0]),
          "r"(rb[1]),
          "f"(c[0]),
          "f"(c[1]),
          "f"(c[2]),
          "f"(c[3])
    );


    // --------------------------------------------------------
    // Step 5: D fragment 写回
    //
    // m16n8 的 D mapping 和前面 1.2 是一样的。
    // --------------------------------------------------------

    int group = lane >> 2;
    int tig   = lane & 3;

    D[group * 8 + tig * 2] =
        d[0];

    D[group * 8 + tig * 2 + 1] =
        d[1];

    D[(group + 8) * 8 + tig * 2] =
        d[2];

    D[(group + 8) * 8 + tig * 2 + 1] =
        d[3];
}


// ============================================================
// main
// ============================================================

int main(int argc, char **argv)
{
    if (argc != 2) {
        printf("FAIL: usage: %s <seed>\n", argv[0]);
        return 1;
    }

    int seed = atoi(argv[1]);
    srand(seed);

    constexpr int M = 16;
    constexpr int N = 8;
    constexpr int K = 32;

    __nv_fp8_e4m3 hA[M * K];
    __nv_fp8_e4m3 hB[K * N];

    float ref[M * N];
    float hD[M * N];


    // ========================================================
    // Step 6: 初始化
    //
    // 用小整数，例如 [-2,2]。
    // ========================================================

    for (int i = 0; i < M * K; ++i) {
        int x = (rand() % 5) - 2;

        // cuda_fp8.h 提供转换
        hA[i] = __nv_fp8_e4m3((float)x);
    }

    for (int i = 0; i < K * N; ++i) {
        int x = (rand() % 5) - 2;
        hB[i] = __nv_fp8_e4m3((float)x);
    }


    // ========================================================
    // Step 7: CPU reference
    // ========================================================

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {

            float sum = 0.0f;

            for (int k = 0; k < K; ++k) {
                float a = (float)hA[m * K + k];
                float b = (float)hB[k * N + n];

                sum += a * b;
            }

            ref[m * N + n] = sum;
        }
    }


    // ========================================================
    // Step 8: device memory
    // ========================================================

    __nv_fp8_e4m3 *dA;
    __nv_fp8_e4m3 *dB;
    float *dD;

    CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
    CUDA_CHECK(cudaMalloc(&dD, sizeof(hD)));

    CUDA_CHECK(cudaMemcpy(
        dA, hA, sizeof(hA),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        dB, hB, sizeof(hB),
        cudaMemcpyHostToDevice));


    // ========================================================
    // Step 9: 一个 warp 做一个 MMA
    // ========================================================

    mma_fp8_kernel<<<1, 32>>>(dA, dB, dD);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        hD, dD, sizeof(hD),
        cudaMemcpyDeviceToHost));


    // ========================================================
    // Step 10: strict comparison
    // ========================================================

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {

            float got = hD[m * N + n];
            float expected = ref[m * N + n];

            if (got != expected) {
                printf(
                    "MISMATCH seed=%d "
                    "D[%d][%d]: got=%f expected=%f\n",
                    seed, m, n, got, expected);

                cudaFree(dA);
                cudaFree(dB);
                cudaFree(dD);

                return 1;
            }
        }
    }

    printf("PASS seed=%d\n", seed);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dD);

    return 0;
}