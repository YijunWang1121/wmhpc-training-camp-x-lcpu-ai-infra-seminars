// 问题 3.2(模块压轴):从零写 tcgen05 单 tile GEMM。
//
// 形状 m128n64k64,bf16 输入,f32 累加,cta_group::1,单 block 128 线程。
// 数据通路:global -> smem(K-major + 128B swizzle)-> tcgen05.mma ->
// TMEM -> tcgen05.ld -> global。判测(main 已给出)用小整数严格对拍。
//
// 给你的材料:课件 F27 的七步流程(下面 kernel 里只留了步骤注释)、
// 你在 2.2 写的 descriptor 编码(SM100 位域)、2.3 的 swizzle_128B
// (staging 布局用它;布局错,结果必错——这里是它的真硬件判测)。
// 其余(TMEM alloc、mbarrier、idesc、tcgen05.mma/ld 的写法)自己查
// PTX ISA 对应章节,课件 C15-C21 讲过每一件的语义,数字换成本题形状。
//
// 两个提醒,直接说明:
// - smem 写完到发射 mma 之间需要 fence.proxy.async(2.1 排序题的答案
//   在这里上真硬件;漏掉的现象自己观察一次,写进报告)
// - tcgen05.ld 每个 warp 只能读自己的 32 条 lane(3.1(a));taddr 高
//   16 bit 是 lane 偏移、低 16 bit 是列偏移;ld 之后要 tcgen05.wait::ld
//
// 运行:make run/m3_tcgen05/02_single_tile;多 seed:./judge_tile.sh
#include <cuda_bf16.h>
#include <cstdio>
#include <random>
#include "../common.h"

constexpr int M = 128, N = 64, K = 64;
#define NUM_COLS 64

// 128B swizzle 的物理偏移(即 2.3 的 swizzle_128B;row 是 K-major 下的
// 行 = M 或 N 维,col 是 K 维字节)。atom = 8 行 × 128B,SBO=1024。
__host__ __device__ inline int swz128(int row, int colByte) {
    int atom = row >> 3, r = row & 7, chunk = colByte >> 4, in16 = colByte & 15;
    return atom * 1024 + r * 128 + ((chunk ^ r) << 4) + in16;
}

__device__ inline uint64_t make_desc_sm100(uint32_t saddr, uint32_t lbo,
                                           uint32_t sbo, uint32_t layout) {
    uint64_t d = 0;
    d |= (uint64_t)((saddr >> 4) & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;             // version = 1(SM100)
    d |= (uint64_t)layout << 61;        // 3 bit layout type
    return d;
}

__device__ inline void mbar_wait(uint32_t mbar, uint32_t phase) {
    uint32_t done = 0;
    while (!done)
        asm volatile(
            "{\n.reg .pred p;\n"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
            "selp.b32 %0, 1, 0, p;\n}"
            : "=r"(done)
            : "r"(mbar), "r"(phase));
}

__global__ void tcgen05_tile(const __nv_bfloat16* gA, const __nv_bfloat16* gB,
                             float* gD) {
    // TODO: 按七步实现。
    // (1) mbarrier 初始化 + TMEM 分配(alloc 结果写到 shared,广播)
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    __shared__ uint64_t mbar[1];
    uint32_t mbar_smem = (uint32_t)__cvta_generic_to_shared(&mbar);
    __shared__ uint32_t s_taddr[1];
    __shared__ __align__(1024) uint8_t sA[M * K * 2];
    __shared__ __align__(1024) uint8_t sB[N * K * 2];
    if (warp_id == 0) {
        if (tid == 0) {
            asm volatile(
                "mbarrier.init.shared::cta.b64 [%0], %1;"
                :: "r"(mbar_smem), "r"(1)
            );
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
        uint32_t dst =(uint32_t)__cvta_generic_to_shared(s_taddr);
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
            :: "r"(dst), "r"(NUM_COLS)
        );
        asm volatile(
            "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
        );

    }
    __syncthreads();
    uint32_t taddr = s_taddr[0];
    // (2) 全体线程把 A/B 按 swizzled 布局写进 smem
    for (int i=tid; i< M*K; i+= blockDim.x){
        int row = i / K;
        int k = i % K;
        *reinterpret_cast<__nv_bfloat16*>(&sA[swz128(row, k*2)]) = gA[i];
    }
    for (int i = tid; i < N * K; i += blockDim.x) {
        int n = i / K, k = i % K;
        *reinterpret_cast<__nv_bfloat16*>(&sB[swz128(n, k * 2)]) = gB[i];
    }
    
    // (3) fence.proxy.async + __syncthreads
    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();
    // (4) 单线程发射 4 条 k16 的 tcgen05.mma(第一条不累加),commit
    uint32_t elected;
    uint32_t sA_addr =(uint32_t)__cvta_generic_to_shared(sA);
    uint32_t sB_addr =(uint32_t)__cvta_generic_to_shared(sB);
    uint64_t a_desc = make_desc_sm100(sA_addr, 0, 1024, 2);
    uint64_t b_desc = make_desc_sm100(sB_addr, 0, 1024, 2);
    uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) | (8u << 17) |
                     (8u << 24);
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "elect.sync _|P, 0xFFFFFFFF;\n"
        "selp.b32 %0, 1, 0, P;\n"
        "}"
        : "=r"(elected)
    );
    if (warp_id ==0 && elected){
        asm volatile("tcgen05.fence::after_thread_sync;");
        #pragma unroll
        for (int ki = 0; ki < 4; ++ki) {
            // 每个 k16 = 16 * sizeof(bf16) = 32 bytes
            uint32_t k_off = ki * 32;
            uint64_t a_desc =make_desc_sm100(sA_addr + k_off, 0, 1024, 2);
            uint64_t b_desc =make_desc_sm100(sB_addr + k_off, 0, 1024, 2);
            asm volatile(
                "{\n"
                ".reg .pred p;\n"
                "setp.ne.b32 p, %4, 0;\n"
                "tcgen05.mma.cta_group::1.kind::f16 "
                "[%0], %1, %2, %3, p;\n"
                "}\n"
                :: "r"(taddr),
                "l"(a_desc),
                "l"(b_desc),
                "r"(idesc),
                "r"((uint32_t)(ki != 0))
            );
        }
        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
            ".shared::cluster.b64 [%0];"
            :
            : "r"(mbar_smem)
            : "memory"
        );
    }
    // (5) mbarrier 等待
    mbar_wait(mbar_smem, 0);
    // (6) epilogue:每 warp tcgen05.ld 自己的 32 条 lane,写回 global
    // (6) 每 warp 读取自己的 32 条 TMEM lane
    uint32_t warp_taddr =
        taddr + ((uint32_t)(warp_id * 32) << 16);

    int row = warp_id * 32 + lane_id;

    #pragma unroll
    for (int col = 0; col < N; col += 4) {
        uint32_t regs[4];

        uint32_t load_taddr = warp_taddr + col;

        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x4.b32 "
            "{%0, %1, %2, %3}, [%4];"
            : "=r"(regs[0]),
            "=r"(regs[1]),
            "=r"(regs[2]),
            "=r"(regs[3])
            : "r"(load_taddr)
        );

        asm volatile("tcgen05.wait::ld.sync.aligned;");

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            gD[row * N + col + j] =
                __uint_as_float(regs[j]);
        }
    }
    // (7) __syncthreads 后 dealloc
    __syncthreads();

    // warp 0 collective dealloc
    if (warp_id == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :
            : "r"(taddr), "r"(NUM_COLS)
        );
    }
}

int main(int argc, char** argv) {
    unsigned seed = argc > 1 ? (unsigned)atoi(argv[1]) : 42;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-3, 3);
    std::vector<__nv_bfloat16> hA(M * K), hB(N * K);
    std::vector<float> ref(M * N, 0.f);
    for (auto& v : hA) v = __float2bfloat16((float)dist(rng));
    for (auto& v : hB) v = __float2bfloat16((float)dist(rng));
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++)
                ref[m * N + n] += __bfloat162float(hA[m * K + k]) *
                                  __bfloat162float(hB[n * K + k]);
    __nv_bfloat16 *dA, *dB;
    float* dD;
    CUDA_CHECK(cudaMalloc(&dA, M * K * 2));
    CUDA_CHECK(cudaMalloc(&dB, N * K * 2));
    CUDA_CHECK(cudaMalloc(&dD, M * N * 4));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * K * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), N * K * 2, cudaMemcpyHostToDevice));
    tcgen05_tile<<<1, 128>>>(dA, dB, dD);
    CUDA_CHECK_KERNEL();
    std::vector<float> got(M * N);
    CUDA_CHECK(cudaMemcpy(got.data(), dD, M * N * 4, cudaMemcpyDeviceToHost));
    long bad = 0;
    for (int i = 0; i < M * N; i++)
        if (got[i] != ref[i]) {
            if (bad < 5)
                printf("MISMATCH D[%d][%d]: got %.1f want %.1f\n", i / N,
                       i % N, got[i], ref[i]);
            bad++;
        }
    printf(bad ? "FAIL seed=%u: %ld / %d\n" : "PASS seed=%u\n", seed,
           bad ? bad : (long)seed, M * N);
    return bad != 0;
}
