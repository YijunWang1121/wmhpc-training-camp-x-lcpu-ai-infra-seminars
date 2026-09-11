#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <vector>
#include "../common.h"

#ifndef STAGES
#define STAGES 3
#endif

constexpr int BM = 128, BN = 64, BK = 64;
constexpr int NSTAGE = STAGES;
static_assert(NSTAGE >= 2, "03_pipeline requires STAGES >= 2");

__device__ inline uint64_t make_desc_sm100(uint32_t saddr, uint32_t lbo,
                                           uint32_t sbo, uint32_t layout) {
    uint64_t d = 0;
    d |= (uint64_t)((saddr >> 4) & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)layout << 61;
    return d;
}

// Potentially blocking wait. try_wait may suspend the issuing thread, which is
// useful for the mandatory wait path.
__device__ inline void mbar_wait(uint32_t mbar, uint32_t phase) {
    uint32_t done = 0;
    while (!done) {
        asm volatile(
            "{\n.reg .pred p;\n"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
            "selp.b32 %0, 1, 0, p;\n}"
            : "=r"(done)
            : "r"(mbar), "r"(phase)
            : "memory");
    }
}

// Truly non-blocking probe for opportunistic prefetch.  PTX try_wait is only
// *potentially* non-blocking; test_wait is the non-blocking instruction.
__device__ inline bool mbar_test(uint32_t mbar, uint32_t phase) {
    uint32_t done;
    asm volatile(
        "{\n.reg .pred p;\n"
        "mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "selp.b32 %0, 1, 0, p;\n}"
        : "=r"(done)
        : "r"(mbar), "r"(phase)
        : "memory");
    return done != 0;
}

__global__ void gemm_pipeline(const __nv_bfloat16* gA,
                              const __nv_bfloat16* gB, float* gD, int M,
                              int N, int K,
                              const __grid_constant__ CUtensorMap tmapA,
                              const __grid_constant__ CUtensorMap tmapB) {
    extern __shared__ uint8_t smem_raw[];
    uint8_t* smem =
        (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
    constexpr uint32_t stageBytes = (uint32_t)(BM + BN) * BK * 2;
    constexpr uint32_t txBytes = stageBytes;

    int tid = threadIdx.x, warp_id = tid / 32, lane_id = tid % 32;

    // Each ring slot has its own producer-complete and consumer-complete
    // barrier. Reusing one empty barrier across slots makes parity ambiguous.
    __shared__ uint64_t mbar_full[NSTAGE];
    __shared__ uint64_t mbar_empty[NSTAGE];
    __shared__ uint32_t s_taddr[1];

    auto sA = [&](int s) { return smem + (size_t)s * stageBytes; };
    auto sB = [&](int s) {
        return smem + (size_t)s * stageBytes + (size_t)BM * BK * 2;
    };
    auto mbarF = [&](int s) {
        return (uint32_t)__cvta_generic_to_shared(&mbar_full[s]);
    };
    auto mbarE = [&](int s) {
        return (uint32_t)__cvta_generic_to_shared(&mbar_empty[s]);
    };

    if (warp_id == 0) {
        if (tid == 0) {
#pragma unroll
            for (int s = 0; s < NSTAGE; ++s) {
                asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                             :
                             : "r"(mbarF(s)), "r"(1));
                asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                             :
                             : "r"(mbarE(s)), "r"(1));
            }
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
        uint32_t dst = (uint32_t)__cvta_generic_to_shared(s_taddr);
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], "
            "%1;"
            :
            : "r"(dst), "r"(BN));
        asm volatile(
            "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    __syncthreads();
    uint32_t taddr = s_taddr[0];

    int tileM = blockIdx.x * BM;
    int tileN = blockIdx.y * BN;
    int kIters = K / BK;

    // Called by tid 0 only. Tile it2 always maps to ring slot it2 % NSTAGE.
    auto issue_tma = [&](int it2) {
        int s = it2 % NSTAGE;
        asm volatile(
            "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
            :
            : "r"(mbarF(s)), "r"(txBytes)
            : "memory");
        uint32_t sA_dst =
            (uint32_t)__cvta_generic_to_shared(sA(s));
        uint32_t sB_dst =
            (uint32_t)__cvta_generic_to_shared(sB(s));
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
            "complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
            :
            : "r"(sA_dst), "l"(reinterpret_cast<uint64_t>(&tmapA)),
              "r"(it2 * BK), "r"(tileM), "r"(mbarF(s))
            : "memory");
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
            "complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
            :
            : "r"(sB_dst), "l"(reinterpret_cast<uint64_t>(&tmapB)),
              "r"(it2 * BK), "r"(tileN), "r"(mbarF(s))
            : "memory");
    };

    int issued = 0;
    if (tid == 0) {
        int warmup = kIters < NSTAGE ? kIters : NSTAGE;
        for (int w = 0; w < warmup; ++w) issue_tma(w);
        issued = warmup;
    }

    uint32_t idesc =
        (1u << 4) | (1u << 7) | (1u << 10) | (8u << 17) | (8u << 24);

    for (int it = 0; it < kIters; ++it) {
        int s = it % NSTAGE;
        if (tid == 0) {
            // Mandatory issue: if tile it was not issued opportunistically,
            // block until its slot is empty and issue it now. Never replace
            // this path with a test-and-skip probe.
            if (issued == it) {
                int reuse = it / NSTAGE;
                if (reuse != 0)
                    mbar_wait(mbarE(s), (reuse - 1) & 1);
                issue_tma(it);
                ++issued;
            }

            // Best-effort deeper prefetch. Stop at the first occupied ring
            // slot; mbarrier.test_wait itself never suspends this thread.
            while (issued < kIters) {
                int s2 = issued % NSTAGE;
                int reuse2 = issued / NSTAGE;
                if (reuse2 != 0 &&
                    !mbar_test(mbarE(s2), (reuse2 - 1) & 1))
                    break;
                issue_tma(issued);
                ++issued;
            }

            // The full-barrier parity is the reuse count of this ring slot.
            mbar_wait(mbarF(s), (it / NSTAGE) & 1);
        }
        __syncthreads();

        uint32_t sA_addr =
            (uint32_t)__cvta_generic_to_shared(sA(s));
        uint32_t sB_addr =
            (uint32_t)__cvta_generic_to_shared(sB(s));
        uint32_t elected;
        asm volatile(
            "{\n"
            ".reg .pred P;\n"
            "elect.sync _|P, 0xFFFFFFFF;\n"
            "selp.b32 %0, 1, 0, P;\n"
            "}"
            : "=r"(elected));
        if (warp_id == 0 && elected) {
            asm volatile("tcgen05.fence::after_thread_sync;");
#pragma unroll
            for (int ki = 0; ki < 4; ++ki) {
                uint32_t k_off = ki * 32;
                uint64_t a_desc =
                    make_desc_sm100(sA_addr + k_off, 0, 1024, 2);
                uint64_t b_desc =
                    make_desc_sm100(sB_addr + k_off, 0, 1024, 2);
                asm volatile(
                    "{\n"
                    ".reg .pred p;\n"
                    "setp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 "
                    "[%0], %1, %2, %3, p;\n"
                    "}\n"
                    :
                    : "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(idesc),
                      "r"((uint32_t)(ki != 0 || it != 0)));
            }
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
                ".shared::cluster.b64 [%0];"
                :
                : "r"(mbarE(s))
                : "memory");
        }
    }

    if (tid == 0) {
        int lastIt = kIters - 1;
        mbar_wait(mbarE(lastIt % NSTAGE), (lastIt / NSTAGE) & 1);
    }
    __syncthreads();

    // The mbarrier wait was performed by tid 0, whereas every thread below
    // issues tcgen05.ld. This fence is required to carry the completion order
    // through __syncthreads() to each TMEM-reading thread.
    asm volatile("tcgen05.fence::after_thread_sync;");

    uint32_t warp_taddr = taddr + ((uint32_t)(warp_id * 32) << 16);
    int row = warp_id * 32 + lane_id;

#pragma unroll
    for (int col = 0; col < BN; col += 4) {
        uint32_t regs[4];
        uint32_t load_taddr = warp_taddr + col;
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x4.b32 "
            "{%0, %1, %2, %3}, [%4];"
            : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
            : "r"(load_taddr));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
#pragma unroll
        for (int j = 0; j < 4; ++j)
            gD[(tileM + row) * N + tileN + col + j] =
                __uint_as_float(regs[j]);
    }

    __syncthreads();
    if (warp_id == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :
            : "r"(taddr), "r"(BN));
    }
    (void)M;
}

int main(int argc, char** argv) {
    int M = argc > 3 ? atoi(argv[1]) : 4096;
    int N = argc > 3 ? atoi(argv[2]) : 4096;
    int K = argc > 3 ? atoi(argv[3]) : 4096;
    if (M % BM || N % BN || K % BK || K == 0) {
        printf("形状需为正数并按 %dx%dx%d 对齐\n", BM, BN, BK);
        return 1;
    }
    size_t nA = (size_t)M * K, nB = (size_t)N * K;
    size_t nD = (size_t)M * N;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-3, 3);
    std::vector<__nv_bfloat16> hA(nA), hB(nB);
    for (auto& v : hA) v = __float2bfloat16((float)dist(rng));
    for (auto& v : hB) v = __float2bfloat16((float)dist(rng));
    __nv_bfloat16 *dA, *dB;
    float *dD, *dRef;
    CUDA_CHECK(cudaMalloc(&dA, nA * 2));
    CUDA_CHECK(cudaMalloc(&dB, nB * 2));
    CUDA_CHECK(cudaMalloc(&dD, nD * 4));
    CUDA_CHECK(cudaMalloc(&dRef, nD * 4));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), nA * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), nB * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dD, 0xFF, nD * 4));

    CUtensorMap tmapA = {}, tmapB = {};
    {
        uint64_t globalDimA[2] = {(uint64_t)K, (uint64_t)M};
        uint64_t globalStridesA[1] = {(uint64_t)K * 2};
        uint32_t boxDimA[2] = {(uint32_t)BK, (uint32_t)BM};
        uint32_t elementStrides[2] = {1, 1};
        CUresult r = cuTensorMapEncodeTiled(
            &tmapA, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, dA, globalDimA,
            globalStridesA, boxDimA, elementStrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (r != CUDA_SUCCESS) {
            const char* err;
            cuGetErrorString(r, &err);
            fprintf(stderr, "tmapA encode failed: %s\n", err);
            exit(1);
        }
    }
    {
        uint64_t globalDimB[2] = {(uint64_t)K, (uint64_t)N};
        uint64_t globalStridesB[1] = {(uint64_t)K * 2};
        uint32_t boxDimB[2] = {(uint32_t)BK, (uint32_t)BN};
        uint32_t elementStrides[2] = {1, 1};
        CUresult r = cuTensorMapEncodeTiled(
            &tmapB, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, dB, globalDimB,
            globalStridesB, boxDimB, elementStrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (r != CUDA_SUCCESS) {
            const char* err;
            cuGetErrorString(r, &err);
            fprintf(stderr, "tmapB encode failed: %s\n", err);
            exit(1);
        }
    }

    dim3 grid(M / BM, N / BN);
    size_t smemBytes =
        (size_t)NSTAGE * (BM + BN) * BK * 2 + 1024;
    CUDA_CHECK(cudaFuncSetAttribute(
        gemm_pipeline, cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smemBytes));
    auto launch = [&] {
        gemm_pipeline<<<grid, 128, smemBytes>>>(dA, dB, dD, M, N, K, tmapA,
                                               tmapB);
    };
    launch();
    CUDA_CHECK_KERNEL();

    cublasHandle_t h;
    cublasCreate(&h);
    float alpha = 1.f, beta = 0.f;
    cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, dB,
                 CUDA_R_16BF, K, dA, CUDA_R_16BF, K, &beta, dRef, CUDA_R_32F,
                 N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> got(nD), ref(nD);
    CUDA_CHECK(cudaMemcpy(got.data(), dD, nD * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ref.data(), dRef, nD * 4, cudaMemcpyDeviceToHost));
    long bad = 0;
    for (size_t i = 0; i < nD; ++i) bad += got[i] != ref[i];

    int timingIters =
        (size_t)M * N >= (size_t)4096 * 4096 ? 20 : 100;
    float ms = time_avg_ms(launch, timingIters);
    double tflops = 2.0 * M * N * K / (ms * 1e9);
    float cub_ms = time_avg_ms(
        [&] {
            cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, dB,
                         CUDA_R_16BF, K, dA, CUDA_R_16BF, K, &beta, dRef,
                         CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);
        },
        timingIters);
    double cub_tflops = 2.0 * M * N * K / (cub_ms * 1e9);
    printf("[4.3 pipeline S=%d] M=%d N=%d K=%d  %s(bad=%ld)  %.2f ms  "
           "%.1f TFLOPS  (cuBLAS %.1f, 达成率 %.0f%%)\n",
           NSTAGE, M, N, K, bad ? "FAIL" : "PASS", bad, ms, tflops,
           cub_tflops, 100.0 * tflops / cub_tflops);
    cublasDestroy(h);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dD);
    cudaFree(dRef);
    return bad != 0;
}
