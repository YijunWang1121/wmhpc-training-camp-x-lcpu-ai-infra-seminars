// E6 microbench: mma.sync.m16n8k16 (SM80 path used by FlashKDA K2) vs tcgen05.mma (SM100) at the
// exact GEMM shapes of one K2 chunk step (D=128, CHUNK=16):
//   G1  [16 x 128] @ [128 x 128]   k_d@S and q_d@S      tcgen05 (transposed): M=128, N=16, K=128 (8 k16 steps)
//   G2  [16 x 16 ] @ [16  x 128]   INV@u, Mqk@U          tcgen05 (transposed): M=128, N=16, K=16  (1 instr)
//   G3  [128 x 16] @ [16  x 128]   k_r^T@U (state upd.)  tcgen05:              M=128, N=128,K=16  (1 instr)
// Everything is 1 CTA per SM (grid = #SM or 1), cycles measured with clock64() inside the kernel.
// smem layouts are FlashKDA's GMMA "INTER" (no swizzle, 8x8 core matrices) so the LBO/SBO values found
// here transfer directly to a K2 rewrite.  Every tcgen05 config is checked against a CPU reference first.
//
// build (login node): nvcc -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a -o mb_mma mb_mma.cu
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>
#include <cmath>

#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA error %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)

// ---------------------------------------------------------------- layouts (FlashKDA K_INTER / MN_INTER)
// K-major tile R x C (C = K dim contiguous within 8-elem core rows), atoms tiled R-first (LayoutLeft):
//   elem(r,c) = ((r/8) + (c/8)*(R/8))*64 + (r%8)*8 + (c%8)
__host__ __device__ inline int kinter(int r, int c, int R) { return ((r >> 3) + (c >> 3) * (R >> 3)) * 64 + (r & 7) * 8 + (c & 7); }
// SW128 K-major: blocks of 64 contiguous-dim elements (128 B rows), row r at r*128 B, 16-B chunk XOR (r&7). Element offset:
__host__ __device__ inline int ksw128(int r, int c, int R) { int blk = c >> 6, cc = c & 63; return blk * R * 64 + r * 64 + ((((cc >> 3) ^ (r & 7)) << 3) | (cc & 7)); }

// ---------------------------------------------------------------- PTX helpers
__device__ inline uint64_t make_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    d |= (uint64_t)((saddr >> 4) & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;        // version 1 (SM100)
    // layout_type = 0 (SWIZZLE_NONE), base_offset 0
    return d;
}
// kind::f16 instruction descriptor: c=F32(1), a=b=BF16(1), majors, N>>3 @17, M>>4 @24
__device__ inline uint32_t make_idesc(int M, int N, int a_mn, int b_mn) {
    return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16) |
           ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}
__device__ inline void mma_ss(uint32_t tmem_d, uint64_t da, uint64_t db, uint32_t idesc, uint32_t accum) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
                 :: "r"(tmem_d), "l"(da), "l"(db), "r"(idesc), "r"(accum));
}
__device__ inline void mma_commit(uint32_t mbar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(mbar) : "memory");
}
__device__ inline void mbar_wait(uint32_t mbar, uint32_t phase) {
    uint32_t done = 0;
    while (!done)
        asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.b32 %0,1,0,p;\n}"
                     : "=r"(done) : "r"(mbar), "r"(phase));
}
__device__ inline uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ inline void hmma16816(float* d, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---------------------------------------------------------------- T1/T2: mma.sync throughput / latency
// NACC independent accumulators per warp; operands constant (registers only) -> pure tensor-pipe issue rate.
template <int NACC, int DEP>
__global__ void k_mmasync(int iters, long long* cyc, float* sink) {
    uint32_t a[4] = {0x3f803f80u + threadIdx.x, 0x3f803f80u, 0x3f803f80u, 0x3f803f80u};
    uint32_t b[2] = {0x3f803f80u, 0x3f803f80u + threadIdx.x};
    float d[NACC][4];
    for (int i = 0; i < NACC; i++) for (int j = 0; j < 4; j++) d[i][j] = 0.f;
    __syncthreads();
    long long t0 = clock64();
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int i = 0; i < NACC; i++) {
            if (DEP) hmma16816(d[0], a, b); else hmma16816(d[i], a, b);
        }
    }
    long long t1 = clock64();
    float s = 0;
    for (int i = 0; i < NACC; i++) for (int j = 0; j < 4; j++) s += d[i][j];
    if (s == 12345.f) sink[threadIdx.x] = s;   // keep
    if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}

// ---------------------------------------------------------------- T3-T5: tcgen05 at K2 shapes
// One CTA = 128 threads (4 warps, as K2's MMA warps). smem holds A (M x K) and B (N x K) tiles in INTER layout.
// mode 0: correctness (single MMA chain, TMEM -> global)
// mode 1: latency   : per iteration issue the chain (KSTEPS mmas into one accumulator) + commit + wait
// mode 2: throughput: issue iters*KSTEPS mmas round-robin over NBUF accumulators, one commit at the end
template <int M, int N, int K, int A_MN, int B_MN, int LAY = 0>
__global__ void k_tcgen05(const __nv_bfloat16* gA, const __nv_bfloat16* gB, float* gD, int mode, int iters, long long* cyc) {
    constexpr int KSTEPS = K / 16;
    constexpr int NBUF = (512 / N) < 8 ? (512 / N) : 8;
    extern __shared__ __align__(1024) uint8_t smem[];
    __nv_bfloat16* sA = (__nv_bfloat16*)smem;                 // M x K
    constexpr int KROW = (LAY && K < 64) ? 64 : K;            // SW128 rows are 128 B even when K=16
    __nv_bfloat16* sB = (__nv_bfloat16*)(smem + M * KROW * 2);   // N x K
    __shared__ uint64_t mbar;
    __shared__ uint32_t s_taddr;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    uint32_t mb = smem_u32(&mbar);
    if (warp == 0) {
        if (tid == 0) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mb)); asm volatile("fence.mbarrier_init.release.cluster;"); }
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem_u32(&s_taddr)), "r"(512));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    // fill smem: A is M x K "K-major logical"; stored K-major INTER if !A_MN else MN-major INTER (transpose of the
    // K-major layout of the transposed tile).  Same for B (N x K).
    for (int i = tid; i < M * K; i += blockDim.x) {
        int m = i / K, k = i % K;
        int off = LAY ? (A_MN ? ksw128(k, m, K) : ksw128(m, k, M)) : (A_MN ? kinter(k, m, K) : kinter(m, k, M));
        sA[off] = gA[i];
    }
    for (int i = tid; i < N * K; i += blockDim.x) {
        int n = i / K, k = i % K;
        int off = LAY ? (B_MN ? ksw128(k, n, K) : ksw128(n, k, N)) : (B_MN ? kinter(k, n, K) : kinter(n, k, N));
        sB[off] = gB[i];
    }
    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();
    uint32_t taddr = s_taddr;
    uint32_t idesc = make_idesc(M, N, A_MN ? 1 : 0, B_MN ? 1 : 0);
    uint32_t aBase = smem_u32(sA), bBase = smem_u32(sB);
    // descriptor strides for INTER layouts (bytes): K-major tile R x C: LBO = between K-adjacent core mats = (R/8)*128,
    // SBO = between M/N-adjacent core mats = 128.  MN-major (stored as K x R K-major tile): LBO = along MN = (K/8)*128
    // ... but "along MN" core mats are adjacent at 128 B and along K at (R/8)*128 -> validated by mode 0.
    auto descA = [&](int ks) {
        if (LAY) {   // SW128: K-major: half-block (ks/4) of M*128 B, +32 B per k16 inside; SBO = 1024 (8-row group); layout type 2
            if (!A_MN) return make_desc(aBase + (ks >> 2) * M * 128 + (ks & 3) * 32, 16, 1024) | ((uint64_t)2 << 61);
            // MN-major SW128 (A stored as [k rows][m], 64-wide MN blocks of K*128 B): k16 step = 16 rows = 2048 B
            else if (A_MN == 1) return make_desc(aBase + ks * 16 * 128, K * 128, 1024) | ((uint64_t)2 << 61);   // (a) LBO = MN-block stride
            else                return make_desc(aBase + ks * 16 * 128, 1024, K * 128) | ((uint64_t)2 << 61);   // (b) swapped
        }
        // K-major: one k16 step = 2 core matrices along K = 2*LBO bytes
        if (!A_MN) return make_desc(aBase + ks * 2 * (M / 8) * 128, (M / 8) * 128, 128);
        // MN-major INTER (validated: hypothesis (b)): LBO = K-adjacent core mats (128 B), SBO = MN-adjacent ((K/8)*128)
        else       return make_desc(aBase + ks * 2 * 128, 128, 128 * (K / 8));
    };
    auto descB = [&](int ks) {
        if (LAY) {
            if (!B_MN) return make_desc(bBase + (ks >> 2) * N * 128 + (ks & 3) * 32, 16, 1024) | ((uint64_t)2 << 61);
            else if (B_MN == 1) return make_desc(bBase + ks * 16 * 128, K * 128, 1024) | ((uint64_t)2 << 61);
            else                return make_desc(bBase + ks * 16 * 128, 1024, K * 128) | ((uint64_t)2 << 61);
        }
        if (!B_MN) return make_desc(bBase + ks * 2 * (N / 8) * 128, (N / 8) * 128, 128);
        else       return make_desc(bBase + ks * 2 * 128, 128, 128 * (K / 8));
    };
    uint32_t phase = 0;
    long long t0 = 0, t1 = 0;
    if (mode == 0) {
        if (warp == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (lane == 0) {
                for (int ks = 0; ks < KSTEPS; ++ks) mma_ss(taddr, descA(ks), descB(ks), idesc, ks != 0);
                mma_commit(mb);
            }
            __syncwarp();
        }
        mbar_wait(mb, phase); phase ^= 1;
    } else if (mode == 1) {
        __syncthreads();
        t0 = clock64();
        for (int it = 0; it < iters; ++it) {
            if (warp == 0) {
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (lane == 0) {
                    for (int ks = 0; ks < KSTEPS; ++ks) mma_ss(taddr, descA(ks), descB(ks), idesc, ks != 0);
                    mma_commit(mb);
                }
                __syncwarp();
            }
            mbar_wait(mb, phase); phase ^= 1;
        }
        t1 = clock64();
    } else {
        __syncthreads();
        t0 = clock64();
        if (warp == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (lane == 0) {
                for (int it = 0; it < iters; ++it) {
                    uint32_t td = taddr + (uint32_t)((it % NBUF) * N);   // rotate accumulators (N fp32 columns each)
                    for (int ks = 0; ks < KSTEPS; ++ks) mma_ss(td, descA(ks), descB(ks), idesc, ks != 0);
                }
                mma_commit(mb);
            }
            __syncwarp();
        }
        mbar_wait(mb, phase); phase ^= 1;
        t1 = clock64();
    }
    asm volatile("tcgen05.fence::after_thread_sync;");
    // epilogue: each warp reads its 32 lanes (rows), N columns, from accumulator 0
    if (mode == 0 || M == 128) {
        int rows_per_warp = 32;
        if (warp * 32 < M) {
            uint32_t wt = taddr + ((uint32_t)(warp * 32) << 16);
            int row = warp * 32 + lane;
            for (int c = 0; c < N; c += 4) {
                uint32_t r[4];
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(wt + c));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (mode == 0) for (int j = 0; j < 4; j++) gD[row * N + c + j] = __uint_as_float(r[j]);
            }
        }
        (void)rows_per_warp;
    }
    __syncthreads();
    if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(512));
    if (tid == 0 && mode) cyc[blockIdx.x] = t1 - t0;
}

// ---------------------------------------------------------------- T6: TMEM load cost (128 lanes x C fp32 columns)
__global__ void k_tmem_ld(int cols, int iters, long long* cyc, float* sink, int dep) {
    __shared__ uint32_t s_taddr;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    if (warp == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem_u32(&s_taddr)), "r"(512));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    __syncthreads();
    uint32_t wt = s_taddr + ((uint32_t)(warp * 32) << 16);
    float acc = 0;
    long long t0 = clock64();
    for (int it = 0; it < iters; ++it) {
        for (int c = 0; c < cols; c += 16) {
            uint32_t r[16];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                         : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                           "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
                         : "r"(wt + c));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            if (dep) { for (int j = 0; j < 16; j++) acc += __uint_as_float(r[j]); } else { acc += __uint_as_float(r[0]) * __uint_as_float(r[15]); }
        }
    }
    long long t1 = clock64();
    if (acc == 12345.f) sink[tid] = acc;
    __syncthreads();
    if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(s_taddr), "r"(512));
    if (tid == 0) cyc[blockIdx.x] = t1 - t0;
    (void)lane;
}

// ---------------------------------------------------------------- host
static int g_nsm = 1;
template <class F> static double run_cycles(F launch, long long* dcyc, int nblk) {
    launch(); CK(cudaDeviceSynchronize());
    launch(); CK(cudaDeviceSynchronize());
    std::vector<long long> h(nblk);
    CK(cudaMemcpy(h.data(), dcyc, nblk * 8, cudaMemcpyDeviceToHost));
    double s = 0; for (auto x : h) s += x; return s / nblk;
}

template <int M, int N, int K, int A_MN, int B_MN, int LAY = 0>
static void test_tcgen05(const char* name, long long* dcyc, float* dsink) {
    std::mt19937 rng(7); std::uniform_int_distribution<int> dist(-3, 3);
    std::vector<__nv_bfloat16> hA(M * K), hB(N * K); std::vector<float> ref(M * N, 0.f), got(M * N);
    for (auto& v : hA) v = __float2bfloat16((float)dist(rng));
    for (auto& v : hB) v = __float2bfloat16((float)dist(rng));
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) { float s = 0; for (int k = 0; k < K; k++) s += __bfloat162float(hA[m * K + k]) * __bfloat162float(hB[n * K + k]); ref[m * N + n] = s; }
    __nv_bfloat16 *dA, *dB; float* dD;
    CK(cudaMalloc(&dA, M * K * 2)); CK(cudaMalloc(&dB, N * K * 2)); CK(cudaMalloc(&dD, M * N * 4));
    CK(cudaMemcpy(dA, hA.data(), M * K * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dB, hB.data(), N * K * 2, cudaMemcpyHostToDevice));
    size_t smem = (size_t)(M + N) * ((LAY && K < 64) ? 64 : K) * 2 + 1024;
    auto kern = k_tcgen05<M, N, K, A_MN, B_MN, LAY>;
    CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    kern<<<1, 128, smem>>>(dA, dB, dD, 0, 1, dcyc); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(got.data(), dD, M * N * 4, cudaMemcpyDeviceToHost));
    long bad = 0; for (int i = 0; i < M * N; i++) if (got[i] != ref[i]) { if (bad < 3) printf("   MISMATCH [%d,%d] got %g want %g\n", i / N, i % N, got[i], ref[i]); bad++; }
    printf("%-34s correctness: %s\n", name, bad ? "FAIL" : "PASS");
    if (bad) { cudaFree(dA); cudaFree(dB); cudaFree(dD); return; }
    const int iters = 2000;
    double lat1 = run_cycles([&] { kern<<<1, 128, smem>>>(dA, dB, dD, 1, iters, dcyc); }, dcyc, 1) / iters;
    double thr1 = run_cycles([&] { kern<<<1, 128, smem>>>(dA, dB, dD, 2, iters, dcyc); }, dcyc, 1) / iters;
    double thrN = run_cycles([&] { kern<<<g_nsm, 128, smem>>>(dA, dB, dD, 2, iters, dcyc); }, dcyc, g_nsm) / iters;
    double macs = (double)M * N * K;
    printf("   latency (issue chain+commit+wait) = %7.1f cyc   throughput 1 SM = %6.1f cyc/instr-chain (%6.0f MAC/cyc)   all-SM = %6.1f cyc (%6.0f MAC/cyc/SM)\n",
           lat1, thr1, macs / thr1, thrN, macs / thrN);
    cudaFree(dA); cudaFree(dB); cudaFree(dD);
}

template <int NACC, int DEP>
static void test_mmasync(int warps, int nblk, long long* dcyc, float* dsink) {
    const int iters = 4000;
    double cyc = run_cycles([&] { k_mmasync<NACC, DEP><<<nblk, warps * 32>>>(iters, dcyc, dsink); }, dcyc, nblk);
    double per = cyc / (iters * NACC);
    double mac_per_cyc_sm = (double)warps * 2048.0 / per;   // all warps of the CTA issue concurrently
    printf("   mma.sync m16n8k16 %s NACC=%d warps/CTA=%2d CTAs=%3d : %6.2f cyc/HMMA/warp  -> %6.0f MAC/cyc/SM\n",
           DEP ? "dependent  " : "independent", NACC, warps, nblk, per, mac_per_cyc_sm);
}

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0)); g_nsm = p.multiProcessorCount;
    printf("device %s cc %d.%d SMs %d\n", p.name, p.major, p.minor, g_nsm);
    long long* dcyc; float* dsink; CK(cudaMalloc(&dcyc, 1024 * 8)); CK(cudaMalloc(&dsink, 4096 * 4));
    printf("== T1/T2 mma.sync (register-resident operands, pure tensor pipe) ==\n");
    test_mmasync<1, 1>(1, 1, dcyc, dsink);
    test_mmasync<1, 1>(4, 1, dcyc, dsink);
    test_mmasync<4, 0>(1, 1, dcyc, dsink);
    test_mmasync<4, 0>(4, 1, dcyc, dsink);     // K2 config: 4 warps, 4 independent accumulators
    test_mmasync<8, 0>(4, 1, dcyc, dsink);
    test_mmasync<8, 0>(8, 1, dcyc, dsink);
    test_mmasync<8, 0>(16, 1, dcyc, dsink);
    test_mmasync<8, 0>(16, g_nsm, dcyc, dsink);
    printf("== T3-T5 tcgen05.mma kind::f16 bf16->f32, INTER (no-swizzle) smem layouts ==\n");
    test_tcgen05<128, 16, 128, 0, 0>("G1 M128 N16 K128 (S^T@k_d^T)", dcyc, dsink);
    test_tcgen05<128, 16, 16, 0, 0>("G2 M128 N16 K16  (u^T@INV^T)", dcyc, dsink);
    test_tcgen05<128, 128, 16, 0, 0>("G3 M128 N128 K16 (U^T@k_r) Kmaj", dcyc, dsink);
    test_tcgen05<128, 128, 16, 0, 1>("G3 M128 N128 K16 B MN-major", dcyc, dsink);
    test_tcgen05<128, 32, 128, 0, 0>("G1s M128 N32 K128 ([kd;qd] stacked)", dcyc, dsink);
    test_tcgen05<128, 64, 128, 0, 0>("ref M128 N64 K128", dcyc, dsink);
    test_tcgen05<128, 128, 128, 0, 0>("ref M128 N128 K128", dcyc, dsink);
    test_tcgen05<128, 256, 16, 0, 0>("ref M128 N256 K16", dcyc, dsink);
    test_tcgen05<64, 16, 128, 0, 0>("G1' M64 N16 K128", dcyc, dsink);
    test_tcgen05<64, 128, 128, 0, 0>("G1t M64 N128 K128 (kd@S, M=64 holds 16 rows)", dcyc, dsink);
    test_tcgen05<64, 128, 16, 0, 0>("G3' M64 N128 K16", dcyc, dsink);
    printf("== T7 SW128 layouts (as used by k2_tc v2) ==\n");
    test_tcgen05<128, 32, 128, 0, 0, 1>("SW128 A Kmaj M128 K128, B Kmaj N32", dcyc, dsink);
    test_tcgen05<128, 16, 16, 0, 0, 1>("SW128 M128 N16 K16", dcyc, dsink);
    test_tcgen05<128, 128, 16, 0, 1, 1>("SW128 B MN-major N128 K16 (a)", dcyc, dsink);
    test_tcgen05<128, 128, 16, 0, 2, 1>("SW128 B MN-major N128 K16 (b)", dcyc, dsink);
    test_tcgen05<128, 128, 128, 0, 0, 1>("SW128 M128 N128 K128", dcyc, dsink);
    printf("== T6 tcgen05.ld 128 lanes x C fp32 (4 warps) ==\n");
    for (int dep : {1, 0}) for (int c : {16, 32, 128}) {
        double cyc = run_cycles([&] { k_tmem_ld<<<1, 128>>>(c, 1000, dcyc, dsink, dep); }, dcyc, 1) / 1000;
        printf("   tcgen05.ld 32x32b.x16 + wait %s, %3d cols: %7.1f cyc per 128x%d fp32 tile (%5.1f B/cyc)\n", dep ? "(+16 dependent FADD)" : "(ld/wait only)      ", c, cyc, c, 128.0 * c * 4 / cyc);
    }
    return 0;
}
