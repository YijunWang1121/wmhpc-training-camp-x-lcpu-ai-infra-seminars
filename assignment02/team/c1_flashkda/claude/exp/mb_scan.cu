// E13 microbench: 分层仿射扫描的"稠密 compose 链步"在 tcgen05 上的代价。
//   每步  S_new^T[v][k] = sum_{k'} S^T[v][k'] M_i[k'][k]  +  sum_t u_i^T[v][t] K'_i[t][k]
//        = A[128 x 144] @ B[128 x 144]^T,  A = [S^T | u^T] 常驻 TMEM(bf16,72 列),B = [M_i^T | K'_i] 在 smem(K-major INTER)
//   9 条 tcgen05.mma kind::f16, A 来自 TMEM(TS 形式)-> commit -> mbarrier 等待 -> 4 warp 把 fp32 累加器(128 列)
//   重打包成 bf16 写回 TMEM 的 A 区(tcgen05.ld -> cvt -> tcgen05.st)-> 下一步。TMEM 共 200 列 -> 每 SM 可 2 个 CTA。
// mode 0: 正确性(2 步,对 CPU 参考,host 同样在步间把 S 舍入到 bf16);mode 1: 计时 iters 步。
// build: nvcc -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a -o mb_scan mb_scan.cu
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <cmath>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA error %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)

constexpr int M = 128, N = 128, KS = 128, KU = 16, K = KS + KU;   // 144
constexpr int TM_ACC = 0, TM_A = 128, TM_COLS = 256;                // A: 72 cols at 128..199

__host__ __device__ inline int kinter(int r, int c, int R) { return ((r >> 3) + (c >> 3) * (R >> 3)) * 64 + (r & 7) * 8 + (c & 7); }
__device__ inline uint64_t make_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0; d |= (uint64_t)((saddr >> 4) & 0x3FFF); d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32; d |= (uint64_t)1 << 46; return d;
}
__device__ inline uint32_t make_idesc(int m, int n) { return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)(n >> 3) << 17) | ((uint32_t)(m >> 4) << 24); }
__device__ inline void mma_ts(uint32_t tmem_d, uint32_t tmem_a, uint64_t db, uint32_t idesc, uint32_t accum) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, {%5, %6, %7, %8}, p;\n}"
                 :: "r"(tmem_d), "r"(tmem_a), "l"(db), "r"(idesc), "r"(accum), "r"(0), "r"(0), "r"(0), "r"(0));
}
__device__ inline void mma_commit(uint32_t mbar) { asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(mbar) : "memory"); }
__device__ inline void mbar_wait(uint32_t mbar, uint32_t phase) {
    uint32_t done = 0;
    while (!done) asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.b32 %0,1,0,p;\n}" : "=r"(done) : "r"(mbar), "r"(phase));
}
__device__ inline uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ inline void tmem_ld32(uint32_t taddr, uint32_t* r) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                 : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),"=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
                   "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),"=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
                 : "r"(taddr));
}
__device__ inline void tmem_st16(uint32_t taddr, const uint32_t* r) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
                 :: "r"(taddr), "r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]),"r"(r[8]),"r"(r[9]),"r"(r[10]),"r"(r[11]),"r"(r[12]),"r"(r[13]),"r"(r[14]),"r"(r[15]));
}
__device__ inline uint32_t pack2(float a, float b) { uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(a), "f"(b)); return r; }

// gB: [N=128][K=144] bf16 K-major logical (row n = output k, cols k' then t);  gS0: [128 v][128 k'] bf16; gU: [128 v][16 t] bf16
__global__ void k_scan(const __nv_bfloat16* gB, const __nv_bfloat16* gS0, const __nv_bfloat16* gU, float* gD, int mode, int iters, long long* cyc) {
    extern __shared__ __align__(1024) uint8_t smem[];
    __nv_bfloat16* sB = (__nv_bfloat16*)smem;          // N x K INTER K-major: 128 x 144 -> 36 KB
    __shared__ uint64_t mbar; __shared__ uint32_t s_taddr;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const uint32_t mb = smem_u32(&mbar);
    if (warp == 0) {
        if (tid == 0) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mb)); asm volatile("fence.mbarrier_init.release.cluster;"); }
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem_u32(&s_taddr)), "r"(TM_COLS));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    for (int i = tid; i < N * K; i += blockDim.x) { int n = i / K, k = i % K; sB[kinter(n, k, N)] = gB[i]; }
    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();
    const uint32_t taddr = s_taddr;
    const uint32_t lane_base = taddr + ((uint32_t)(warp * 32) << 16);
    const int row = warp * 32 + lane;
    // A init in TMEM: lane = v, col c holds k = 2c (low) | 2c+1 (high)  [guess; validated by mode 0]
    {
        uint32_t r[16];
        for (int c0 = 0; c0 < 64; c0 += 16) {
            for (int j = 0; j < 16; ++j) { int k = 2 * (c0 + j); r[j] = pack2(__bfloat162float(gS0[row * KS + k]), __bfloat162float(gS0[row * KS + k + 1])); }
            tmem_st16(lane_base + TM_A + c0, r);
        }
        for (int j = 0; j < 8; ++j) r[j] = pack2(__bfloat162float(gU[row * KU + 2 * j]), __bfloat162float(gU[row * KU + 2 * j + 1]));
        for (int j = 8; j < 16; ++j) r[j] = 0;
        tmem_st16(lane_base + TM_A + 64, r);     // cols 64..71 = u^T (72..79 zero, unused)
        asm volatile("tcgen05.wait::st.sync.aligned;");
    }
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    const uint32_t idesc = make_idesc(M, N);
    const uint32_t bBase = smem_u32(sB);
    uint32_t phase = 0;
    const int steps = mode == 0 ? 2 : iters;
    long long t0 = clock64();
    for (int it = 0; it < steps; ++it) {
        if (warp == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (lane == 0) {
                for (int ks = 0; ks < K / 16; ++ks)      // 9 k16 steps: A cols advance 8 per k16; B K-major INTER: 2 core mats per k16
                    mma_ts(taddr + TM_ACC, taddr + TM_A + ks * 8, make_desc(bBase + ks * 2 * (N / 8) * 128, (N / 8) * 128, 128), idesc, ks != 0);
                mma_commit(mb);
            }
            __syncwarp();
        }
        mbar_wait(mb, phase); phase ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;");
        // repack: acc fp32 (128 cols) -> bf16 pairs -> A cols 0..63 (in TMEM)
        uint32_t r[32], o[16];
        for (int c0 = 0; c0 < 128; c0 += 32) {
            tmem_ld32(lane_base + TM_ACC + c0, r);
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            for (int j = 0; j < 16; ++j) o[j] = pack2(__uint_as_float(r[2 * j]), __uint_as_float(r[2 * j + 1]));
            tmem_st16(lane_base + TM_A + c0 / 2, o);
        }
        asm volatile("tcgen05.wait::st.sync.aligned;");
        asm volatile("tcgen05.fence::before_thread_sync;");
        __syncthreads();
    }
    long long t1 = clock64();
    if (mode == 0) {    // dump S after 2 steps (bf16-rounded, as stored in A) as float
        uint32_t r[32];
        for (int c0 = 0; c0 < 64; c0 += 32) {
            tmem_ld32(lane_base + TM_A + c0, r); asm volatile("tcgen05.wait::ld.sync.aligned;");
            for (int j = 0; j < 32; ++j) { gD[row * KS + 2 * (c0 + j)] = __uint_as_float(r[j] << 16); gD[row * KS + 2 * (c0 + j) + 1] = __uint_as_float(r[j] & 0xFFFF0000u); }
        }
    }
    __syncthreads();
    if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(TM_COLS));
    if (tid == 0) cyc[blockIdx.x] = t1 - t0;
}

static float bf(float x) { return __bfloat162float(__float2bfloat16(x)); }
int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0)); int nsm = p.multiProcessorCount;
    printf("device %s cc %d.%d SMs %d\n", p.name, p.major, p.minor, nsm);
    std::mt19937 rng(3); std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<__nv_bfloat16> hB(N * K), hS(M * KS), hU(M * KU);
    std::vector<float> B(N * K), S(M * KS), U(M * KU);
    for (int i = 0; i < N * K; ++i) { float v = dist(rng) * (i % K < KS ? 0.09f : 0.25f); hB[i] = __float2bfloat16(v); B[i] = bf(v); }
    for (int i = 0; i < M * KS; ++i) { float v = dist(rng); hS[i] = __float2bfloat16(v); S[i] = bf(v); }
    for (int i = 0; i < M * KU; ++i) { float v = dist(rng); hU[i] = __float2bfloat16(v); U[i] = bf(v); }
    // CPU reference: 2 steps, S rounded to bf16 between steps
    std::vector<float> ref = S;
    for (int st = 0; st < 2; ++st) {
        std::vector<float> nx(M * KS);
        for (int v = 0; v < M; ++v) for (int k = 0; k < N; ++k) {
            float a = 0; for (int kp = 0; kp < KS; ++kp) a += ref[v * KS + kp] * B[k * K + kp];
            for (int t = 0; t < KU; ++t) a += U[v * KU + t] * B[k * K + KS + t];
            nx[v * KS + k] = bf(a);
        }
        ref = nx;
    }
    __nv_bfloat16 *dB, *dS, *dU; float* dD; long long* dcyc;
    CK(cudaMalloc(&dB, N * K * 2)); CK(cudaMalloc(&dS, M * KS * 2)); CK(cudaMalloc(&dU, M * KU * 2)); CK(cudaMalloc(&dD, M * KS * 4)); CK(cudaMalloc(&dcyc, 4096 * 8));
    CK(cudaMemcpy(dB, hB.data(), N * K * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dS, hS.data(), M * KS * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dU, hU.data(), M * KU * 2, cudaMemcpyHostToDevice));
    size_t smem = (size_t)N * K * 2 + 1024;
    CK(cudaFuncSetAttribute(k_scan, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    k_scan<<<1, 128, smem>>>(dB, dS, dU, dD, 0, 2, dcyc); CK(cudaDeviceSynchronize());
    std::vector<float> got(M * KS); CK(cudaMemcpy(got.data(), dD, M * KS * 4, cudaMemcpyDeviceToHost));
    double md = 0, mr = 0; long bad = 0;
    for (int i = 0; i < M * KS; ++i) { double d = fabs(got[i] - ref[i]); md = fmax(md, d); mr = fmax(mr, fabs(ref[i])); if (d > 1e-2 * fmax(1.0, fabs(ref[i]))) { if (bad < 4) printf("  MISMATCH [%d,%d] got %g want %g\n", i / KS, i % KS, got[i], ref[i]); bad++; } }
    printf("correctness (2 dependent steps, A from TMEM, bf16 repack): max|diff| %.3e (max|ref| %.3e)  %s\n", md, mr, bad ? "FAIL" : "PASS");
    if (bad) return 1;
    auto run = [&](int nblk, int iters) {
        k_scan<<<nblk, 128, smem>>>(dB, dS, dU, dD, 1, iters, dcyc); CK(cudaDeviceSynchronize());
        k_scan<<<nblk, 128, smem>>>(dB, dS, dU, dD, 1, iters, dcyc); CK(cudaDeviceSynchronize());
        std::vector<long long> h(nblk); CK(cudaMemcpy(h.data(), dcyc, nblk * 8, cudaMemcpyDeviceToHost));
        double s = 0; for (auto x : h) s += x; return s / nblk / iters;
    };
    const int iters = 2000;
    double c1 = run(1, iters), cN = run(nsm, iters), c2N = run(2 * nsm, iters), c4N = run(4 * nsm, iters);
    printf("cycles per compose step (9 x tcgen05.mma K16 TS + commit/wait + fp32->bf16 TMEM repack):\n");
    printf("   1 CTA on 1 SM      : %7.1f cyc/step   (%.0f MAC/cyc)\n", c1, (double)M * N * K / c1);
    printf("   1 CTA per SM (all) : %7.1f cyc/step\n", cN);
    printf("   2 CTA per SM       : %7.1f cyc/step per CTA  -> SM throughput %.0f cyc/step\n", c2N, c2N / 2);
    printf("   4 CTA per SM (queued, TMEM allows 2): %7.1f cyc/step per CTA\n", c4N);
    printf("reference: FlashKDA K2 mma.sync chunk = 2732 cyc (1 CTA/SM), 2139 cyc/chunk-wave at 2 CTA/SM (E10)\n");
    return 0;
}
