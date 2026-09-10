// Challenge ("只换指令不动算法"): FlashKDA K2 (chunk recurrence) re-implemented with tcgen05.mma (SM100),
// consuming the UNCHANGED K1 workspace (k_decayed / q_decayed / k_restored / g_total / INV / Mqk per 16-token tile),
// same CHUNK=16, same bf16 state, same rounding points as the SM80 kernel except that the two out contributions
// are accumulated in fp32 inside TMEM instead of being added in bf16.
//
// Orientation: every GEMM is transposed so that D (=128) is the MMA M dimension and CHUNK=16 is N or K:
//   G1  [uT | outT] (128 x 32)  = S^T (128x128)  @ [kd ; qd]^T      M=128 N=32  K=128  (8 tcgen05.mma, one B tile)
//   G2  u2T (128 x 16)          = uT (128x16)    @ INV^T            M=128 N=16  K=16
//   G2' outT (128 x 16)        += UT (128x16)    @ Mqk^T            M=128 N=16  K=16   (accumulate into G1's cols 16..31)
//   G3  kU^T (128 x 128)        = UT (128x16)    @ k_r              M=128 N=128 K=16   (B is MN-major)
// S^T stays in smem (bf16, INTER/no-swizzle K-major, the same physical layout FlashKDA uses); the state update
// S^T = bf16(S^T * g[k] + kU^T) is done by the 128 threads (thread = row v) reading kU^T back from TMEM.
// One CTA (128 threads) per (sequence, head); non-varlen only (bench "fixed" case) — enough for the question asked.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {
constexpr int D = 128, C = 16, NTHREADS = 128;
constexpr int64_t WS_KD = C * D * 2, WS_QD = C * D * 2, WS_KR = C * D * 2, WS_GT = D * 4, WS_INV = C * C * 2, WS_MQK = C * C * 2;
constexpr int64_t WS_PER_TILE = WS_KD + WS_QD + WS_KR + WS_GT + WS_INV + WS_MQK;   // 13824 (WorkspaceSizes<16,128>)

// ---- smem map (bytes)
constexpr int SM_S = D * D * 2;                                                     // 32768 S^T INTER
constexpr int ST_KQ = 32 * D * 2, ST_KR = C * D * 2, ST_V = C * D * 2, ST_INV = 512, ST_MQK = 512, ST_GT = 512, ST_BETA = 128;
constexpr int OFF_KQ = 0, OFF_KR = OFF_KQ + ST_KQ, OFF_V = OFF_KR + ST_KR, OFF_INV = OFF_V + ST_V, OFF_MQK = OFF_INV + ST_INV,
              OFF_GT = OFF_MQK + ST_MQK, OFF_BETA = OFF_GT + ST_GT;
constexpr int ST_BYTES = OFF_BETA + ST_BETA;                                        // 18048
constexpr int STAGES = 2;
constexpr int SM_UT = D * C * 2;                                                    // 4096 (uT and UT tiles)
constexpr int SM_OUT = C * D * 2;
constexpr int SM_TOTAL = 1024 + SM_S + STAGES * ST_BYTES + 2 * SM_UT + SM_OUT;      // 1024 = alignment slack
// TMEM columns
constexpr int TM_G1 = 0, TM_G2 = 32, TM_G2B = 48, TM_G3 = 128, TM_COLS = 256;

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint64_t make_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    d |= (uint64_t)((saddr >> 4) & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;   // SM100 descriptor version; layout_type 0 = no swizzle
    return d;
}
__device__ __forceinline__ constexpr uint32_t make_idesc(int M, int N, int b_mn) {
    return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)b_mn << 16) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}
__device__ __forceinline__ void mma_ss(uint32_t tmem_d, uint64_t da, uint64_t db, uint32_t idesc, uint32_t accum) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
                 :: "r"(tmem_d), "l"(da), "l"(db), "r"(idesc), "r"(accum));
}
__device__ __forceinline__ void mma_commit(uint32_t mbar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(mbar) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t mbar, uint32_t phase) {
    uint32_t done = 0;
    while (!done)
        asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.b32 %0,1,0,p;\n}"
                     : "=r"(done) : "r"(mbar), "r"(phase));
}
__device__ __forceinline__ void tmem_ld16(uint32_t taddr, uint32_t* r) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                   "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
                 : "r"(taddr));
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}
__device__ __forceinline__ void cp_async16(uint32_t saddr, const void* g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(saddr), "l"(g));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ float bf2f(uint16_t b) { return __uint_as_float(((uint32_t)b) << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16(f)); }
__device__ __forceinline__ float sigmoid_tanh(float x) { float th; asm("tanh.approx.f32 %0, %1;" : "=f"(th) : "f"(x * 0.5f)); return th * 0.5f + 0.5f; }

// INTER (no-swizzle) K-major tile with R rows: byte offset of the 16-B chunk (row r, k-chunk c) = ((r/8) + c*(R/8))*128 + (r%8)*16
__device__ __forceinline__ int inter_off(int r, int c, int R) { return ((r >> 3) + c * (R >> 3)) * 128 + (r & 7) * 16; }

// issue cp.async for tile `tile` of head h into stage buffer `st`
__device__ __forceinline__ void load_tile(uint8_t* st, const uint8_t* ws, int64_t ws_idx, const __nv_bfloat16* v, int64_t v_row0, int H, int tid) {
    const uint8_t* tp = ws + ws_idx * WS_PER_TILE;
    const uint8_t* kd = tp; const uint8_t* qd = tp + WS_KD; const uint8_t* kr = qd + WS_QD;
    const uint8_t* gt = kr + WS_KR; const uint8_t* inv = gt + WS_GT; const uint8_t* mqk = inv + WS_INV;
    uint32_t sb = smem_u32(st);
    // KQ: 32 rows (kd 0..15, qd 16..31) x 16 chunks = 512 chunks
    for (int i = tid; i < 512; i += NTHREADS) {
        int n = i >> 4, c = i & 15;
        const uint8_t* src = (n < 16) ? kd + (n * 16 + c) * 16 : qd + ((n - 16) * 16 + c) * 16;
        cp_async16(sb + OFF_KQ + inter_off(n, c, 32), src);
    }
    for (int i = tid; i < 256; i += NTHREADS) {   // KR 16 x 16 chunks, INTER 16-row tile
        int t = i >> 4, c = i & 15;
        cp_async16(sb + OFF_KR + inter_off(t, c, 16), kr + i * 16);
    }
    for (int i = tid; i < 256; i += NTHREADS) {   // V rows t, 16 chunks, plain row-major [t][v]
        int t = i >> 4, c = i & 15;
        cp_async16(sb + OFF_V + i * 16, (const uint8_t*)(v + (v_row0 + t) * (int64_t)H * D) + c * 16);
    }
    if (tid < 32) cp_async16(sb + OFF_INV + inter_off(tid >> 1, tid & 1, 16), inv + tid * 16);     // 16x16 -> 2 chunks/row
    else if (tid < 64) { int j = tid - 32; cp_async16(sb + OFF_MQK + inter_off(j >> 1, j & 1, 16), mqk + j * 16); }
    else if (tid < 96) { int j = tid - 64; cp_async16(sb + OFF_GT + j * 16, gt + j * 16); }
}

template <bool TIMING>
__global__ void __launch_bounds__(NTHREADS) k2_tc_kernel(
    const uint8_t* __restrict__ ws, int total_tiles, const __nv_bfloat16* __restrict__ v, const __nv_bfloat16* __restrict__ beta,
    const __nv_bfloat16* __restrict__ h0, __nv_bfloat16* __restrict__ hT, __nv_bfloat16* __restrict__ out,
    int T, int H, long long* __restrict__ dbg, float* __restrict__ dump)
{
    extern __shared__ uint8_t smem_raw[];
    uint8_t* smem = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
    uint8_t* sS = smem;                                  // S^T INTER (128 rows v, K = k)
    uint8_t* sStage = sS + SM_S;                         // STAGES x ST_BYTES
    uint8_t* sUT0 = sStage + STAGES * ST_BYTES;          // uT tile (A operand of G2)
    uint8_t* sUT1 = sUT0 + SM_UT;                        // UT tile (A operand of G2', G3)
    uint8_t* sOut = sUT1 + SM_UT;                        // out staging [t][v]
    __shared__ uint64_t mbar[3];
    __shared__ uint32_t s_taddr;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int h = blockIdx.x;                            // head (N = 1)
    const int t_tiles = (T + C - 1) / C;
    const int64_t bos = 0;

    // --- TMEM alloc + barriers
    if (warp == 0) {
        if (lane == 0) {
            for (int i = 0; i < 3; i++) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem_u32(&mbar[i])));
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem_u32(&s_taddr)), "r"(TM_COLS));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    // --- initial state: h0[h][v][k] row-major -> INTER
    {
        const uint8_t* src = (const uint8_t*)(h0 + (int64_t)h * D * D);
        for (int i = tid; i < D * 16; i += NTHREADS) {   // 128 rows x 16 chunks
            int r = i >> 4, c = i & 15;
            *(uint4*)(sS + inter_off(r, c, 128)) = *(const uint4*)(src + i * 16);
        }
    }
    // --- prologue: load tile 0
    load_tile(sStage, ws, (int64_t)h * total_tiles + 0, v, bos, H, tid);
    cp_async_commit();
    if (tid < C) {   // beta for tile 0
        ((uint16_t*)(sStage + OFF_BETA))[tid] = __bfloat16_as_ushort(beta[(bos + tid) * (int64_t)H + h]);
    }
    __syncthreads();
    const uint32_t taddr = s_taddr;
    const uint32_t mb0 = smem_u32(&mbar[0]), mb1 = smem_u32(&mbar[1]), mb2 = smem_u32(&mbar[2]);
    uint32_t phase = 0;
    const uint32_t sS_a = smem_u32(sS), sUT0_a = smem_u32(sUT0), sUT1_a = smem_u32(sUT1);
    constexpr uint32_t ID_G1 = make_idesc(128, 32, 0), ID_G2 = make_idesc(128, 16, 0), ID_G3 = make_idesc(128, 128, 1);
    const uint32_t lane_base = taddr + ((uint32_t)(warp * 32) << 16);
    long long tm[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; long long tc = 0;
    auto stamp = [&](int i) { if (TIMING && tid == 0) { long long n = clock64(); tm[i] += n - tc; tc = n; } };

    for (int t = 0; t < t_tiles; ++t) {
        uint8_t* st = sStage + (t & 1) * ST_BYTES;
        const uint32_t st_a = smem_u32(st);
        cp_async_wait_all();
        fence_async_smem();          // generic-proxy smem writes (cp.async data, S update) -> visible to tcgen05
        __syncthreads();
        if (TIMING && tid == 0) tc = clock64();
        // prefetch next tile into the other stage (its previous consumers all finished before the barrier above)
        if (t + 1 < t_tiles) {
            uint8_t* nst = sStage + ((t + 1) & 1) * ST_BYTES;
            load_tile(nst, ws, (int64_t)h * total_tiles + t + 1, v, bos + (int64_t)(t + 1) * C, H, tid);
            cp_async_commit();
            if (tid < C) ((uint16_t*)(nst + OFF_BETA))[tid] = __bfloat16_as_ushort(beta[(bos + (int64_t)(t + 1) * C + tid) * (int64_t)H + h]);
        }
        if (dump && t == 1 && h == 0) {
            for (int i = tid; i < D * D; i += NTHREADS) { int r = i >> 7, k = i & 127; dump[40960 + i] = bf2f(((const uint16_t*)(sS + inter_off(r, k >> 3, 128)))[k & 7]); }
            for (int i = tid; i < 32 * D; i += NTHREADS) { int n = i >> 7, k = i & 127; dump[57344 + i] = bf2f(((const uint16_t*)(st + OFF_KQ + inter_off(n, k >> 3, 32)))[k & 7]); }
            for (int i = tid; i < 16 * D; i += NTHREADS) dump[61440 + i] = bf2f(((const uint16_t*)(st + OFF_V))[i]);
        }
        stamp(7);
        // ---- G1: [uT|outT] = S^T @ [kd;qd]^T  (M=128, N=32, K=128)
        if (tid == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;");
#pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                uint64_t da = make_desc(sS_a + ks * 2 * 2048, 2048, 128);            // S^T: 16 core mats along M -> LBO 2048
                uint64_t db = make_desc(st_a + OFF_KQ + ks * 2 * 512, 512, 128);    // KQ : 4 core mats along N -> LBO 512
                mma_ss(taddr + TM_G1, da, db, ID_G1, ks != 0);
            }
            mma_commit(mb0);
            if (TIMING) { long long n = clock64(); tm[8] += n - tc; }
        }
        mbar_wait(mb0, phase);
        asm volatile("tcgen05.fence::after_thread_sync;");
        stamp(0);
        // ---- E1: u[t] = bf16((v[t][vrow] - bf16(uT[t])) * beta[t]) ; write uT tile (A operand, row = my lane's v)
        {
            uint32_t r[16];
            tmem_ld16(lane_base + TM_G1, r);
            if (dump && t == 0 && h == 0) for (int j = 0; j < 16; ++j) dump[tid * 16 + j] = __uint_as_float(r[j]);
            const int vrow = tid;                            // thread = TMEM lane = v index (warp w covers 32w..32w+31)
            const uint16_t* V = (const uint16_t*)(st + OFF_V);
            const uint16_t* B = (const uint16_t*)(st + OFF_BETA);
            uint32_t packed[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                float b0 = sigmoid_tanh(bf2f(B[2 * j])), b1 = sigmoid_tanh(bf2f(B[2 * j + 1]));
                float u0 = bf2f(V[(2 * j) * D + vrow]) - bf2f(f2bf(__uint_as_float(r[2 * j])));
                float u1 = bf2f(V[(2 * j + 1) * D + vrow]) - bf2f(f2bf(__uint_as_float(r[2 * j + 1])));
                // FlashKDA: (v - u) in bf16, then * beta in bf16
                u0 = bf2f(f2bf(bf2f(f2bf(u0)) * bf2f(f2bf(b0))));
                u1 = bf2f(f2bf(bf2f(f2bf(u1)) * bf2f(f2bf(b1))));
                packed[j] = (uint32_t)f2bf(u0) | ((uint32_t)f2bf(u1) << 16);
            }
            uint8_t* dst = sUT0 + inter_off(vrow, 0, 128);
            *(uint4*)dst = make_uint4(packed[0], packed[1], packed[2], packed[3]);
            *(uint4*)(sUT0 + inter_off(vrow, 1, 128)) = make_uint4(packed[4], packed[5], packed[6], packed[7]);
        }
        fence_async_smem();
        __syncthreads();
        stamp(1);
        // ---- G2: u2T = uT @ INV^T (M=128, N=16, K=16)
        if (tid == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;");
            mma_ss(taddr + TM_G2, make_desc(sUT0_a, 2048, 128), make_desc(st_a + OFF_INV, 256, 128), ID_G2, 0);
            mma_commit(mb1);
        }
        mbar_wait(mb1, phase);
        asm volatile("tcgen05.fence::after_thread_sync;");
        stamp(2);
        // ---- E2: U = bf16(u2T) -> UT tile
        {
            uint32_t r[16];
            tmem_ld16(lane_base + TM_G2, r);
            uint32_t packed[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) packed[j] = (uint32_t)f2bf(__uint_as_float(r[2 * j])) | ((uint32_t)f2bf(__uint_as_float(r[2 * j + 1])) << 16);
            if (dump && t == 0 && h == 0) for (int j = 0; j < 16; ++j) dump[6144 + tid * 16 + j] = __uint_as_float(r[j]);
            *(uint4*)(sUT1 + inter_off(tid, 0, 128)) = make_uint4(packed[0], packed[1], packed[2], packed[3]);
            *(uint4*)(sUT1 + inter_off(tid, 1, 128)) = make_uint4(packed[4], packed[5], packed[6], packed[7]);
        }
        fence_async_smem();
        __syncthreads();
        stamp(3);
        // ---- G2' (outT += UT @ Mqk^T) and G3 (kU^T = UT @ k_r, B MN-major)
        if (tid == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;");
            mma_ss(taddr + TM_G2B, make_desc(sUT1_a, 2048, 128), make_desc(st_a + OFF_MQK, 256, 128), ID_G2, 0);
            mma_ss(taddr + TM_G3, make_desc(sUT1_a, 2048, 128), make_desc(st_a + OFF_KR, 128, 256), ID_G3, 0);
            mma_commit(mb2);
        }
        mbar_wait(mb2, phase);
        asm volatile("tcgen05.fence::after_thread_sync;");
        phase ^= 1;
        stamp(4);
        // ---- E3: out tile (bf16) -> staging [t][v]
        {
            uint32_t r[16], r2[16];
            tmem_ld16(lane_base + TM_G1 + 16, r);
            tmem_ld16(lane_base + TM_G2B, r2);
            uint16_t* O = (uint16_t*)sOut;
#pragma unroll
            for (int j = 0; j < 16; ++j) O[j * D + tid] = f2bf(bf2f(f2bf(__uint_as_float(r[j]))) + bf2f(f2bf(__uint_as_float(r2[j]))));
            if (dump && t == 0 && h == 0) { for (int j = 0; j < 16; ++j) { dump[2048 + tid * 16 + j] = __uint_as_float(r[j]); dump[4096 + tid * 16 + j] = __uint_as_float(r2[j]); } }
        }
        // ---- E4: S^T[v][k] = bf16(S^T[v][k] * g[k] + kU^T[v][k]); TMEM read in 2 batches of 4 x16 loads (one wait each)
        {
            const float* G = (const float*)(st + OFF_GT);
#pragma unroll
            for (int b = 0; b < 2; ++b) {
                uint32_t r[64];
#pragma unroll
                for (int q4 = 0; q4 < 4; ++q4) {
                    uint32_t* rr = r + q4 * 16;
                    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                                 : "=r"(rr[0]), "=r"(rr[1]), "=r"(rr[2]), "=r"(rr[3]), "=r"(rr[4]), "=r"(rr[5]), "=r"(rr[6]), "=r"(rr[7]),
                                   "=r"(rr[8]), "=r"(rr[9]), "=r"(rr[10]), "=r"(rr[11]), "=r"(rr[12]), "=r"(rr[13]), "=r"(rr[14]), "=r"(rr[15])
                                 : "r"(lane_base + TM_G3 + (b * 4 + q4) * 16));
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (dump && t == 0 && h == 0) for (int j = 0; j < 64; ++j) dump[8192 + tid * 128 + b * 64 + j] = __uint_as_float(r[j]);
                uint4 sv[8];
#pragma unroll
                for (int c8 = 0; c8 < 8; ++c8) sv[c8] = *(const uint4*)(sS + inter_off(tid, b * 8 + c8, 128));
                float g[64];
#pragma unroll
                for (int j = 0; j < 64; ++j) g[j] = G[b * 64 + j];
#pragma unroll
                for (int c8 = 0; c8 < 8; ++c8) {                   // 8 k-chunks of 8 in this batch
                    int c = b * 8 + c8;
                    uint32_t w[4] = {sv[c8].x, sv[c8].y, sv[c8].z, sv[c8].w};
                    uint32_t o[4];
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        float s0 = bf2f((uint16_t)(w[q] & 0xFFFF)), s1 = bf2f((uint16_t)(w[q] >> 16));
                        float n0 = s0 * g[c8 * 8 + 2 * q] + __uint_as_float(r[c8 * 8 + 2 * q]);
                        float n1 = s1 * g[c8 * 8 + 2 * q + 1] + __uint_as_float(r[c8 * 8 + 2 * q + 1]);
                        o[q] = (uint32_t)f2bf(n0) | ((uint32_t)f2bf(n1) << 16);
                    }
                    *(uint4*)(sS + inter_off(tid, c, 128)) = make_uint4(o[0], o[1], o[2], o[3]);
                    if (dump && t == 0 && h == 0) for (int q = 0; q < 4; ++q) { dump[8192 + 16384 + tid * 128 + c * 8 + 2 * q] = bf2f((uint16_t)(o[q] & 0xFFFF)); dump[8192 + 16384 + tid * 128 + c * 8 + 2 * q + 1] = bf2f((uint16_t)(o[q] >> 16)); }
                }
            }
        }
        asm volatile("tcgen05.fence::before_thread_sync;");
        __syncthreads();
        stamp(5);
        if (dump && t == 0 && h == 0) for (int i = tid; i < 16 * D; i += NTHREADS) dump[63488 + i] = bf2f(((const uint16_t*)sOut)[i]);
        // ---- out staging -> gmem (coalesced 16-B chunks)
        {
            uint8_t* og = (uint8_t*)(out + (bos + (int64_t)t * C) * (int64_t)H * D + (int64_t)h * D);
            for (int i = tid; i < 256; i += NTHREADS) {
                int tt = i >> 4, c = i & 15;
                *(uint4*)(og + (int64_t)tt * H * D * 2 + c * 16) = *(const uint4*)(sOut + i * 16);
            }
        }
        stamp(6);
    }
    __syncthreads();
    // --- final state S^T INTER -> hT[h][v][k] row-major
    {
        uint8_t* dst = (uint8_t*)(hT + (int64_t)h * D * D);
        for (int i = tid; i < D * 16; i += NTHREADS) {
            int r = i >> 4, c = i & 15;
            *(uint4*)(dst + i * 16) = *(const uint4*)(sS + inter_off(r, c, 128));
        }
    }
    if (TIMING && tid == 0) for (int i = 0; i < 10; i++) dbg[h * 10 + i] = tm[i];
    __syncthreads();
    if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(TM_COLS));
}
}  // namespace

void k2_tc_fwd(torch::Tensor workspace, int64_t total_tiles, torch::Tensor v, torch::Tensor beta, torch::Tensor h0,
               torch::Tensor hT, torch::Tensor out, torch::Tensor dbg, bool timing, torch::Tensor dump) {
    TORCH_CHECK(v.dim() == 4 && v.size(0) == 1 && v.size(3) == D, "v must be [1,T,H,128]");
    TORCH_CHECK(v.dtype() == torch::kBFloat16 && beta.dtype() == torch::kBFloat16 && h0.dtype() == torch::kBFloat16);
    int T = v.size(1), H = v.size(2);
    TORCH_CHECK(T % C == 0, "T must be a multiple of 16 (non-varlen fixed case)");
    TORCH_CHECK(total_tiles == T / C);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto kern = timing ? k2_tc_kernel<true> : k2_tc_kernel<false>;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_TOTAL);
    kern<<<H, NTHREADS, SM_TOTAL, stream>>>(
        workspace.data_ptr<uint8_t>(), (int)total_tiles, (const __nv_bfloat16*)v.data_ptr(), (const __nv_bfloat16*)beta.data_ptr(),
        (const __nv_bfloat16*)h0.data_ptr(), (__nv_bfloat16*)hT.data_ptr(), (__nv_bfloat16*)out.data_ptr(), T, H,
        timing ? reinterpret_cast<long long*>(dbg.data_ptr<int64_t>()) : nullptr,
        dump.numel() > 0 ? dump.data_ptr<float>() : nullptr);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "k2_tc launch failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("k2_tc_fwd", &k2_tc_fwd, "FlashKDA K2 with tcgen05 (SM100)");
    m.attr("SMEM_BYTES") = SM_TOTAL;
}
