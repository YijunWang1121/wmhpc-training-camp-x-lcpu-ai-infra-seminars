// k2_tc v2: FlashKDA K2 with tcgen05 — corrected workspace layout (six separate arrays, FlashKDA fwd_launch.cu)
// and TMA loads (cp.async.bulk.tensor 2-D/3-D boxes + 1-D bulk copies) instead of per-thread cp.async.
// Layouts: S (128x128) / KQ ([kd;qd] 32x128) / KR (16x128) / V (16x128) are SWIZZLE_128B tiles produced natively by
// TMA (box = 64 elems x rows); uT/UT (128x16) are thread-written INTER tiles; INV/Mqk are 1-D bulk-loaded row-major
// and relaid to INTER by 32 threads.  GEMM orientation as in v1 (D=128 is M, CHUNK is N or K).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {
constexpr int D = 128, C = 16, NTHREADS = 128;
constexpr int64_t WS_KD = C * D * 2, WS_GT = D * 4, WS_LM = C * C * 2;   // per-tile region sizes (4096 / 512 / 512)

// ---- smem map (bytes).  SW128 tile of R rows and 128 k: two halves (k<64, k>=64) of R*128 B each.
constexpr int SM_S = D * D * 2;                                           // 32768 = 2 halves x 16 KB
constexpr int ST_KQ = 32 * D * 2, ST_KR = C * D * 2, ST_V = C * D * 2, ST_RAW = 3 * 512, ST_INV = 512, ST_MQK = 512, ST_BETA = 128;
constexpr int OFF_KQ = 0, OFF_KR = OFF_KQ + ST_KQ, OFF_V = OFF_KR + ST_KR, OFF_RAW = OFF_V + ST_V,
              OFF_INV = OFF_RAW + ST_RAW, OFF_MQK = OFF_INV + ST_INV, OFF_BETA = OFF_MQK + ST_MQK;
constexpr int ST_BYTES = ((OFF_BETA + ST_BETA + 1023) / 1024) * 1024;      // 19456: SW128 tiles need 1024-B aligned stage bases
constexpr int STAGES = 2;
constexpr int SM_UT = D * C * 2;
constexpr int SM_OUT = C * D * 2;
constexpr int SM_TOTAL = 1024 + SM_S + STAGES * ST_BYTES + 2 * SM_UT + SM_OUT;
constexpr uint32_t TX_BYTES = 3 * 4096 + 4096 + 3 * 512 + 32;            // per-chunk TMA bytes = 17952 (incl. beta)
constexpr int TM_G1 = 0, TM_G2 = 32, TM_G2B = 48, TM_G3 = 128, TM_COLS = 256;
#ifndef KR_LBO
#define KR_LBO 2048   // hypothesis (a): MN-major SW128 LBO = MN-block stride
#endif
#ifndef KR_SBO
#define KR_SBO 1024   //                 SBO = 8-row K-group stride
#endif

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint64_t make_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo, uint32_t layout) {
    uint64_t d = 0;
    d |= (uint64_t)((saddr >> 4) & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)layout << 61;
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
__device__ __forceinline__ void mbar_expect_tx(uint32_t mbar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(mbar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void tma_2d(uint32_t dst, const void* tmap, int c0, int c1, uint32_t mbar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(dst), "l"(reinterpret_cast<uint64_t>(tmap)), "r"(c0), "r"(c1), "r"(mbar) : "memory");
}
__device__ __forceinline__ void tma_3d(uint32_t dst, const void* tmap, int c0, int c1, int c2, uint32_t mbar) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
                 :: "r"(dst), "l"(reinterpret_cast<uint64_t>(tmap)), "r"(c0), "r"(c1), "r"(c2), "r"(mbar) : "memory");
}
__device__ __forceinline__ void bulk_1d(uint32_t dst, const void* src, uint32_t bytes, uint32_t mbar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(mbar) : "memory");
}
__device__ __forceinline__ void tmem_ld16(uint32_t taddr, uint32_t* r) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                   "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
                 : "r"(taddr));
}
__device__ __forceinline__ void tmem_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;"); }
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ float bf2f(uint16_t b) { return __uint_as_float(((uint32_t)b) << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16(f)); }
// packed round-to-nearest fp32x2 -> bf16x2 (cvt.rn.bf16x2.f32 -> F2FP.PACK_AB, fast pipe); lo = a, hi = b
__device__ __forceinline__ uint32_t pack2(float a, float b) { uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float lo_f(uint32_t p) { return __uint_as_float(p << 16); }
__device__ __forceinline__ float hi_f(uint32_t p) { return __uint_as_float(p & 0xFFFF0000u); }
__device__ __forceinline__ float sigmoid_tanh(float x) { float th; asm("tanh.approx.f32 %0, %1;" : "=f"(th) : "f"(x * 0.5f)); return th * 0.5f + 0.5f; }
// INTER K-major tile with R rows: 16-B chunk (row r, k-chunk c) at ((r/8) + c*(R/8))*128 + (r%8)*16
__device__ __forceinline__ int inter_off(int r, int c, int R) { return ((r >> 3) + c * (R >> 3)) * 128 + (r & 7) * 16; }
// SW128 tile with R rows: 16-B chunk c (0..15) of row r: half c/8 (R*128 B each), row r*128, chunk (c%8)^(r%8)
__device__ __forceinline__ int sw_off(int r, int c, int R) { return (c >> 3) * R * 128 + r * 128 + (((c & 7) ^ (r & 7)) << 4); }

struct Params {
    const uint8_t* ws; const __nv_bfloat16* beta_t; __nv_bfloat16* hT; __nv_bfloat16* out;
    int T, H, total_tiles; long long* dbg; float* dump;
};

template <bool TIMING>
__global__ void __launch_bounds__(NTHREADS) k2_tc2_kernel(
    const __grid_constant__ CUtensorMap tm_ws, const __grid_constant__ CUtensorMap tm_v, const __grid_constant__ CUtensorMap tm_h0,
    Params p)
{
    extern __shared__ uint8_t smem_raw[];
    uint8_t* smem = smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u);   // 1024-aligned, still a __shared__ pointer -> LDS/STS
    uint8_t* sS = smem;
    uint8_t* sStage = sS + SM_S;
    uint8_t* sUT0 = sStage + STAGES * ST_BYTES;
    uint8_t* sUT1 = sUT0 + SM_UT;
    uint8_t* sOut = sUT1 + SM_UT;
    __shared__ __align__(8) uint64_t mbar[3];       // MMA commit barriers
    __shared__ __align__(8) uint64_t ldbar[2];      // TMA stage barriers
    __shared__ __align__(8) uint64_t sbar;          // initial-state barrier
    __shared__ uint32_t s_taddr;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int h = blockIdx.x;
    const int T = p.T, H = p.H, total_tiles = p.total_tiles;
    const int t_tiles = T / C;
    const int64_t n_ht = (int64_t)H * total_tiles;
    const uint8_t* ws = p.ws;
    long long* dbg = p.dbg; float* dump = p.dump;

    if (warp == 0) {
        if (lane == 0) {
            for (int i = 0; i < 3; i++) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem_u32(&mbar[i])));
            for (int i = 0; i < 2; i++) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem_u32(&ldbar[i])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem_u32(&sbar)));
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem_u32(&s_taddr)), "r"(TM_COLS));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    __syncthreads();
    const uint32_t mb0 = smem_u32(&mbar[0]), mb1 = smem_u32(&mbar[1]), mb2 = smem_u32(&mbar[2]);
    const uint32_t lb0 = smem_u32(&ldbar[0]), lb1 = smem_u32(&ldbar[1]), sb = smem_u32(&sbar);

    // issue all loads of tile `tile` into stage buffer st (thread 0 only)
    auto issue_tile = [&](uint8_t* st, int tile, uint32_t bar) {
        const uint32_t sa = smem_u32(st);
        const int64_t ws_idx = (int64_t)h * total_tiles + tile;
        mbar_expect_tx(bar, TX_BYTES);
        // kd -> KQ rows 0..15, qd -> KQ rows 16..31 (per half: 32 rows x 128 B = 4096 B; kd at +0, qd at +2048)
        for (int hf = 0; hf < 2; ++hf) {
            tma_2d(sa + OFF_KQ + hf * 4096, &tm_ws, hf * 64, (int)(ws_idx * 16), bar);                       // kd
            tma_2d(sa + OFF_KQ + hf * 4096 + 2048, &tm_ws, hf * 64, (int)((n_ht + ws_idx) * 16), bar);       // qd
            tma_2d(sa + OFF_KR + hf * 2048, &tm_ws, hf * 64, (int)((2 * n_ht + ws_idx) * 16), bar);          // kr
            tma_3d(sa + OFF_V + hf * 2048, &tm_v, hf * 64, h, tile * C, bar);                                // v rows
        }
        bulk_1d(sa + OFF_RAW, ws + n_ht * 3 * WS_KD + ws_idx * WS_GT, 512, bar);                            // g_total (fp32)
        bulk_1d(sa + OFF_RAW + 512, ws + n_ht * (3 * WS_KD + WS_GT) + ws_idx * WS_LM, 512, bar);            // INV row-major
        bulk_1d(sa + OFF_RAW + 1024, ws + n_ht * (3 * WS_KD + WS_GT + WS_LM) + ws_idx * WS_LM, 512, bar);   // Mqk row-major
        bulk_1d(sa + OFF_BETA, p.beta_t + (int64_t)h * T + (int64_t)tile * C, 32, bar);                     // beta_t[h][t0..t0+15]
    };
    // --- prologue: initial state (TMA, 2 halves of 128 rows x 128 B) + tile 0
    if (tid == 0) {
        mbar_expect_tx(sb, SM_S);
        tma_2d(smem_u32(sS), &tm_h0, 0, h * D, sb);
        tma_2d(smem_u32(sS) + 16384, &tm_h0, 64, h * D, sb);
        issue_tile(sStage, 0, lb0);
    }
    mbar_wait(sb, 0);

    const uint32_t taddr = s_taddr;
    uint32_t phase = 0;
    const uint32_t sS_a = smem_u32(sS), sUT0_a = smem_u32(sUT0), sUT1_a = smem_u32(sUT1);
    constexpr uint32_t ID_G1 = make_idesc(128, 32, 0), ID_G2 = make_idesc(128, 16, 0), ID_G3 = make_idesc(128, 128, 1);
    const uint32_t lane_base = taddr + ((uint32_t)(warp * 32) << 16);
    long long tm[16] = {0}; long long tc = 0;
    auto stamp = [&](int i) { if (TIMING && tid == 0) { long long n = clock64(); tm[i] += n - tc; tc = n; } };

    for (int t = 0; t < t_tiles; ++t) {
        uint8_t* st = sStage + (t & 1) * ST_BYTES;
        const uint32_t tmA = (t & 1) * 64;                 // this chunk's accumulator set (G1 @ tmA, G2 @ tmA+32, G2B @ tmA+48)
        const uint32_t st_a = smem_u32(st);
        const uint32_t lbar = (t & 1) ? lb1 : lb0;
        if (TIMING && tid == 0) tc = clock64();
        mbar_wait(lbar, (t >> 1) & 1);                    // tile t landed (TMA = async proxy, no fence needed for MMA)
        stamp(9);
        // relayout INV / Mqk (row-major 16x16, 32 B rows) -> INTER for the B operand; 32 threads each
        if (tid < 64) {
            int which = tid >> 5, j = tid & 31, r = j >> 1, c = j & 1;
            const uint8_t* src = st + OFF_RAW + 512 + which * 512 + j * 16;
            *(uint4*)(st + (which ? OFF_MQK : OFF_INV) + inter_off(r, c, 16)) = *(const uint4*)src;
        }
        fence_async_smem();                                // relayout + S update (generic writes) -> async proxy
        __syncthreads();
        // prefetch tile t+1 into the other stage (freed: its MMAs finished at the previous mb2 wait, threads passed the barrier)
        if (t + 1 < t_tiles) {
            uint8_t* nst = sStage + ((t + 1) & 1) * ST_BYTES;
            if (tid == 32) issue_tile(nst, t + 1, (t & 1) ? lb0 : lb1);
        }
        if (dump && t == 1 && h == 0) {
            for (int i = tid; i < D * D; i += NTHREADS) { int r = i >> 7, k = i & 127; dump[40960 + i] = bf2f(((const uint16_t*)(sS + sw_off(r, k >> 3, 128)))[k & 7]); }
            for (int i = tid; i < 32 * D; i += NTHREADS) { int n = i >> 7, k = i & 127; dump[57344 + i] = bf2f(((const uint16_t*)(st + OFF_KQ + sw_off(n, k >> 3, 32)))[k & 7]); }
            for (int i = tid; i < 16 * D; i += NTHREADS) { int n = i >> 7, k = i & 127; dump[61440 + i] = bf2f(((const uint16_t*)(st + OFF_V + sw_off(n, k >> 3, 16)))[k & 7]); }
        }
        stamp(7);
        // ---- G1: [uT|outT] = S^T @ [kd;qd]^T  (M=128, N=32, K=128), SW128 K-major both
        if (tid == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;");
#pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                uint64_t da = make_desc(sS_a + (ks >> 2) * 16384 + (ks & 3) * 32, 16, 1024, 2);
                uint64_t db = make_desc(st_a + OFF_KQ + (ks >> 2) * 4096 + (ks & 3) * 32, 16, 1024, 2);
                mma_ss(taddr + tmA + TM_G1, da, db, ID_G1, ks != 0);
            }
            mma_commit(mb0);
            if (TIMING) { long long n = clock64(); tm[8] += n - tc; }
        }
        if (t > 0) {   // deferred epilogue of chunk t-1 (out tile) overlaps this chunk's G1
            const uint32_t tmP = ((t - 1) & 1) * 64;
        // ---- E3: out tile = bf16(bf16(outT) + bf16(out2)) -> staging [t][v]
        {
            uint32_t r[16], r2[16];
            uint16_t* O = (uint16_t*)sOut;
#ifdef K2TC_NO_E3
            if (false) {
#else
            tmem_ld16(lane_base + tmP + TM_G1 + 16, r); tmem_ld16(lane_base + tmP + TM_G2B, r2); tmem_wait();
            {
#endif
#pragma unroll
            for (int j = 0; j < 16; j += 2) {
                uint32_t a = pack2(__uint_as_float(r[j]), __uint_as_float(r[j + 1])), b = pack2(__uint_as_float(r2[j]), __uint_as_float(r2[j + 1]));
                uint32_t o = pack2(lo_f(a) + lo_f(b), hi_f(a) + hi_f(b));
                O[j * D + tid] = (uint16_t)(o & 0xFFFF); O[(j + 1) * D + tid] = (uint16_t)(o >> 16);
            }
            }
            if (dump && t == 1 && h == 0) { for (int j = 0; j < 16; ++j) { dump[2048 + tid * 16 + j] = __uint_as_float(r[j]); dump[4096 + tid * 16 + j] = __uint_as_float(r2[j]); } }
        }
        stamp(10);
            __syncthreads();
        if (dump && t == 1 && h == 0) for (int i = tid; i < 16 * D; i += NTHREADS) dump[63488 + i] = bf2f(((const uint16_t*)sOut)[i]);
        // ---- out staging -> gmem
        {
            uint8_t* og = (uint8_t*)(p.out + ((int64_t)(t - 1) * C) * H * D + (int64_t)h * D);
            for (int i = tid; i < 256; i += NTHREADS) {
                int tt = i >> 4, c = i & 15;
                *(uint4*)(og + (int64_t)tt * H * D * 2 + c * 16) = *(const uint4*)(sOut + i * 16);
            }
        }
        stamp(6);
        }
        mbar_wait(mb0, phase);
        asm volatile("tcgen05.fence::after_thread_sync;");
        stamp(0);
        // ---- E1: u[t] = bf16((v[t][v] - bf16(uT[t])) * beta[t]) -> uT tile (INTER, row = v)
        {
            uint32_t r[16];
            tmem_ld16(lane_base + tmA + TM_G1, r); tmem_wait();
            stamp(11 + 1);
            if (dump && t == 0 && h == 0) for (int j = 0; j < 16; ++j) dump[tid * 16 + j] = __uint_as_float(r[j]);
            const int vrow = tid;
            const uint16_t* B = (const uint16_t*)(st + OFF_BETA);
            float vv[16];
            uint32_t packed[8];
#ifdef K2TC_NO_E1
#pragma unroll
            for (int j = 0; j < 8; ++j) packed[j] = r[2 * j] & 0x3f803f80u;
            if (false)
#else
#pragma unroll
            for (int j = 0; j < 16; ++j) vv[j] = bf2f(*(const uint16_t*)(st + OFF_V + sw_off(j, vrow >> 3, 16) + (vrow & 7) * 2));
#pragma unroll
#endif
            for (int j = 0; j < 8; ++j) {
                uint32_t bb = pack2(sigmoid_tanh(bf2f(B[2 * j])), sigmoid_tanh(bf2f(B[2 * j + 1])));      // beta in bf16
                uint32_t uu = pack2(__uint_as_float(r[2 * j]), __uint_as_float(r[2 * j + 1]));            // bf16(u_acc)
                uint32_t dd = pack2(vv[2 * j] - lo_f(uu), vv[2 * j + 1] - hi_f(uu));                      // bf16(v - u)
                packed[j] = pack2(lo_f(dd) * lo_f(bb), hi_f(dd) * hi_f(bb));                              // bf16(.. * beta)
            }
            *(uint4*)(sUT0 + inter_off(vrow, 0, 128)) = make_uint4(packed[0], packed[1], packed[2], packed[3]);
            *(uint4*)(sUT0 + inter_off(vrow, 1, 128)) = make_uint4(packed[4], packed[5], packed[6], packed[7]);
        }
        stamp(13);
        fence_async_smem();
        stamp(14);
        __syncthreads();
        stamp(1);
        // ---- G2: u2T = uT @ INV^T (M=128, N=16, K=16); A INTER (LBO 2048, SBO 128), B INTER (LBO 256, SBO 128)
        if (tid == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;");
            mma_ss(taddr + tmA + TM_G2, make_desc(sUT0_a, 2048, 128, 0), make_desc(st_a + OFF_INV, 256, 128, 0), ID_G2, 0);
            mma_commit(mb1);
        }
        mbar_wait(mb1, phase);
        asm volatile("tcgen05.fence::after_thread_sync;");
        stamp(2);
        // ---- E2: U = bf16(u2T) -> UT tile
        {
            uint32_t r[16];
            tmem_ld16(lane_base + tmA + TM_G2, r); tmem_wait();
            uint32_t packed[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) packed[j] = pack2(__uint_as_float(r[2 * j]), __uint_as_float(r[2 * j + 1]));
            if (dump && t == 0 && h == 0) for (int j = 0; j < 16; ++j) dump[6144 + tid * 16 + j] = __uint_as_float(r[j]);
            *(uint4*)(sUT1 + inter_off(tid, 0, 128)) = make_uint4(packed[0], packed[1], packed[2], packed[3]);
            *(uint4*)(sUT1 + inter_off(tid, 1, 128)) = make_uint4(packed[4], packed[5], packed[6], packed[7]);
        }
        fence_async_smem();
        __syncthreads();
        stamp(3);
        // ---- G2' (out2 = UT @ Mqk^T) and G3 (kU^T = UT @ k_r; B = KR tile as MN-major SW128: LBO = MN-block 2048, SBO = 1024)
        if (tid == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;");
            mma_ss(taddr + tmA + TM_G2B, make_desc(sUT1_a, 2048, 128, 0), make_desc(st_a + OFF_MQK, 256, 128, 0), ID_G2, 0);
            mma_ss(taddr + TM_G3, make_desc(sUT1_a, 2048, 128, 0), make_desc(st_a + OFF_KR, KR_LBO, KR_SBO, 2), ID_G3, 0);
            mma_commit(mb2);
        }
        mbar_wait(mb2, phase);
        asm volatile("tcgen05.fence::after_thread_sync;");
        phase ^= 1;
        stamp(4);
        // ---- E4: S^T[v][k] = bf16(S^T[v][k] * g[k] + kU^T[v][k]); all 8 TMEM loads issued first, one wait
        {
            const float* G = (const float*)(st + OFF_RAW);
            uint32_t r[128];
#ifdef K2TC_NO_E4
            if (false)
#endif
#pragma unroll
            for (int q = 0; q < 8; ++q) tmem_ld16(lane_base + TM_G3 + q * 16, r + q * 16);
            uint4 sv[16];
#pragma unroll
            for (int c = 0; c < 16; ++c) sv[c] = *(const uint4*)(sS + sw_off(tid, c, 128));
            tmem_wait();
            stamp(11);
            if (dump && t == 0 && h == 0) for (int j = 0; j < 128; ++j) dump[8192 + tid * 128 + j] = __uint_as_float(r[j]);
#ifdef K2TC_NO_E4
            if (false)
#endif
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                const float4 ga = ((const float4*)G)[c * 2], gb = ((const float4*)G)[c * 2 + 1];
                const float g8[8] = {ga.x, ga.y, ga.z, ga.w, gb.x, gb.y, gb.z, gb.w};
                uint32_t w[4] = {sv[c].x, sv[c].y, sv[c].z, sv[c].w};
                uint32_t o[4];
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    float n0 = lo_f(w[q]) * g8[2 * q] + __uint_as_float(r[c * 8 + 2 * q]);
                    float n1 = hi_f(w[q]) * g8[2 * q + 1] + __uint_as_float(r[c * 8 + 2 * q + 1]);
                    o[q] = pack2(n0, n1);
                }
                *(uint4*)(sS + sw_off(tid, c, 128)) = make_uint4(o[0], o[1], o[2], o[3]);
                if (dump && t == 0 && h == 0) for (int q = 0; q < 4; ++q) { dump[24576 + tid * 128 + c * 8 + 2 * q] = lo_f(o[q]); dump[24576 + tid * 128 + c * 8 + 2 * q + 1] = hi_f(o[q]); }
            }
        }
        asm volatile("tcgen05.fence::before_thread_sync;");
        __syncthreads();
        stamp(5);
    }
    {   // epilogue of the last chunk
        const int t = t_tiles; const uint32_t tmP = ((t - 1) & 1) * 64;
        asm volatile("tcgen05.fence::after_thread_sync;");
        // ---- E3: out tile = bf16(bf16(outT) + bf16(out2)) -> staging [t][v]
        {
            uint32_t r[16], r2[16];
            uint16_t* O = (uint16_t*)sOut;
#ifdef K2TC_NO_E3
            if (false) {
#else
            tmem_ld16(lane_base + tmP + TM_G1 + 16, r); tmem_ld16(lane_base + tmP + TM_G2B, r2); tmem_wait();
            {
#endif
#pragma unroll
            for (int j = 0; j < 16; j += 2) {
                uint32_t a = pack2(__uint_as_float(r[j]), __uint_as_float(r[j + 1])), b = pack2(__uint_as_float(r2[j]), __uint_as_float(r2[j + 1]));
                uint32_t o = pack2(lo_f(a) + lo_f(b), hi_f(a) + hi_f(b));
                O[j * D + tid] = (uint16_t)(o & 0xFFFF); O[(j + 1) * D + tid] = (uint16_t)(o >> 16);
            }
            }
            if (false) { for (int j = 0; j < 16; ++j) { dump[2048 + tid * 16 + j] = __uint_as_float(r[j]); dump[4096 + tid * 16 + j] = __uint_as_float(r2[j]); } }
        }
        stamp(10);
        __syncthreads();
        if (false) for (int i = tid; i < 16 * D; i += NTHREADS) dump[63488 + i] = bf2f(((const uint16_t*)sOut)[i]);
        // ---- out staging -> gmem
        {
            uint8_t* og = (uint8_t*)(p.out + ((int64_t)(t - 1) * C) * H * D + (int64_t)h * D);
            for (int i = tid; i < 256; i += NTHREADS) {
                int tt = i >> 4, c = i & 15;
                *(uint4*)(og + (int64_t)tt * H * D * 2 + c * 16) = *(const uint4*)(sOut + i * 16);
            }
        }
        stamp(6);
    }
    __syncthreads();
    {   // final state: SW128 -> row-major [v][k]
        uint8_t* dst = (uint8_t*)(p.hT + (int64_t)h * D * D);
        for (int i = tid; i < D * 16; i += NTHREADS) {
            int r = i >> 4, c = i & 15;
            *(uint4*)(dst + i * 16) = *(const uint4*)(sS + sw_off(r, c, 128));
        }
    }
    if (TIMING && tid == 0) for (int i = 0; i < 16; i++) dbg[h * 16 + i] = tm[i];
    __syncthreads();
    if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(TM_COLS));
}

// ---------------------------------------------------------------- host: tensor maps via the driver entry point
typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                  const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                  CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeTiledFn get_encode() {
    static EncodeTiledFn fn = nullptr;
    if (!fn) {
        void* p = nullptr; cudaDriverEntryPointQueryResult q;
        cudaError_t e = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q);
        TORCH_CHECK(e == cudaSuccess && p != nullptr, "cuTensorMapEncodeTiled not available");
        fn = (EncodeTiledFn)p;
    }
    return fn;
}
static CUtensorMap make_map(void* base, int rank, const cuuint64_t* dims, const cuuint64_t* strides_bytes, const cuuint32_t* box) {
    CUtensorMap m; cuuint32_t es[5] = {1, 1, 1, 1, 1};
    CUresult r = get_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, base, dims, strides_bytes, box, es,
                              CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                              CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
    return m;
}
}  // namespace

void k2_tc2_fwd(torch::Tensor workspace, int64_t total_tiles, torch::Tensor v, torch::Tensor beta, torch::Tensor h0,
                torch::Tensor hT, torch::Tensor out, torch::Tensor dbg, bool timing, torch::Tensor dump) {
    TORCH_CHECK(v.dim() == 4 && v.size(0) == 1 && v.size(3) == D, "v must be [1,T,H,128]");
    int T = v.size(1), H = v.size(2);
    TORCH_CHECK(T % C == 0 && total_tiles == T / C);
    const int64_t n_ht = (int64_t)H * total_tiles;
    // ws region [kd|qd|kr] as a 2-D tensor: 3*n_ht*16 rows of 128 bf16
    cuuint64_t d_ws[2] = {(cuuint64_t)D, (cuuint64_t)(3 * n_ht * 16)}; cuuint64_t s_ws[1] = {(cuuint64_t)D * 2}; cuuint32_t b_ws[2] = {64, 16};
    CUtensorMap tm_ws = make_map(workspace.data_ptr(), 2, d_ws, s_ws, b_ws);
    cuuint64_t d_v[3] = {(cuuint64_t)D, (cuuint64_t)H, (cuuint64_t)T}; cuuint64_t s_v[2] = {(cuuint64_t)D * 2, (cuuint64_t)H * D * 2}; cuuint32_t b_v[3] = {64, 1, 16};
    CUtensorMap tm_v = make_map(v.data_ptr(), 3, d_v, s_v, b_v);
    cuuint64_t d_h[2] = {(cuuint64_t)D, (cuuint64_t)H * D}; cuuint64_t s_h[1] = {(cuuint64_t)D * 2}; cuuint32_t b_h[2] = {64, 128};
    CUtensorMap tm_h0 = make_map(h0.data_ptr(), 2, d_h, s_h, b_h);
    TORCH_CHECK(beta.dim() == 2 && beta.size(0) == H && beta.size(1) == T && beta.is_contiguous(), "beta must be beta_t [H,T] contiguous");
    Params p{workspace.data_ptr<uint8_t>(), (const __nv_bfloat16*)beta.data_ptr(), (__nv_bfloat16*)hT.data_ptr(), (__nv_bfloat16*)out.data_ptr(),
             T, H, (int)total_tiles, timing ? reinterpret_cast<long long*>(dbg.data_ptr<int64_t>()) : nullptr,
             dump.numel() > 0 ? dump.data_ptr<float>() : nullptr};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto kern = timing ? k2_tc2_kernel<true> : k2_tc2_kernel<false>;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_TOTAL);
    kern<<<H, NTHREADS, SM_TOTAL, stream>>>(tm_ws, tm_v, tm_h0, p);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "k2_tc2 launch failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("k2_tc_fwd", &k2_tc2_fwd, "FlashKDA K2 with tcgen05 v2 (TMA loads, SW128)");
    m.attr("SMEM_BYTES") = SM_TOTAL;
}
