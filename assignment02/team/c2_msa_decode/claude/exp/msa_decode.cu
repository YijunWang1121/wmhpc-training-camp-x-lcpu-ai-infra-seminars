// Fused small-batch MSA decode kernel, v2 (challenge (a)).
//
// One work unit = (query token t, kv head kh): 16 query heads (GQA group) x top-k<=16 blocks x 128 keys.
// A cluster of CL CTAs shares one work unit: CTA `rank` handles top-k slots [rank*B, rank*B+B), B = 16/CL.
//   CL = 1 : no split, one CTA streams all 16 blocks through a STAGES-deep TMA ring (CUTLASS-like shape)
//   CL = 4 : 4 blocks per CTA, partial states merged in-cluster through distributed shared memory (DSMEM)
// Inside a CTA: STAGES warp-groups of 4 warps; group g owns smem stage g and consumes blocks i with
// i % STAGES == g, so that several independent instruction streams share every SM sub-partition (v1 with
// one warp per scheduler was issue-latency bound). One stage per group keeps every mbarrier's phase
// sequence strictly in consumption order (a shared ring with interleaved groups let a group wait two
// phases ahead, which mbarrier parity cannot express -> v2 race / launch failure, see LOG.md S5).
// Each warp owns a 32-key slice of its block and keeps its own online-softmax state; the 8 warp states are
// merged through smem, the CL CTA states through DSMEM (rank 0 writes the output).
// Loads: cp.async.bulk.tensor.4d (TMA) on a 4-D tensor map over kv_cache [pages][kvh][128][256], page
// coordinate computed at runtime from topk_idx -> block_table (discussion point 3), 128B swizzle so ldmatrix
// is bank-conflict free. MMA: mma.sync m16n8k16 bf16 (M = 16 = GQA group, discussion point 1).
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cstdio>
namespace cg = cooperative_groups;

namespace msa {
constexpr int HD = 128, PAGE = 128, ROW = 2 * HD, GQA = 16, TOPK = 16;
constexpr uint32_t TILE_BYTES = PAGE * 64 * 2;  // one TMA box: 128 rows x 64 bf16 = 16 KiB
constexpr uint32_t BLK_BYTES = 4 * TILE_BYTES;  // K lo/hi + V lo/hi = 64 KiB
constexpr int MAX_STAGES = 3;
constexpr int SCR_WARP = 32 + GQA * HD;  // floats per warp slot in the merge scratch (m16,l16,acc)

struct Params {
  const __nv_bfloat16* q;  // [total_q][nheads][128]
  __nv_bfloat16* out;      // same layout
  const int32_t* topk;     // [nkv][total_q][16]
  const int32_t* bt;       // [num_reqs][bt_stride]
  const int32_t* seq_lens; // [num_reqs]
  int bt_stride, total_q, nkv, dql;
  long q_stride_t, q_stride_h;  // in elements
  float scale_log2;             // sm_scale * log2(e)
};

__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(su32(b)), "r"(cnt));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(su32(b)), "r"(bytes));
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @p bra DONE_%=;\n bra WAIT_%=;\n DONE_%=:\n}" ::"r"(su32(b)), "r"(parity));
}
__device__ __forceinline__ void tma_4d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, int c2, int c3) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3,%4,%5,%6}], [%2];" ::"r"(dst),
      "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2), "r"(c3)
      : "memory");
}
__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ void group_bar(int g) { asm volatile("bar.sync %0, %1;" ::"r"(g + 1), "r"(128) : "memory"); }
// byte offset of 16-byte chunk `c` (0..7) of row `r` inside a 128B-swizzled tile (row pitch 128 B)
__device__ __forceinline__ uint32_t swz(int r, int c) { return r * 128 + ((c ^ (r & 7)) << 4); }

// smem layout: [stages x 64K (1024-aligned)] [q 4K] [bars] [pages]
template <int CL, int STAGES>
__device__ __forceinline__ void decode_body(const CUtensorMap& map, const Params& p) {
  static_assert(TOPK % CL == 0, "cluster must divide topk");
  constexpr int B = TOPK / CL;  // slots per CTA
  constexpr int NG = STAGES, NWARPS = 4 * NG, NTHREADS = 32 * NWARPS;
  extern __shared__ __align__(1024) unsigned char smem[];
  const uint32_t smem_u32 = su32(smem);
  __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(smem + STAGES * BLK_BYTES);
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + STAGES * BLK_BYTES + GQA * HD * 2);
  int32_t* pages = reinterpret_cast<int32_t*>(bars + MAX_STAGES);
  float* scratch = reinterpret_cast<float*>(smem);  // reused after the main loop: 8 warps x (m16, l16, acc 16x128)

  int rank = 0;
  [[maybe_unused]] cg::cluster_group cluster = cg::this_cluster();
  if constexpr (CL > 1) rank = (int)cluster.block_rank();
  const int unit = blockIdx.x / CL;
  const int t = unit % p.total_q, kh = unit / p.total_q;
  const int req = t / p.dql;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int grp = warp >> 2;          // warp-group == owned stage
  const int key0 = (warp & 3) * 32;   // this warp's key slice inside every block

  const int seq_len = p.seq_lens[req];
  const int qpos = seq_len - p.dql + (t - req * p.dql);
  const int kv_len = max(qpos + 1, 0);
  const int real_topk = min(TOPK, (kv_len + PAGE - 1) / PAGE);
  const int slot0 = rank * B;
  const int nblk = max(0, min(B, real_topk - slot0));
  const int32_t* topk_row = p.topk + ((size_t)kh * p.total_q + t) * TOPK + slot0;

  // --- prologue: barriers, page indirection (two dependent scalar loads, all slots in parallel), Q tile ---
  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) mbar_init(&bars[s], 1);
    asm volatile("fence.proxy.async.shared::cta;");
  }
  if (tid < nblk) pages[tid] = p.bt[(size_t)req * p.bt_stride + topk_row[tid]];
  {  // Q: 16 heads x 128 d, bf16 -> smem (row pitch 256 B). 4 KiB = 256 x 16 B
    const __nv_bfloat16* qg = p.q + (size_t)t * p.q_stride_t + (size_t)kh * GQA * p.q_stride_h;
    for (int i = tid; i < GQA * HD / 8; i += NTHREADS) {
      const int h = i / (HD / 8), c = i % (HD / 8);
      *reinterpret_cast<int4*>(q_s + h * HD + c * 8) = *reinterpret_cast<const int4*>(qg + (size_t)h * p.q_stride_h + c * 8);
    }
  }
  __syncthreads();
  auto issue = [&](int i, int s) {  // load slot i into stage s (one thread)
    const uint32_t dst = smem_u32 + s * BLK_BYTES, bar = su32(&bars[s]);
    mbar_expect_tx(&bars[s], BLK_BYTES);
    const int pg = pages[i];
    tma_4d(dst + 0 * TILE_BYTES, &map, bar, 0, 0, kh, pg);
    tma_4d(dst + 1 * TILE_BYTES, &map, bar, 64, 0, kh, pg);
    tma_4d(dst + 2 * TILE_BYTES, &map, bar, 128, 0, kh, pg);
    tma_4d(dst + 3 * TILE_BYTES, &map, bar, 192, 0, kh, pg);
  };
  if (tid == 0)
    for (int i = 0; i < min(nblk, STAGES); ++i) issue(i, i);

  // --- Q fragments (A operand) for all 8 k-steps: 32 regs ---
  uint32_t qa[8][4];
  {
    const int m = lane >> 3, i = lane & 7;
    const int row = i + 8 * (m & 1);
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) ldsm_x4(qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], su32(q_s + row * HD + ks * 16 + 8 * (m >> 1)));
  }
  // loop-invariant ldmatrix offsets (relative to the stage base)
  uint32_t koff[8][2], voff[2][8];
  {
    const int m = lane >> 3, r = lane & 7;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks)
#pragma unroll
      for (int np = 0; np < 2; ++np)
        koff[ks][np] = (ks >> 2) * TILE_BYTES + swz(key0 + np * 16 + 8 * (m >> 1) + r, (ks & 3) * 2 + (m & 1));
#pragma unroll
    for (int ks = 0; ks < 2; ++ks)
#pragma unroll
      for (int np = 0; np < 8; ++np)
        voff[ks][np] = (2 + (np >> 2)) * TILE_BYTES + swz(key0 + ks * 16 + 8 * (m & 1) + r, (np & 3) * 2 + (m >> 1));
  }

  // --- per-warp online softmax state over this warp's 32-key slice of every block it sees ---
  float m_r[2] = {-INFINITY, -INFINITY};  // rows lane/4 and lane/4+8
  float l_r[2] = {0.f, 0.f};              // per-thread partial (reduced across the quad at the end)
  float acc[16][4];
#pragma unroll
  for (int n = 0; n < 16; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;

  const uint32_t kt = smem_u32 + grp * BLK_BYTES;  // this group's stage
  for (int i = grp; i < nblk; i += NG) {
    mbar_wait(&bars[grp], (i / NG) & 1);  // tiles: K[0:64], K[64:128], V[0:64], V[64:128]
    const int valid = min(PAGE, kv_len - topk_row[i] * PAGE);  // keys < valid are real (topk row is L1-hot)

    // S = Q K^T for this warp's 32 keys: 4 n-tiles
    float sc[4][4];
#pragma unroll
    for (int n = 0; n < 4; ++n) sc[n][0] = sc[n][1] = sc[n][2] = sc[n][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
#pragma unroll
      for (int np = 0; np < 2; ++np) {
        uint32_t b0, b1, b2, b3;
        ldsm_x4(b0, b1, b2, b3, kt + koff[ks][np]);
        mma_bf16(sc[np * 2 + 0], qa[ks], b0, b1);
        mma_bf16(sc[np * 2 + 1], qa[ks], b2, b3);
      }
    }
    // scale (+ mask only in the tail block) + row max
    float mx[2] = {-INFINITY, -INFINITY};
    if (valid < PAGE) {
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        const int col = key0 + n * 8 + (lane & 3) * 2;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          sc[n][j] = (col + (j & 1)) < valid ? sc[n][j] * p.scale_log2 : -INFINITY;
          mx[j >> 1] = fmaxf(mx[j >> 1], sc[n][j]);
        }
      }
    } else {
#pragma unroll
      for (int n = 0; n < 4; ++n)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          sc[n][j] *= p.scale_log2;
          mx[j >> 1] = fmaxf(mx[j >> 1], sc[n][j]);
        }
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffff, mx[h], 1));
      mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffff, mx[h], 2));
    }
    float alpha[2], mu[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float mn = fmaxf(m_r[h], mx[h]);
      mu[h] = (mn == -INFINITY) ? 0.f : mn;
      alpha[h] = exp2f(m_r[h] - mu[h]);
      m_r[h] = mn;
      l_r[h] *= alpha[h];
    }
    uint32_t pa[2][4];  // P as A fragments for the 2 k-steps (16 keys each)
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      const float p0 = exp2f(sc[n][0] - mu[0]), p1 = exp2f(sc[n][1] - mu[0]);
      const float p2 = exp2f(sc[n][2] - mu[1]), p3 = exp2f(sc[n][3] - mu[1]);
      l_r[0] += p0 + p1;
      l_r[1] += p2 + p3;
      pa[n >> 1][(n & 1) * 2 + 0] = pack_bf16(p0, p1);
      pa[n >> 1][(n & 1) * 2 + 1] = pack_bf16(p2, p3);
    }
    if (!__all_sync(0xffffffff, alpha[0] == 1.f && alpha[1] == 1.f)) {  // skip the 64 rescales when no max moved
#pragma unroll
      for (int n = 0; n < 16; ++n) {
        acc[n][0] *= alpha[0]; acc[n][1] *= alpha[0];
        acc[n][2] *= alpha[1]; acc[n][3] *= alpha[1];
      }
    }
    // O += P V : 2 k-steps over 16 keys, 16 n-tiles over 128 d
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
#pragma unroll
      for (int np = 0; np < 8; ++np) {
        uint32_t b0, b1, b2, b3;
        ldsm_x4_t(b0, b1, b2, b3, kt + voff[ks][np]);
        mma_bf16(acc[np * 2 + 0], pa[ks], b0, b1);
        mma_bf16(acc[np * 2 + 1], pa[ks], b2, b3);
      }
    }
    group_bar(grp);  // the 4 warps of this group are done with the stage -> refill it with the group's next block
    if ((tid & 127) == 0 && i + NG < nblk) issue(i + NG, grp);
  }
  // finish per-thread l: reduce across the quad
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    l_r[h] += __shfl_xor_sync(0xffffffff, l_r[h], 1);
    l_r[h] += __shfl_xor_sync(0xffffffff, l_r[h], 2);
  }
  // --- merge the 8 warps through smem (stage memory is free now: all TMA loads were consumed) ---
  __syncthreads();
  {
    float* w = scratch + warp * SCR_WARP;
    if ((lane & 3) == 0) {
      w[lane >> 2] = m_r[0]; w[(lane >> 2) + 8] = m_r[1];
      w[16 + (lane >> 2)] = l_r[0]; w[16 + (lane >> 2) + 8] = l_r[1];
    }
    float* a = w + 32;
#pragma unroll
    for (int n = 0; n < 16; ++n) {
      const int row = lane >> 2, col = n * 8 + (lane & 3) * 2;
      *reinterpret_cast<float2*>(a + row * HD + col) = make_float2(acc[n][0], acc[n][1]);
      *reinterpret_cast<float2*>(a + (row + 8) * HD + col) = make_float2(acc[n][2], acc[n][3]);
    }
  }
  __syncthreads();
  // combined CTA state -> warp-0 slot (in place). work item idx (0..255): row = idx/16, cols (idx%16)*8 .. +8
  constexpr int ITEMS = (256 + NTHREADS - 1) / NTHREADS;
  float o[ITEMS][8];
#pragma unroll
  for (int it = 0; it < ITEMS; ++it) {
    const int idx = tid + it * NTHREADS;
    if (idx < 256) {
      const int row = idx >> 4, c0 = (idx & 15) * 8;
      float mw[NWARPS], M = -INFINITY;
#pragma unroll
      for (int w = 0; w < NWARPS; ++w) { mw[w] = scratch[w * SCR_WARP + row]; M = fmaxf(M, mw[w]); }
      const float Mu = (M == -INFINITY) ? 0.f : M;
      float L = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) o[it][j] = 0.f;
#pragma unroll
      for (int w = 0; w < NWARPS; ++w) {
        const float wt = exp2f(mw[w] - Mu);
        L += wt * scratch[w * SCR_WARP + 16 + row];
        const float4* a = reinterpret_cast<const float4*>(scratch + w * SCR_WARP + 32 + row * HD + c0);
        const float4 x = a[0], y = a[1];
        o[it][0] += wt * x.x; o[it][1] += wt * x.y; o[it][2] += wt * x.z; o[it][3] += wt * x.w;
        o[it][4] += wt * y.x; o[it][5] += wt * y.y; o[it][6] += wt * y.z; o[it][7] += wt * y.w;
      }
      // stash M, L in registers via a tiny trick: reuse mw[0], mw[1]
      mw[0] = M; mw[1] = L;
      __syncthreads();  // all reads of the NWARPS slots done before overwriting slot 0 (uniform: every thread reaches it)
      if ((idx & 15) == 0) { scratch[row] = mw[0]; scratch[16 + row] = mw[1]; }
      float4* a0 = reinterpret_cast<float4*>(scratch + 32 + row * HD + c0);
      a0[0] = make_float4(o[it][0], o[it][1], o[it][2], o[it][3]);
      a0[1] = make_float4(o[it][4], o[it][5], o[it][6], o[it][7]);
    } else {
      __syncthreads();  // keep the barrier count uniform for threads without a work item
    }
  }
  // --- cluster merge through DSMEM; rank 0 writes the output ---
  __nv_bfloat16* og = p.out + (size_t)t * p.q_stride_t + (size_t)kh * GQA * p.q_stride_h;
  if constexpr (CL == 1) {
    __syncthreads();
#pragma unroll
    for (int it = 0; it < ITEMS; ++it) {
      const int idx = tid + it * NTHREADS;
      if (idx < 256) {
        const int row = idx >> 4, c0 = (idx & 15) * 8;
        const float L = scratch[16 + row];
        const float inv = (L > 0.f) ? 1.f / L : 0.f;
        int4* dst = reinterpret_cast<int4*>(og + (size_t)row * p.q_stride_h + c0);
        *dst = make_int4(pack_bf16(o[it][0] * inv, o[it][1] * inv), pack_bf16(o[it][2] * inv, o[it][3] * inv),
                         pack_bf16(o[it][4] * inv, o[it][5] * inv), pack_bf16(o[it][6] * inv, o[it][7] * inv));
      }
    }
  } else {
    cluster.sync();  // all ranks' combined states are visible cluster-wide
    if (rank == 0) {
#pragma unroll
      for (int it = 0; it < ITEMS; ++it) {
        const int idx = tid + it * NTHREADS;
        if (idx < 256) {
          const int row = idx >> 4, c0 = (idx & 15) * 8;
          float mr[CL], M = -INFINITY;
#pragma unroll
          for (int r = 0; r < CL; ++r) { const float* rs = cluster.map_shared_rank(scratch, r); mr[r] = rs[row]; M = fmaxf(M, mr[r]); }
          const float Mu = (M == -INFINITY) ? 0.f : M;
          float L = 0.f, oo[8];
#pragma unroll
          for (int j = 0; j < 8; ++j) oo[j] = 0.f;
#pragma unroll
          for (int r = 0; r < CL; ++r) {
            const float* rs = cluster.map_shared_rank(scratch, r);
            const float wt = exp2f(mr[r] - Mu);
            L += wt * rs[16 + row];
            const float4* a = reinterpret_cast<const float4*>(rs + 32 + row * HD + c0);
            const float4 x = a[0], y = a[1];
            oo[0] += wt * x.x; oo[1] += wt * x.y; oo[2] += wt * x.z; oo[3] += wt * x.w;
            oo[4] += wt * y.x; oo[5] += wt * y.y; oo[6] += wt * y.z; oo[7] += wt * y.w;
          }
          const float inv = (L > 0.f) ? 1.f / L : 0.f;
          int4* dst = reinterpret_cast<int4*>(og + (size_t)row * p.q_stride_h + c0);
          *dst = make_int4(pack_bf16(oo[0] * inv, oo[1] * inv), pack_bf16(oo[2] * inv, oo[3] * inv),
                           pack_bf16(oo[4] * inv, oo[5] * inv), pack_bf16(oo[6] * inv, oo[7] * inv));
        }
      }
    }
    cluster.sync();  // keep every rank's smem alive until rank 0 is done reading
  }
}

template <int CL, int STAGES>
__global__ void __cluster_dims__(CL, 1, 1) __launch_bounds__(128 * STAGES) decode_kernel_cluster(const __grid_constant__ CUtensorMap map, Params p) {
  decode_body<CL, STAGES>(map, p);
}
template <int STAGES>
__global__ void __launch_bounds__(128 * STAGES) decode_kernel_plain(const __grid_constant__ CUtensorMap map, Params p) {
  decode_body<1, STAGES>(map, p);
}

template <int CL, int STAGES>
static int launch(const CUtensorMap& map, const Params& p, int grid, cudaStream_t stream) {
  constexpr size_t smem = (size_t)STAGES * BLK_BYTES + GQA * HD * 2 + MAX_STAGES * 8 + TOPK * 4;
  auto k = (CL == 1) ? decode_kernel_plain<STAGES> : decode_kernel_cluster<CL, STAGES>;
  static bool configured = false;  // per template instance
  if (!configured) {
    if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) != cudaSuccess) return -2;
    configured = true;
  }
  k<<<grid, 128 * STAGES, smem, stream>>>(map, p);
  return (int)cudaGetLastError();
}
}  // namespace msa

extern "C" int msa_decode_launch(const void* q, const void* kv, void* out, const int32_t* topk, const int32_t* bt, int bt_stride,
                                 const int32_t* seq_lens, int total_q, int nkv, int npages, int dql, float sm_scale,
                                 long q_stride_t, long q_stride_h, int cluster, int stages, void* stream_) {
  using namespace msa;
  cudaStream_t stream = (cudaStream_t)stream_;
  CUtensorMap map;
  cuuint64_t gdim[4] = {ROW, PAGE, (cuuint64_t)nkv, (cuuint64_t)npages};
  cuuint64_t gstr[3] = {ROW * 2, (cuuint64_t)PAGE * ROW * 2, (cuuint64_t)nkv * PAGE * ROW * 2};
  cuuint32_t box[4] = {64, PAGE, 1, 1}, estr[4] = {1, 1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(kv), gdim, gstr, box, estr,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) return -1;
  Params p;
  p.q = (const __nv_bfloat16*)q; p.out = (__nv_bfloat16*)out; p.topk = topk; p.bt = bt; p.seq_lens = seq_lens;
  p.bt_stride = bt_stride; p.total_q = total_q; p.nkv = nkv; p.dql = dql;
  p.q_stride_t = q_stride_t; p.q_stride_h = q_stride_h; p.scale_log2 = sm_scale * 1.4426950408889634f;
  const int grid = total_q * nkv * cluster;
#define L(CL, ST) if (cluster == CL && stages == ST) return launch<CL, ST>(map, p, grid, stream);
  L(1, 1) L(1, 2) L(1, 3) L(2, 1) L(2, 2) L(2, 3) L(4, 1) L(4, 2) L(4, 3)
#undef L
  return -4;
}
