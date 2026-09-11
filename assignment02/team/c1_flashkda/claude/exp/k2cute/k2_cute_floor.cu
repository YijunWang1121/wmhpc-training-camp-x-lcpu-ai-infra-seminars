// E9: K2 的 tcgen05 依赖链,用 CUTLASS/CuTe SM100 原语搭(不手写 PTX 描述符):
//   TMA(cute::make_tma_atom/SM90_TMA_LOAD) + UMMA(SM100_MMA_F16BF16_SS atom, cute::gemm) + TMEM(Allocator1Sm,
//   partition_fragment_C) + umma_arrive/wait_barrier。每 chunk 的链与 k2_tc2.cu §7 完全同形:
//     G1  [128(V) x 32]  = S^T[128x128] @ [kd;qd]^T          8 条 K16 原子, TMEM col 0..31
//     E1' TMEM 回读 128x16 -> bf16 -> smem uT (INTER, K-major)   (真依赖: G2 的 A 操作数)
//     G2  [128 x 16]     = uT @ INV^T                        1 条,    TMEM col 32..47
//     E2' TMEM 回读 128x16 -> bf16 -> smem UT
//     G2' [128 x 16]     = UT @ Mqk^T                        1 条,    TMEM col 48..63
//     G3  [128(D) x 128(V)] = KR^T(MN-major SW128) @ UT^T     1 条,    TMEM col 128..255
//     E4' TMEM 回读 128x128 (状态更新的读侧; 地板版不做 RMW, 与 §7 "v3-E1-E3-E4" 口径对齐)
//   2 级 TMA 预取(KQ/KR/INV/Mqk 每 chunk 17 KB), S 常驻 smem。
// 模式:DEBUG=1 -> 跑 1 个 chunk,把 4 个 GEMM 的 TMEM 结果 dump 到 global,host 用 fp32 对拍(验证 CuTe 生成的
//       描述符/布局/majorness 正确);否则计时 H 个 CTA × NT 个 chunk,报 cycle/chunk。
// 编译: nvcc -O3 -std=c++17 --expt-relaxed-constexpr -gencode arch=compute_100f,code=sm_100f -I$CUTLASS/include
//       -I$CUTLASS/tools/util/include k2_cute_floor.cu -o k2_cute_floor -lcuda
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cmath>
#include <cutlass/bfloat16.h>
#include <cutlass/arch/barrier.h>
#include <cute/tensor.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

using namespace cute;
using bf16 = cutlass::bfloat16_t;

constexpr int D = 128, C = 16, KQ_ROWS = 32;
constexpr int TM_G1 = 0, TM_G2 = 32, TM_G2B = 48, TM_G3 = 128, TM_COLS = 256;

// ---- MMA atoms ---------------------------------------------------------------------------------------------
using MmaG1 = decltype(make_tiled_mma(SM100_MMA_F16BF16_SS<bf16, bf16, float, 128, 32,  UMMA::Major::K,  UMMA::Major::K>{}));
using MmaG2 = decltype(make_tiled_mma(SM100_MMA_F16BF16_SS<bf16, bf16, float, 128, 16,  UMMA::Major::K,  UMMA::Major::K>{}));
using MmaG3 = decltype(make_tiled_mma(SM100_MMA_F16BF16_SS<bf16, bf16, float, 128, 128, UMMA::Major::MN, UMMA::Major::K>{}));

// ---- smem layouts (CuTe 从 atom 生成 UMMA 描述符,不用手算 LBO/SBO/swizzle 相位) --------------------------------
using SLayoutS   = decltype(UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<bf16>{},  partition_shape_A(MmaG1{}, Shape<_128,_128>{})));
using SLayoutKQ  = decltype(UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<bf16>{},  partition_shape_B(MmaG1{}, Shape<_32,_128>{})));
using SLayoutU   = decltype(UMMA::tile_to_mma_shape(UMMA::Layout_K_INTER_Atom<bf16>{},  partition_shape_A(MmaG2{}, Shape<_128,_16>{})));
using SLayoutINV = decltype(UMMA::tile_to_mma_shape(UMMA::Layout_K_INTER_Atom<bf16>{},  partition_shape_B(MmaG2{}, Shape<_16,_16>{})));
using SLayoutKR  = decltype(UMMA::tile_to_mma_shape(UMMA::Layout_MN_SW128_Atom<bf16>{}, partition_shape_A(MmaG3{}, Shape<_128,_16>{})));
using SLayoutUTB = decltype(UMMA::tile_to_mma_shape(UMMA::Layout_K_INTER_Atom<bf16>{},  partition_shape_B(MmaG3{}, Shape<_128,_16>{})));
// 线程写 uT/UT 时用的 2-D 视图(同一个 atom、同一个整体形状 -> 同一物理布局;DEBUG 对拍会验证这一点)
using SLayoutU2D = decltype(tile_to_shape(UMMA::Layout_K_INTER_Atom<bf16>{}, Shape<_128,_16>{}));

struct SharedStorage {
  alignas(1024) ArrayEngine<bf16, cosize_v<SLayoutS>>   S;
  alignas(1024) ArrayEngine<bf16, cosize_v<SLayoutKQ>>  KQ[2];
  alignas(1024) ArrayEngine<bf16, cosize_v<SLayoutKR>>  KR[2];
  alignas(128)  ArrayEngine<bf16, cosize_v<SLayoutINV>> INV[2];
  alignas(128)  ArrayEngine<bf16, cosize_v<SLayoutINV>> MQK[2];
  alignas(1024) ArrayEngine<bf16, cosize_v<SLayoutU>>   U;
  alignas(1024) ArrayEngine<bf16, cosize_v<SLayoutU>>   UT;
  alignas(16) uint64_t tma_bar[2];
  alignas(16) uint64_t s_bar;
  alignas(16) uint64_t mma_bar[3];
  alignas(16) uint32_t tmem_base;
};

__device__ __forceinline__ void tmem_ld16(uint32_t taddr, uint32_t* r) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                 "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
               : "r"(taddr));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}
__device__ __forceinline__ void tmem_fence_after_sync() { asm volatile("tcgen05.fence::after_thread_sync;"); }
__device__ __forceinline__ void tmem_fence_before_sync() { asm volatile("tcgen05.fence::before_thread_sync;"); }

struct Params { int NT; float* dump; float* sink; };

template <bool DEBUG, class TmaS, class TmaKQ, class TmaKR, class TmaINV, class TmaMQK,
          class GS, class GKQ, class GKR, class GINV, class GMQK>
__global__ void __launch_bounds__(128) k2_cute_kernel(
    CUTE_GRID_CONSTANT TmaS const tma_S, CUTE_GRID_CONSTANT TmaKQ const tma_KQ, CUTE_GRID_CONSTANT TmaKR const tma_KR,
    CUTE_GRID_CONSTANT TmaINV const tma_INV, CUTE_GRID_CONSTANT TmaMQK const tma_MQK,
    GS mS, GKQ mKQ, GKR mKR, GINV mINV, GMQK mMQK, Params p)
{
  extern __shared__ char smem_raw[];
  SharedStorage& st = *reinterpret_cast<SharedStorage*>((smem_raw + 1023) - ((uintptr_t)(smem_raw + 1023) & 1023));
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, h = blockIdx.x;
  const uint32_t elect_thr = cute::elect_one_sync();
  const bool warp0 = (warp == 0);

  MmaG1 mma1; MmaG2 mma2; MmaG3 mma3;
  ThrMMA thr1 = mma1.get_slice(0), thr2 = mma2.get_slice(0), thr3 = mma3.get_slice(0);

  // ---- global tiles for this head: (rows, 128, chunks) -> local_tile keeps the chunk mode
  Tensor gS   = local_tile(mS,   Shape<_128,_128>{}, make_coord(0, 0, h));          // (128,128)
  Tensor gKQ  = local_tile(mKQ,  Shape<_32,_128>{},  make_coord(0, 0, _));          // (32,128,NCH)
  Tensor gKR  = local_tile(mKR,  Shape<_128,_16>{},  make_coord(0, 0, _));          // (128,16,NCH)  MN-major
  Tensor gINV = local_tile(mINV, Shape<_16,_16>{},   make_coord(0, 0, _));          // (16,16,NCH)
  Tensor gMQK = local_tile(mMQK, Shape<_16,_16>{},   make_coord(0, 0, _));

  Tensor tCsS   = make_tensor(make_smem_ptr(st.S.begin()),      SLayoutS{});
  Tensor tCsKQ0 = make_tensor(make_smem_ptr(st.KQ[0].begin()),  SLayoutKQ{});
  Tensor tCsKQ1 = make_tensor(make_smem_ptr(st.KQ[1].begin()),  SLayoutKQ{});
  Tensor tCsKR0 = make_tensor(make_smem_ptr(st.KR[0].begin()),  SLayoutKR{});
  Tensor tCsKR1 = make_tensor(make_smem_ptr(st.KR[1].begin()),  SLayoutKR{});
  Tensor tCsI0  = make_tensor(make_smem_ptr(st.INV[0].begin()), SLayoutINV{});
  Tensor tCsI1  = make_tensor(make_smem_ptr(st.INV[1].begin()), SLayoutINV{});
  Tensor tCsM0  = make_tensor(make_smem_ptr(st.MQK[0].begin()), SLayoutINV{});
  Tensor tCsM1  = make_tensor(make_smem_ptr(st.MQK[1].begin()), SLayoutINV{});
  Tensor tCsU   = make_tensor(make_smem_ptr(st.U.begin()),      SLayoutU{});
  Tensor tCsUTa = make_tensor(make_smem_ptr(st.UT.begin()),     SLayoutU{});      // as A of G2'
  Tensor tCsUTb = make_tensor(make_smem_ptr(st.UT.begin()),     SLayoutUTB{});    // as B of G3
  Tensor sU2d   = make_tensor(make_smem_ptr(st.U.begin()),      SLayoutU2D{});    // (128,16) thread-write view
  Tensor sUT2d  = make_tensor(make_smem_ptr(st.UT.begin()),     SLayoutU2D{});

  Tensor tCgS   = thr1.partition_A(gS);
  Tensor tCgKQ  = thr1.partition_B(gKQ);
  Tensor tCgKR  = thr3.partition_A(gKR);
  Tensor tCgINV = thr2.partition_B(gINV);
  Tensor tCgMQK = thr2.partition_B(gMQK);

  // fragments (UMMA smem descriptors, generated by CuTe)
  Tensor tCrS    = thr1.make_fragment_A(tCsS);
  Tensor tCrKQ0  = thr1.make_fragment_B(tCsKQ0);  Tensor tCrKQ1 = thr1.make_fragment_B(tCsKQ1);
  Tensor tCrU    = thr2.make_fragment_A(tCsU);
  Tensor tCrI0   = thr2.make_fragment_B(tCsI0);   Tensor tCrI1  = thr2.make_fragment_B(tCsI1);
  Tensor tCrUTa  = thr2.make_fragment_A(tCsUTa);
  Tensor tCrM0   = thr2.make_fragment_B(tCsM0);   Tensor tCrM1  = thr2.make_fragment_B(tCsM1);
  Tensor tCrKR0  = thr3.make_fragment_A(tCsKR0);  Tensor tCrKR1 = thr3.make_fragment_A(tCsKR1);
  Tensor tCrUTb  = thr3.make_fragment_B(tCsUTb);

  // TMEM accumulators
  Tensor tCtG1  = partition_fragment_C(mma1, Shape<_128,_32>{});
  Tensor tCtG2  = partition_fragment_C(mma2, Shape<_128,_16>{});
  Tensor tCtG2b = partition_fragment_C(mma2, Shape<_128,_16>{});
  Tensor tCtG3  = partition_fragment_C(mma3, Shape<_128,_128>{});

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};
  if (warp0) tmem_allocator.allocate(TM_COLS, &st.tmem_base);
  __syncthreads();
  const uint32_t tmem_base = st.tmem_base;
  tCtG1.data()  = tmem_base + TM_G1;
  tCtG2.data()  = tmem_base + TM_G2;
  tCtG2b.data() = tmem_base + TM_G2B;
  tCtG3.data()  = tmem_base + TM_G3;

  // TMA partitions (1 CTA, no multicast)
  auto [tSgS, tSsS]     = tma_partition(tma_S,   Int<0>{}, Layout<_1>{}, group_modes<0,3>(tCsS),   group_modes<0,3>(tCgS));
  auto [tKgKQ, tKsKQ0]  = tma_partition(tma_KQ,  Int<0>{}, Layout<_1>{}, group_modes<0,3>(tCsKQ0), group_modes<0,3>(tCgKQ));
  auto [tKgKQ_, tKsKQ1] = tma_partition(tma_KQ,  Int<0>{}, Layout<_1>{}, group_modes<0,3>(tCsKQ1), group_modes<0,3>(tCgKQ));
  auto [tRgKR, tRsKR0]  = tma_partition(tma_KR,  Int<0>{}, Layout<_1>{}, group_modes<0,3>(tCsKR0), group_modes<0,3>(tCgKR));
  auto [tRgKR_, tRsKR1] = tma_partition(tma_KR,  Int<0>{}, Layout<_1>{}, group_modes<0,3>(tCsKR1), group_modes<0,3>(tCgKR));
  auto [tIgI, tIsI0]    = tma_partition(tma_INV, Int<0>{}, Layout<_1>{}, group_modes<0,3>(tCsI0),  group_modes<0,3>(tCgINV));
  auto [tIgI_, tIsI1]   = tma_partition(tma_INV, Int<0>{}, Layout<_1>{}, group_modes<0,3>(tCsI1),  group_modes<0,3>(tCgINV));
  auto [tMgM, tMsM0]    = tma_partition(tma_MQK, Int<0>{}, Layout<_1>{}, group_modes<0,3>(tCsM0),  group_modes<0,3>(tCgMQK));
  auto [tMgM_, tMsM1]   = tma_partition(tma_MQK, Int<0>{}, Layout<_1>{}, group_modes<0,3>(tCsM1),  group_modes<0,3>(tCgMQK));
  const int stage_bytes = sizeof(make_tensor_like(tKsKQ0)) + sizeof(make_tensor_like(tRsKR0))
                        + sizeof(make_tensor_like(tIsI0)) + sizeof(make_tensor_like(tMsM0));
  const int s_bytes = sizeof(make_tensor_like(tSsS));

  if (warp0 && elect_thr) {
    cute::initialize_barrier(st.tma_bar[0], 1); cute::initialize_barrier(st.tma_bar[1], 1);
    cute::initialize_barrier(st.s_bar, 1);
    cute::initialize_barrier(st.mma_bar[0], 1); cute::initialize_barrier(st.mma_bar[1], 1); cute::initialize_barrier(st.mma_bar[2], 1);
  }
  __syncthreads();

  const int NT = p.NT;
  const int chunk0 = h * NT;
  auto issue_tma = [&](int it) {   // one thread
    int s = it & 1; int ch = chunk0 + it;
    cute::set_barrier_transaction_bytes(st.tma_bar[s], stage_bytes);
    if (s == 0) {
      copy(tma_KQ.with(st.tma_bar[0]),  tKgKQ(_, ch), tKsKQ0);
      copy(tma_KR.with(st.tma_bar[0]),  tRgKR(_, ch), tRsKR0);
      copy(tma_INV.with(st.tma_bar[0]), tIgI(_, ch),  tIsI0);
      copy(tma_MQK.with(st.tma_bar[0]), tMgM(_, ch),  tMsM0);
    } else {
      copy(tma_KQ.with(st.tma_bar[1]),  tKgKQ(_, ch), tKsKQ1);
      copy(tma_KR.with(st.tma_bar[1]),  tRgKR(_, ch), tRsKR1);
      copy(tma_INV.with(st.tma_bar[1]), tIgI(_, ch),  tIsI1);
      copy(tma_MQK.with(st.tma_bar[1]), tMgM(_, ch),  tMsM1);
    }
  };
  if (warp0 && elect_thr) {
    cute::set_barrier_transaction_bytes(st.s_bar, s_bytes);
    copy(tma_S.with(st.s_bar), tSgS, tSsS);
    issue_tma(0);
  }
  cute::wait_barrier(st.s_bar, 0);

  const uint32_t lane_base = tmem_base + ((uint32_t)(warp * 32) << 16);
  float sink = 0.f;
  uint32_t r[16];
  int mma_phase[3] = {0, 0, 0};

  for (int it = 0; it < NT; ++it) {
    const int s = it & 1;
    if (it + 1 < NT && warp0 && elect_thr) issue_tma(it + 1);      // stage (it+1)&1 was freed by G3 of it-1
    cute::wait_barrier(st.tma_bar[s], (it >> 1) & 1);

    // ---- G1: 8 K16 atoms, single warp issues
    if (warp0) {
      tmem_fence_after_sync();
      mma1.accumulate_ = UMMA::ScaleOut::Zero;
      auto& tCrKQ = (s == 0) ? tCrKQ0 : tCrKQ1;
      for (int k = 0; k < size<2>(tCrS); ++k) { gemm(mma1, tCrS(_,_,k), tCrKQ(_,_,k), tCtG1); mma1.accumulate_ = UMMA::ScaleOut::One; }
      cutlass::arch::umma_arrive(&st.mma_bar[0]);
    }
    cute::wait_barrier(st.mma_bar[0], mma_phase[0]); mma_phase[0] ^= 1;
    tmem_fence_after_sync();
    // E1': read u (first 16 cols) -> bf16 -> sU (each warp: its 32 rows)
    tmem_ld16(lane_base + TM_G1, r);
    if (DEBUG && it == 0) for (int j = 0; j < 16; ++j) p.dump[(warp*32+lane)*32 + j] = __uint_as_float(r[j]);
    if (DEBUG && it == 0) { uint32_t r2[16]; tmem_ld16(lane_base + TM_G1 + 16, r2); for (int j = 0; j < 16; ++j) p.dump[(warp*32+lane)*32 + 16 + j] = __uint_as_float(r2[j]); }
    for (int j = 0; j < 16; ++j) sU2d(warp*32 + lane, j) = bf16(__uint_as_float(r[j]));
    cutlass::arch::fence_view_async_shared();
    tmem_fence_before_sync();
    __syncthreads();

    // ---- G2: uT @ INV^T
    if (warp0) {
      tmem_fence_after_sync();
      mma2.accumulate_ = UMMA::ScaleOut::Zero;
      auto& tCrI = (s == 0) ? tCrI0 : tCrI1;
      gemm(mma2, tCrU(_,_,0), tCrI(_,_,0), tCtG2);
      cutlass::arch::umma_arrive(&st.mma_bar[1]);
    }
    cute::wait_barrier(st.mma_bar[1], mma_phase[1]); mma_phase[1] ^= 1;
    tmem_fence_after_sync();
    tmem_ld16(lane_base + TM_G2, r);
    if (DEBUG && it == 0) for (int j = 0; j < 16; ++j) p.dump[128*32 + (warp*32+lane)*16 + j] = __uint_as_float(r[j]);
    for (int j = 0; j < 16; ++j) sUT2d(warp*32 + lane, j) = bf16(__uint_as_float(r[j]));
    cutlass::arch::fence_view_async_shared();
    tmem_fence_before_sync();
    __syncthreads();

    // ---- G2' + G3
    if (warp0) {
      tmem_fence_after_sync();
      mma2.accumulate_ = UMMA::ScaleOut::Zero;
      auto& tCrM = (s == 0) ? tCrM0 : tCrM1;
      gemm(mma2, tCrUTa(_,_,0), tCrM(_,_,0), tCtG2b);
      mma3.accumulate_ = UMMA::ScaleOut::Zero;
      auto& tCrKR = (s == 0) ? tCrKR0 : tCrKR1;
      gemm(mma3, tCrKR(_,_,0), tCrUTb(_,_,0), tCtG3);
      cutlass::arch::umma_arrive(&st.mma_bar[2]);
    }
    cute::wait_barrier(st.mma_bar[2], mma_phase[2]); mma_phase[2] ^= 1;
    tmem_fence_after_sync();
    // E4': read the 128x128 state-update operand (8 x 16 cols)
    for (int c = 0; c < 128; c += 16) {
      tmem_ld16(lane_base + TM_G3 + c, r);
      if (DEBUG && it == 0) for (int j = 0; j < 16; ++j) p.dump[128*32 + 128*16 + 128*16 + (warp*32+lane)*128 + c + j] = __uint_as_float(r[j]);
      sink += __uint_as_float(r[0]) + __uint_as_float(r[15]);
    }
    if (DEBUG && it == 0) { tmem_ld16(lane_base + TM_G2B, r); for (int j = 0; j < 16; ++j) p.dump[128*32 + 128*16 + (warp*32+lane)*16 + j] = __uint_as_float(r[j]); }
    tmem_fence_before_sync();
    __syncthreads();       // stage s free for TMA (it+2), next iteration
  }
  p.sink[h * 128 + tid] = sink;
  __syncthreads();
  if (warp0) { tmem_allocator.release_allocation_lock(); tmem_allocator.free(tmem_base, TM_COLS); }
}

// ------------------------------------------------------------------------------------------------ host
static float bf2f(bf16 b) { return float(b); }
static uint32_t rng_state = 12345u;
static float frand() { rng_state = rng_state * 1664525u + 1013904223u; return ((rng_state >> 8) & 0xFFFF) / 65536.f * 2.f - 1.f; }

int main(int argc, char** argv) {
  int H = argc > 1 ? atoi(argv[1]) : 12;
  int NT = argc > 2 ? atoi(argv[2]) : 512;
  int debug = argc > 3 ? atoi(argv[3]) : 0;
  float mhz = argc > 4 ? atof(argv[4]) : 1095.f;
  if (debug) { NT = 1; }
  const int NCH = H * NT;
  std::vector<bf16> hS((size_t)H * D * D), hKQ((size_t)NCH * KQ_ROWS * D), hKR((size_t)NCH * C * D), hI((size_t)NCH * C * C), hM((size_t)NCH * C * C);
  for (auto& x : hS)  x = bf16(frand() * 0.5f);
  for (auto& x : hKQ) x = bf16(frand());
  for (auto& x : hKR) x = bf16(frand());
  for (auto& x : hI)  x = bf16(frand());
  for (auto& x : hM)  x = bf16(frand());
  bf16 *dS, *dKQ, *dKR, *dI, *dM; float *dDump, *dSink;
  cudaMalloc(&dS, hS.size() * 2); cudaMalloc(&dKQ, hKQ.size() * 2); cudaMalloc(&dKR, hKR.size() * 2);
  cudaMalloc(&dI, hI.size() * 2); cudaMalloc(&dM, hM.size() * 2);
  cudaMalloc(&dDump, (128*32 + 128*16*2 + 128*128) * 4); cudaMalloc(&dSink, (size_t)H * 128 * 4);
  cudaMemcpy(dS, hS.data(), hS.size() * 2, cudaMemcpyHostToDevice); cudaMemcpy(dKQ, hKQ.data(), hKQ.size() * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dKR, hKR.data(), hKR.size() * 2, cudaMemcpyHostToDevice); cudaMemcpy(dI, hI.data(), hI.size() * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dM, hM.data(), hM.size() * 2, cudaMemcpyHostToDevice);

  // global tensors: (rows, k, chunks)
  Tensor mS   = make_tensor(make_gmem_ptr(dS),  make_layout(make_shape(_128{}, _128{}, H),   make_stride(_128{}, _1{}, D*D)));
  Tensor mKQ  = make_tensor(make_gmem_ptr(dKQ), make_layout(make_shape(_32{},  _128{}, NCH), make_stride(_128{}, _1{}, KQ_ROWS*D)));
  Tensor mKR  = make_tensor(make_gmem_ptr(dKR), make_layout(make_shape(_128{}, _16{},  NCH), make_stride(_1{}, _128{}, C*D)));   // (d, c): MN-major A
  Tensor mINV = make_tensor(make_gmem_ptr(dI),  make_layout(make_shape(_16{},  _16{},  NCH), make_stride(_16{}, _1{}, C*C)));
  Tensor mMQK = make_tensor(make_gmem_ptr(dM),  make_layout(make_shape(_16{},  _16{},  NCH), make_stride(_16{}, _1{}, C*C)));

  Copy_Atom tma_S   = make_tma_atom(SM90_TMA_LOAD{}, mS,   SLayoutS{},   Shape<_128,_128>{});
  Copy_Atom tma_KQ  = make_tma_atom(SM90_TMA_LOAD{}, mKQ,  SLayoutKQ{},  Shape<_32,_128>{});
  Copy_Atom tma_KR  = make_tma_atom(SM90_TMA_LOAD{}, mKR,  SLayoutKR{},  Shape<_128,_16>{});
  Copy_Atom tma_INV = make_tma_atom(SM90_TMA_LOAD{}, mINV, SLayoutINV{}, Shape<_16,_16>{});
  Copy_Atom tma_MQK = make_tma_atom(SM90_TMA_LOAD{}, mMQK, SLayoutINV{}, Shape<_16,_16>{});
  Tensor mS_t = tma_S.get_tma_tensor(shape(mS));   Tensor mKQ_t = tma_KQ.get_tma_tensor(shape(mKQ));
  Tensor mKR_t = tma_KR.get_tma_tensor(shape(mKR)); Tensor mI_t = tma_INV.get_tma_tensor(shape(mINV));
  Tensor mM_t = tma_MQK.get_tma_tensor(shape(mMQK));

  int smem = sizeof(SharedStorage) + 1024;
  Params p{NT, dDump, dSink};
  auto launch = [&](auto dbg) {
    auto* kern = &k2_cute_kernel<decltype(dbg)::value, decltype(tma_S), decltype(tma_KQ), decltype(tma_KR), decltype(tma_INV), decltype(tma_MQK),
                                 decltype(mS_t), decltype(mKQ_t), decltype(mKR_t), decltype(mI_t), decltype(mM_t)>;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kern<<<H, 128, smem>>>(tma_S, tma_KQ, tma_KR, tma_INV, tma_MQK, mS_t, mKQ_t, mKR_t, mI_t, mM_t, p);
  };
  printf("smem/CTA = %d B, H=%d NT=%d %s\n", smem, H, NT, debug ? "DEBUG" : "TIMING");

  if (debug) {
    launch(cute::true_type{});
    cudaError_t e = cudaDeviceSynchronize(); if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }
    std::vector<float> dump(128*32 + 128*16*2 + 128*128);
    cudaMemcpy(dump.data(), dDump, dump.size() * 4, cudaMemcpyDeviceToHost);
    // host reference, chunk 0 head 0, mirroring the bf16 rounding points
    std::vector<float> g1(128*32), u(128*16), g2(128*16), ut(128*16), g2b(128*16), g3(128*128);
    for (int v = 0; v < 128; ++v) for (int n = 0; n < 32; ++n) { float a = 0; for (int k = 0; k < 128; ++k) a += bf2f(hS[v*128+k]) * bf2f(hKQ[n*128+k]); g1[v*32+n] = a; }
    for (int v = 0; v < 128; ++v) for (int j = 0; j < 16; ++j) u[v*16+j] = bf2f(bf16(g1[v*32+j]));
    for (int v = 0; v < 128; ++v) for (int n = 0; n < 16; ++n) { float a = 0; for (int k = 0; k < 16; ++k) a += u[v*16+k] * bf2f(hI[n*16+k]); g2[v*16+n] = a; }
    for (int i = 0; i < 128*16; ++i) ut[i] = bf2f(bf16(g2[i]));
    for (int v = 0; v < 128; ++v) for (int n = 0; n < 16; ++n) { float a = 0; for (int k = 0; k < 16; ++k) a += ut[v*16+k] * bf2f(hM[n*16+k]); g2b[v*16+n] = a; }
    for (int d = 0; d < 128; ++d) for (int v = 0; v < 128; ++v) { float a = 0; for (int c = 0; c < 16; ++c) a += bf2f(hKR[c*128+d]) * ut[v*16+c]; g3[d*128+v] = a; }
    auto cmp = [&](const char* nm, const float* got, const float* ref, int n) {
      double md = 0, mr = 0; for (int i = 0; i < n; ++i) { md = fmax(md, fabs(got[i]-ref[i])); mr = fmax(mr, fabs(ref[i])); }
      printf("  %-4s max|diff| = %.3e  (max|ref| = %.3e)  %s\n", nm, md, mr, md <= 2e-3 * fmax(mr, 1.0) ? "OK" : "MISMATCH");
      return md <= 2e-3 * fmax(mr, 1.0);
    };
    bool ok = true;
    ok &= cmp("G1", dump.data(), g1.data(), 128*32);
    ok &= cmp("G2", dump.data() + 128*32, g2.data(), 128*16);
    ok &= cmp("G2'", dump.data() + 128*32 + 128*16, g2b.data(), 128*16);
    ok &= cmp("G3", dump.data() + 128*32 + 128*16*2, g3.data(), 128*128);
    printf("E9 DEBUG: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
  }
  launch(cute::false_type{});
  cudaError_t e = cudaDeviceSynchronize(); if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  const int iters = 10;
  cudaEventRecord(a);
  for (int i = 0; i < iters; ++i) launch(cute::false_type{});
  cudaEventRecord(b); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms, a, b); ms /= iters;
  printf("k2_cute floor: H=%d NT=%d  %.1f us/launch  =>  %.0f cycle/chunk @ %.0f MHz\n", H, NT, ms * 1e3, ms * 1e3 * mhz / NT, mhz);
  return 0;
}
