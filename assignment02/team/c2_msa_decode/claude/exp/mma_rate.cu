// Microbenchmark: throughput of legacy warp-level mma.sync on this GPU (cycles per instruction per SM sub-partition),
// vs CUDA-core FFMA. Each warp runs a loop of independent MMAs (8 accumulators) so latency is hidden by ILP.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_bf16_k8(float* c, const uint32_t* a, uint32_t b0) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(b0));
}
__device__ __forceinline__ void mma_e4m3(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
template <int MODE>
__global__ void k(float* out, long long* cyc, int iters) {
  uint32_t a[4] = {threadIdx.x * 0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u};
  uint32_t b0 = 0x3c003c00u, b1 = 0x3c003c00u;
  float c[8][4] = {};
  __syncthreads();
  long long t0 = clock64();
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (MODE == 0) mma_bf16(c[j], a, b0, b1);
      else if (MODE == 1) mma_f16(c[j], a, b0, b1);
      else if (MODE == 2) mma_bf16_k8(c[j], a, b0);
      else if (MODE == 3) mma_e4m3(c[j], a, b0, b1);
      else {  // 4 dependent FFMA chains x 8 = same instruction count shape
        c[j][0] = fmaf(c[j][0], 1.0001f, 0.5f); c[j][1] = fmaf(c[j][1], 1.0001f, 0.5f);
        c[j][2] = fmaf(c[j][2], 1.0001f, 0.5f); c[j][3] = fmaf(c[j][3], 1.0001f, 0.5f);
      }
    }
  }
  long long t1 = clock64();
  float s = 0; for (int j = 0; j < 8; ++j) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
  if (threadIdx.x % 32 == 0) cyc[blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32] = t1 - t0;
}
int main() {
  const int iters = 2000, nblk = 148;
  float* out; long long* cyc; cudaMalloc(&out, nblk * 1024 * 4); cudaMalloc(&cyc, nblk * 32 * 8);
  const char* names[] = {"mma.sync m16n8k16 bf16", "mma.sync m16n8k16 f16", "mma.sync m16n8k8 bf16", "mma.sync m16n8k32 e4m3", "FFMA x4 (per inst 4 FMA/thread)"};
  const double flop_per_inst[] = {2.0 * 16 * 8 * 16, 2.0 * 16 * 8 * 16, 2.0 * 16 * 8 * 8, 2.0 * 16 * 8 * 32, 2.0 * 4 * 32};
  for (int mode = 0; mode < 5; ++mode)
    for (int warps : {4, 8, 16}) {
      void (*f)(float*, long long*, int) = mode == 0 ? k<0> : mode == 1 ? k<1> : mode == 2 ? k<2> : mode == 3 ? k<3> : k<4>;
      f<<<nblk, warps * 32>>>(out, cyc, iters); cudaDeviceSynchronize();
      cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
      cudaEventRecord(e0); f<<<nblk, warps * 32>>>(out, cyc, iters); cudaEventRecord(e1); cudaDeviceSynchronize();
      float ms; cudaEventElapsedTime(&ms, e0, e1);
      long long c[32]; cudaMemcpy(c, cyc, warps * 8, cudaMemcpyDeviceToHost);
      double inst_per_warp = 8.0 * iters;
      double cyc_per_inst_warp = (double)c[0] / inst_per_warp;                 // per warp
      double warps_per_smsp = warps / 4.0;
      double tflops = nblk * warps * inst_per_warp * flop_per_inst[mode] / (ms * 1e-3) / 1e12;
      printf("%-34s warps/CTA=%2d (%.0f/SMSP): %6.1f cyc/inst per warp -> %5.1f cyc/inst per SMSP | %7.1f TFLOPS (148 SMs)\n",
             names[mode], warps, warps_per_smsp, cyc_per_inst_warp, cyc_per_inst_warp / warps_per_smsp, tflops);
    }
  return 0;
}
