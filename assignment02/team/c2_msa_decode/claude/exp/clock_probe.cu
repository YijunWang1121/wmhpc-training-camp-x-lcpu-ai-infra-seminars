// Measure the actual SM clock during (a) a tiny latency-bound kernel loop like decode, (b) sustained heavy load.
#include <cuda_runtime.h>
#include <cstdio>
__global__ void probe(long long* out, int spin) {
  long long c0 = clock64(); unsigned long long g0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  float x = threadIdx.x;
  for (int i = 0; i < spin; ++i) x = fmaf(x, 1.0001f, 0.5f);
  long long c1 = clock64(); unsigned long long g1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
  if (threadIdx.x == 0) { out[blockIdx.x * 2] = c1 - c0; out[blockIdx.x * 2 + 1] = g1 - g0; }
  if (x == 12345.f) out[0] = 0;
}
int main() {
  long long* d; cudaMalloc(&d, 148 * 16); long long h[2];
  auto report = [&](const char* name, int blocks, int spin, int reps) {
    for (int r = 0; r < reps; ++r) probe<<<blocks, 128>>>(d, spin);
    cudaDeviceSynchronize(); cudaMemcpy(h, d, 16, cudaMemcpyDeviceToHost);
    printf("%-40s cycles=%lld ns=%lld -> SM clock %.0f MHz\n", name, h[0], h[1], (double)h[0] / h[1] * 1e3);
  };
  report("cold: 4 CTAs, 20k iters, 1 launch", 4, 20000, 1);
  report("4 CTAs, 20k iters, after 200 launches", 4, 20000, 200);
  report("148 CTAs, 200k iters (heavy), 50 launches", 148, 200000, 50);
  report("4 CTAs, 20k iters, right after heavy", 4, 20000, 1);
  return 0;
}
