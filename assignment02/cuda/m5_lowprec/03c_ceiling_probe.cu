// 问题 5.3(c):ceiling 探针。
//
// 目标:测出"5.3(b) 这种访存形状的上限带宽"。方法是写一个和你的
// quant kernel 访存完全同形(读同样的 bf16、写同样位置的 8 byte 数据
// 与 1 byte SF)、但不做任何数学的 kernel——读进来的位 xor 一下直接
// 写出去即可。它的耗时就是这个访存模式在这块卡上的地板。
//
// 跑完把三个数放在一起:探针 GB/s、你的 quant kernel GB/s(03b 的
// 输出)、两者比值。报告里回答:你的 kernel 离自己的上限还有多远,
// 差距是访存还是计算(ncu 的 SM% / DRAM% 可以佐证)。
//
// 在下面实现探针 kernel 和 launch;main 不需要修改。
#include <vector>
#include <random>
#include "../common.h"
#include "nvfp4_common.h"

template <int BLOCK>
__global__ void probe_kernel(const __nv_bfloat16* __restrict__ in,
                             uint8_t* __restrict__ dataOut,
                             uint8_t* __restrict__ sfOut, int M, int K) {
    // 和 nvfp4_quant_kernel 完全同形的访存:同样的线程->组映射,读同一
    // 组的 16 个 bf16(32 B),写同一位置的 8 B 数据 + 1 B swizzled
    // SF,只是把"编码"换成"xor 直通",没有 fmaxf/除法/cvt 这些数学。
    int groupsPerRow = K / NVFP4_GROUP;
    long total = (long)M * groupsPerRow;
    long idx = (long)blockIdx.x * BLOCK + threadIdx.x;
    if (idx >= total) return;
    int r = (int)(idx / groupsPerRow);
    int g = (int)(idx % groupsPerRow);
    int numKTiles = nvfp4_num_ktiles(K);

    const __nv_bfloat16* base = in + (size_t)r * K + (size_t)g * NVFP4_GROUP;
    uint8_t* dst = dataOut + (size_t)r * K / 2 + (size_t)g * 8;
    uint8_t acc = 0;
#pragma unroll
    for (int i = 0; i < NVFP4_GROUP; i += 2) {
        uint16_t b0 = __bfloat16_as_ushort(base[i]);
        uint16_t b1 = __bfloat16_as_ushort(base[i + 1]);
        uint8_t byte = (uint8_t)(b0 ^ b1 ^ (b0 >> 8) ^ (b1 >> 8));
        dst[i / 2] = byte;
        acc ^= byte;
    }
    int64_t sfOffset = sf_swizzled_offset(r, g, numKTiles);
    sfOut[sfOffset] = acc;
}

static void launch_probe(const __nv_bfloat16* in, uint8_t* dataOut,
                         uint8_t* sfOut, int M, int K, int sms) {
    constexpr int BLOCK = 128;
    long total = (long)M * (K / NVFP4_GROUP);
    long grid = (total + BLOCK - 1) / BLOCK;
    if (grid < sms) grid = sms;
    probe_kernel<BLOCK><<<(unsigned int)grid, BLOCK>>>(in, dataOut, sfOut, M,
                                                        K);
}

int main() {
    int sms;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    for (const auto& shape :
         {std::pair{4096, 7168}, {16384, 4096}, {16384, 8192}}) {
        int M = shape.first;
        int K = shape.second;
        size_t n = (size_t)M * K;
        __nv_bfloat16* dx;
        uint8_t *dd, *dsf;
        CUDA_CHECK(cudaMalloc(&dx, n * 2));
        CUDA_CHECK(cudaMalloc(&dd, n / 2));
        CUDA_CHECK(cudaMalloc(&dsf, nvfp4_sf_bytes(M, K)));
        CUDA_CHECK(cudaMemset(dx, 0x3c, n * 2));
        float ms = time_avg_ms(
            [&] { launch_probe(dx, dd, dsf, M, K, sms); }, 50);
        CUDA_CHECK_KERNEL();
        double bytes = n * 2.0 + n * 0.5 + n / 16.0;
        printf("M=%-6d K=%-5d  probe %8.2f us  %6.0f GB/s\n", M, K, ms * 1e3,
               effective_gbps(bytes, ms));
        cudaFree(dx); cudaFree(dd); cudaFree(dsf);
    }
    return 0;
}
