// 问题 5.3(b):实现 NVFP4 quant kernel。
//
// 输入 bf16 矩阵 [M, K](K 是 16 的倍数),输出:
//   dataOut:e2m1 打包数据,每行 K/2 byte,低 nibble 放偶数下标元素
//   sfOut:  e4m3 SF,swizzled 布局(偏移用 nvfp4_common.h 的
//           sf_swizzled_offset;整个 SF 张量已在调用前清零)
//
// 每组的计算顺序(判测按同一顺序生成真值,逐 byte 严格相等):
//   amax = 组内 16 个值的绝对值最大
//   sf8  = __nv_fp8_e4m3(amax / 6.0f)
//   sf   = float(sf8)
//   inv  = sf != 0 ? 1.0f / sf : 0.0f
//   nibble[i] = encode(v[i] * inv)
//
// 设备侧的 e2m1 转换直接用 cuda_fp4.h 的 __nv_fp4x2_e2m1(float2 的 .x
// 进低 nibble),它在 sm_100 家族上是单条硬件指令;你在 5.3(a) 写的
// 编码器语义与它一致,host 参考用的就是它。
//
// 组织建议:16 元素 = 32 byte,一个线程恰好负责一个组,天然免掉组内
// 线程协作;quant 没有行间依赖,grid 怎么铺完全自由。
#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include "nvfp4_common.h"

// 一个线程 = 一个 16 元素组(32 B 输入,8 B e2m1 输出 + 1 B SF),
// 组间没有依赖,线性索引直接铺满 grid。
template <int BLOCK>
__global__ void nvfp4_quant_kernel(const __nv_bfloat16* __restrict__ in,
                                   uint8_t* __restrict__ dataOut,
                                   uint8_t* __restrict__ sfOut, int M, int K) {
    int groupsPerRow = K / NVFP4_GROUP;
    long total = (long)M * groupsPerRow;
    long idx = (long)blockIdx.x * BLOCK + threadIdx.x;
    if (idx >= total) return;
    int r = (int)(idx / groupsPerRow);
    int g = (int)(idx % groupsPerRow);
    int numKTiles = nvfp4_num_ktiles(K);

    const __nv_bfloat16* base = in + (size_t)r * K + (size_t)g * NVFP4_GROUP;
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < NVFP4_GROUP; i++)
        amax = fmaxf(amax, fabsf(__bfloat162float(base[i])));
    __nv_fp8_e4m3 sf8 = __nv_fp8_e4m3(amax / 6.0f);
    float s = float(sf8);
    float inv = s != 0.f ? 1.0f / s : 0.f;

    int64_t sfOffset = sf_swizzled_offset(r, g, numKTiles);
    sfOut[sfOffset] = *reinterpret_cast<uint8_t*>(&sf8);

    uint8_t* dst = dataOut + (size_t)r * K / 2 + (size_t)g * 8;
#pragma unroll
    for (int i = 0; i < NVFP4_GROUP; i += 2) {
        float v0 = __bfloat162float(base[i]) * inv;      // 低 nibble
        float v1 = __bfloat162float(base[i + 1]) * inv;  // 高 nibble
        __nv_fp4x2_e2m1 p(make_float2(v0, v1));
        dst[i / 2] = *reinterpret_cast<uint8_t*>(&p);
    }
}

// 判测和 5.4 会按这个签名调用;grid 大小你自己定,写在这里。
inline void launch_nvfp4_quant(const __nv_bfloat16* in, uint8_t* dataOut,
                               uint8_t* sfOut, int M, int K, int sms) {
    constexpr int BLOCK = 128;
    long total = (long)M * (K / NVFP4_GROUP);
    long grid = (total + BLOCK - 1) / BLOCK;
    // 纯逐组独立工作、无跨组依赖,sms 只用来给一个下限,避免形状很小
    // 时 grid 比 SM 数还少导致明显跑不满。
    if (grid < sms) grid = sms;
    nvfp4_quant_kernel<BLOCK><<<(unsigned int)grid, BLOCK>>>(in, dataOut,
                                                             sfOut, M, K);
}
