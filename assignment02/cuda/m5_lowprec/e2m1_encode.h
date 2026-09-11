// 问题 5.3(a):实现 e2m1 的 round-to-nearest-even 编码。
//
// e2m1 的幅值格点是 0, 0.5, 1, 1.5, 2, 3, 4, 6(编码 0-7),bit 3 是
// 符号位。要求与硬件 cvt 指令(cvt.rn.satfinite.e2m1x2.f32)的语义
// 一致:round to nearest,恰好落在两个格点中点时取尾数为偶的那个,
// 大于 6 饱和到 6(satfinite)。输入保证是有限值。
//
// 这个函数是后面所有题目 host 参考实现的基石:5.3(b) 的判测、5.4 的
// 判测都用它生成真值,所以先用 03a_encode_check 把它和硬件逐点对齐。
//
// 提示:先把每个中点(0.25、0.75、1.25、1.75、2.5、3.5、5.0)该落到
// 哪边推清楚,再写代码。__host__ __device__ 两侧都要能编译。
#pragma once
#include <cstdint>
#include <math.h>

__host__ __device__ inline uint8_t e2m1_encode(float v) {
    // 幅值格点(下标 0-7):0, 0.5, 1, 1.5, 2, 3, 4, 6。相邻格点中点
    // 依次是 0.25/0.75/1.25/1.75/2.5/3.5/5.0;round-to-nearest-even
    // 在每个中点上都恰好偏向下标为偶数的那一侧(0/2/4/6 对应
    // 0/1/2/4,都是"整数"格点),所以每段边界该并入哪一侧、包不包含
    // 等号,直接由这条奇偶规则决定,不需要单独查表。
    // signbit 而不是 v<0.f:要把 -0.0 的符号位也保留下来(判测里对
    // 全部候选值取了负号镜像,包含 -0.0)。
    uint8_t sign = signbit(v) ? 0x8 : 0x0;
    float a = fabsf(v);
    uint8_t mag;
    if (a <= 0.25f)      mag = 0;  // -> 0
    else if (a < 0.75f)  mag = 1;  // -> 0.5
    else if (a <= 1.25f) mag = 2;  // -> 1
    else if (a < 1.75f)  mag = 3;  // -> 1.5
    else if (a <= 2.5f)  mag = 4;  // -> 2
    else if (a < 3.5f)   mag = 5;  // -> 3
    else if (a <= 5.0f)  mag = 6;  // -> 4
    else                 mag = 7;  // -> 6,satfinite 饱和
    return sign | mag;
}
