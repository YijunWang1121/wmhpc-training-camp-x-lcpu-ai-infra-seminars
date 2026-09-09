# Assignment 02 报告

本文件只收纳**没有对应代码文件的纸面题**(handout 中 `.prob` 标注不带
`file=` 的题):0.2、0.3、2.1、3.1、5.5。

其余题目(DERIVE/EXPERIMENT/FROM-SCRATCH/DEBUG 等)自带 `file=` 代码文件的,
报告内容按仓库已有惯例放在各自代码旁边即可,例如:

- 1.5 的 wavefront/cycle 表与解释 → `cuda/m1_sm80/05.md`(已存在)
- 1.1 的附加问题、1.4(a)(b)、2.2 的 MN-major 问题、3.2 的 fence 实验、
  3.3(a)(b)、3.4(a)-(d)、4.1-4.3 的性能表与归因、4.5(a)-(d)、5.1(a)-(c)、
  5.2(a)-(c)、5.3(c) 的三个数字、5.4 的逐形状结果、6.1 的表与(a)(b)
  → 建议各开一个同名 `.md`(参照 `05.md` 的做法),或者告诉我你想统一放
  在这一个文件里,我再补一版汇总模板。

`0.1`、`0.3(c)`、`0.3(d)` 你已经直接写在 `handout/src/assignment02.md`
里了,如果想统一收口到这份报告,自己誊过来就行,我不替你搬。

提交要求原文(见 handout 末尾「提交」一节): 报告需包含纸面题解答、实验
表格与性能归因，以及 DEBUG 题的现象记录和修改说明；所有实验数据注明
使用的 GPU。

---

## 0.2 Tensor Core 理论峰值推导 {.prob type=DERIVE}

口径声明(dense/sparse、boost/base 频率、FMA 是否算 2 FLOP 等):

> 

| 量 | 5090 | B300 |
|---|---|---|
| bf16 FLOP/cycle/SM | | |
| bf16 峰值(TFLOPS) | | |
| fp8 峰值(TFLOPS) | | |
| fp4 峰值(TFLOPS) | | |
| datasheet 对照值与口径差异 | | |
| HBM/GDDR 带宽(GB/s) | | |
| 机器平衡点(FLOP/byte，bf16) | | |

与单条 mma 的计算强度(m16n8k16 fp16 为 3.2 FLOP/byte)比较,差距意味着
什么?为什么后续 M2-M4 要从数据供给路径入手优化?

> 

---

## 0.3 mma / 计算强度 概念判断 {.prob type=CONCEPT}

(a) 一条 mma 的计算强度,分子是 $2MNK$,分母按 A、B 读入与 D 写回的字节
总和计。

> 对/错:
> 理由:

(b) mma.sync 是 warp 级协作指令:32 个 lane 各持 fragment 的一部分,要求
全 warp 一致地执行这条指令;有 lane 发散时行为未定义。

> 对/错:
> 理由:

(c) 增大 mma 的形状 M/N/K 能提高单条指令的计算强度,而且没有代价,所以
指令形状越大越好。

> 对/错:
> 理由:

(d) 只要单条 mma 的计算强度低于机器平衡点,GEMM kernel 就不可能逼近
计算峰值。

> 对/错:
> 理由:

---

## 2.1 wgmma 操作定序 {.prob type=CONCEPT}

(a) 排出下面六个操作的正确顺序,并说明每一步用于避免哪两个参与者之间
的哪种乱序:

`wgmma.mma_async` / `st.shared` / `wgmma.commit_group` /
`fence.proxy.async` / `wgmma.fence` / `wgmma.wait_group`

> 顺序:
>
> 每一步的作用:
> 1.
> 2.
> 3.
> 4.
> 5.
> 6.

(b) 判断下列说法是否正确,并给出一句理由。

1. `fence.proxy.async` 是 wgmma 专属的指令,TMA 与 tcgen05 的场景不需要它。

> 对/错:
> 理由:

2. `wgmma.commit_group` 会阻塞,直到它之前发射的 wgmma 全部完成。

> 对/错:
> 理由:

3. 不加 `fence.proxy.async` 时,wgmma 可能读到 shared memory 中的旧值,
   因为 `st.shared` 的写经过 generic proxy,而 wgmma 的读经过 async proxy。

> 对/错:
> 理由:

---

## 3.1 tcgen05 / TMEM / mbarrier 概念判断 {.prob type=CONCEPT}

(a) `tcgen05.ld` 读取 TMEM 时,每个 warp 只能读取自己对应的 32 条 lane,
warp 之间不能互相读取。

> 对/错:错
> 理由:warp只能读取lane 32i-32i+31.

(b) 与 `mma.sync` 由 warp 协作、wgmma 由 warpgroup 协作不同,`tcgen05.mma`
由单个线程发射,随后由硬件异步执行。

> 对/错:对
> 理由:

(c) TMEM 中的累加结果可以直接通过 TMA 搬回 global memory,不需要经过
寄存器。

> 对/错:错
> 理由:要经过register先

(d) TMEM 每个 SM 包含 128 lane × 512 column × 4 B;一个 m128n256 的 f32
accumulator 恰好占用其中一半。(需要写出计算过程)

> 对/错:对
> 计算过程:128×256×4刚好为一半

(e) `tcgen05.commit` 会阻塞直到之前发射的 mma 全部完成,因此 commit 返回
后即可安全读取 TMEM。

> 对/错:错
> 理由:还要等mbarrier

---

## 5.5 W4A16+Marlin vs NVFP4 {.prob type=CONCEPT}

(a) 两者分别属于存储量化还是计算量化?

> 

(b) 两种方法分别节省哪些资源:显存容量、显存带宽还是计算吞吐?

> 

(c) 在 4.5 中的小 batch decode 场景下,哪一类量化的收益更加直接?

> 
