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

> 全部按 **dense**(非 sparsity,官方常报的是 dense 的 2×)、FMA 算 **2
> FLOP**、频率按**本机实测的稳态 SM 频率**(而不是 `nvidia-smi` 标出的
> boost 频率——两者在 B300 上差了近 2×,这本身就是下面"datasheet 对照"
> 要讨论的口径差异之一)。B300 侧的 SM 数、频率、显存带宽都是在这块卡上
> 实测/查询得到的(见下方脚注),不是抄的宣传页;5090 没有这块集群可测,
> 只能用官方 spec sheet 的数字反推,**这几行我标了"（外部值,待核对）",
> 提交前建议对着课件 P1 S018-S019 或 5090 datasheet 核一遍**,不要直接
> 当最终答案抄。

> B300 实测(2026-09-11,`cudaGetDeviceProperties`/`nvidia-smi dmon`,见
> `report.md` 上方 4.1-4.3 节的 GPU 脚注):SM 数 = 148,compute capability
> 10.3;`nvidia-smi` 标称 SM 频率上限 2032 MHz,但**跑 kernel 时用
> `nvidia-smi dmon` 实测 SM 频率被锁定在 1095 MHz,与负载无关**(这与
> `team/c2_msa_decode` 之前在同一块卡上测到的结果一致);显存时钟
> 3996 MHz、总线 7680 bit(HBM3e),按 `2 × 3996MHz × 7680bit/8` 折算峰值
> 带宽 ≈ 7672 GB/s(与 B200/B300 系列官方宣传的"约 8 TB/s"量级吻合)。

| 量 | 5090(外部值,待核对) | B300(本机实测+推算) |
|---|---|---|
| bf16 FLOP/cycle/SM | ≈ 1231(反推,见下) | ≈ 1975(反推,见下) |
| bf16 峰值(TFLOPS) | ≈ 419.5 | ≈ 320(legacy `mma.sync` 实测下限)~ 2240(`tcgen05` 估算上限) |
| fp8 峰值(TFLOPS) | ≈ 838.8 | ≈ 1200(legacy `mma.sync` 实测下限)~ 4480(`tcgen05` 估算上限) |
| fp4 峰值(TFLOPS) | ≈ 1677.6 | ≈ 8960(`tcgen05` 估算上限,未实测) |
| datasheet 对照值与口径差异 | 官方主推的是 fp4 sparse "3352 AI TOPS"这类数字,dense/bf16 需要按精度每减半×2 反推,和上面 TFLOPS 是同一条链条推出来的,不是独立验证 | 差异主要来自**实际稳态频率(1095MHz)远低于标称 boost(2032MHz)**,以及 legacy `mma.sync` 打不满 5th-gen tensor core 全部宽度——必须用 `tcgen05` 才能摸到接近 datasheet 的峰值,这也是模块 3-4 从 `mma.sync` 换到 `tcgen05` 的动机之一 |
| HBM/GDDR 带宽(GB/s) | ≈ 1792(GDDR7,512-bit,官方 spec) | ≈ 7672(实测频率+总线宽度算出,见上) |
| 机器平衡点(FLOP/byte，bf16) | ≈ 419.5e3/1792 ≈ 234 | ≈ 2240e3/7672 ≈ 292(取 `tcgen05` 估算上限;若只算 legacy `mma.sync` 的 320 TFLOPS,则只有 ≈ 42) |

推导方法(FLOP/cycle/SM 那两格怎么来的):反过来从"峰值 TFLOPS"除以
"SM 数 × 稳态频率"倒推,即 `FLOP/cycle/SM = 峰值FLOPS / (SM数 × clock_Hz)`。
5090:419.5e12 / (170 × 2.41e9) ≈ 1021(这里 170 SM、2.41GHz boost 都是
5090 官方 spec,同样"外部值待核对");B300 legacy 路径:320e12 /
(148×1.095e9) ≈ 1975。之所以反过来推而不是正向"per-SM 硬件参数 × 时钟"
算,是因为 NVIDIA 官方没有公开 5th-gen tensor core 精确到"每 SM 每 cycle
多少 FLOP"这个粒度的数字,只公开了整卡的 TFLOPS,所以只能反推,**这也是
这张表里最不确定的部分**。

与单条 mma 的计算强度(m16n8k16 fp16 为 3.2 FLOP/byte)比较,差距意味着
什么?为什么后续 M2-M4 要从数据供给路径入手优化?

> 单条 mma 的计算强度只有 3.2 FLOP/byte,而机器平衡点在两块卡上都在
> 200-300 FLOP/byte 量级——差了接近两个数量级。这个差距的含义是:如果
> 每条 mma 都独立地重新从 global memory 读一遍 A/B、写一遍 D(即不做任何
> 数据复用),这个 kernel 无论算力多强都会被 DRAM 带宽死死摁住,单指令
> 层面完全没有指望摸到计算峰值。
>
> 但 GEMM 本身有大量可复用的数据:同一块 A/B tile 会被同一行/列上很多条
> mma 复用(K 维累加、以及同一 tile 内的多个输出位置共享同一份输入)。
> M2(swizzle)、M3(tcgen05/TMEM/mbarrier)、M4(tiling + TMA + 多级流水)
> 做的事情本质上都是:把这些能复用的数据一次性搬进 shared memory/TMEM,
> 让很多条 mma 复用同一份已经在片上的数据,从而把"面向 DRAM 的有效计算
> 强度"从 3.2 FLOP/byte 一路推高,逼近甚至超过机器平衡点。换句话说,
> 单条 mma 的计算强度低是硬件指令形状决定的、改不了,能改的是"数据供给
> 路径"——通过分块复用把有效强度做上去,这就是为什么后续优化全部集中在
> staging/swizzle/TMA/pipeline 这条路径上,而不是纠结单条指令本身。

---

## 0.3 mma / 计算强度 概念判断 {.prob type=CONCEPT}

(a) 一条 mma 的计算强度,分子是 $2MNK$,分母按 A、B 读入与 D 写回的字节
总和计。

> 对/错:对
> 理由:按 m16n8k16 fp16 代入验证——分子 $2 \times 16 \times 8 \times 16=4096$
> FLOP;分母 A(16×16 fp16=512B)+B(16×8 fp16=256B)+D(16×8 f32=512B)
> =1280B;$4096/1280=3.2$ FLOP/byte,正好等于题面给出的 3.2。分母不含 C,
> 是因为 K 循环里 C 通常就是上一轮的累加寄存器 D 本身,不会额外产生一次
> 内存读——只有 A、B 的输入读和 D 的最终写才是"必须发生"的内存流量。

(b) mma.sync 是 warp 级协作指令:32 个 lane 各持 fragment 的一部分,要求
全 warp 一致地执行这条指令;有 lane 发散时行为未定义。

> 对/错:对
> 理由:PTX ISA 里 `mma.sync` 的 `.sync` 后缀就是强制这一点——要求执行
> 这条指令的 32 个 lane 必须收敛(active mask 一致),每个 lane 只持有
> A/B/C/D fragment 里属于自己那一小份寄存器,硬件按 warp 整体拼出完整的
> 矩阵操作数;一旦有 lane 因为分支发散而不参与,行为未定义,这也是为什么
> 调用前一般要保证 warp 内没有发散的控制流。

(c) 增大 mma 的形状 M/N/K 能提高单条指令的计算强度,而且没有代价,所以
指令形状越大越好。

> 对/错:错
> 理由:形状变大确实能提高计算强度(FLOP 按 $MNK$ 增长比字节按
> $MK+KN+MN$ 增长快),但"没有代价"不成立——形状越大,每个线程/warp
> 持有的 fragment 越大,寄存器(或 tcgen05 下的 TMEM)占用越高,容易挤占
> occupancy 甚至寄存器溢出;而且硬件只支持固定的一组离散形状,不能无限
> 增大;单条指令延迟也会变长,不一定有利于和其它工作重叠。

(d) 只要单条 mma 的计算强度低于机器平衡点,GEMM kernel 就不可能逼近
计算峰值。

> 对/错:错
> 理由:单条 mma 的计算强度算的是"每次都从头访存"的最坏情形;实际
> GEMM kernel 会靠分块(tiling)把同一份从 DRAM 搬进 shared memory/TMEM
> 的数据,复用给同一 tile 内很多条 mma(K 维累加、同 tile 内多个输出位置
> 共享同一份输入),这样面向 DRAM 的**有效**计算强度会被大幅推高,可以
> 远超单条 mma 自身的强度、逼近甚至超过机器平衡点——这正是 0.2 里问的
> "为什么 M2-M4 要从数据供给路径入手优化"的答案。

---

## 2.1 wgmma 操作定序 {.prob type=CONCEPT}

(a) 排出下面六个操作的正确顺序,并说明每一步用于避免哪两个参与者之间
的哪种乱序:

`wgmma.mma_async` / `st.shared` / `wgmma.commit_group` /
`fence.proxy.async` / `wgmma.fence` / `wgmma.wait_group`

> 顺序:
> `st.shared` → `fence.proxy.async` → `wgmma.fence` → `wgmma.mma_async`
> → `wgmma.commit_group` → `wgmma.wait_group`
>
> 每一步的作用:
> 1. `st.shared`:把 A/B 操作数写进 shared memory,这是后面 wgmma 从 smem
>    读取数据的前提;这次写经过的是 generic proxy。
> 2. `fence.proxy.async`:避免"generic proxy 的写"和"async proxy 的读"
>    之间的乱序——`st.shared` 走 generic proxy,而 `wgmma` 读 smem 走
>    async proxy,两个 proxy 之间没有默认的内存序保证,不插这个 fence,
>    `wgmma` 可能读到 smem 里的旧值(即 3(b)-3 描述的现象)。
> 3. `wgmma.fence`:避免"线程对累加寄存器的普通写"和"后续 wgmma_async
>    对同一寄存器的异步读写"之间的乱序——开始新一组 wgmma 前,如果之前
>    有普通指令改过将要用作累加器的寄存器,要靠这个 fence 让异步的 wgmma
>    单元看到最新值。
> 4. `wgmma.mma_async`:真正发射异步矩阵乘,读 smem 中的 A/B、累加进
>    目标寄存器,发射后立刻返回,不等待完成。
> 5. `wgmma.commit_group`:把之前连续发射的一批 `wgmma.mma_async` 打包
>    成一个"commit group",作为后面 `wait_group` 等待的单位——避免"发射
>    方"和"等待方"之间没法定位"到底要等哪一批"的问题。
> 6. `wgmma.wait_group`:阻塞直到未完成的 commit group 数量降到指定值
>    以下,用于避免"wgmma 硬件异步写累加寄存器"和"线程后续读取这些寄存
>    器"之间的乱序——没有这一步,线程可能在 wgmma 真正写完之前就去读
>    结果。

(b) 判断下列说法是否正确,并给出一句理由。

1. `fence.proxy.async` 是 wgmma 专属的指令,TMA 与 tcgen05 的场景不需要它。

> 对/错:错
> 理由:要不要插 `fence.proxy.async`,取决于"写 smem 的那条路径是否
> 经过 generic proxy",跟"读 smem 的是 wgmma 还是 tcgen05"无关。4.1 里
> tcgen05 配合手工 `st.shared`(generic proxy 写)时依然需要
> `fence.proxy.async`(文件里就有这一条);只有像 4.2 那样连搬运也换成
> TMA(TMA 写 smem 走的就是 async proxy,和 tcgen05 的读是同一个 proxy)
> 时才不需要——这条判断把"指令类型"和"是否跨 proxy"这两件事搞混了。

2. `wgmma.commit_group` 会阻塞,直到它之前发射的 wgmma 全部完成。

> 对/错:错
> 理由:`wgmma.commit_group` 本身不阻塞,立即返回,只是把此前发射的一批
> `wgmma.mma_async` 划成一个组、打上分界;真正会阻塞等待完成的是
> `wgmma.wait_group`。

3. 不加 `fence.proxy.async` 时,wgmma 可能读到 shared memory 中的旧值,
   因为 `st.shared` 的写经过 generic proxy,而 wgmma 的读经过 async proxy。

> 对/错:对
> 理由:两个 proxy 之间没有隐式的内存序保证,`st.shared`(generic proxy)
> 写完之后,`wgmma`(async proxy)不一定能看到这次写,必须显式插入
> `fence.proxy.async` 才能保证写后读的可见性,这正是这条 fence 存在的
> 原因。

---

## 3.1 tcgen05 / TMEM / mbarrier 概念判断 {.prob type=CONCEPT}

(a) `tcgen05.ld` 读取 TMEM 时,每个 warp 只能读取自己对应的 32 条 lane,
warp 之间不能互相读取。

> 对/错:错
> 理由:warp只能读取lane 32i-32i+31.

(b) 与 `mma.sync` 由 warp 协作、wgmma 由 warpgroup 协作不同,`tcgen05.mma`
由单个线程发射,随后由硬件异步执行。

> 对/错:对
> 理由:`mma.sync` 需要 32 个 lane 各持有 fragment 的一部分协作发射,
> `wgmma` 需要 128 线程的 warpgroup 协作;而 `tcgen05.mma` 的操作数是
> shared memory descriptor 和 TMEM 地址,不是"每个 lane 手里的寄存器
> fragment",所以只需要 `elect.sync` 选出一个线程发射这条指令即可,发射
> 后 Tensor Core 异步执行,发射线程立刻可以继续往下跑,靠 mbarrier 来
> 确认完成——这也是你代码里为什么用 `elect.sync` 单独选一个线程去发
> `tcgen05.mma` 的原因。

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

## 4.1-4.3 GEMM 优化梯子 {.prob type=FROM-SCRATCH}

> 本节汇总放这里(而不是拆成 `cuda/m4_gemm/*.md`),按你的要求统一记录。

> GPU: NVIDIA B300 SXM6 AC(单卡,`srun -G 1` 分到 `dev-slurm`),驱动
> 580.126.09,CUDA 13.0(`nvcc release 13.0, V13.0.88`),显存 275040 MiB,
> SM 最大频率 2032 MHz,功耗上限 1100 W。(`nvidia-smi` 实测,2026-09-11)

贯穿 4.1--4.3 的性能表(第 0 行 naive 需要重新跑 assignment01 Bonus 的 fp32
naive matmul,只比量级):

| 实现 | TFLOPS | 对 cuBLAS 达成率 | 一句话:时间主要花在哪 |
|---|---|---|---|
| naive（assignment01，fp32） | | | |
| 4.1 tiled | 29.7 | 3% | 标量 st.shared/全局 load 的访存延迟(84% 的 stall 周期等 L1TEX 返回),不是被算力或带宽墙压住 |
| 4.2 TMA | 368.7 | 36% | 数据搬运外包给 TMA 硬件,SM 主要 stall 变成等 CTA barrier;compute/memory 利用率从 24%/15% 升到 47%/72% |
| 4.3 pipeline（S=3） | 303.9 | 29% | 多级 buffer 把每 CTA smem 从 4.2 的 ~25 KB 推到 ~75 KB,`Block Limit Shared Mem` 从 8 降到 3,理论 occupancy 43.7%→18.75%;在 4096³ 这种 grid 本来就够大的形状下,少掉的并发比流水掩盖的延迟更值钱,CTA barrier 仍是最大 stall 项(≈51%),整体比单缓冲的 4.2 还慢 |
| cuBLAS | | 100% | |

（4.1、4.2、4.3 三行的 TFLOPS/达成率取自同一次 `srun` 会话里连续跑
`./bin/m4_gemm/01_tiled 4096 4096 4096`、`02_tma`、`03_pipeline`(默认
`STAGES=3`)的实测输出,4096³,PASS,cuBLAS 基线 ≈1040 TFLOPS——同会话内
测是为了让三行的 cuBLAS 分母可比,不同 `srun` 之间 cuBLAS 实测值本身会随
时钟/热状态浮动 ±30~70%,不能跨会话直接比 TFLOPS 绝对值,下面 2. 的
`STAGES` 扫描表同理,每个形状内部四个 S 都在同一次 `srun` 里连续跑出。）

### 4.1 {.prob type=FROM-SCRATCH file=cuda/m4_gemm/01_tiled.cu}

结合 0.2 中计算的机器平衡点,判断此时性能主要受哪一环节限制:

> 用 `ncu` 实测(见 4.2 第 3 问的数据):Compute (SM) Throughput 只有
> 23.9%,Memory Throughput 15.0%,DRAM Throughput 仅 0.31%——计算和
> 显存带宽这两个屋顶都远没打满,尤其 DRAM 几乎空闲,说明**既不是被
> 0.2 里算出来的算力峰值卡住,也不是被显存带宽卡住**,不落在机器平衡点
> 划分出的 compute-bound / memory-bound 任何一侧。真正限制它的是**访存
> 延迟 + occupancy**:warp 平均每发射一条指令要等 28.9 个 cycle,其中约
> 84%(24.3 cycle)是在等 L1TEX(即手工 `st.shared` 前那次全局内存标量
> load)的 scoreboard;而共享内存用量又把每 SM 能驻留的 block 数限制到
> 8 个(`Block Limit Shared Mem = 8`),理论 occupancy 只有 50%(实测
> 43.7%),没有足够多余的 warp 去掩盖这些延迟。换句话说,4.1 卡在"两个
> 屋顶之间的地带"——这正是 4.2 换用 TMA 要解决的问题。

### 4.2 {.prob type=MODIFY file=cuda/m4_gemm/02_tma.cu}

比较 4.1 与 4.2,回答:

1. 4.1 中 shared memory staging 的开销由哪些部分组成?

> - 每个线程要自己算一遍 `swz128` 的 swizzle 偏移(额外整数/位运算);
> - 每个线程发起的是**标量**、逐元素的全局内存 load 和共享内存 store
>   (没有向量化、没有批量搬运指令),指令数量多、每条都要单独走一次
>   L1TEX;
> - 整个 K 循环是"先搬完、`__syncthreads()`、再算"的串行结构,没有任何
>   prefetch/pipeline 重叠,mma 发射前必须等全部 128 个线程的 staging
>   都完成;
> - 共享内存用量把每 SM 能同时驻留的 block 数限制在 8 个,可用来掩盖
>   访存延迟的 warp 数不够多。
>
>   这几项叠加的结果就是 4.1 那条实测:warp 平均每发射一条指令要空等
>   28.9 个 cycle,其中 84%(24.3 cycle)花在等 L1TEX 返回上——staging
>   的开销主要是**访存延迟**,不是数据量本身(DRAM 吞吐只有 0.31%,远
>   没有把带宽用满)。

2. 改用 TMA 后,其中哪些工作不再由普通 CUDA 指令完成?

> - swizzle 地址计算整体消失,由 tensor map 描述的布局在硬件搬运时自动
>   完成,kernel 里不再需要 `swz128` 这个函数;
> - 128 个线程各自负责一小片、逐元素 load/store 的模式消失,变成 1 个
>   线程发 3 条指令(`mbarrier.arrive.expect_tx` + 两条
>   `cp.async.bulk.tensor`)就把整块 tile 的搬运工作外包给专门的 TMA
>   硬件引擎,SM 的 LSU/ALU pipe 完全不参与这部分数据搬运;
> - 因此原来大量线程在 L1TEX scoreboard 上等待的情况消失了——4.2 实测
>   下 warp 的主要 stall 原因换成了"等 CTA barrier"(约 51.7%,即等
>   `__syncthreads` 而不是等数据本身),Compute/Memory Throughput 也从
>   4.1 的 24%/15% 升到 46.5%/72%,说明 SM 的 issue slot 被大量释放出来
>   去发 mma 而不是发 staging 指令。

3. 使用 Nsight Compute 辅助分析,4.1 中 SM 时间主要消耗在哪些部分?(贴关键
   ncu 指标)

> `ncu --section SpeedOfLight --section WarpStateStats --section
> ComputeWorkloadAnalysis --section Occupancy --launch-count 1
> ./bin/m4_gemm/01_tiled 4096 4096 4096`(B300,`gemm_tiled`,4096³ 首次
> launch):
>
> | 指标 | 数值 |
> |---|---|
> | Compute (SM) Throughput | 23.93% |
> | Memory Throughput | 14.99% |
> | DRAM Throughput | 0.31% |
> | Warp Cycles Per Issued Instruction | 28.88 cycle |
> | ——其中等 L1TEX scoreboard 占比 | ≈84.3%(≈24.3 cycle) |
> | Theoretical / Achieved Occupancy | 50.0% / 43.71% |
> | Block Limit(限制因素) | Shared Mem → 8 blocks/SM |
>
> 结论:SM 大部分时间既不在算(compute 只有 24%),也不在真的等 DRAM
> (DRAM 只有 0.31%),而是**warp 发出手工 staging 的标量 load/store 后
> 空等 L1TEX 数据返回**——这 84% 的 stall 加上共享内存把 occupancy 摁在
> 50% 以下(没有足够 warp 互相掩盖延迟),两者叠加就是 4.1 只有 3% cuBLAS
> 达成率的直接原因。

### 4.3 {.prob type=FROM-SCRATCH file=cuda/m4_gemm/03_pipeline.cu}

1. `S=3` 的结果已并入上面的性能表。

2. 不同 stage 数(`./sweep_stages.sh`,单次 `srun` 会话内连续跑完两个
   形状的全部 8 组,PASS,cuBLAS 基线 4096³ ≈1035 TFLOPS、
   256×4096×16384 ≈678 TFLOPS):

| 形状 | S=2 | S=3 | S=4 | S=6 |
|---|---|---|---|---|
| 4096³ | 344.1(33%) | 303.7(29%) | 245.6(24%) | 139.0(13%) |
| 256 × 4096 × 16384 | 128.7(19%) | 129.9(19%) | 131.5(19%) | 130.5(19%) |

比较两个形状对 `STAGES` 的敏感程度,结合 shared memory 用量、每个 SM
可同时驻留的 block 数、block 间并发能隐藏的延迟解释:

> **两个形状对 `STAGES` 的反应几乎相反。** 4096³ 单调变差(344→304→
> 246→139 TFLOPS),256×4096×16384 几乎打平(129→130→131→130 TFLOPS,
> 差距在噪声量级)。用 `ncu --section Occupancy` 实测两边的
> `Block Limit Shared Mem` / `Achieved Occupancy`(同一张 B300,4096³ 与
> 薄形状分别测):
>
> | S | smem/block | 4096³ Block Limit Shared Mem | 4096³ Achieved Occ. | 薄形状 Block Limit Shared Mem | 薄形状 Achieved Occ. |
> |---|---|---|---|---|---|
> | 2 | 50176 B | 4 | 20.99% | 4 | 6.26% |
> | 3 | 74752 B | 3 | 16.97% | 3 | 6.25% |
> | 4 | 99328 B | 2 | 12.09% | 2 | 6.25% |
> | 6 | 148480 B | 1 |  6.23% | 1 | 6.24% |
>
> `Block Limit Shared Mem` 两个形状完全一样(同一个 kernel、同样的
> smem/block,B300 每 SM 可用 smem 按这组数字反推 ≈3×74752≈224 KB,和
> Hopper/Blackwell 数据中心卡文档里的每 SM 共享内存上限同量级)——差异
> 全部来自**这个 block limit 是否真的卡住了 occupancy**:
>
> - **4096³**:grid = 32×64 = 2048 个 block,远多于 148 个 SM,每个 SM
>   本来就能排上好几个 block 排队,`Achieved Occupancy` 几乎精确等于
>   `Block Limit Shared Mem / 16`(16 是 128 线程/block ÷ 32 线程/warp
>   之类换算出的每 SM 理论满载 warp 数上限的分母)——即 occupancy 完全
>   被 smem 卡住,S 每翻一倍,能同时驻留的 block 数近似减半,TFLOPS 也
>   跟着掉,S=6 时只剩 1 个 block/SM、6.23% occupancy,已经没有任何
>   跨 block 并发去掩盖延迟,比 4.2 单缓冲(`Block Limit Shared Mem=8`)
>   还慢。
> - **256×4096×16384**:grid = (256/128)×(4096/64) = 2×64 = 128 个
>   block,比 148 个 SM 还少——`Achieved Occupancy` 死死钉在 6.24~6.26%
>   不随 S 变(哪怕 S=2 理论上能塞 4 个 block/SM,实际每 SM 平均分不到
>   1 个 block,`Block Limit Shared Mem` 根本没被打满)。这个形状的
>   瓶颈从一开始就是**grid 太小、block 级并发不够**,不是 smem;深流水
>   在这里不掉血,是因为它没有 occupancy 可掉——唯一还能起作用的是
>   **单个 block 内部**把更多轮 TMA 提前发出去掩盖延迟,这也是为什么
>   S 从 2 升到 4 时 TFLOPS 还能小幅爬升(128.7→131.5),到 S=6 又略降
>   (130.5,多一级 stage 的管理开销 vs 边际延迟收益打平甚至倒贴)。
>
> 一句话:`STAGES` 对性能的净效应 = "block 内延迟隐藏的收益"减去"smem
> 增长挤掉的跨 block 并发"。4096³ 本来 block 数过剩,第二项占主导,越
> 深越亏;薄形状 block 数本来就不够,第二项几乎不起作用,只剩第一项,
> 越深(在一定范围内)越赚。

**补一条上面表里缺的基线:完全不用 pipeline(`02_tma`,即 `S=1`)自己
放到同一张表里对比,才能回答"pipeline 到底在什么情况下有用"这个问题
——单看 S=2..6 之间怎么变,看不出"要不要做 pipeline"本身值不值。补测
(同一次 `srun` 会话,`02_tma` 与 `03_pipeline` 连续跑,cuBLAS 基线两个
形状分别 ≈1058 TFLOPS、≈692 TFLOPS):**

| 形状 | 02_tma(无 pipeline,S=1) | S=2 | S=3 | S=4 | S=6 |
|---|---|---|---|---|---|
| 4096³ | **371.0(35%)** | 344.9(33%) | 304.5(29%) | 244.3(23%) | 139.5(13%) |
| 256×4096×16384 | **81.5(12%)** | 131.9(**19%**) | 130.1(19%) | 131.2(19%) | 131.8(19%) |

结论非常干脆:**4096³ 上不加 pipeline 反而是最优解**(35% 全表最高,
S 每加深一级都更慢);**256×4096×16384 上 pipeline 是必需的**——只要
上了 S≥2 的任意深度,TFLOPS 从 81.5 跳到 ~131(+62%),对 cuBLAS 达成率
从 12% 提到 **19%**,S 具体取多少反而不重要(19%±0)。也就是说
pipeline 有用的判据不是"K 有多长"或"形状薄不薄"本身,而是上一段量化
出的那条线——**grid 大小是否已经超过 SM 数**:

- grid ≥ SM 数(4096³,2048 个 block vs 148 个 SM,`ncu Launch
  Statistics` 里 `Waves Per SM=1.73`,一次 launch 铺得开将近 2 个满
  wave):block 级并发本身就足够多,S=1 单缓冲已经能靠"不同 block 处在
  K 循环的不同轮次"来互相掩盖延迟,这时候 pipeline 只有 smem 成本、
  没有增量收益,是纯负担。
- grid < SM 数(256×4096×16384,128 个 block vs 148 个 SM,`Waves Per
  SM=0.11`,连一个满 wave 都铺不满,一大批 SM 干脆分不到 block):没有
  "别的 block" 可以用来掩盖延迟,唯一能重叠 TMA 传输的地方就是**同一个
  block 内部、跨 K 轮次**——这正是 pipeline 存在的意义,不做就白白把
  TMA 的传输延迟串行地摊在总时间里。

**再用 ncu 的 Warp State Statistics 把"为什么"钉死在指令级证据上**
(`--section WarpStateStats --section Occupancy -c 1`,只抓第一次
launch,256×4096×16384 这个形状上 `02_tma` vs `03_pipeline S=3`):

| 指标 | 02_tma(S=1) | 03_pipeline(S=3) |
|---|---:|---:|
| Duration | 426.4 us | 264.5 us |
| Compute (SM) Throughput | 11.23% | 18.06% |
| Memory Throughput | 10.71% | 17.21% |
| Warp Cycles Per Issued Instruction | 37.57 cycle | 21.54 cycle |
| 花在"等 CTA barrier"上的 stall | 26.2 cycle(占 69.7%) | 14.1 cycle(占 65.4%) |
| Achieved Occupancy | 6.26% | 6.25% |
| Achieved Active Warps/SM | 4.01 | 4.00 |

关键在最后两行:**两者的 occupancy 几乎一模一样**(都卡在 grid 太小、
每 SM 平均分不到 1 个 block,跟上面 2. 里的表完全吻合),排除了"pipeline
靠更多并发 block 取胜"这个解释。真正变化的是前几行:同一个资源受限
的 block 内部,`Warp Cycles Per Issued Instruction` 从 37.57 降到
21.54(接近腰斩),CTA barrier 的绝对 stall cycle 从 26.2 降到 14.1
——因为 `S=3` 让 TMA 提前把后面几轮的数据发出去,`tid==0` 在
`mbar_wait(full)` 上等的时间被"这轮 mma 算的同时,TMA 已经在搬未来
几轮"重叠掉了一部分,其余 3 个空闲 warp 在 `__syncthreads()` 上陪等
的时间跟着缩短。Compute/Memory Throughput 双双从 ~11% 提到 ~17-18%,
也是同一件事的另一种度量:单位时间内 tensor core 和 TMA 引擎都比
单缓冲版本更少空闲。**这就是"为什么要做 pipeline"的 profile 证据:
不是因为它增加了 block 级并发(occupancy 相同),而是因为它把原本
`TMA 传输 → 等待 → mma 计算` 这条单 block 内的串行链条,改造成了
`mma(第 it 轮)` 与 `TMA(第 it+1..it+S-1 轮)` 的并行链条,直接压缩了
`__syncthreads()` 处的空等时间。**

3. 稳态阶段各 stage 中 TMA 与 mma 的流水时空图(贴图或 ASCII):

> 取 `S=3`。按 mbarrier 的强制发射/机会式预取逻辑推演(见 kernel 里的
> `issued` 计数):预热发完 3 轮 TMA 后,每进一轮 `it` 的消费,机会式
> 预取都恰好能再往前发一轮——稳态下 `issued` 始终领先 `it` 正好
> `NSTAGE=3` 轮,即"当前在算第 `it` 轮时,TMA 引擎已经在搬第 `it+2`
> 轮的数据"。三个 stage(s0/s1/s2)轮流复用,拷贝(TMA 硬件的
> copy engine)与计算(tensor core)是两条独立流水线,不互相抢占:
>
> ```
> 轮次 it :   0    1    2    3    4    5    6    7    8
> 用的 stage:  s0   s1   s2   s0   s1   s2   s0   s1   s2
>
> s0  TMA  : [==]                [==]                [==]        (轮0/3/6 搬入)
> s0  mma  :      [--]                [--]                [--]   (轮0/3/6 消费,晚 TMA 一轮起步)
> s1  TMA  :      [==]                [==]                [==]
> s1  mma  :           [--]                [--]                [--]
> s2  TMA  :           [==]                [==]                [==]
> s2  mma  :                [--]                [--]                [--]
>                ↑                        ↑
>          预热发完 s0/s1/s2         稳态:每格里始终有 1 个
>          三轮 TMA 之后进入稳态     stage 在 mma、另有 stage 的
>                                    TMA 已经在背后搬下一轮数据
> ```
>
> 关键点:`[==]`(TMA)与紧跟着的同 stage `[--]`(mma)之间空出的格子,
> 就是"本该阻塞等待、但被更早发出的 TMA 提前掩盖掉"的延迟——这正是
> 4.2 单缓冲版本里没有的部分(4.2 每轮都是 TMA 发完当轮就地等 full,
> 等价于这里 S=1、`[==]` 和 `[--]` 紧挨在同一格,没有提前量)。代价在
> 图外:三个 stage 同时占着 smem,S 越大能重叠的轮数越多,但每 SM 能
> 同时跑的 block 数也越少——这就是 2. 里量化的那笔账。

4. 回答:

   (a) 从 4.1 到 4.3,主要瓶颈发生了哪些变化?

> 4.1:标量 staging 的**访存延迟**(84% 的 warp stall 在等 L1TEX)叠加
> **occupancy 不足**(smem 把 block 数摁在 8,理论 occupancy 只有
> 50%)——两者都没打满,卡在"两个屋顶之间"。
>
> 4.2:把 staging 外包给 TMA 硬件后,访存延迟这部分基本消失
> (compute/memory throughput 从 24%/15% 升到 46.5%/72%),但因为还是
> 单缓冲,瓶颈变成**同步等待**:每轮都要在本地 block 内老老实实等这
> 一轮的 TMA 落地才能算,`CTA barrier` 变成头号 stall 原因
> (≈51.7%)——本质是"延迟没法和下一轮的搬运重叠"。
>
> 4.3:多级缓冲想解决的正是 4.2 这个"不能重叠"的问题,3. 里的时空图
> 显示它确实做到了(TMA 提前 `NSTAGE-1` 轮发出)。但瓶颈没有消失,而是
> **搬到了另一个维度**:换 occupancy 换重叠——多 stage 让每个 CTA 占用
> 更多 smem,`Block Limit Shared Mem` 直接下降(8→3@S=3→1@S=6),CTA
> barrier 仍然是最大 stall 项且占比基本没变(≈51%,ncu 实测),只是
> 现在"barrier 空等"背后能用来掩盖它的**跨 block 并发**变少了。也就是
> 说 4.1→4.2 解决的是"单轮内的延迟",4.2→4.3 解决的是"轮与轮之间的
> 延迟",但 4.3 引入的新代价(occupancy)在 grid 足够大、并发本来就
> 充裕的形状上(4096³)会反过来吃掉收益——这正是 2. 里两个形状表现相反
> 的原因,瓶颈的性质从"延迟"逐级转移成了"容量(smem)换并发"的权衡。

   (b) 梯子表中每一级优化分别减少了哪部分开销?

> - **4.1→4.2**:去掉了标量 staging 本身(swizzle 地址计算、逐元素
>   load/store 指令、以及 84% 花在等 L1TEX scoreboard 上的延迟),把
>   搬运工作从 SM 的 LSU/ALU pipe 转移到专用 TMA 硬件引擎,SM 的 issue
>   slot 从发 staging 指令改成发 mma 指令。29.5→369.3 TFLOPS(3%→35%)。
> - **4.2→4.3(在 occupancy 够用的场景下)**:去掉了"每轮 TMA 落地前
>   block 干等"的同步开销——用多级 buffer 把下一轮(乃至下 `NSTAGE-1`
>   轮)的 TMA 提前发出去,和当前这一轮的 mma 计算重叠,理论上能把
>   "TMA 传输时间"从"计算时间 + 传输时间"的串行链条里减掉,只要
>   `mma` 时间 ≥ `TMA` 时间就能被完全掩盖。**但这一级省下来的开销要用
>   smem/occupancy 去换**,4096³ 这种大 grid 场景换亏了(303.9 TFLOPS
>   反而低于 4.2 的 369.3);S 小一点、或者 grid 本身不够大掩盖不了
>   occupancy 损失的场景(256×4096×16384)才真的划算(129~131 对
>   4.2 单缓冲版本在同形状下的表现,见下条(c)前置的 occupancy 分析
>   ——薄形状下深流水没有 occupancy 代价,是纯收益)。

   (c) 如果继续增大 tile 或增加 stage 数,shared memory 与 TMEM 哪一个会
   先成为容量限制?结合 3.4(c) 的结果说明。

> **两条路都是 shared memory 先顶到墙,TMEM 还早得很。**
>
> - **加 stage 数**:TMEM 只存 `BM×BN` 的 f32 accumulator,大小完全由
>   输出 tile 形状决定,和 K 方向的流水深度 `STAGES` 无关——`tcgen05.
>   alloc` 的宽度在这个 kernel 里恒为 `BN=64` 列,不随 `STAGES` 变化。
>   而 smem 是 `STAGES*(BM+BN)*BK*2` 字节,随 `STAGES` **线性增长**,
>   已经实测到 S=6 时 `Block Limit Shared Mem` 掉到 1 block/SM(smem
>   用量 148480 B,推算的每 SM 可用 smem ≈224 KB,S=9 左右
>   `9*24576+1024≈222 KB` 就会顶到这个上限、kernel 直接无法启动)——
>   TMEM 全程没有参与这场竞争,shared memory 是唯一在涨的资源,自然是
>   它先撞墙。
> - **加输出 tile(增大 BN)**:这次 TMEM 才会真的开始涨——列数
>   随 `BN` 涨(3.1(d) 已给出 TMEM 规格:每 SM 128 lane × 512 column ×
>   4 B,当前 `BN=64` 只占了 512 列里的 64 列,即 12.5%)。但 smem 端
>   `stageBytes=(BM+BN)*BK*2` 里 `BN` 只是加法项,`BM=128` 才是大头,
>   BN 从 64 涨到比如 256 时 smem 才涨 `(256-64)*64*2=24576 B`(一级
>   stage),而对应的 TMEM 占用是从 64 列涨到 256 列,涨到了 512 列
>   预算的一半——对比 3.4(c) 的结论(`cta_group::2` 把 B 侧 smem
>   减半也只换回 4096 B、连一级 stage 都买不起,因为 A 那部分
>   `128×64×2=16384 B` 是大头,不随 N 切分变化),同一条逻辑在这里
>   反过来看:**A 侧的固定开销(16 KB/stage)让 smem 对 BN 的增长本来
>   就不敏感,而 TMEM 对 BN 的增长是线性且直接吃满 512 列预算的**——
>   单看"扩 BN"这一个方向,TMEM 反而会比 smem 更早顶到墙(BN 到 512
>   就是硬上限,再大要么切两个 tile 要么上 `cta_group::2` 把 M/N
>   分给两个 CTA 分摊)。
>
> 综合:这个 kernel 家族里,**扩 K 方向的 stage 数完全是 smem 说了算**
> (TMEM 零参与,S 到个位数就已经把占用率压得很低了);**扩 N 方向的
> tile 才轮到 TMEM 先喊停**(smem 对 BN 不敏感,TMEM 512 列的硬预算
> 才是真正的上限)——两种"变大"撞的是两堵不同的墙,不能一概而论。

---

# 低精度与 block scaling

## 5.1 per-tensor scale 与 outlier {.prob type=EXPERIMENT file=kernels/quant_outlier.py}

`uv run python kernels/quant_outlier.py` 的输出(1 万个 [-1,1) 均匀值 +
一个 3000 的 outlier,E4M3,scale=amax/448):

| x≈ | 含 outlier rel_err |
|---|---|
| 0.5 | 4.611e-02 |
| 0.1 | 4.634e-02 |
| 0.01 | 3.085e-01 |
| 0.005 | 1.000e+00(量化成 0,相对误差封顶) |
| 3000.0 | 0.000e+00(outlier 自己被精确保留) |

scale = 3000/448 ≈ 6.696。outlier 把 scale 顶到了原本该有值的 ~3000
倍,代价是其余全部正常范围的值都被这个过大的 scale 挤到了 E4M3 格点
很稀的区域——0.5 这种"正常"值的相对误差高达 4.6%,0.005 直接量化成 0。

(a) 去掉 outlier 重新量化:scale 掉到 ≈0.002232(约为含 outlier 时的
1/3000,而且 0.002232 ≈ 1/448,说明此时 amax≈1,符合均匀分布 [-1,1)
的期望),x≈0.5 处 rel_err 从 4.611e-02 降到 **3.086e-04**,提升了近
**150 倍**。可见 per-tensor scale 对单个 outlier 极度敏感:一个离群点
就能让全张量的有效精度崩掉两个数量级。

(b) 量化到 0 的阈值:含 outlier 时 ≈6.539e-3,去掉 outlier 时
≈2.180e-6;两种情况下阈值/scale 都 ≈**0.0010**,即阈值 = scale ×
2^-10。这正是 round-to-nearest-even 的舍入边界:E4M3 最小正次正规数是
2^-9(=0.001953125),小于其一半(2^-10)的值舍入到 0。这个阈值只随
scale 线性变化,scale 越大(outlier 越极端),被"吃成 0"的正常值范围
就越宽——这是 per-tensor scale 的第二个代价:不仅精度整体变差,还会
有一整段小值被直接清零,而不只是变粗糙。

(c) 换成 1×128 per-block scale:outlier 所在的 block(scale≈6.696,跟
per-tensor 时相同,因为这个 block 里恰好含 outlier)x≈0.5 处
rel_err=1.312e-2;不含 outlier 的 block(scale≈0.002227,基本等于
"没有 outlier 时的 per-tensor scale")x≈0.5 处 rel_err=1.104e-3——比
含 outlier 的 block 低一个数量级,且已经接近(a)中"整张量都没有
outlier"时的水平。**block scale 把 outlier 的影响限制在了它所在的那
一个 128 元素小段内**,其余 127/128 的数据完全不受这一个离群点拖累,
这就是 block scaling(以及 5.2、NVFP4 的组内 16 元素 scale)存在的
核心动机。

## 5.2 block scaling 的两种乘回范围 {.prob type=DERIVE file=kernels/block_scale_sim.py}

判测:`PYTHONPATH=. uv run pytest tests/test_block_scale.py` → 3 passed。

- `gemm_scale_per_row_col`:sA(每行一个)、sB(每列/每个输出通道一个)
  在整个 K 归约里都是常数,可以直接从 $\sum_k$ 里提出来:先归一化
  A、B,做完整点积,最后在 [M,N] 输出上乘回一次 $s_A \otimes s_B$。
- `gemm_scale_along_k`:scale 每 SEG=128 个 K 元素换一次,不再是整个
  K 求和的公因子,只能逐 K block 提:每段内归一化、做该段的 partial
  点积、乘回**这一段自己的** $s_A[block] \cdot s_B[block]$,再把各段
  结果相加。
- 反例 `gemm_scale_along_k_one_restore` 之所以错:它把没乘回 scale 的
  各段 partial sum 先加在一起,再用**第一段**的 scale 乘回整体。这一步
  隐含假设了"所有 K block 的 scale 乘积相同,可以整体提出",但沿 K
  分段量化的前提恰恰是不同段的 scale 不同——这个假设不成立,提出来的
  就是错误的公因子。判测里 `test_along_k_cannot_restore_only_once`
  用随机 scale 验证了这一点(误差 > 1e-3,远超浮点噪声)。
- 一般规律:一个 scale 能不能从归约里整体提出来,只取决于它在被归约的
  维度上**是否为常数**。per-row/per-col scale 对 K 求和是常数,能提;
  per-K-block scale 恰好是在被求和的那个维度上分段变化的,不能整体提,
  只能跟着分段一起做"归一化-求和-乘回"的循环。这也是 NVFP4(K 方向
  每 16 个元素一个 SF)必须在 tensor core 内部按组处理、不能简单套用
  per-tensor/per-row scale 那套"算完再乘一次"写法的原因。

## 5.3 NVFP4 量化通路 {.prob type=FROM-SCRATCH file=cuda/m5_lowprec/}

GPU:B300 SXM6(`srun -G 1`)。

### (a) e2m1 编码器 `e2m1_encode.h`

幅值格点 {0, 0.5, 1, 1.5, 2, 3, 4, 6} 的中点依次是
0.25/0.75/1.25/1.75/2.5/3.5/5.0;round-to-nearest-even 在每个中点上都
偏向下标为偶数的一侧(下标 0/2/4/6 对应 0/1/2/4,都是"整数"格点),
于是每段边界该并入哪一侧、该不该带等号,直接由这条奇偶规则决定,不用
分别查表验证。符号位用 `signbit` 而不是 `v<0.f` 取,是为了让 -0.0 也
保留符号位(判测对全部候选值做了负号镜像,包含 -0.0)。

判测(`03a_encode_check`,与硬件 `__nv_fp4x2_e2m1`/
`F2FP.SATFINITE.E2M1` 逐点比对,候选含全部中点邻域、格点、饱和区、
均匀网格与随机采样共 202864 个值):**PASS,202864/202864 全部与硬件
逐位一致**。

### (b) NVFP4 quant kernel `nvfp4_quant_kernel.h`

一个线程负责一个 16 元素组(读 32 B bf16,写 8 B e2m1 + 1 B swizzled
SF),组间无依赖,线性索引铺满 grid,`BLOCK=128`,grid 下限为 SM 数
(避免小形状 grid 比 SM 数还少)。判测(`03b_nvfp4_quant`,与 host
参考逐 byte 严格比较,三个形状):

| M | K | 结果 | 耗时 | 有效带宽 |
|---:|---:|---|---:|---:|
| 128 | 1024 | PASS | 4.11 us | 82 GB/s |
| 200 | 4096 | PASS | 6.16 us | 341 GB/s |
| 4096 | 7168 | PASS | 64.68 us | 1163 GB/s |

全部 **PASS,逐 byte 严格相等,bad=0**。

消费端终验(`test_fp4_gemm`,把这个 kernel 的产物原样喂给 cuBLASLt
的 block-scaled FP4 matmul,和 host 反量化的 double GEMM 对拍):

| M | N | K | maxrel | 结果 |
|---:|---:|---:|---:|---|
| 128 | 128 | 1024 | 3.880e-3 | PASS |
| 256 | 512 | 4096 | 3.891e-3 | PASS |
| 200 | 128 | 1024(M 非 128 倍数) | 3.880e-3 | PASS |

maxrel 落在预期的 ≈4e-3(2^-8,bf16 输出舍入)附近,说明 swizzled SF
布局和 e2m1 打包顺序与 cuBLASLt/tensor core 实际读取的完全一致——如果
布局错位,scale 会配错组,结果会整块崩掉而不是差 4e-3 这种量级。

### (c) ceiling 探针 `03c_ceiling_probe.cu`

探针 kernel 与 quant kernel 访存完全同形(同样的线程→组映射、同样的
读 32B/写 8B+1B),只把编码换成 xor 直通,量出这套访存模式在 B300 上
的带宽地板:

| M | K | 探针 GB/s | quant kernel GB/s(同/邻近形状) | 比值 |
|---:|---:|---:|---:|---:|
| 4096 | 7168 | 1183 | 1163(同形状) | **98.3%** |
| 16384 | 4096 | 1252 | — | — |
| 16384 | 8192 | 1275 | — | — |

在 M=4096,K=7168 这个两边都测过的形状上,quant kernel 已经跑到探针
地板的 **98.3%**——说明这套 quant kernel 已经几乎完全被这个访存模式
本身的带宽限制住,不是被"没写好"限制住;ncu 层面的解读是
`sm__throughput` 应该显著低于 `dram__throughput`(计算量小,搬运量
大),差距主要是访存,不是计算——5.4 节里对同一个 kernel 有 ncu 实测
数字可以直接印证。

### (d) 消费端(可选)

未实现:时间优先分配给了必做的 5.4 与 module 6,tcgen05
`kind::mxf4nvf4` 直接消费自己 NVFP4 数据这一项先跳过。

## 5.4 融合 rms_norm + NVFP4 quant {.prob type=FROM-SCRATCH file=cuda/m5_lowprec/04_fused_rms_nvfp4.cu}

### 设计

融合 kernel(`fused_rms_nvfp4_kernel`)结构:block-per-row,先用和两步
基线完全相同的两级 warp-shuffle 归约算出 `rnorm`,再按 16 元素一组直接
算 `y=x*rnorm*w` 并编码成 e2m1+SF——`y` 只活在寄存器里,不落 bf16 中间
张量。字节账:两步基线读 x(2B)、写中间值(2B)、读中间值(2B)、写
e2m1(0.5B)+SF(1/16B)=6.56 B/elem;融合只读 x、w(近似 2B,w 很小可
忽略)、写 e2m1+SF=2.56 B/elem,预言加速比 6.56/2.56≈2.56x。

### 调参(保证基线公平)

两步基线的 `rms_norm_baseline_kernel` 与融合 kernel 分别在
`BLOCK∈{128,256,512,1024} × grid 倍率∈{1,2,4,8}×SM数` 上扫过(M=16384
的 K=4096/7168 与 M=4096,K=4096 三个代表形状),结果都是 **BLOCK=128、
grid=8×SM 数** 附近最快或接近最快(例如融合 kernel 在 M=16384,K=7168
上:BLOCK=128 483us / BLOCK=256 480us / BLOCK=512 509us / BLOCK=1024
650us,grid 倍率从 1 到 8 单调变快;基线的 rms_norm 半程也是同一组合
最快)。两者最终都固定在 `BLOCK=128,grid=min(M, 8×SM数)`,通过
`FUSED_BLOCK/FUSED_GRID_MULT`、`BASELINE_BLOCK/BASELINE_GRID_MULT` 三
个宏可覆盖复测。

### 结果(B300,`04_fused_rms_nvfp4`)

| M | K | 两步(us) | 融合(us) | 加速比 | 结果 |
|---:|---:|---:|---:|---:|---|
| 1 | 4096 | 8.21 | 4.61 | **1.78x** | PASS |
| 16 | 4096 | 8.21 | 4.62 | **1.78x** | PASS |
| 256 | 4096 | 8.21 | 6.16 | 1.33x | PASS |
| 1024 | 4096 | 14.34 | 12.30 | 1.17x | PASS |
| 4096 | 4096 | 34.57 | 40.40 | 0.86x | PASS |
| 16384 | 4096 | 121.40 | 150.71 | 0.81x | PASS |
| 4096 | 7168 | 61.39 | 67.59 | 0.91x | PASS |
| 16384 | 7168 | 215.20 | 262.28 | 0.82x | PASS |
| 4096 | 8192 | 65.64 | 75.78 | 0.87x | PASS |
| 16384 | 8192 | 233.68 | 291.49 | 0.80x | PASS |

判测口径允许 1e-4 比例的 byte 不一致(sumsq 归约顺序不同导致极少数
舍入边界值翻转),全部 10 个形状 PASS。

### 差距归因:预言 2.56x,实测从 1.78x(小 M)一路降到 0.80x(大 M)

小 M(1、16)确实拿到了预言量级的加速(1.78x,虽然没到 2.56x——小 M
时两个 kernel 都被固定 launch 开销主导,省下的字节数本身占比也变小,
这和 4.5 里"小 M 时固定开销主导"是同一条逻辑)。但 M≥4096 之后融合
kernel 反而比两步基线**更慢**,且 M 越大越慢——这是需要解释的现象,
不是把它调得更快就能消失的效果,ncu 证据如下(M=16384, K=7168,同一张
B300):

| kernel | grid×block | `sm__warps_active` | `sm__throughput` | `dram__throughput` | 耗时 |
|---|---|---:|---:|---:|---:|
| 融合 kernel | 1184×128 | **38.9%** | 20.5% | 7.7% | 480 us |
| 独立 quant kernel(同形状) | 57344×128 | **68.8%** | 34.8% | 15.5% | 238 us |

两者在同一张卡、同样的 clock 状态下测,融合 kernel 反而比"只做 quant
这一半工作"的独立 kernel 还慢一倍,占用率(warps active)只有对方的
约 56%。原因是**两个阶段对 grid 形状的需求完全不同,融合后被迫共用
同一套 launch 配置**:

- reduction 阶段(算 rnorm)是 block-per-row,一行一个 block 完全够用,
  grid 只需要覆盖 SM 数量级(148×8≈1184)就能把带宽打满——这也是它
  单独调参时选出 grid≈8×SM 的原因;
- encode 阶段(quant)天然是"一个线程一组"的细粒度并行,独立 quant
  kernel 的 grid 高达 57344(=M×K/16/128),充分利用了 SM 的深度
  并发去隐藏每个线程里 `fmaxf`/除法/`cvt.e2m1x2` 这几步延迟;
- 融合 kernel 把 encode 阶段塞进了 reduction 阶段那套只有 1184 个
  block 的 grid 里,每个 block 要靠 `for (g = tid; g < groupsPerRow;
  g += BLOCK)` 网格跨步把整行的全部 group 串行处理完——同样多的组数,
  并发线程数骤降到独立 kernel 的 1184/57344 ≈ 1/48,占用率随之腰斩,
  单条 group 的延迟没有足够多的 in-flight warp 来隐藏,`sm__throughput`
  和 `dram__throughput` 反而比两个"半个 kernel"加起来还低。

结论:融合确实省下了中间值的一次写、一次读(4 B/elem),但代价是
reduction 和 encode 两个形状迥异的并行模式被迫共用一套 grid/block——
在 M 足够大、encode 阶段本可以靠海量并发去逼近访存地板时,这个代价
超过了省下的字节数带来的收益,融合反而更慢。这正是题面提到的上游 PR
"收益卡在噪声里"现象的一个具体、可测的版本:**融合类优化的收益不能只
按字节数简单相减,还要看两段工作对并行度的需求是否兼容**——warp
specialization(题面已明确排除在外)是这类问题的标准解法,允许同一个
kernel 内部让不同的 warp 组各自使用适合自己那一半工作的并行粒度,但
这超出了本题范围。

---

## 5.5 W4A16+Marlin vs NVFP4 {.prob type=CONCEPT}

(a) 两者分别属于存储量化还是计算量化?

> W4A16+Marlin 是**存储量化**:只有权重被压成 4 bit 存储/搬运,实际
> 乘累加仍在 fp16 精度下进行(Marlin 的贡献是把"INT4→FP16 反量化"
> 融合进 kernel,避免多一趟显存往返),tensor core 吃到的始终是 fp16
> 操作数。NVFP4(5.3 里跑通的 `CUDA_R_4F_E2M1` + block-scaled tensor
> core 路径)是**计算量化**:A、B 两个操作数都真正被转成 e2m1 存储,
> 且 tensor core 本身按 FP4 精度执行乘累加,吃的是 5.3(a)(b) 里那条
> 编码通路产出的数据,不是先解量化回高精度再算。

(b) 两种方法分别节省哪些资源:显存容量、显存带宽还是计算吞吐?

> W4A16+Marlin:只压缩权重,省**显存容量**(4x)和**显存带宽**(权重
> 读取量降到 1/4,这对 4.5 里发现的"小 M decode 时间被读权重主导"直接
> 有效);不省**计算吞吐**——tensor core 峰值 FLOPS 仍是 fp16 那一档,
> Marlin 的加速来自省掉的带宽与访存延迟,不是更高的算力上限。NVFP4:
> A、B 都转 FP4,三者都省——存储容量(4x)、显存带宽(权重+激活都变
> 成 1/4 大小)、以及**计算吞吐**本身(tensor core 原生跑 FP4,峰值
> FLOPS 比 bf16 高一档,见 0.2 表里 B300 tcgen05 估算的 8960 vs 2240
> TFLOPS)。

(c) 在 4.5 中的小 batch decode 场景下,哪一类量化的收益更加直接?

> **W4A16+Marlin 更直接**。4.5 的结论是:M≤16 时全部七层的 %TCpeak
> 都在 4.4% 以下,tensor core 算力从来没被打满过,此时的瓶颈是把整块
> 权重矩阵从 HBM 读一遍这个和 M 无关的固定成本(以及 kernel/TMA 的
> setup 开销)。NVFP4 的核心卖点——更高的 tensor core FLOPS——在这个
> regime 完全用不上,因为算力从来不是瓶颈;而 W4A16 把权重压到 1/4
> 大小,直接砍掉的正是 4.5 里被反复验证过的那个真实瓶颈(显存带宽/
> 权重读取量)。等到 M 足够大、进入 4.5 图里的"平台期"(compute-bound
> 一侧),NVFP4 的计算吞吐优势才会真正体现出来。

---

# TileLang 对照

## 6.1 sm_90a vs sm_100a lowering {.prob type=EXPERIMENT}

复用 assignment01 的 `kernels/tilelang_matmul.py`(`make_matmul`,fp16
输入/fp32 累加,`M=N=K=1024`,`BLOCK_M=128,BLOCK_N=128,BLOCK_K=32,
threads=128,num_stages=3`),分别以 `Target({"kind":"cuda",
"arch":"sm_90a"})` 与 `"sm_100a"` 为 target 调用 `tilelang.compile`
(只编译,没有实际跑 H 卡/B 卡执行,`get_kernel_source()` 拿生成的 CUDA
源码)。生成的完整源码存在
`kernels/tilelang_lowering/matmul_sm_{90a,100a}.cu`。

| | sm_90a | sm_100a(tilelang 0.1.13 实测) |
|---|---|---|
| 选中的 Tensor Core 指令 | `wgmma.mma_async`(生成代码里的 `tl::wgmma_ss<fp16,fp16,fp32,64,128,16,...>`) | **仍是 `mma.sync.m16n8k16`**(`tl::mma_sync`,配 `tl::ptx_ldmatrix_x4`/`_trans`),不是 SM100 的 `tcgen05.mma` |
| descriptor 在哪里、由谁生成 | wgmma descriptor(`tl::GmmaDescriptor`),由生成代码里的 `tl::initialize_wgmma_descriptor`/`increase_descriptor_offset` 现算——等价于我们 2.2 手推的 64 位 descriptor 编码,但由 TileLang 后端生成 | 没有 descriptor:`mma.sync` 走寄存器操作数,数据先用 `ldmatrix` 读进寄存器,不涉及 smem descriptor 这套机制 |
| smem swizzle 布局在哪一步确定 | lowering 阶段由后端根据 wgmma 操作数布局要求确定,体现为 `A_shared`/`B_shared` 的地址算式里烘死的偏移常数(如 `(k%3)*4096`、`+2048`),我们看不到显式的 swizzle 公式 | 同样在 lowering 阶段确定,体现为 `ptx_ldmatrix_x4`/`_trans` 里那一长串按位与/异或拼出来的地址(本质就是 swizzle 后的偏移),同样编译期算好写死 |
| 数据由谁搬入 smem | TMA(`tl::tma_load`),由 `threadIdx.x<128` 的"生产者" warpgroup 负责,配合 6 个 mbarrier(3 个 full + 3 个 empty,对应 `num_stages=3`)做多级流水,消费者 warpgroup(`threadIdx.x>=128`)专职发 `wgmma` | **和 sm_90a 几乎一模一样**:同样是 TMA + 同一套生产者/消费者 warpgroup + 6 个 mbarrier 的三级流水结构,TMA 路径完全没有随 target 切换 |

两份生成代码在结构上几乎是同一份代码的两个变体:kernel 签名相同
(`CUtensorMap A_desc, B_desc` 作为 `__grid_constant__` 参数)、
`__launch_bounds__(256,1)`、`mbarrier_mem[6]`、同样的
prologue(`prefetch_tma_descriptor`+`init`+`fence_barrier_init`)、同样
的 `threadIdx.x<128`/`>=128` warp 分工、同样的 `for(k<32)` K 循环结构。
唯一的实质差别集中在"消费者"分支内部算 `wgmma`(sm_90a)还是
`ldmatrix`+`mma.sync`(sm_100a)这几行——**TileLang 0.1.13 对
`T.gemm` 的 sm_100a 后端还没有接入 tcgen05/TMEM,遇到 `sm_100a` 会
退化到 Ampere/Hopper 世代的 legacy `mma.sync` 路径,只有数据搬运
(TMA)部分是"面向未来"的**。这印证了 handout 编者注里提到的
"TileLang 对 sm_100 codegen 的支持范围需要核实"——核实结论是:截至
这个 pin 住的版本,`sm_100a` target 不能假设会自动拿到 SM100 的
tensor core 指令世代。

(a) 哪些硬件相关的决策已经由 DSL 自动完成?

> - **warp specialization**(生产者/消费者 warpgroup 划分)——这正是
>   我们在 4.1-4.3 里被明确排除在考核范围外的部分,DSL 两个 target 都
>   自动做了。
> - **多级流水的搭建**:prologue 预热、`full`/`empty` mbarrier 的
>   arrive/wait、按 `num_stages` 取模复用 stage——完全对应我们 4.3
>   手写的那套状态机,不需要我们推 phase parity。
> - **TMA descriptor 的接入方式**:`CUtensorMap` 以 `__grid_constant__`
>   传参,`T.copy` 背后直接生成 `tl::tma_load`——host 侧建 tensor map
>   这一步被 `T.Buffer`/`T.copy` 封装了,不需要我们手写
>   `cuTensorMapEncodeTiled`。
> - **wgmma/mma 的 descriptor 或 fragment 布局计算**:sm_90a 的
>   `GmmaDescriptor` 位域编码(2.2)、sm_100a 的 `ldmatrix` 地址公式,
>   都不需要我们手写。
> - **smem swizzle 的具体地址映射公式**(2.3 的 `swizzle_128B`)——两个
>   target 都被编译器烘进了地址表达式里。

(b) 哪些参数仍然需要程序员决定,例如 tile 尺寸和 stages?

> - `BLOCK_M/BLOCK_N/BLOCK_K`、`threads`、`num_stages`——`make_matmul`
>   的这些参数完全由调用者传入,DSL 不会替你选,决定了 smem/寄存器
>   占用和最终性能(这正是 assignment01 7.6 的 `bench()` 在扫的东西,
>   对应我们在 4.3 里手动扫 `STAGES` 的那件事)。
> - **目标硬件的 tensor core 指令世代**这件事,理论上应该完全由
>   `target` 字符串决定,但上面的实测说明这条假设目前不成立——换成
>   `sm_100a` 并没有自动拿到 `tcgen05`,程序员仍然需要去读生成代码
>   确认真正选中的指令,不能只看 target 字符串就认定拿到了对应硬件
>   最新的 tensor core 能力。这也是这道题让我们"保存 lowering 输出"
>   而不是只填表的原因——不亲自看生成代码,会凭空多出一格"DSL 自动
>   完成"的假信心。

补 assignment01 7.5 的"谁负责"表(仓库里没有单独的 assignment01
report 文件,新增的一行整理在这里):

| 谁负责 | CUDA SIMT | cuTile | Triton | TileLang |
|---|---|---|---|---|
| 线程到数据的映射 | 用户 | 编译器 | 编译器 | 编译器 |
| 边界处理 | 用户 | 编译器 | 用户(`tl.where`/mask) | 编译器(`T.Buffer` 越界由 `T.copy`/`T.gemm` 处理) |
| tile / block 尺寸的选择 | 用户 | 用户 | 用户 | 用户 |
| block 内同步 | 用户 | 编译器 | 编译器 | 编译器(`T.Pipelined`/`T.gemm` 内部的 barrier 由 DSL 插入) |
| **Tensor Core 指令选择与供数布局**(新增) | 用户 | 编译器 | 编译器 | **编译器,但会随 target 支持程度变化**——本题实测显示同一份 `T.gemm` 代码换 target 字符串,选中的指令世代可能不随硬件世代同步更新(sm_100a 拿到的还是 sm_90a 那一代的 `mma.sync`),需要用户自己核实生成代码 | 
