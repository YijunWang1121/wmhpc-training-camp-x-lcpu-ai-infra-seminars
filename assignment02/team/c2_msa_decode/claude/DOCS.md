# 文档考证摘录(原文引用,来源 + 章节)

## PTX ISA(docs.nvidia.com/cuda/parallel-thread-execution,2026-09-09 抓取,文档版本 PTX ISA 9.4 / CUDA 13.4)

### A. `cp.async.bulk.tensor` 的坐标是运行期寄存器(§9.7.10.28.5)
> "The vector operand tensorCoords specifies the starting coordinates in the tensor data in the global memory
> from or to which the copy operation has to be performed. The individual tensor coordinates in tensorCoords
> are of type .s32."

→ 坐标是 .s32 寄存器向量,**不需要编译期常量**。对本题:把 kv_cache 建成 4D tensor map
`[num_pages, num_kv_heads, 128, 256]`,`page` 由 `topk_idx → block_table` 两次标量 load 得到后作为最高维坐标即可。
TMA 描述的是"规则张量的某个 box",间接寻址由 SM 线程算好坐标再发指令;TMA 本身不做 gather。

### B. `.tile::gather4`(§5.5.3.4;PTX ISA 8.6,sm_100a / sm_100f)
> "These modes are similar to the tiled mode with restriction that these modes work only on 2D tensor data.
> Tile::scatter4 and Tile::gather4 modes are used to access multiple non-contiguous rows of tensor data. ...
> the instruction will take: four tensor coordinates across the dimension 0, one tensor coordinate across the dimension 1."
> "Qualifiers .tile::gather4 and .im2col::w require: sm_100a when destination state space is .shared::cluster and is
> supported on sm_100f from PTX ISA version 8.8. sm_100 or higher when destination state space is .shared::cta."

→ 唯一带"gather"语义的 TMA 模式只做 **4 行**、仅 2D。本题一块 = 页内连续 128 行 × 256 列(K|V 拼接),
不需要 gather 行;若把 kv_cache 视作 2D `[num_pages*4*128, 256]`,每块是连续 128 行,普通 .tile 即可。

### C. `.override::global_address`(§9.7.10.28.5.2.1;PTX ISA 9.4 = CUDA 13.4,本机 CUDA 13.0 → PTX 9.0,**不可用**)
> "The optional qualifier .override::global_address allows overriding of the tensor base address. When
> .override::global_address is specified, the global memory address present in the opaque tensorMap object is
> ignored and instead the address specified by the explicit 64-bit operand gAddrToOverride is used.
> The gAddrToOverride address must be 16B aligned ... the memory address range [gAddrToOverride, gAddrToOverride + 128 KiB)
> must be allocated and accessible during execution"

→ 最新 PTX 允许 per-instruction 覆盖 tensormap 基址,即"一个 tensormap 描述一页、基址运行期换页",
这正是为间接寻址设计的。但要 CUDA 13.4;本机 CUDA 13.0 用不了。

### D. 1-D `cp.async.bulk`(§9.7.10.28.4.1;PTX ISA 8.0,sm_90+)
> `cp.async.bulk.dst.src.completion_mechanism [dstMem], [srcMem], size, [mbar]`  .dst = {.shared::cluster | .shared::cta}, .src = {.global}
> "The 32-bit operand size specifies the amount of memory to be copied, in terms of number of bytes. size must be
> a multiple of 16. ... The addresses dstMem and srcMem must be aligned to 16 bytes."

→ 不需要 tensormap:一块 (page, kv_head) 的 K|V 在 vLLM 布局里是**连续 128×256×2 B = 64 KiB**
(`stride_kv_pos = 256`,`stride_kv_h = 128*256`),用 1-D bulk copy 直接给全局地址即可,
两级间接寻址只是"算地址",这是最简单的搬法。代价:没有 swizzle(mma 读 smem 需自己处理 bank conflict)
和不能只取 K 半边(K/V 在每行内交错,想只搬 K 要 128 条 256 B 的拷贝,或者用 2D tensormap 的 box)。

### E. CUDA → PTX ISA 版本表(§Release Notes)
CUDA 12.9 → PTX 8.8;**CUDA 13.0 → PTX 9.0**;13.1 → 9.1;… 13.4 → 9.4。本机 nvcc 13.0.88。

## B300 规格(公开资料,用于 roofline 上限;实测值以本机 nsys/ncu 为准)
- HBM3e 8 TB/s;BF16 dense ≈ 2.25–2.8 PFLOPS(与 B200 同级,B300 主要提升 FP4);FP8 dense ≈ 4.5–7 PFLOPS。
  来源:SemiAnalysis InferenceX (inferencex.semianalysis.com/chips/b300)、glennklockwood.com/garden/processors/b300。
  本机 `torch.cuda.get_device_properties`:148 SM,cc 10.3,L2 126.5 MB,SM 2032 MHz,mem 3996 MHz(=HBM3e 8 Gbps×8192 bit ≈ 8.2 TB/s)。

## Thread block cluster / DSMEM(讨论点 2)
### CUDA Programming Guide §2.3.3.8 Distributed Shared Memory(docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)
> "Thread Block Clusters, introduced in compute capability 9.0 ... provide the ability for threads in a thread block cluster to
> access shared memory of all the participating thread blocks in that cluster. ... Threads that belong to a thread block cluster
> can read, write or perform atomics in the distributed address space, regardless of whether the address belongs to the local
> thread block or a remote thread block."
> "Accessing data in distributed shared memory requires all the thread blocks to exist. A user can guarantee that all thread
> blocks have started executing using cluster.sync() ... The user also needs to ensure that all distributed shared memory
> operations happen before the exit of a thread block, e.g., if a remote thread block is trying to read a given thread block's
> shared memory, the program needs to ensure that the shared memory read by the remote thread block is completed before it can exit."

### Blackwell Tuning Guide §1.4.1.2(docs.nvidia.com/cuda/blackwell-tuning-guide)
> "The maximum portable cluster size supported is 8; however, NVIDIA Blackwell B200 GPU allows for a nonportable cluster size
> of 16 by opting in. Launching a kernel with a nonportable cluster size requires setting the
> cudaFuncAttributeNonPortableClusterSizeAllowed function attribute. Using larger cluster sizes may reduce the maximum number
> of active blocks across the GPU."
> "Distributed Shared Memory can be used by an SM simultaneously with L2 cache accesses."
> "For applications using Thread Block Clusters, it is always recommended to compute the occupancy using
> cudaOccupancyMaxActiveClusters and launch cluster-based kernels accordingly."

### PTX ISA §9.7.10.26 `mapa` / §9.7.15.3 `barrier.cluster`
> "mapa: Map the address of the shared variable in the target CTA." `mapa.shared::cluster.type d, a, b;`
> "A barrier.cluster.arrive operation synchronizes with a barrier.cluster.wait operation."
→ `cooperative_groups::cluster_group::map_shared_rank()` 就是 `mapa`,`cluster.sync()` 就是 `barrier.cluster.arrive + wait`。

### 与本题的对应(讨论点 2:merge 放 cluster / mbarrier 里做)
- 一个 (token, kv_head) 的 16 个 top-k 块分给一个 cluster 的 CL 个 CTA;每个 CTA 算完自己的 (m, l, acc[16×128] fp32 = 8 KB + 128 B)
  留在自己 smem;`cluster.sync()` 后 rank 0 用 `map_shared_rank` 读其余 CL−1 份做 LSE 合并,写最终输出;再 `cluster.sync()` 保证
  被读 CTA 不提前退出(上面引用的要求)。partial 不落 global,也没有第二个 kernel。
- mbarrier 在这里的角色是 TMA 完成计数(`cp.async.bulk.tensor ... mbarrier::complete_tx::bytes`),不是 CTA 间同步;
  CTA 间同步用 cluster barrier。要用 mbarrier 做跨 CTA 同步也可以(remote mbarrier arrive 需 `.shared::cluster` 地址),
  但 cluster.sync 已足够且更简单。
- CL=16 需要 non-portable 选项(上引);CL ≤ 8 可移植。实现见 `exp/msa_decode.cu`。
