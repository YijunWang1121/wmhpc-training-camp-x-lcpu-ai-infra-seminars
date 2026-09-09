# 实验日志(时间顺序,原始发现)

## 2026-09-09 S0 — 环境(login 节点,无 GPU)
- 工具链:CUDA 13.0 (V13.0.88),torch 2.14.0+cu130,triton 3.8.0,ncu/nsys 在 /usr/local/cuda/bin
- Slurm:partition `gpu` MaxTime=01:00:00,节点 dev-slurm gpu:8;`~/bin/gpu` = srun/sbatch -G 1 --time 30min
- 已读:harness/*.py、vllm_msa_ref/sparse_attn.py(全文)、msa_cutlass_sparse_decode.py、sparse_attention_msa.py、
  test_sparse_attn_fp8_scale.py、index_topk.py(头部与 block_table 用法)

### 基线 kernel 结构(读代码得出,待实验验证)
- decode kernel grid = (total_q × NUM_TOPK_CHUNKS, num_kv_heads);
  NUM_TOPK_CHUNKS = 2^floor(log2(min(16, 256 // (total_q×4))))
  → b=1:16 chunks(每 CTA 1 块);b=4:16;b=8:8(每 CTA 2 块);b=16:4(4 块);b=32:2;b=64:1
  → grid CTA 数 = b×chunks×4:b=1:64,b=4:256,b=8:256,b=16:256,b=32:256,b=64:256
- 每 CTA:q tile [16 heads, 128] bf16;每块 K [128d,128pos] + V [128pos,128d] 各 32 KB;两次 tl.dot M=16,N=128,K=128
- 每 CTA 写 partial o [16,128] bf16 + lse [16] fp32;merge kernel grid=(total_q, 64 heads),每 CTA 读 chunks×128 bf16
- 一步 decode 总 KV 读量 = b × 4 kv_heads × 16 blocks × 64 KB = b × 4 MiB(b=1:4 MiB;b=16:64 MiB)

## S1 — job 24073:E1/E2/E5 基线测量(`exp/e1_baseline.py`,原始输出 `logs/e1_final.out`)
GPU:**NVIDIA B300 SXM6 AC**,148 SM,cc 10.3,HBM 268 GB,L2 126.5 MB,SM 2032 MHz,mem 3996 MHz。
harness 三组 check 全 PASS(err_ratio 3.2e-3 / 2.8e-3 / 3.2e-3)。

### 原始数据(seq=8192, topk=16, kvh=4, gqa=16, dql=1;us)
| b | chunks | CTA 数 | eager | graph | decode | merge | KV MiB | 有效 GB/s |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 16 | 64 | 32.5 | 8.2 | 5.9 | 2.7 | 4 | 510 |
| 2 | 16 | 128 | 32.4 | 9.2 | 6.4 | 2.8 | 8 | 910 |
| 4 | 16 | 256 | 32.4 | 11.2 | 8.0 | 3.1 | 16 | 1495 |
| 8 | 8 | 256 | 32.7 | 14.9 | 11.4 | 3.4 | 32 | 2252 |
| 16 | 4 | 256 | 32.5 | 21.9 | 18.5 | 3.2 | 64 | 3068 |
| 32 | 2 | 256 | 47.3 | 42.9 | 39.0 | 3.7 | 128 | 3127 |
| 64 | 1 | 256 | 78.7 | 74.4 | 70.1 | 4.0 | 256 | 3607 |

### 发现
1. **eager 时间 b≤16 恒 ≈32.5 us**,而 CUDA graph 下 b=1 只要 8.2 us → 非 graph 时 ~24 us 是 Python/Triton launcher 开销。
   vLLM 生产用 CUDA graph,所以后续以 graph 数字为准;但 eager 数字说明 host 侧才是最大的那块。
2. graph 下 b=1..16 时间 8.2→21.9 us,而数据量 4→64 MiB。有效带宽 0.5→3.1 TB/s,远低于 B300 HBM3e 峰值(标称 8 TB/s)。
   即使 b=64(256 MiB)也只有 3.6 TB/s ≈ 45% 峰值 → 这个 kernel 在大 batch 也不是带宽饱和的。
3. **E5 chunks 扫描揭示每 CTA 逐块串行代价**:b=1, chunks=1(1 CTA 处理 16 块)= 40.3 us → ≈2.4 us/块;
   chunks=16(每 CTA 1 块)= 5.9 us。时间几乎正比于「每 CTA 块数」:16→40, 8→22, 4→12.7, 2→8.2, 1→5.9。
   线性拟合:t ≈ 3.6 us + 2.3 us × (块/CTA)。→ 3.6 us 是 kernel 固定开销(launch + 序言 + 收尾),2.3 us 是**每块的串行延迟链**
   (topk_idx 标量 load → block_table 标量 load → K 载入 → dot → softmax → V 载入 → dot,块间无重叠)。
   一块 64 KB 若按 CTA 级带宽算不该要 2.3 us,这是延迟不是带宽。
4. **merge kernel 固定 ~2.7–4 us**,b=1 时占总 8.2 us 的 1/3;它只读 16×64×16×128×2 B = 4 MiB… 不对,b=1 时 partial 为 16 chunks×64 heads×128×2 B = 256 KB,纯固定开销。
5. b=8 chunks=16(512 CTA)比 chunks=8 慢(14.0 vs 11.3),b=16 chunks=16(1024 CTA)26 us:CTA 数超过一波后
   反而变慢 → 单 CTA 资源占用大(smem),每 SM 驻留 CTA 数少,多波排队。待 ncu 确认 occupancy。
6. seq 长度 256→32768 对时间无影响(10.2→11.2)→ 数据只来自 top-k 块,与 seq 无关(符合预期)。

### 由此得到的瓶颈判断(待 E3 ncu/nsys 佐证)
小 batch:**延迟链暴露**(每块 2.3 us 串行 × 块/CTA)+ **固定开销**(decode kernel ~3.6 us + merge ~2.7 us),
不是算力,也不是带宽(0.5 TB/s ≪ 8 TB/s)。

## S2 — job 24074:E3 nsys + ncu(`exp/job_e3.sh`,原始文件 `profiles/`)
### nsys(eager,20 步/batch,中位数,us)
| b | decode grid | decode | merge | gap dec→mrg | gap mrg→next dec | 周期 |
|--:|--|--:|--:|--:|--:|--:|
| 1 | 16×4 | 6.11 | 2.98 | 5.01 | 16.13 | 30.2 |
| 4 | 64×4 | 8.40 | 3.41 | 3.30 | 15.68 | 30.8 |
| 8 | 64×4 | 11.68 | 3.74 | 2.34 | 13.12 | 30.9 |
| 16 | 64×4 | 18.98 | 3.65 | 1.94 | 5.98 | 30.5 |
- eager 周期恒 ≈30.5 us,与 E1 的 32.5 一致:GPU 空闲(两个 gap 之和)b=1 时 21 us = 70%。
  gap 来自 Triton launcher(Python 侧参数打包 + 驱动 launch)。**graph 是前提**,否则 kernel 优化毫无意义。
- decode kernel:128 线程 / CTA(4 warps),195 reg/thread,动态 smem 73.7 KB。

### ncu(单次 launch,`--set full`)decode kernel
| 指标 | b=1 | b=4 | b=8 | b=16 | b=1 ch=1 | b=1 ch=4 |
|--|--:|--:|--:|--:|--:|--:|
| Grid | 64 | 256 | 256 | 256 | 4 | 16 |
| Duration (us, ncu 时钟锁定下) | 9.92 | 13.25 | 18.37 | 26.66 | 51.23 | 17.92 |
| DRAM read | 4.23 MB | — | — | 67.4 MB | 4.23 MB | — |
| DRAM Throughput % | 5.6 | 16.7 | 24.0 | 33.4 | 1.1 | 3.1 |
| Compute(SM) % | 4.8 | 16.1 | 16.9 | 19.3 | 0.5 | 1.6 |
| tensor pipe (hmma) % of active | 7.8 | — | — | 16.5 | 16.0 | — |
| Theoretical occupancy | 12.5% (8 warps/SM) | ← | ← | ← | ← | ← |
| Achieved occupancy | 6.4% | 10.9% | 11.1% | 10.8% | 6.3% | 6.4% |
| Block limit: registers / smem | 2 / 3 | ← | ← | ← | ← | ← |
| Waves per SM | 0.22 | 0.86 | 0.86 | 0.86 | 0.01 | 0.05 |
| No Eligible % | 81 | 75 | 75 | 74 | 79 | 79 |
| Warp cycles / issued inst | 5.25 | 6.89 | 6.99 | 6.73 | 4.80 | 4.86 |
| 主要 stall(cycles/inst) | long_sb 1.32, wait 1.22, short_sb 0.75 | | | long_sb 1.97, wait 1.45, short_sb 1.20 | wait 1.56, long_sb 1.07 | |
- DRAM 读 4.23 MB = 4 MiB KV + 索引,**数据量与手算完全一致**(没有多读)。L2 命中 ~1–3%:纯流式。
- 所有 batch 都是:占用率 ≤ 11%(每 SM 最多 2 CTA = 8 warps,寄存器 195 限制),Waves ≤ 0.86 → **一波都填不满 148 SM**;
  每调度器只有 ~1–1.7 个活跃 warp,80% 周期无可发射 warp。DRAM/SM/tensor 三条 pipe 都 < 35%。
  → 典型的**延迟受限**(latency-bound),不是吞吐受限。
- ch=1(1 CTA 做 16 块)51 us、DRAM 1%:一条 warp 链的串行延迟暴露到极致。

### Triton 生成代码结构(~/.triton/cache/.../_gqa_sparse_decode_kernel.{ptx,ttgir})
- `.target sm_103a`,`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` ×64,`cp.async.cg.shared.global` ×64,
  `ldmatrix` ×48,3× `cp.async.wait_group`。→ 在 B300 上 Triton 3.8 对这个 kernel 用的仍是 **Ampere 风格 mma.sync + cp.async**,
  不是 tcgen05/TMA。M=16 正好等于 GQA 组 16 = m16n8k16 的 M,tensor core 一行都不浪费。
- TTGIR:num_stages=3 但 K/V smem 只有 **1 级缓冲**(`memdesc<1x128x128xbf16>`),循环体是
  `async_wait(k,v,page) → 计算 → 发起下一块 k/v/page 的 cp.async`;下一轮开头立刻 wait。
  即**下一块的载入与本块计算不重叠**,只有 page 索引提前一轮。每块 = 完整一次 DRAM 往返 + 计算,串行。
  这解释了 E5 的 2.3 us/块。

### 瓶颈结论(任务 1 / 讨论点 5)
小 batch(b≤16)decode 的时间构成(graph 模式,b=1 共 8.2 us):
1. decode kernel 固定开销 ~3.6 us(launch + 序言 + 索引两级 load + 收尾)
2. 每块串行延迟链 ~2.3 us × (块/CTA)(b=1: 1 块)
3. merge kernel ~2.7 us(几乎全是固定开销:b=1 只读 272 KB)
4. eager 模式再加 ~22 us host launch gap
不是算力(tensor pipe <17%),不是带宽(DRAM <34%,b=1 仅 5.6%)。是 **latency + launch**。
- 教训:B300 是 cc 10.3,sm_100a 的 cubin 不能跑(no kernel image);要用 family 目标 sm_100f(与 Makefile 默认一致)。job 24075 因此全失败,重编后重跑。

## S3 — job 24076:讨论点 3 TMA 间接寻址实验(`exp/tma_indirect.cu`,`logs/tma_24076.out`)
每 CTA 做一次 topk_idx → block_table → page 两级标量 load,然后用三种方式把一块 (page, kh) 64 KiB 搬进 smem,
与 host 端按同样间接寻址算的校验和比对。1 块/CTA 结果(b=1/4/16 基本一致):
| 搬法 | 校验 | CTA 内载入耗时(clock64) |
|--|--|--:|
| (A) `cp.async.bulk.tensor.4d` + 4D tensormap,page 作为运行期最高维坐标 | OK | 1645 cyc ≈ 0.83 us |
| (B) 1-D `cp.async.bulk`,运行期全局地址 | OK | 1832 cyc ≈ 0.92 us |
| (C) 128 线程 `ld.global.v4` 搬 64 KiB | OK | 5911 cyc ≈ 2.96 us |
- **结论:TMA 能表达**——tensormap 描述 kv_cache 整体 [pages][kvh][128][256],间接寻址的结果(page)当坐标即可
  (与 DOCS.md A 条一致:坐标是 .s32 寄存器)。TMA 不做 gather,但本题不需要 gather:一块就是页内连续 tile。
- 单块延迟 0.83 us 是 "一次 DRAM 往返 + 64 KiB 进 smem" 的下限;Triton 每块 2.3 us 中,载入本身只占 ~0.9 us,
  其余是 cp.async 由 128 线程发 4096 条 16 B 请求的发射开销 + 计算 + 不重叠。
- 4/16 块/CTA 配置首轮因 smem 超 227 KB 失败(实验设计错误),已改成 3 级 ring 流式,job 24077 重跑。

## S4 — job 24077/24078:融合 kernel v1(`exp/msa_decode_v1.cu`)正确但慢
- 正确性:ACCEPTANCE 形状矩阵 108 个 (case × 配置) 全 PASS(err_R0 2.4e-3–2.6e-3,**优于** Triton 的 2.5e-3–3.3e-3;
  逐元素对 Triton max|Δ| ≤ 9.8e-4;无 NaN)。含 real_topk<16、尾块、dql=2、cl∈{1,2,4,8,16}。
- 性能(graph,us):b=1 Triton 8.1,v1 最好 cl4s2 21.0;cl1s3 26.9;cl16s1 40.4。**慢 2.6–5×**。
- TMA 流式实验(job 24077)证明载入不是问题:1 CTA 用 3 级 ring 流 16 块只要 4.6 us(≈ 14 GB/s… 即 1 MiB/4.6us = 228 GB/s 单 CTA)。
- ncu(b=1 cl1s3):4 CTA,SM 一直 active(sm__cycles_active ≈ 时长),DRAM 1.8%;每 warp 执行 7,340 条指令
  (= 16 块 × ~460 条),Warp Cycles/Issued Inst = 4.0,stall 首位 `wait`(固定延迟依赖)。
  → 瓶颈是 **单 warp 串行发射**:每 SM 只有 4 warps(每调度器 1 个),ldmatrix→mma→softmax→mma 的依赖链
  延迟全部暴露,~460 条/块 × 4–6 cycle。这与 Triton 基线是同一种病(它也是 4 warps/CTA、1–2 CTA/SM)。
- cl16 随 CTA 数线性变慢(b=1 40 us → b=16 242 us):16-CTA cluster 要在同一 GPC 里凑 16 个空 SM,
  200 KB smem/CTA 时一个 GPC 只能放一个 cluster,cluster 之间近乎串行 → **cluster=16 不可取**,cl≤4 才合理。

### v2 设计决定(据上面数据)
1. 8 warps/CTA,分 2 个 warp-group,各处理奇/偶块 → 每 warp 指令减半、每调度器 2 warps 可互相掩盖延迟。
2. 去掉每块都做的掩码/重缩放:valid==128 时跳过掩码;alpha 全为 1 时跳过 64 条 acc 重缩放(warp 一致分支)。
3. 循环不变的 swizzle 地址提到循环外。
4. cluster 只保留 1/2/4。

## S5 — job 24079/24080:v2(8 warps 双 group 共享 ring)崩溃分析
- v2 只有 cl4s1 报 `unspecified launch failure`(compute-sanitizer:Unknown Error 在 thread 0/128 处);其它配置数值正确。
- 根因(mbarrier 语义):`mbarrier.try_wait.parity` 只能区分"当前相位"与"上一相位";两个 group 交错消费同一个
  stage ring 时,group 1 可能在 group 0 还没消费完 block i 时就去等 block i+STAGES(同一 stage、两个相位之后),
  parity 恰好等于"上一相位"→ 立即返回真,读到旧数据(STAGES=3 是潜在竞态),而 STAGES=1 时更进一步让第二个
  arrive.expect_tx 落在未完成的相位上 → 硬件错误。
- 修正(v3):**每个 warp-group 独占一个 stage**(NG = STAGES,4·STAGES warps),group g 只消费 i ≡ g (mod NG) 的块,
  每个 barrier 的相位严格按消费顺序推进,不存在跨 group 的相位歧义。STAGES=3 → 12 warps(384 线程 × 170 reg = 65,280 ≤ 65,536)。

## S5b — job 24080:讨论点 4 FP8 scale 实验(`exp/e8_fp8scale.py`,scale 0.3/0.7)
| 放法 | 对反量化参照 max|Δ| / err_ratio | 上游阈值 2e-2 |
|--|--|--|
| (A) kernel 内逐元素(上游 mode 1) | 3e-5 / 3e-5 | OK |
| (B) host 折进 Q + epilogue 乘 v_scale,kernel 不感知 scale | 2.3e-4 / 4.2e-3 | OK |
| (C) kernel 内 per-token(上游 mode 2) | 3e-5 / 3e-5 | OK |
| 不加 scale | 8e-2 / 1.25 | FAIL(应当) |
计时(graph,us):b=1 bf16 8.3 / fp8 无 scale 11.0 / fp8 标量 11.8 / fp8 per-token 12.5;b=8:14.9 / 19.8 / 24.2 / 21.7。
→ 上游 Triton 的 fp8 路径比 bf16 **慢 30–60%**:省的是带宽(小 batch 根本不缺),付的是 fp8→bf16 转换 + 乘 scale 的指令
(每块 2×128×128 个元素,恰好加在串行链上)。
