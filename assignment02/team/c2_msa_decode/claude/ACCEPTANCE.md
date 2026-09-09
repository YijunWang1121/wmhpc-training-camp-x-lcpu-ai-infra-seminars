# 验收方案(讨论点 6)— 先贴出来,接受挑战

适用对象:任何替代 Triton 基线的小 batch decode 实现(本组的 CUDA 融合 kernel `exp/msa_decode.cu`)。

## 1. 参照(三档)
| 档 | 参照 | 用途 |
|--|--|--|
| R0 | `harness/ref_sdpa.py`:fp32、逐 (token, kv_head) 聚齐选中块做标准 softmax attention | 语义真值(块选择、block_table、尾块掩码、因果位置) |
| R1 | 上游 Triton `minimax_m3_sparse_attn_decode`(bf16 输入,fp32 累加,exp2 softmax) | 数值口径对齐:新 kernel 相对 R0 的误差不得显著大于 Triton 相对 R0 的误差 |
| R2 | 与 R1 逐元素比较 | 找系统性 bug(单 head / 单块错位这类误差会被 R0 的全局范数比掩盖) |

## 2. 误差口径与阈值
- 全局:`err_ratio = ||x − R0||₂ / ||R0||₂ < 2e-2`(harness 口径;Triton 实测 3.2e-3,新 kernel 目标同量级 ≤ 5e-3)。
- 逐 token/逐 head:`max_h ||x[t,h] − R0[t,h]||₂ / ||R0[t,h]||₂ < 5e-2`(防止"平均好、个别 head 烂")。
- 逐元素对 R1:`max |x − R1| ≤ 2^-7·max|R1| + 1e-2`(bf16 输出 1 ulp 量级 + 累加顺序差)。
- 输出中 **不得有 NaN/Inf**(包括 kv_len<128 的单块、real_topk<16 的短序列)。

## 3. 形状矩阵(每个 cell 至少 3 个 seed)
| 维度 | 取值 |
|--|--|
| batch | 1, 2, 3, 4, 8, 16 |
| seq_len | 50–300(real_topk<16 + 尾块掩码),1024–8192,32768(block_table 大) |
| decode_query_len | 1, 2(投机 decode:同一请求两个 query 位置、因果不同) |
| topk 内容 | 打乱的物理页(harness 默认);恒含当前块;各 kv_head 不同选择 |
| dtype | bf16 KV(主);fp8 e4m3 KV + scalar scale(讨论点 4 扩展,参照 = 反量化后 bf16 跑 R1,阈值同上游 test:atol=rtol=2e-2) |
不支持的形状(如 gqa≠16、head_dim≠128、topk≠16)必须 **显式报错**,不能静默算错。

## 4. 性能口径
- 计时:CUDA graph 内 20 次调用取平均(排除 host launch 开销),每个 batch 预热 20 次,重复 200 次取平均;
  另报 eager 数字说明 host 开销。
- 对照:同一 case、同一 graph 方法下的 Triton 基线(decode+merge 两 kernel)。
- 报告 b ∈ {1,4,8,16} 的 us/step 与加速比;并给出 b=32/64 说明超出适用范围后的行为(允许比 Triton 慢,但要说明为什么)。
- 所有数字来自 B300 本机实测,附原始日志。

## 5. 自曝弱点(欢迎挑战)
- R0 是 fp32 但 K/V 本身是 bf16 量化过的,所以"真值"只精确到 bf16 输入;阈值 2e-2 偏松,靠 R2 逐元素对比补。
- 合成 top-k 是随机块,真实 indexer 的块有局部性(相邻块常被一起选),真实 L2 命中率可能更高 → 实测带宽利用率偏保守。
- 未覆盖 KV_SCALE_MODE=2(per-token scale)和 fp8 e5m2。
