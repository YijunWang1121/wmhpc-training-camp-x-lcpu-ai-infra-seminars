#!/bin/sh
# 3.4 (b): ncu 对比 cta_group::1 vs ::2 的 staging store 和 Tensor Core shared 读。
# 用法(必须在 GPU allocation 里):
#   srun -G 1 --time=00:20:00 --pty bash
#   cd cuda/m3_tcgen05 && ./prof_cta_pair.sh
#
# 04_cta_pair 的 main() 会把每个 kernel 调用 ~201 次(1 次校验 + 200 次计时),
# 所以用 -c 2 只抓前两次 launch:第 1 次 = tile_kernel<1>,第 2 次 = tile_kernel<2>。
set -e
cd "$(dirname "$0")/.."          # -> cuda/

BIN=bin/m3_tcgen05/04_cta_pair
OUT="${OUT:-/tmp/cta_pair_prof}"

make "$BIN"                      # 只 build,不 run

METRICS="l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
smsp__inst_executed_op_shared_st.sum,\
smsp__inst_executed_op_shared_ld.sum,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,\
smsp__cycles_active.avg"

echo "==================== 关键计数器(终端速览) ===================="
ncu -k "tile_kernel" -c 2 --metrics "$METRICS" "$BIN"

echo
echo "==================== 完整 section -> $OUT.ncu-rep ===================="
ncu -k "tile_kernel" -c 2 \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    --section SchedulerStats \
    --section SourceCounters \
    -f -o "$OUT" "$BIN"

echo
echo "看报告:  ncu-ui $OUT.ncu-rep       (或)  ncu --import $OUT.ncu-rep --page details"
