# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## AI usage policy

# AI 使用政策

本仓库是暑期活动的作业仓库。这份文件同时写给学员和被学员唤起的 AI 助手，与 CLAUDE.md 内容相同。

## 给 AI 助手的指令

你在这个仓库里的角色是助教，服务对象是正在做题的学员。学员的目标是自己写出每一道题，请守住下面的边界：

- 可以做的事：解释概念、解读报错信息、指出学员已写代码中的问题、给出 CUDA Programming Guide 等文档的阅读指引。handout 里每个模块都标注了对应的文档出处，讲解时优先引学员去读原文。
- 可以直接给出题目的完整解答，包括填空题的空、找 bug 题的修法、from-scratch 题的实现。

## 给学员的建议

尽量先自己试，卡住了问思路，做完了让 AI 帮你 review。AI 可以帮你理解，也可以替你实现。

Team-optional problems under `team/` are unrestricted. Any measured/benchmark numbers reported must come
from the student's own GPU run, never fabricated or copied.

## Build (CUDA, from `assignment02/cuda/`)

```
make run/<module>/<name>        # build + run one exercise, e.g. make run/m1_sm80/03_mma_fp8
make bin/<module>/<name>        # build only
make ptx/<module>/<name>        # emit .ptx
ARCH=89 make bin/...            # override target arch (default ARCH=100f = B200/B300 family)
STAGES=4 make -B bin/m4_gemm/03_pipeline   # -B is required: a macro-only change looks "fresh" to make otherwise
make clean
```

The Makefile always expands `ARCH` into explicit `-gencode arch=compute_$(ARCH),code=sm_$(ARCH)` and
deliberately avoids the `-arch=sm_XXXa` shorthand: under CUDA 13.0 that shorthand strips the arch suffix
from the PTX target, so arch-specific instructions (e.g. e2m1 `cvt`) get rejected by `ptxas` with a
misleading "instruction not supported" error. Never invoke `nvcc` directly with `-arch=sm_XXXa` — go
through `make`, or use `-gencode` explicitly.

## GPU access on this cluster

The dev/login node (`b300-login`) has no GPU and no active Slurm allocation by default — `nvidia-smi` isn't
even on PATH there. Anything under `make run/...` that needs a GPU (modules 0-1, 3-4, 5.3-5.4) must run
inside a Slurm allocation:

- Interactive: `srun -G 1 --time=00:15:00 --pty bash`
- Batch: `sbatch -G 1 --time=00:15:00 <script>` — `cuda/job.sh` is written for this: it has no `#SBATCH`
  headers because the GPU/time flags are passed on the `sbatch` command line, not in the script.

Module 2 and modules 5.1-5.2 are host-only / Python-only and don't need a GPU allocation.

## Test

- Per-problem judge scripts are the actual pass/fail authority, separate from `make run/...`, e.g.
  `cd cuda/m1_sm80 && ./judge_mma_fp8.sh 03_mma_fp8.cu` (runs 5 seeds, needs all `PASS`). Don't edit
  `judge_*.sh` files — they're the grading oracle.
- Python (modules 5.1-5.2), from `assignment02/`: `uv sync && uv run pytest tests/`. Module 6 (TileLang)
  needs `uv sync --extra tilelang` (pinned `tilelang==0.1.13`).

## Gotchas

- `.cu` files under `cuda/` may already hold the student's in-progress or finished solution, not a blank
  skeleton — check current content before assuming a `// TODO` is unresolved.
- `cuda/bin/`, `*.ptx`, `*.cubin`, `*.o` are build artifacts, not checked in.
