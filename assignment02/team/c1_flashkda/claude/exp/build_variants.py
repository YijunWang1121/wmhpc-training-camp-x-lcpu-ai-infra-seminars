#!/usr/bin/env python3
"""Build ablation variants of FlashKDA (K2 attribution, 'profile -> next step' loop).

Each variant = pristine upstream tree (pin 1ce47ea) + text patches on csrc/smxx/fwd_kernel2.cuh, compiled for
sm_103a only into a module named flash_kda_C_<variant> (loaded directly by exp/e4_ablate.py, bypassing the
python wrapper).  Login node only (no GPU needed).

Variants:
  stock      : unmodified (sanity: same timing as the pip-installed module)
  notma      : -DTMA_DISABLE_ALL   (upstream macro: no load/store warps, no pipeline waits; MMA warps compute on
                                     whatever is in smem -> compute-chain floor of K2)
  notma_noP1 : notma + Phase 1 (dual GEMM k@S, q@S, 64 HMMA/warp) removed
  notma_noP6 : notma + Phase 6 (state update, 32 HMMA/warp + full-state LDSM/STSM) removed
  notma_noP34: notma + Phase 3/4 GEMMs (INV@u, Mqk@U, 8 HMMA/warp + MOVM transposes) removed
  notma_noMMA: notma + all three removed (what is left: TMA-less loop, casts, barriers)
  noP6       : stock pipeline + Phase 6 removed (does the load pipeline hide behind compute?)
"""
import os, re, shutil, subprocess, sys, pathlib

SRC = pathlib.Path.home() / "flashkda-build" / "FlashKDA"
OUT = pathlib.Path.home() / "flashkda-build" / "variants"
ROOT = pathlib.Path("/home/lcpu/00737767/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02")
DEST = ROOT / "team/c1_flashkda/claude/exp/variants"
PY = ROOT / ".venv/bin/python"

P1_BEGIN = "            // ======== Phase 1: Dual GEMM k@s and q@s (k-loop, 2 blocks per warp) ========"
P1_END = "            // ======== Phase 2: Cast out (keep in regs), load v/INV/beta ========"
P6_BEGIN = "            // ======== Phase 6: s_acc update ========"
P6_END = "            }\n            compute_barrier.arrive_and_wait();"
P3_GEMM = "                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB_u_tmp(_,_,Int<0>{}), u_acc[i]);"
P4_GEMM = "                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB_u_arr[i](_,_,Int<0>{}), out_acc[i]);"


def patch(text, name):
    if "noP1" in name or "noMMA" in name:
        a, b = text.index(P1_BEGIN), text.index(P1_END)
        text = text[:a] + "            // [ablation] Phase 1 removed\n" + text[b:]
    if "noP6" in name or "noMMA" in name:
        a, b = text.index(P6_BEGIN), text.index(P6_END)
        text = text[:a] + "            // [ablation] Phase 6 removed\n" + text[b:]
    if "noP34" in name or "noMMA" in name:
        assert P3_GEMM in text and P4_GEMM in text
        text = text.replace(P3_GEMM, "                // [ablation] P3 gemm removed").replace(P4_GEMM, "                // [ablation] P4 gemm removed")
    return text


def build(name):
    tree = OUT / name
    if tree.exists():
        shutil.rmtree(tree)
    shutil.copytree(SRC, tree, symlinks=True, ignore=shutil.ignore_patterns(".git", "build", "*.egg-info", "cutlass"))
    os.symlink(SRC / "cutlass", tree / "cutlass")
    k2 = tree / "csrc/smxx/fwd_kernel2.cuh"
    k2.write_text(patch(k2.read_text(), name))
    setup = tree / "setup.py"
    s = setup.read_text()
    s = s.replace("name='flash_kda_C'", f"name='flash_kda_C_{name}'")
    s = s.replace('subprocess.run(["git", "submodule", "update", "--init", "cutlass"])', "pass")
    if "notma" in name:
        s = s.replace("'-O3',\n                '-U__CUDA_NO_HALF_OPERATORS__'", "'-O3', '-DTMA_DISABLE_ALL',\n                '-U__CUDA_NO_HALF_OPERATORS__'")
        assert "-DTMA_DISABLE_ALL" in s
    setup.write_text(s)
    env = dict(os.environ, FLASH_KDA_CUDA_ARCHS="103a", NVCC_THREADS="16")
    log = tree / "build.log"
    with open(log, "w") as f:
        r = subprocess.run([str(PY), "setup.py", "build_ext", "--inplace"], cwd=tree, env=env, stdout=f, stderr=subprocess.STDOUT)
    sos = list(tree.glob(f"flash_kda_C_{name}*.so"))
    if r.returncode or not sos:
        print(f"[{name}] BUILD FAILED, see {log}"); return False
    DEST.mkdir(parents=True, exist_ok=True)
    shutil.copy(sos[0], DEST / sos[0].name)
    regs = re.findall(r"_flash_kda_fwd_recurrence.*?\n.*?Used (\d+) registers", open(log).read(), re.S)
    print(f"[{name}] ok -> {DEST / sos[0].name}  (K2 regs sample: {regs[:2]})")
    return True


if __name__ == "__main__":
    names = sys.argv[1:] or ["stock", "notma", "notma_noP1", "notma_noP6", "notma_noP34", "notma_noMMA", "noP6"]
    for n in names:
        build(n)
