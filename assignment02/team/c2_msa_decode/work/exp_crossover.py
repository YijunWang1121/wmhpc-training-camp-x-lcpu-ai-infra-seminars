"""Discussion point 7 — reproduce the Q-outer vs KV-outer crossover.

Fireworks' I/O cost model: KV-outer (persistent 128x128, ~ CUTLASS decode path)
beats Q-outer (per-query, ~ Triton split-K) iff  nsb / N < 2.85
  nsb = number of DISTINCT (kv_head, page) blocks selected across the whole batch
  N   = number of query tokens (= batch, dql=1)

We sweep batch, build the synthetic top-k under two selection models, and report
nsb/N.  We also give the model's predicted runtimes:
  T_qouter  ~ c * (N * topk)         (each query loads its 16 blocks)
  T_kvouter ~ c * (nsb + N*topk/128 * overhead)  (each distinct block loaded once,
              MMA over the queries that picked it)
and mark the crossover batch.
"""
import math
import torch

PAGE = 128
TOPK = 16
KVH = 4


def build_topk(batch, seq, model, seed):
    """Return list of sets: selected[(kv_head)] -> set of (page) logical ids.
    model: 'indep' random uniform;  'prefix' shared-prefix locality."""
    g = torch.Generator().manual_seed(seed)
    nblk = math.ceil(seq / PAGE)
    per_head_blocks = [set() for _ in range(KVH)]
    total_sel = 0
    for r in range(batch):
        for kh in range(KVH):
            if model == "indep":
                sel = torch.randperm(nblk, generator=g)[:TOPK].tolist()
            else:  # prefix: 8 shared "hot" blocks + 8 random, models prompt reuse
                hot = list(range(min(8, nblk)))
                rest = torch.randperm(nblk, generator=g)[:8].tolist()
                sel = set(hot + rest)
            sel = set(sel)
            sel.add(nblk - 1)  # current block always
            per_head_blocks[kh] |= {(r_and := p) for p in sel} if False else set()
            # distinct across batch: tag by (kh, page) — page shared across reqs
            for p in sel:
                per_head_blocks[kh].add(p)
            total_sel += len(sel)
    nsb = sum(len(s) for s in per_head_blocks)
    return nsb, total_sel


print("KV-stationary decode (fmha_sm100) gathers, per distinct (kv_head,page),")
print("the query TOKENS that selected it and runs ONE MMA with")
print("   M = avg_reuse * GQA(16).   A Blackwell tcgen05 MMA tile is up to 128 rows.")
print("Below M ~ 64 the MMA is <50% full -> the KV-outer machinery (TMA pipeline,")
print("TMEM, persistent scheduler, no-merge) doesn't pay for its fixed prologue.\n")
print(f"{'batch':>6} {'model':>7} {'sum_sel':>8} {'nsb':>7} {'avg_reuse':>10} "
      f"{'M=reuse*16':>11} {'MMA_fill':>9} {'nsb/N':>7}")
for model in ("indep", "prefix"):
    for n in (1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64):
        nsb, ssel = build_topk(n, 8192, model, seed=0)
        reuse = ssel / nsb
        M = reuse * 16
        fill = min(1.0, M / 128)
        mark = "  <-- M crosses 64" if 3.8 < reuse < 4.3 else ""
        print(f"{n:>6} {model:>7} {ssel:>8} {nsb:>7} {reuse:>10.2f} "
              f"{M:>11.0f} {fill:>8.0%} {nsb/n:>7.2f}{mark}")
