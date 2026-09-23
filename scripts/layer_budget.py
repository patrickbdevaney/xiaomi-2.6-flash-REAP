"""Search a PER-LAYER prune budget instead of pruning every layer by the same fraction.

Uniform allocation is the obvious choice and it is measurably wrong: this project has already
recorded that a uniform per-layer budget costs about 5 points against a searched one. Layers are
not equally redundant -- some carry many near-duplicate experts and can give up more than half,
others carry specialists and cannot give up a third -- and a global ratio ignores all of that.

WHAT IS HELD FIXED
------------------
The TOTAL number of pruned experts. The Thor fit is what forces 50% overall (94.7 GB of weights
against a 125.6 GB envelope), so the budget may move freely between layers but its sum may not
move at all. Every candidate in the search satisfies that exactly, by construction, rather than
by penalty -- a penalty would let the search buy fitness with a checkpoint that does not fit.

OBJECTIVE
---------
Worst-domain retention, the same quantity criterion_compare ranks by. Optimising the mean would
reproduce the averaging failure the whole pipeline is built to avoid: a budget that is excellent
on eight domains and catastrophic on audio would win on the mean and destroy the model's one
distinguishing capability.

DEPLOYABILITY: A NON-UNIFORM BUDGET IS NOT PORTABLE
---------------------------------------------------
Checked against the checkpoint's own modeling code before relying on this: `n_routed_experts`
is a SCALAR in config.json and is read globally --

    self.experts = nn.ModuleList(
        [MiMoV2MLP(config, ...) for _ in range(config.n_routed_experts)])    # line 192

-- so every MoE layer is built with the same expert count. llama.cpp reads a single `n_expert`
hparam as well. A per-layer budget therefore CANNOT be expressed in stock transformers, in vLLM,
or in a GGUF; it requires patched modeling code and would only run on our own CUDA server.

Padding the short layers back up to a uniform count would restore portability and throw away the
entire point, since the Thor fit is a SIZE constraint.

So this search produces a number to decide with, not automatically a checkpoint to ship:
    * if the measured gain is small, take the uniform budget and stay portable;
    * if it is large, the gain has to be weighed against losing GGUF and vLLM.
`--verify-with-hope` and the reported `gain` exist so that decision is made on our own
measurement rather than the ~5% carried over from the GLM project.

WHY THE SEARCH RUNS IN REAP MODE
--------------------------------
Fitness is evaluated thousands of times. A HOPE QP per layer per candidate is far too expensive,
while a score sort is microseconds, and the two agree on the ORDERING of layer budgets even
where they disagree on which experts to drop. So the budget is searched with the cheap selector
and then APPLIED with HOPE. That is an explicit approximation, stated here so it can be checked:
`--verify-with-hope` re-scores the winning budget under HOPE and reports the difference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import reap_select as RS


def _layer_tables(acc_path: Path, criterion: str):
    """Per-layer (scores, domain mass, F) -- computed once, reused by every candidate."""
    acc, f_sum, f_cnt, buckets = RS.load_acc(acc_path)
    names = sorted(acc, key=RS.layer_index)
    tab = []
    for n in names:
        a = acc[n]
        li = RS.layer_index(n)
        tab.append({
            "name": n,
            "scores": RS.scores(a, criterion),
            "mass": RS.domain_mass(a),
            "F": (f_sum[li] / f_cnt[li].clamp(min=1)).numpy(),
            "n_exp": a["cnt"].shape[1],
        })
    return tab, buckets


def fitness(tab, budget: np.ndarray, mode: str = "reap") -> tuple:
    """-> (worst-domain retention, per-domain retention vector)."""
    rets = []
    for t, k in zip(tab, budget):
        pruned = RS.select_layer(t["F"], t["scores"], int(k), mode)
        rets.append(RS.retention(t["mass"], pruned))
    R = np.stack(rets)                        # [n_layer, n_bucket]
    per_domain = R.mean(0)
    return float(per_domain.min()), per_domain


def search(tab, total_prune: int, n_exp: int, generations: int = 300, pop: int = 16,
           seed: int = 0, min_frac: float = 0.20, max_frac: float = 0.75,
           mode: str = "reap") -> tuple:
    """Evolutionary search over integer per-layer budgets with a fixed sum.

    `min_frac`/`max_frac` bound each layer. Without them the search will happily strip one layer
    to nothing to buy a fraction of a point elsewhere, which is not a trade the forward pass
    survives -- a layer with almost no experts left is a bottleneck no retention number sees.
    """
    rng = np.random.default_rng(seed)
    L = len(tab)
    lo = np.full(L, int(n_exp * min_frac))
    hi = np.full(L, int(n_exp * max_frac))

    base = np.full(L, total_prune // L)
    base[: total_prune - base.sum()] += 1
    base = np.clip(base, lo, hi)

    def repair(x):
        """Restore the exact total after mutation, respecting the per-layer bounds."""
        x = np.clip(x, lo, hi).astype(int)
        while x.sum() != total_prune:
            d = total_prune - x.sum()
            idx = rng.permutation(L)
            for i in idx:
                if d == 0:
                    break
                step = 1 if d > 0 else -1
                if lo[i] <= x[i] + step <= hi[i]:
                    x[i] += step
                    d -= step
        return x

    population = [repair(base.copy())]
    for _ in range(pop - 1):
        population.append(repair(base + rng.integers(-3, 4, L)))
    scored = [(fitness(tab, p, mode)[0], p) for p in population]
    scored.sort(key=lambda t: -t[0])
    best0 = scored[0][0]

    for _ in range(generations):
        parent = scored[rng.integers(0, max(2, pop // 2))][1]
        child = parent.copy()
        # Move budget BETWEEN layers: the sum is the invariant, so every mutation is a transfer.
        for _ in range(int(rng.integers(1, 4))):
            i, j = rng.integers(0, L, 2)
            amt = int(rng.integers(1, 5))
            child[i] += amt
            child[j] -= amt
        child = repair(child)
        f = fitness(tab, child, mode)[0]
        if f > scored[-1][0]:
            scored[-1] = (f, child)
            scored.sort(key=lambda t: -t[0])
    return scored[0][1], scored[0][0], best0


def run(acc_path: Path, out_path: Path, ratio: float, criterion: str,
        generations: int, verify_with_hope: bool = False) -> dict:
    tab, buckets = _layer_tables(acc_path, criterion)
    n_exp = tab[0]["n_exp"]
    total = int(round(len(tab) * n_exp * ratio))
    budget, best, uniform = search(tab, total, n_exp, generations=generations)
    _, per_dom = fitness(tab, budget)
    _, per_dom_u = fitness(tab, np.full(len(tab), total // len(tab)))
    res = {
        "criterion": criterion, "ratio": ratio,
        "total_pruned": int(budget.sum()), "total_required": total,
        "layers": len(tab), "experts_per_layer": n_exp,
        "uniform_worst": float(uniform), "searched_worst": float(best),
        "gain": float(best - uniform),
        "searched_by_domain": {b: float(v) for b, v in zip(buckets, per_dom)},
        "uniform_by_domain": {b: float(v) for b, v in zip(buckets, per_dom_u)},
        "budget": {t["name"]: int(k) for t, k in zip(tab, budget)},
    }
    if verify_with_hope:
        h, hd = fitness(tab, budget, mode="hope")
        res["hope_worst_on_searched_budget"] = float(h)
        res["hope_by_domain"] = {b: float(v) for b, v in zip(buckets, hd)}
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc", default="artifacts/saliency/accumulators.pt")
    ap.add_argument("--out", default="artifacts/masks/layer_budget.json")
    ap.add_argument("--ratio", type=float, default=0.50)
    ap.add_argument("--criterion", default="reap_1_1_1", choices=list(RS.CRITERIA))
    ap.add_argument("--generations", type=int, default=300)
    ap.add_argument("--verify-with-hope", action="store_true")
    a = ap.parse_args()
    r = run(Path(a.acc), Path(a.out), a.ratio, a.criterion, a.generations, a.verify_with_hope)
    print(json.dumps({k: v for k, v in r.items() if k != "budget"}, indent=1))
    print(f"total pruned {r['total_pruned']} (required {r['total_required']}) "
          f"| uniform {r['uniform_worst']:.4f} -> searched {r['searched_worst']:.4f} "
          f"({r['gain']:+.4f})")
