"""Turn the calibration pass's statistics into a prune mask.

This is the step where the 34-hour pass becomes a decision. Nothing here touches the model or
the corpus: it reads `accumulators.pt` and writes a per-layer keep/prune set, so it is cheap to
re-run under different criteria, which is the whole point -- the criterion is a choice we must
MEASURE, not assume.

THE CRITERION FAMILY (arXiv 2606.15716)
---------------------------------------
    S(b, a, B) = (1 / N^b) * sum over the expert's routed tokens of  g^a * ||f||^B

with g the router gate and ||f|| the expert's output norm. Four members are recoverable from
what the pass accumulated, which is why those seven statistics were collected together rather
than just REAP's:

    (1,1,1)  sum/cnt   REAP as published -- the MEAN gated output norm
    (0,1,1)  sum       the TOTAL, so a rarely-routed expert cannot score like a busy one
    (1,2,2)  sq/cnt    mean of the squared contribution: rewards consistency
    (0,2,2)  sq        total squared contribution

At 50% on coding workloads the published task-specific winners are (0,1,1) and (0,2,2), NOT
REAP's (1,1,1). That is a large enough claim to check on our own model rather than adopt.

HOPE vs REAP
------------
REAP scores each expert alone. HOPE (arXiv 2609.18916) minimises p^T F p over the prune set,
where F carries the PAIRWISE interactions -- two experts that duplicate each other are cheap to
prune together, two that complement each other are not. REAP is exactly HOPE with the
off-diagonal zeroed, so both run from the same F and the comparison is free.

WHY EVALUATION IS PER DOMAIN, NEVER AVERAGED
--------------------------------------------
arXiv 2606.03328: an averaged score moved 2.85 points while code retention moved 51.9. An
average over domains would have called that configuration fine. Every number this module
reports is therefore per bucket, and the summary line is the WORST domain, not the mean.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

CRITERIA = {
    "reap_1_1_1":  dict(b=1, a=1, B=1, key="sum"),
    "total_0_1_1": dict(b=0, a=1, B=1, key="sum"),
    "mean_1_2_2":  dict(b=1, a=2, B=2, key="sq"),
    "total_0_2_2": dict(b=0, a=2, B=2, key="sq"),
}


def load_acc(path: Path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    acc = {k: {kk: vv.double() for kk, vv in v.items()} for k, v in d["acc"].items()}
    return acc, d["f_sum"].double(), d["f_cnt"].double(), list(d["buckets"])


def layer_index(name: str) -> int:
    return int(name.split(".")[2])


def scores(a: dict, crit: str, bucket_weights=None) -> np.ndarray:
    """Per-expert criterion score for one layer, summed over domains.

    `bucket_weights` lets a domain be up-weighted in the DECISION. It never affects the
    evaluation, which stays per domain -- mixing those two is how an average hides a collapse.
    """
    c = CRITERIA[crit]
    num = a[c["key"]]                      # [n_bucket, n_expert]
    cnt = a["cnt"]
    if bucket_weights is not None:
        w = torch.as_tensor(bucket_weights, dtype=torch.float64)[:, None]
        num, cnt = num * w, cnt * w
    num, cnt = num.sum(0), cnt.sum(0)
    s = num / cnt.clamp(min=1) if c["b"] == 1 else num
    return s.numpy()


def domain_mass(a: dict) -> np.ndarray:
    """[n_bucket, n_expert] of gated output mass -- the quantity retention is measured in."""
    return a["sum"].numpy()


def select_layer(F: np.ndarray, s: np.ndarray, n_prune: int, mode: str,
                 protect=None) -> np.ndarray:
    """-> indices to PRUNE."""
    if mode == "reap":
        # REAP is HOPE with interactions zeroed; do it explicitly so the control is exact.
        order = np.argsort(s)
        if protect is not None and len(protect):
            order = np.array([i for i in order if i not in set(protect)])
        return np.sort(order[:n_prune])
    if mode == "hope":
        from hope_fmatrix import solve_prune_set
        return np.sort(solve_prune_set(F, n_prune, protect=protect))
    raise ValueError(mode)


def retention(mass: np.ndarray, pruned: np.ndarray) -> np.ndarray:
    """Per-domain fraction of gated output mass carried by the KEPT experts."""
    total = mass.sum(1)
    keep = np.ones(mass.shape[1], dtype=bool)
    keep[pruned] = False
    kept = mass[:, keep].sum(1)
    return kept / np.maximum(total, 1e-30)


def run(acc_path: Path, out_path: Path, ratio: float, mode: str, crit: str,
        bucket_weights=None, protect_frac: float = 0.0) -> dict:
    acc, f_sum, f_cnt, buckets = load_acc(acc_path)
    per_layer, ret_by_layer = {}, []
    for name in sorted(acc, key=layer_index):
        li = layer_index(name)
        a = acc[name]
        n_exp = a["cnt"].shape[1]
        n_prune = int(round(n_exp * ratio))
        s = scores(a, crit, bucket_weights)
        F = (f_sum[li] / f_cnt[li].clamp(min=1)).numpy()
        protect = None
        if protect_frac > 0:
            # Protect the top slice of every DOMAIN separately. A domain that is a small share
            # of the corpus still has experts it cannot lose, and a global ranking will not see
            # them -- which is precisely how a media capability disappears silently.
            m = domain_mass(a)
            k = max(1, int(n_exp * protect_frac))
            protect = np.unique(np.concatenate([np.argsort(-m[b])[:k] for b in range(m.shape[0])]))
        pruned = select_layer(F, s, n_prune, mode, protect=protect)
        per_layer[name] = sorted(int(i) for i in pruned)
        ret_by_layer.append(retention(domain_mass(a), pruned))
    R = np.stack(ret_by_layer)                      # [n_layer, n_bucket]
    per_domain = R.mean(0)
    result = {
        "criterion": crit, "mode": mode, "ratio": ratio,
        "protect_frac": protect_frac,
        "buckets": buckets,
        "retention_by_domain": {b: float(v) for b, v in zip(buckets, per_domain)},
        "worst_domain": buckets[int(np.argmin(per_domain))],
        "worst_retention": float(per_domain.min()),
        "mean_retention": float(per_domain.mean()),
        "layers": len(per_layer),
        "pruned_per_layer": {k: len(v) for k, v in per_layer.items()},
    }
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({**result, "mask": per_layer}, indent=1))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc", default="artifacts/saliency/accumulators.pt")
    ap.add_argument("--out", default="artifacts/masks/mask.json")
    ap.add_argument("--ratio", type=float, default=0.50)
    ap.add_argument("--mode", choices=["reap", "hope"], default="hope")
    ap.add_argument("--criterion", choices=list(CRITERIA), default="reap_1_1_1")
    ap.add_argument("--protect-frac", type=float, default=0.0)
    a = ap.parse_args()
    r = run(Path(a.acc), Path(a.out), a.ratio, a.mode, a.criterion,
            protect_frac=a.protect_frac)
    print(json.dumps(r, indent=1))
