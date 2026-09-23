"""Per-chunk invariants for the calibration pass. Cheap, and fatal when violated.

A 34-hour pass whose F-matrix cannot be recovered from a partial result must not be allowed to
spend 30 of those hours accumulating garbage. Everything here runs on tensors already in memory
(or a checkpoint on disk), costs milliseconds, and is called after every chunk.

The load-bearing check is the first one. F's diagonal and `sq/cnt` are computed by two
INDEPENDENT paths over the same tokens -- the outer-product scatter and the per-expert reduction
-- so their agreement is evidence that the per-token slot bookkeeping behind HOPE's off-diagonal
is correct. Nothing else in the pipeline tests that, and if it were wrong the off-diagonal would
be silently meaningless while every shape, count and dtype stayed plausible.

Run standalone against a checkpoint at any time:  python scripts/verify_pass.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


class VerifyError(RuntimeError):
    pass


def verify(acc: dict, f_sum: torch.Tensor, f_cnt: torch.Tensor, buckets: list,
           top_k: int, prev: dict | None = None) -> dict:
    """Raise VerifyError on any broken invariant; return a status dict on success."""
    problems, stats = [], {}
    cov = []
    for name, a in sorted(acc.items()):
        li = int(name.split(".")[2])
        cnt = a["cnt"].sum(0).double().cpu()
        live = cnt > 0
        if not bool(live.any()):
            problems.append(f"{name}: no expert was ever routed to")
            continue
        cov.append(float(live.float().mean()))

        for k, v in a.items():
            if not torch.isfinite(v).all():
                problems.append(f"{name}.{k}: contains NaN or Inf")
            if bool((v < 0).any()):
                problems.append(f"{name}.{k}: negative value in a sum of non-negative terms")

        F = (f_sum[li] / f_cnt[li].clamp(min=1)).double().cpu()
        if not torch.allclose(F, F.T, rtol=1e-9, atol=1e-12):
            problems.append(f"{name}: F is not symmetric")
        d_f = torch.diagonal(F)[live]
        d_a = (a["sq"].sum(0).double().cpu() / cnt.clamp(min=1))[live]
        rel = ((d_f - d_a).abs() / d_a.clamp(min=1e-30)).max().item()
        if rel > 1e-6:
            problems.append(f"{name}: F diagonal disagrees with sq/cnt by {rel:.2e} relative "
                            f"-- the F outer-product slot bookkeeping is wrong")
        # Cauchy-Schwarz on the conditional means: |F_ij| <= sqrt(F_ii * F_jj) need not hold
        # exactly under conditional averaging, but a gross violation means index corruption.
        off = F - torch.diag(torch.diagonal(F))
        bound = torch.sqrt(torch.outer(torch.diagonal(F).clamp(min=0),
                                       torch.diagonal(F).clamp(min=0)))
        viol = (off.abs() > 10 * bound + 1e-9).sum().item()
        if viol:
            problems.append(f"{name}: {viol} off-diagonal entries exceed 10x sqrt(Fii*Fjj)")

        if prev and name in prev:
            for k in ("sum", "sq", "cnt"):
                if bool((a[k].double().cpu() < torch.as_tensor(prev[name][k]) - 1e-6).any()):
                    problems.append(f"{name}.{k}: DECREASED since the last chunk; these are "
                                    f"running sums and must never shrink")
        stats[name] = {"routed": float(cnt.sum()), "coverage": float(live.float().mean())}

    if problems:
        raise VerifyError("; ".join(problems[:6]))
    return {"layers": len(acc), "coverage_min": min(cov) if cov else 0.0,
            "coverage_mean": sum(cov) / len(cov) if cov else 0.0, "per_layer": stats}


def snapshot(acc: dict) -> dict:
    return {n: {k: a[k].double().cpu().clone() for k in ("sum", "sq", "cnt")}
            for n, a in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/saliency")
    ap.add_argument("--top-k", type=int, default=8)
    a = ap.parse_args()
    d = torch.load(Path(a.out) / "accumulators.pt", map_location="cpu")
    acc = {k: {kk: vv for kk, vv in v.items()} for k, v in d["acc"].items()}
    try:
        st = verify(acc, d["f_sum"], d["f_cnt"], d["buckets"], a.top_k)
    except VerifyError as e:
        print(f"VERIFY FAIL: {e}")
        sys.exit(1)
    print(f"VERIFY PASS  layers {st['layers']}  coverage min {st['coverage_min']:.1%} "
          f"mean {st['coverage_mean']:.1%}")


if __name__ == "__main__":
    main()
