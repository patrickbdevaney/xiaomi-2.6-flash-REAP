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
        # THE OFF-DIAGONAL MAGNITUDE BOUND WAS REMOVED. It is ill-posed, and two successive
        # attempts to keep it both rejected correct data and cost the run a restart each.
        #
        # Cauchy-Schwarz bounds F_ij by sqrt(E_Xij[a_i^2] * E_Xij[a_j^2]) -- expectations over
        # the CO-ACTIVATION set. The diagonal we have is F_ii = E_Xi[a_i^2], an expectation over
        # a DIFFERENT, larger set. If an expert fires harder than its own average precisely when
        # its partner fires, |F_ij| exceeds sqrt(F_ii*F_jj) by an arbitrary margin with nothing
        # wrong. Gating on co-activation count did not help, because the effect is conditional
        # structure and not sampling noise: real chunk 1 still violated it on pairs with >= 64
        # co-activations. Making it sound would need per-pair second moments, which the pass
        # does not accumulate.
        #
        # Replaced by two statements about the SLOT BOOKKEEPING that are exactly true, which is
        # what the magnitude bound was a proxy for in the first place:
        #
        #   f_cnt[i,i] == cnt_i     an expert co-activates with itself on exactly its own tokens
        #   f_cnt[i,j] <= min(cnt_i, cnt_j)   a pair cannot co-fire more often than either fires
        #
        # A wrong index in the outer-product scatter breaks both immediately, and neither can
        # be tripped by legitimate statistics.
        fc = f_cnt[li].cpu()
        dc = torch.diagonal(fc)
        bad = (dc[live] - cnt[live]).abs().max().item() if bool(live.any()) else 0.0
        if bad > 0.5:
            problems.append(f"{name}: f_cnt diagonal disagrees with the routed-token count by "
                            f"{bad:.0f} -- the F slot bookkeeping is wrong")
        # The SOUND cap is 2*min, not min: FAccumulator adds both orders (i,j) and (j,i), so
        # f_cnt[i,j] = 2 * (tokens where both fire) and that is bounded by 2*min(cnt_i, cnt_j).
        # Measured values sit far below even 1*min on real data -- co-activation is rare at
        # top-8 of 256 -- but "observed on one sample" is exactly the reasoning that produced
        # two false failures already, so the bound used here is the one that is guaranteed.
        pair_cap = 2.0 * torch.minimum(cnt[:, None], cnt[None, :])
        over = int((fc > pair_cap + 0.5).sum().item())
        if over:
            worst = float((fc - pair_cap).max().item())
            problems.append(f"{name}: {over} expert pairs co-activate more often than one of "
                            f"them activates at all (worst excess {worst:.0f}) -- impossible, "
                            f"the F slot bookkeeping is wrong")

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
