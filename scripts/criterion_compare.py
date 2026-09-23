"""Run every criterion x selector and report retention PER DOMAIN.

The output of this is the decision: which criterion and which selector produce the mask we
actually prune with. It is deliberately a table and not a scalar, because the one thing the
literature is unambiguous about here is that a scalar hides the failure -- 2.85 points of
averaged movement concealed 51.9 points of code retention (arXiv 2606.03328).

Ranking is by WORST DOMAIN. A configuration that is excellent on eight buckets and catastrophic
on audio is not a good configuration for an omnimodal model; it is the exact failure this whole
corpus was built to prevent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import reap_select as RS


def compare(acc: Path, ratio: float, out_dir: Path, modes=("reap", "hope"),
            criteria=None, protect_frac: float = 0.0) -> list:
    criteria = criteria or list(RS.CRITERIA)
    rows = []
    for crit in criteria:
        for mode in modes:
            out = out_dir / f"mask_{crit}_{mode}.json"
            r = RS.run(acc, out, ratio, mode, crit, protect_frac=protect_frac)
            rows.append(r)
    rows.sort(key=lambda r: -r["worst_retention"])
    return rows


def render(rows: list) -> str:
    buckets = rows[0]["buckets"]
    w = max(12, max(len(b) for b in buckets) + 1)
    head = f"{'criterion':<14}{'sel':<6}" + "".join(f"{b:>{w}}" for b in buckets)
    head += f"{'WORST':>10}{'mean':>8}"
    lines = [head, "-" * len(head)]
    for r in rows:
        line = f"{r['criterion']:<14}{r['mode']:<6}"
        for b in buckets:
            line += f"{r['retention_by_domain'][b]:>{w}.3f}"
        line += f"{r['worst_retention']:>10.3f}{r['mean_retention']:>8.3f}"
        lines.append(line)
    best = rows[0]
    lines.append("")
    lines.append(f"BEST BY WORST DOMAIN: {best['criterion']} / {best['mode']} "
                 f"-- worst = {best['worst_domain']} at {best['worst_retention']:.3f}")
    # The comparison that matters most: does the second-order term buy anything?
    by_key = {(r["criterion"], r["mode"]): r for r in rows}
    for crit in {r["criterion"] for r in rows}:
        a, b = by_key.get((crit, "reap")), by_key.get((crit, "hope"))
        if a and b:
            d = b["worst_retention"] - a["worst_retention"]
            lines.append(f"  HOPE vs REAP on {crit:<14} worst-domain {d:+.4f}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc", default="artifacts/saliency/accumulators.pt")
    ap.add_argument("--out-dir", default="artifacts/masks")
    ap.add_argument("--ratio", type=float, default=0.50)
    ap.add_argument("--protect-frac", type=float, default=0.0)
    a = ap.parse_args()
    rows = compare(Path(a.acc), a.ratio, Path(a.out_dir), protect_frac=a.protect_frac)
    txt = render(rows)
    print(txt)
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(a.out_dir) / "comparison.txt").write_text(txt + "\n")
    (Path(a.out_dir) / "comparison.json").write_text(json.dumps(rows, indent=1))
