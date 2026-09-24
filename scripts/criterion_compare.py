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
            criteria=None, protect_frac: float = 0.0, allow_dead: bool = False,
            select_by: str = "hope") -> list:
    criteria = criteria or list(RS.CRITERIA)
    _acc, _fs, _fc, _bk = RS.load_acc(acc)
    RS.assert_domains_live(_acc, _bk, "criterion comparison", allow_dead)
    del _acc, _fs, _fc
    rows = []
    for crit in criteria:
        for mode in modes:
            out = out_dir / f"mask_{crit}_{mode}.json"
            r = RS.run(acc, out, ratio, mode, crit, protect_frac=protect_frac,
                       allow_dead=allow_dead)
            rows.append(r)
    # SORTED BY REAP'S METRIC, but the HOPE objective is carried in every row and printed, so a
    # selection is never chosen without both numbers in view. Ranking by retention alone is what
    # would discard HOPE unseen; see reap_select.run for why the two rulers disagree by design.
    # SELECTION RULER. Default HOPE, because ranking by retained mass is ranking HOPE by the
    # objective REAP maximises by construction -- measured here, REAP scores 0.9998 retention at
    # p^T F p = 3.03e-02 while HOPE scores 0.9882 at 1.21e-02, a 2.5x smaller interaction-aware
    # output error. Retention alone would pick REAP every time and discard the off-diagonal that
    # the entire calibration pass exists to accumulate and that cannot be recovered afterwards.
    # arXiv 2609.18916 measures HOPE at +2.8-6.1% agentic over REAP at 40-50%.
    #
    # NOTE: in HOPE mode every criterion gives an IDENTICAL selection, because solve_prune_set
    # uses F and never reads the scalar criterion. The criterion only matters under REAP.
    if select_by == "hope":
        rows.sort(key=lambda r: (r["interaction_cost"], -r["worst_retention"]))
    else:
        rows.sort(key=lambda r: -r["worst_retention"])
    return rows


def render(rows: list) -> str:
    buckets = rows[0]["buckets"]
    w = max(12, max(len(b) for b in buckets) + 1)
    head = f"{'criterion':<14}{'sel':<6}" + "".join(f"{b:>{w}}" for b in buckets)
    head += f"{'WORST':>12}{'mean':>10}{'pFp (HOPE)':>12}"
    lines = [head, "-" * len(head)]
    for r in rows:
        line = f"{r['criterion']:<14}{r['mode']:<6}"
        for b in buckets:
            line += f"{r['retention_by_domain'][b]:>{w}.3f}"
        line += (f"{r['worst_retention']:>12.6f}{r['mean_retention']:>10.6f}"
                 f"{r['interaction_cost']:>12.3e}")
        lines.append(line)
    best = rows[0]
    lines.append("")
    bh = min(rows, key=lambda r: r["interaction_cost"])
    lines.append(f"BEST BY HOPE OBJECTIVE (p^T F p, lower is better): {bh['criterion']} / "
                 f"{bh['mode']} at {bh['interaction_cost']:.3e}")
    lines.append("  The two rulers disagree BY CONSTRUCTION: worst-domain retention is the "
                 "quantity REAP's greedy")
    lines.append("  ranking maximises directly, so ranking by it alone always favours REAP and "
                 "discards the")
    lines.append("  interaction terms the F accumulation exists to provide.")
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
    ap.add_argument("--select-by", choices=["hope", "retention"], default="hope",
                    help="which ruler picks the winner written to comparison.json (default hope)")
    ap.add_argument("--allow-dead-domains", action="store_true",
                    help="rank using only the domains that have mass")
    a = ap.parse_args()
    rows = compare(Path(a.acc), a.ratio, Path(a.out_dir), protect_frac=a.protect_frac,
                   allow_dead=a.allow_dead_domains, select_by=a.select_by)
    txt = render(rows)
    print(txt)
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(a.out_dir) / "comparison.txt").write_text(txt + "\n")
    (Path(a.out_dir) / "comparison.json").write_text(json.dumps(rows, indent=1))
