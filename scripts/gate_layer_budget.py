"""Gate the per-layer budget search.

The load-bearing property is the CONSTRAINT, not the objective: the total number of pruned
experts fixes the checkpoint size, and the Thor fit is why 50% is forced at all. A search that
buys retention by quietly pruning fewer experts is not an improvement, it is a checkpoint that
does not fit -- and every number it reports would still look good.

Second: the search must actually beat uniform on a fixture where the right answer is known --
one layer full of near-duplicate experts, another full of specialists.
"""
from __future__ import annotations
import sys, tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import layer_budget as LB      # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(name)


def make(td: Path, n_exp=16):
    """Layer 1: flat (every expert equal -> pruning costs mass linearly).
       Layer 2: heavy-tailed (4 experts carry nearly everything -> pruning 12 is nearly free).
    A good search moves budget OFF layer 1 and ONTO layer 2."""
    flat = torch.ones(2, n_exp, dtype=torch.float64) * 10.0
    heavy = torch.zeros(2, n_exp, dtype=torch.float64)
    heavy[:, :4] = 100.0
    heavy[:, 4:] = 0.5
    cnt = torch.full((2, n_exp), 10.0, dtype=torch.float64)
    mk = lambda m: {"sum": m, "sq": m * m, "cnt": cnt, "nrm": m, "nsq": m * m,
                    "gat": cnt, "gsq": cnt}
    acc = {"model.layers.1.mlp": mk(flat), "model.layers.2.mlp": mk(heavy)}
    fs = torch.zeros(3, n_exp, n_exp, dtype=torch.float64)
    fc = torch.ones(3, n_exp, n_exp, dtype=torch.float64)
    for li, m in ((1, flat), (2, heavy)):
        d = m.sum(0) / cnt.sum(0)
        fs[li] = torch.diag(d * d)
    torch.save({"acc": acc, "f_sum": fs, "f_cnt": fc, "buckets": ["x", "y"]},
               td / "accumulators.pt")


with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    make(td)

    print("[1] the total pruned count is EXACTLY preserved")
    r = LB.run(td / "accumulators.pt", td / "b.json", 0.5, "total_0_1_1", generations=150)
    check("searched budget prunes exactly the required total",
          r["total_pruned"] == r["total_required"],
          f"{r['total_pruned']} vs {r['total_required']}")
    check("budget is 50% of all experts", r["total_required"] == 16, str(r["total_required"]))

    print("\n[2] the search beats uniform where the answer is known")
    print(f"    layer budgets: {r['budget']}")
    check("search improves worst-domain retention over uniform", r["gain"] >= 0,
          f"uniform {r['uniform_worst']:.4f} -> searched {r['searched_worst']:.4f}")
    heavy_budget = r["budget"]["model.layers.2.mlp"]
    flat_budget = r["budget"]["model.layers.1.mlp"]
    check("more pruning is allocated to the REDUNDANT layer than the flat one",
          heavy_budget > flat_budget, f"heavy={heavy_budget} flat={flat_budget}")

    print("\n[3] per-layer bounds are respected")
    check("no layer below the floor or above the ceiling",
          all(int(16 * 0.20) <= v <= int(16 * 0.75) for v in r["budget"].values()),
          str(r["budget"]))

    print("\n[4] evaluation stays per domain")
    check("per-domain retention reported for both buckets",
          set(r["searched_by_domain"]) == {"x", "y"}, str(r["searched_by_domain"]))

    print("\n[5] a budget that cheats the total is rejected by construction")
    tab, _ = LB._layer_tables(td / "accumulators.pt", "total_0_1_1")
    cheat = np.array([2, 2])          # prunes far fewer than required
    f_cheat, _ = LB.fitness(tab, cheat)
    f_true, _ = LB.fitness(tab, np.array([r["budget"]["model.layers.1.mlp"],
                                          r["budget"]["model.layers.2.mlp"]]))
    check("under-pruning would score better, which is why the SUM is a hard constraint",
          f_cheat > f_true,
          f"cheat {f_cheat:.4f} > honest {f_true:.4f} -- the search may never consider it")

print("\n" + ("GATE FAIL: " + ", ".join(FAIL) if FAIL else
              "GATE PASS: budget search respects the size constraint and beats uniform"))
sys.exit(1 if FAIL else 0)
