"""Gate the mask selection against cases whose answers are known in advance.

Selection is the step where a silent bug is most expensive: a wrong mask still has the right
shape, the right count, and plausible-looking retention, and nothing downstream would catch it.
So every check here is constructed so the correct answer is known independently of the code.
"""
from __future__ import annotations
import json, sys, tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import reap_select as RS          # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(name)


def synth(n_layer=3, n_bucket=4, n_exp=16, seed=0):
    """Accumulators whose per-expert ordering we control exactly."""
    g = torch.Generator().manual_seed(seed)
    # Layer 0 of MiMo is DENSE, so MoE layer indices start at 1 and F must be sized for the
    # highest index, not the count -- the same off-by-one the real pass would hit.
    acc = {}
    f_sum = torch.zeros(n_layer + 1, n_exp, n_exp, dtype=torch.float64)
    f_cnt = torch.zeros(n_layer + 1, n_exp, n_exp, dtype=torch.float64)
    for li in range(n_layer):
        cnt = torch.randint(50, 500, (n_bucket, n_exp), generator=g).double()
        s = torch.rand(n_bucket, n_exp, generator=g).double() * cnt
        a = {"sum": s, "sq": s * s / cnt.clamp(min=1), "cnt": cnt,
             "nrm": s, "nsq": s * s, "gat": cnt, "gsq": cnt}
        acc[f"model.layers.{li+1}.mlp"] = a
        d = (s.sum(0) / cnt.sum(0))
        f_sum[li + 1] = torch.outer(d, d) * 100
        f_cnt[li + 1] = torch.full((n_exp, n_exp), 100.0, dtype=torch.float64)
    return acc, f_sum, f_cnt, [f"b{i}" for i in range(n_bucket)]


with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    acc, f_sum, f_cnt, buckets = synth()
    torch.save({"acc": acc, "f_sum": f_sum, "f_cnt": f_cnt, "buckets": buckets},
               td / "accumulators.pt")

    print("[1] the mask prunes exactly the requested count")
    r = RS.run(td / "accumulators.pt", td / "m.json", 0.5, "reap", "reap_1_1_1")
    m = json.loads((td / "m.json").read_text())["mask"]
    check("50% of 16 experts pruned in every layer",
          all(v == 8 for v in r["pruned_per_layer"].values()), str(r["pruned_per_layer"]))
    check("no duplicate indices", all(len(set(v)) == len(v) for v in m.values()))
    check("indices in range", all(all(0 <= i < 16 for i in v) for v in m.values()))

    print("\n[2] REAP keeps the HIGHEST-scoring experts (known answer)")
    a0 = acc["model.layers.1.mlp"]
    s = RS.scores(a0, "reap_1_1_1")
    want_pruned = set(np.argsort(s)[:8].tolist())
    got = set(m["model.layers.1.mlp"])
    check("pruned set == the 8 lowest scores", got == want_pruned,
          f"{sorted(got)} vs {sorted(want_pruned)}")

    print("\n[3] the criteria genuinely differ")
    s_reap = RS.scores(a0, "reap_1_1_1")
    s_tot = RS.scores(a0, "total_0_1_1")
    o1, o2 = list(np.argsort(-s_reap)), list(np.argsort(-s_tot))
    check("(1,1,1) and (0,1,1) rank experts differently", o1 != o2,
          "identical rankings would mean the normalisation is not applied")

    print("\n[4] HOPE with zeroed off-diagonal reproduces REAP (the published control)")
    f_diag = f_sum.clone()
    for li in range(f_diag.shape[0]):
        d = torch.diagonal(f_diag[li]).clone()
        f_diag[li] = torch.diag(d)
    torch.save({"acc": acc, "f_sum": f_diag, "f_cnt": f_cnt, "buckets": buckets},
               td / "acc_diag.pt")
    rh = RS.run(td / "acc_diag.pt", td / "mh.json", 0.5, "hope", "reap_1_1_1")
    mh = json.loads((td / "mh.json").read_text())["mask"]
    # With interactions zeroed, minimising p^T F p == taking the smallest diagonal entries.
    a1 = acc["model.layers.1.mlp"]
    d = (f_diag[1].diagonal()).numpy()
    expect = set(np.argsort(d)[:8].tolist())
    check("HOPE on a diagonal F picks the 8 smallest diagonal entries",
          set(mh["model.layers.1.mlp"]) == expect,
          f"{sorted(mh['model.layers.1.mlp'])} vs {sorted(expect)}")

    print("\n[5] retention is computed against real mass, and is per domain")
    mass = RS.domain_mass(a0)
    pruned = np.array(sorted(m["model.layers.1.mlp"]))
    ret = RS.retention(mass, pruned)
    manual = np.array([mass[b][[i for i in range(16) if i not in set(pruned)]].sum()
                       / mass[b].sum() for b in range(mass.shape[0])])
    check("retention matches a hand computation", np.allclose(ret, manual),
          f"max diff {np.abs(ret-manual).max():.2e}")
    check("retention is reported per bucket, not averaged", len(ret) == 4, f"{len(ret)} values")
    # A COMPARISON WHOSE SIGN IS KNOWN IN ADVANCE. Retention is measured in TOTAL gated mass,
    # so the criterion that ranks by total mass -- (0,1,1) -- must retain at least as much of it
    # as the one that ranks by the MEAN, (1,1,1). An expert with a high mean but few routed
    # tokens carries little mass, so REAP's mean criterion can and does discard mass; on this
    # random fixture it drops to 0.475, which is not a bug but the exact effect that makes the
    # criterion choice worth measuring on the real model.
    RS.run(td / "accumulators.pt", td / "m_tot.json", 0.5, "reap", "total_0_1_1")
    m_tot = json.loads((td / "m_tot.json").read_text())["mask"]
    ret_tot = RS.retention(mass, np.array(sorted(m_tot["model.layers.1.mlp"])))
    check("ranking by TOTAL mass retains more mass than ranking by the MEAN",
          ret_tot.sum() >= ret.sum() - 1e-9,
          f"total-criterion {ret_tot.mean():.3f} vs mean-criterion {ret.mean():.3f}")
    # On the UNIFORM fixture, retention near 0.5 is the correct answer, not a failure: if every
    # expert carries the same mass, dropping half the experts drops half the mass by
    # construction. The property worth asserting is that the selector keeps the heavy hitters
    # when there are any -- so assert it on a fixture that HAS a heavy tail, which is what real
    # MoE routing looks like.
    n_exp = 16
    heavy = torch.zeros(2, n_exp, dtype=torch.float64)
    heavy[0] = torch.tensor([100.0] * 4 + [1.0] * 12, dtype=torch.float64)
    heavy[1] = torch.tensor([1.0] * 12 + [100.0] * 4, dtype=torch.float64)   # different domain!
    cnt_h = torch.full((2, n_exp), 10.0, dtype=torch.float64)
    acc_h = {"model.layers.1.mlp": {"sum": heavy, "sq": heavy * heavy, "cnt": cnt_h,
                                    "nrm": heavy, "nsq": heavy * heavy,
                                    "gat": cnt_h, "gsq": cnt_h}}
    fs = torch.zeros(2, n_exp, n_exp, dtype=torch.float64)
    fc = torch.ones(2, n_exp, n_exp, dtype=torch.float64)
    d = heavy.sum(0) / cnt_h.sum(0)
    fs[1] = torch.diag(d * d)
    torch.save({"acc": acc_h, "f_sum": fs, "f_cnt": fc, "buckets": ["x", "y"]},
               td / "acc_heavy.pt")
    rh2 = RS.run(td / "acc_heavy.pt", td / "mheavy.json", 0.5, "reap", "total_0_1_1")
    check("with a heavy tail, the kept half carries most of the mass",
          rh2["worst_retention"] > 0.85, f"worst {rh2['worst_retention']:.3f}")
    check("both domains keep their own heavy experts (a global ranking could drop one)",
          rh2["retention_by_domain"]["x"] > 0.85 and rh2["retention_by_domain"]["y"] > 0.85,
          str({k: round(v, 3) for k, v in rh2["retention_by_domain"].items()}))

    print("\n[6] protect_frac actually protects each domain's top experts")
    rp = RS.run(td / "accumulators.pt", td / "mp.json", 0.5, "reap", "reap_1_1_1",
                protect_frac=0.10)
    mp = json.loads((td / "mp.json").read_text())["mask"]
    top_each = {int(np.argmax(mass[b])) for b in range(mass.shape[0])}
    overlap = top_each & set(mp["model.layers.1.mlp"])
    check("no domain's top expert is pruned when protected", not overlap, f"pruned {overlap}")
    check("protected run still prunes the full count",
          all(v == 8 for v in rp["pruned_per_layer"].values()), str(rp["pruned_per_layer"]))

print("\n" + ("GATE FAIL: " + ", ".join(FAIL) if FAIL else
              "GATE PASS: selection is correct on cases with known answers"))
sys.exit(1 if FAIL else 0)
