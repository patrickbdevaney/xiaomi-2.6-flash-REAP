"""Gate router repair on a problem whose answer is known by construction.

The trap this is written against: if the objective were routing-distribution matching, the
sliced router already achieves zero loss and a training loop would silently do nothing while
printing a falling curve on noise. So the first check is that the UNTRAINED sliced router has a
NON-ZERO output loss -- if it did not, there would be nothing to repair and this whole module
would be theatre.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import router_kd as RK      # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(name)


torch.manual_seed(0)
N, H, E, K = 512, 32, 16, 4          # tokens, hidden, experts, top-k
hidden = torch.randn(N, H)
router_w = torch.randn(E, H) * 0.5
router_b = torch.zeros(E)
# Each expert has a distinct, structured output so substituting one for another is detectable.
expert_out = torch.randn(N, E, H)
keep = torch.tensor([i for i in range(E) if i % 2 == 0])     # prune the odd experts

print("[1] there is genuinely something to repair")
base = RK.baseline_loss(hidden, router_w, router_b, expert_out, keep, K)
check("the untrained sliced router has non-zero output loss", base > 1e-6,
      f"baseline MSE {base:.6e} -- zero here would mean the objective is vacuous")

print("\n[2] routing-distribution matching would have been vacuous (the trap)")
with torch.no_grad():
    tl = hidden @ router_w.T + router_b
    sl = hidden @ router_w[keep].T + router_b[keep]
check("student logits equal the teacher's on kept experts",
      torch.allclose(tl[:, keep], sl, atol=1e-6),
      "so a KL on routing distributions is identically zero and trains nothing")

print("\n[3] training reduces the output loss")
w, b, first, last = RK.fit_layer(hidden, router_w, router_b, expert_out, keep, K, steps=250)
print(f"    loss {first:.6e} -> {last:.6e}")
check("loss decreases", last < first, f"{first:.6e} -> {last:.6e}")
check("loss improves on the untrained slice", last < base,
      f"trained {last:.6e} vs sliced {base:.6e}")
check("shapes are the student's, not the teacher's",
      tuple(w.shape) == (len(keep), H) and tuple(b.shape) == (len(keep),),
      f"{tuple(w.shape)}, {tuple(b.shape)}")
check("trained weights actually moved", not torch.allclose(w, router_w[keep], atol=1e-6))

print("\n[4] a control whose sign is known: no pruning means nothing to learn")
keep_all = torch.arange(E)
base_all = RK.baseline_loss(hidden, router_w, router_b, expert_out, keep_all, K)
check("with NO experts pruned the sliced router is already exact", base_all < 1e-10,
      f"{base_all:.3e} -- a non-zero value here would mean the teacher path is wrong")

print("\n[5] pruning more hurts more (monotonicity)")
losses = []
for n_keep in (14, 10, 6):
    kp = torch.arange(E)[:n_keep]
    losses.append(RK.baseline_loss(hidden, router_w, router_b, expert_out, kp, K))
check("baseline loss grows as more experts are removed",
      losses[0] <= losses[1] <= losses[2],
      " <= ".join(f"{v:.4e}" for v in losses))

print("\n" + ("GATE FAIL: " + ", ".join(FAIL) if FAIL else
              "GATE PASS: router repair targets the real damage and improves on the slice"))
sys.exit(1 if FAIL else 0)
