"""Gate for hope_fmatrix.py. CPU-only, seconds, no checkpoint -- runs before the pass.

The pass it guards is the expensive one and its F-matrix is unrecoverable afterwards, so every
property HOPE depends on is asserted here rather than inspected later.
"""
import itertools
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts")
from hope_fmatrix import FAccumulator, solve_prune_set

rng = np.random.default_rng(0)
fail = 0


def check(name, ok, extra=""):
    global fail
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{('  ' + extra) if extra else ''}")
    fail += (not ok)


# ---- 1. accumulator vs brute force -------------------------------------------------------
E, K, N = 7, 3, 400
idx = torch.from_numpy(np.stack([rng.choice(E, K, replace=False) for _ in range(N)]))
s = torch.from_numpy(rng.random((N, K)) + 0.1)
acc = FAccumulator(1, E)
acc.update(0, idx, s)
F = acc.finalize(0)

bs = np.zeros((E, E)); bc = np.zeros((E, E))
for t in range(N):
    for a in range(K):
        for b in range(K):
            bs[idx[t, a], idx[t, b]] += s[t, a].item() * s[t, b].item()
            bc[idx[t, a], idx[t, b]] += 1
bf = bs / np.maximum(bc, 1)
check("F matches brute force", np.allclose(F, bf), f"max|d| {np.abs(F-bf).max():.2e}")
check("F is symmetric", np.allclose(F, F.T))

# Diagonal must be the SECOND MOMENT over the expert's own tokens, which is what the paper
# gives as F_kk = E[(g||f||)^2] and what distinguishes it from the squared REAP score.
d_ref = np.zeros(E)
for e in range(E):
    vals = [s[t, a].item() for t in range(N) for a in range(K) if idx[t, a] == e]
    d_ref[e] = np.mean([v * v for v in vals]) if vals else 0.0
check("diagonal is E[(g*||f||)^2]", np.allclose(np.diag(F), d_ref),
      f"max|d| {np.abs(np.diag(F)-d_ref).max():.2e}")

# ---- 2. the paper's own control: zero off-diagonals -> REAP -------------------------------
D = np.diag(np.diag(F))
got = solve_prune_set(D, 3)
want = np.sort(np.argsort(np.diag(F))[:3])
check("off-diagonal zeroed reproduces the REAP prune-set", np.array_equal(got, want),
      f"{got.tolist()} vs {want.tolist()}")

# ---- 3. known answer where interactions MUST change the decision --------------------------
# Four experts. Diagonals say prune {0,1} (smallest). But 0 and 1 reinforce each other
# strongly, so removing both together is the expensive choice; the true optimum is {0,2}.
G = np.array([[1.0, 9.0, 0.0, 0.0],
              [9.0, 1.1, 0.0, 0.0],
              [0.0, 0.0, 1.2, 0.0],
              [0.0, 0.0, 0.0, 5.0]])
best = min(itertools.combinations(range(4), 2),
           key=lambda c: sum(G[i, j] for i in c for j in c))
got = solve_prune_set(G, 2)
reap = np.sort(np.argsort(np.diag(G))[:2])
check("interaction term changes the decision", not np.array_equal(got, reap),
      f"HOPE {got.tolist()} vs REAP {reap.tolist()}")
check("matches brute-force optimum", np.array_equal(got, np.sort(np.array(best))),
      f"{got.tolist()} vs {sorted(best)}")

# ---- 4. exhaustive optimality on random small problems ------------------------------------
wins = 0
for trial in range(40):
    E2, kp = 8, 3
    A = rng.random((E2, E2)); M = (A + A.T) / 2
    opt = min(itertools.combinations(range(E2), kp),
              key=lambda c: sum(M[i, j] for i in c for j in c))
    ov = sum(M[i, j] for i in opt for j in opt)
    g = solve_prune_set(M, kp)
    gv = sum(M[i, j] for i in g for j in g)
    wins += gv <= ov + 1e-9
check("relaxation reaches the exact optimum on random 8x8", wins >= 36, f"{wins}/40")

# ---- 5. shapes and guards ------------------------------------------------------------------
check("protect is never pruned", 3 not in solve_prune_set(G, 2, protect=[3]).tolist())
big = FAccumulator(47, 256)
mb = (big.sum.numel() + big.cnt.numel()) * 8 / 2**20
check("MiMo-scale accumulator is small", mb < 128, f"{mb:.1f} MiB for 47 layers x 256 experts")

print("\nGATE " + ("PASS" if not fail else f"FAIL ({fail})"))
sys.exit(1 if fail else 0)
