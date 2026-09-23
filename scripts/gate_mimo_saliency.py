"""Gate for mimo_saliency.py. CPU, seconds, no checkpoint. Runs BEFORE the expensive pass.

The two properties that matter most:
  * the patched forward is arithmetically IDENTICAL to the upstream one (we are instrumenting a
    model, not changing it);
  * F's diagonal agrees with the independently-accumulated `sq`/`cnt`. Those two quantities are
    computed by different code paths over the same tokens, so agreement is real evidence that
    the per-token slot bookkeeping behind the F outer product is right. If the slot indices were
    wrong, the off-diagonal would be silently garbage and nothing downstream would notice.
"""
import sys, types

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "scripts")
import mimo_saliency as MS

torch.manual_seed(0)
fail = 0


def check(name, ok, extra=""):
    global fail
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{('  ' + extra) if extra else ''}")
    fail += (not ok)


H, E, K, N = 16, 9, 3, 64


class Expert(nn.Module):
    def __init__(self):
        super().__init__(); self.l = nn.Linear(H, H, bias=False)
    def forward(self, x): return self.l(x)


class MoE(nn.Module):
    """Upstream MiMoV2MoE.moe, copied verbatim as the reference."""
    def __init__(self):
        super().__init__(); self.experts = nn.ModuleList([Expert() for _ in range(E)])
    def moe(self, hidden_states, topk_indices, topk_weights):
        final = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        m = torch.nn.functional.one_hot(topk_indices, num_classes=len(self.experts)).permute(2, 0, 1)
        for i, ex in enumerate(self.experts):
            ti, wi = torch.where(m[i])
            if ti.numel() > 0:
                final.index_add_(0, ti, ex(hidden_states[ti]) * topk_weights[ti, wi].unsqueeze(-1))
        return final.type(hidden_states.dtype)


mod = types.ModuleType("fake"); mod.MiMoV2MoE = MoE
ref = MoE()
x = torch.randn(N, H)
idx = torch.stack([torch.randperm(E)[:K] for _ in range(N)])
w = torch.rand(N, K) + 0.1
w = w / w.sum(-1, keepdim=True)

want = ref.moe(x, idx, w)                       # before patching

MS.configure(["a", "b"], n_layers=1, n_experts=E)
MS.LAYER_INDEX["L0"] = 0
MS.patch(mod)
MS.CTX.update({"layer": "L0", "bucket": 0, "valid": None})
got = ref.moe(x, idx, w)                        # after patching, same instance

check("patched forward is bit-identical to upstream", torch.equal(want, got),
      f"max|d| {(want-got).abs().max().item():.2e}")

acc = MS.ACC["L0"]
F = MS.FACC.finalize(0)

# brute force the per-expert stats
bs = np.zeros(E); bc = np.zeros(E); bq = np.zeros(E)
for t in range(N):
    for a in range(K):
        e = idx[t, a].item()
        f = ref.experts[e](x[t:t+1])
        s = w[t, a].item() * float(f.float().norm())
        bs[e] += s; bq[e] += s * s; bc[e] += 1
check("cnt matches brute force", np.allclose(acc["cnt"][0].numpy(), bc))
check("sum matches brute force", np.allclose(acc["sum"][0].numpy(), bs, rtol=1e-9),
      f"max|d| {np.abs(acc['sum'][0].numpy()-bs).max():.2e}")
check("sq matches brute force", np.allclose(acc["sq"][0].numpy(), bq, rtol=1e-9))

# THE CROSS-CHECK: F_kk must equal sq/cnt, computed by a different path over the same tokens.
diag_from_acc = acc["sq"][0].numpy() / np.maximum(acc["cnt"][0].numpy(), 1)
check("F diagonal == sq/cnt (independent paths agree)",
      np.allclose(np.diag(F), diag_from_acc, rtol=1e-9),
      f"max|d| {np.abs(np.diag(F)-diag_from_acc).max():.2e}")
check("F is symmetric", np.allclose(F, F.T))

# every criterion in the unified family is derivable
for nm, v in (("Frequency(0,0,0)", acc["cnt"]), ("SEER(0,1,0)", acc["gat"]),
              ("EAN(0,0,1)", acc["nrm"]), ("REAP(1,1,1)", acc["sum"] / acc["cnt"].clamp(min=1)),
              ("MAN(1,0,1)", acc["nrm"] / acc["cnt"].clamp(min=1)),
              ("MSAN(1,0,2)", acc["nsq"] / acc["cnt"].clamp(min=1)),
              ("(0,1,1)", acc["sum"]), ("(0,2,2)", acc["sq"])):
    if not torch.isfinite(v).all():
        check(f"{nm} derivable", False); break
else:
    check("all 8 unified-family criteria derivable from the accumulators", True)

# valid-mask path: masked rows must not be counted anywhere, F included
MS.ACC.clear(); MS.configure(["a"], 1, E); MS.LAYER_INDEX["L0"] = 0
keep = torch.zeros(N, dtype=torch.bool); keep[: N // 2] = True
MS.CTX.update({"layer": "L0", "bucket": 0, "valid": keep})
ref.moe(x, idx, w)
a2 = MS.ACC["L0"]; F2 = MS.FACC.finalize(0)
check("valid mask drops rows from cnt", a2["cnt"][0].sum().item() == (N // 2) * K,
      f"{a2['cnt'][0].sum().item():.0f} vs {(N//2)*K}")
d2 = a2["sq"][0].numpy() / np.maximum(a2["cnt"][0].numpy(), 1)
check("valid mask keeps F and acc consistent", np.allclose(np.diag(F2), d2, rtol=1e-9),
      f"max|d| {np.abs(np.diag(F2)-d2).max():.2e}")

# a misaligned mask must raise, not guess
MS.CTX.update({"valid": torch.ones(N + 1, dtype=torch.bool)})
try:
    ref.moe(x, idx, w); check("misaligned mask raises", False)
except RuntimeError as e:
    check("misaligned mask raises", "not 1:1" in str(e))

print("\nGATE " + ("PASS" if not fail else f"FAIL ({fail})"))
sys.exit(1 if fail else 0)
