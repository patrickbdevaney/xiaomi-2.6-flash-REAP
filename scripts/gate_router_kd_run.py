"""Gate the driver's three pure pieces on CPU, while the GPU is busy with the calibration pass.

Each check is a statement whose answer is known before it runs:
  - the reservoir is UNIFORM (a biased one silently trains on one domain);
  - the teacher occupies candidate slots 0..k-1 BY CONSTRUCTION, so `t_slot` need not be searched
    for -- if that ever stops holding, the teacher mixture silently reads the wrong experts;
  - an expert appearing at two slots for one token gives the SAME value at both.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import router_kd_run as D    # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(name)


def main():
    torch.manual_seed(0)
    # ---- 1. reservoir uniformity ----
    # The MAX deviation over M bins is a bad statistic here: at B/M = 0.04 and 400 trials each
    # bin is a count of ~16 with sigma ~3.9, so a max |deviation| near 80% is ordinary noise, and
    # a first version of this gate failed the correct code for exactly that reason. What has
    # power is the EARLY-vs-LATE mean, because every way this can break -- keeping the first N,
    # or the wrong acceptance rate -- tilts the stream position. It is compared against a
    # deliberately broken sampler whose sign is known in advance.
    B, M, TRIALS = 16, 400, 400

    def sample(fn):
        counts = torch.zeros(M)
        for t in range(TRIALS):
            D.CAP.update({"hid": torch.zeros(B, 1), "budget": B, "seen": 0,
                          "gen": torch.Generator().manual_seed(t)})
            for s0 in range(0, M, 37):                   # ragged batches, as real ones are
                fn(torch.arange(s0, min(s0 + 37, M)).float().unsqueeze(-1))
            counts += torch.bincount(D.CAP["hid"].squeeze(-1).long(), minlength=M).float()
        return counts / TRIALS

    def first_n_wins(h):                                 # the bug this must be able to see
        n, buf = D.CAP["seen"], D.CAP["hid"]
        take = max(0, min(B - n, h.shape[0]))
        if take:
            buf[n:n + take] = h[:take]
        D.CAP["seen"] = n + h.shape[0]

    freq, bad = sample(D._reservoir_fast), sample(first_n_wins)
    half = M // 2
    tilt = float((freq[:half].mean() - freq[half:].mean()) / (B / M))
    tilt_bad = float((bad[:half].mean() - bad[half:].mean()) / (B / M))
    se = float((freq.std() / (half ** 0.5)) / (B / M)) * 2      # 2 s.e. on the difference
    check("reservoir does not favour early tokens", abs(tilt) < max(0.10, se),
          f"early-late tilt {tilt:+.1%} (2 s.e. = {se:.1%}); the first-N sampler reads "
          f"{tilt_bad:+.0%}")
    check("the uniformity test can actually see a biased sampler", tilt_bad > 0.5,
          f"first-N tilt {tilt_bad:+.0%}")
    check("every position is reachable", float((freq == 0).float().mean()) == 0.0,
          f"{int((freq == 0).sum())} of {M} positions never sampled")

    # ---- 2. candidate construction ----
    N, E, H, K, CK = 64, 64, 32, 8, 12
    hid = torch.randn(N, H)
    W, b = torch.randn(E, H) * 0.1, torch.randn(E) * 0.5
    keep = torch.sort(torch.randperm(E)[:E // 2]).values
    cand, t_slot, t_w, kept_slot, kept_local = D.candidates(hid, W, b, keep, K, CK)
    import router_kd as RK
    t_idx_ref, t_w_ref = RK.gate_forward(hid, W, b, K)
    check("teacher occupies candidate slots 0..k-1",
          torch.equal(t_slot, torch.arange(K).expand(N, K)) and
          torch.equal(cand[:, :K], t_idx_ref), "cand[:, :k] is the teacher's own selection")
    check("teacher weights match the gate", torch.allclose(t_w, t_w_ref, atol=1e-6))
    check("kept candidate block is exactly CK wide", kept_local.shape == (N, CK),
          f"{tuple(kept_local.shape)}")
    check("every kept candidate is actually a kept expert",
          bool(torch.isin(cand.gather(1, kept_slot), keep).all()))
    check("kept_local indexes the student's rows correctly",
          torch.equal(keep[kept_local], cand.gather(1, kept_slot)))

    # ---- 3. expert outputs, including duplicates ----
    class Lin(torch.nn.Module):
        def __init__(self, e):
            super().__init__()
            self.w = torch.randn(H, H) * 0.1 + e
        def forward(self, x):
            return x @ self.w
    experts = torch.nn.ModuleList([Lin(e) for e in range(E)])
    out = D.expert_outputs(experts, hid, cand, torch.float32)
    ref = torch.stack([torch.stack([experts[int(cand[n, c])](hid[n]) for c in range(cand.shape[1])])
                       for n in range(N)])
    check("expert outputs match a direct per-(token,expert) computation",
          torch.allclose(out, ref, atol=1e-4), f"max|diff| {(out-ref).abs().max():.2e}")
    dup = (cand[:, :K].unsqueeze(-1) == cand[:, K:].unsqueeze(1))
    ns, ks, cs = dup.nonzero(as_tuple=True)
    check("a duplicated expert gives the same value at both slots", len(ns) > 0 and
          torch.allclose(out[ns, ks], out[ns, K + cs], atol=1e-6),
          f"{len(ns)} duplicate slot pairs exercised")

    print(("GATE FAIL: " + ", ".join(FAIL)) if FAIL else "GATE PASS")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
