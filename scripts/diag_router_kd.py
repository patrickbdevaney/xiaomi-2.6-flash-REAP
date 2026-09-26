"""Why router KD rejected 47/47 layers: is it the token budget, or the step size?

The shipped run fed 2,048 rows and Adam(lr=1e-3) for 300 steps, and every layer came back
worse than the teacher slice. Two candidate explanations, and they call for opposite fixes:

  (a) sample starvation -- 2,048 rows to fit [128, 4096]. Fix: more tokens.
  (b) the step is too large for the parameter. Measured on the shipped checkpoint, the router
      weight has RMS 0.032. Adam's update is scale-free -- roughly `lr` per entry per step
      regardless of gradient magnitude -- so 300 steps at 1e-3 walks each entry up to 0.3,
      about 10x the entire weight. Fix: a smaller step, not more data.

This separates them on a fixture with the checkpoint's real scales, where the answer is known
because the fixture is built with a recoverable optimum. Run on CPU on purpose: the GPU belongs
to the GLM saliency pass.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_kd as RK

# Shapes scaled down from MiMo (256 experts, top-8, H=4096) but with the MEASURED weight and
# bias scales kept exactly, because those are what set the step-size argument.
E, KEEP, H, TOPK, C = 256, 128, 512, 8, 32
N_EXTRA = C - TOPK                       # kept candidates the student chooses among
W_RMS = 0.032202
# MEASURED on the shipped checkpoint, and the number that matters is the SPREAD, not the RMS.
# Layer 7: bias mean +1.3831, std 0.0588. Layer 20: mean +10.4696, std 0.0173. A constant added
# to every expert's score cannot change a top-k, so the bias is very nearly inert and SELECTION
# IS DRIVEN BY THE TRAINABLE WEIGHT. An earlier fixture drew the bias as N(0, 2.04) from the RMS
# alone; that made one fixed expert set win for every token, no kept expert was ever displaced,
# and the affected rate came out 0.00% for reasons that were entirely an artifact of the fixture.
B_MEAN, B_STD = 1.3831, 0.0588


def fixture(n_tokens, seed=0, aligned=False):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(n_tokens, H, generator=g)
    w = torch.randn(E, H, generator=g)
    w *= W_RMS / w.pow(2).mean().sqrt()
    b = B_MEAN + torch.randn(E, generator=g) * B_STD
    # Scale hidden so the gate pre-activation has unit spread: a saturated sigmoid would make
    # the gradient vanish for reasons that have nothing to do with the question being asked.
    hidden = hidden / (hidden @ w.T).std()

    if aligned:
        # What REAP actually produces. The mask keeps the experts carrying the most gated
        # output mass -- which are, by construction, the ones the router selects most often.
        # A random keep-set is a different problem entirely, and assuming they behave the
        # same is how a stage gets declared broken when it is merely out of work.
        t0, _ = RK.gate_forward(hidden, w, b, TOPK)
        freq = torch.bincount(t0.flatten(), minlength=E).float()
        keep = torch.topk(freq, KEEP).indices.sort().values
    else:
        keep = torch.randperm(E, generator=g)[:KEEP].sort().values
    t_idx, t_w = RK.gate_forward(hidden, w, b, TOPK)              # teacher: top-8 of all E

    # Candidate set: the teacher's top-8 (slots 0..7) then the best-scoring KEPT experts
    # (slots 8..31). Making the kept candidates exactly the extras keeps every tensor
    # rectangular; it only ever understates what the student may choose from, so `boundary`
    # still bounds the approximation honestly.
    sc = torch.sigmoid(hidden @ w.T) + b
    extra_local = torch.topk(sc[:, keep], N_EXTRA, dim=-1).indices            # into keep
    cand_id = torch.cat([t_idx, keep[extra_local]], dim=1)                    # [N, C] global ids

    # Per-expert ELEMENTWISE map. A full [E,H,H] of linear maps is [N,C,H,H] once gathered --
    # 275 GB at these shapes, which is how the first version of this fixture nearly took the
    # box down. Elementwise is enough: distinct experts still produce distinct outputs, so
    # re-weighting them is still a real optimisation problem.
    gg = torch.Generator().manual_seed(seed + 1)
    e_scale = torch.randn(E, H, generator=gg) * 0.5 + 1.0
    e_shift = torch.randn(E, H, generator=gg) * 0.1
    cand_out = hidden.unsqueeze(1) * e_scale[cand_id] + e_shift[cand_id]      # [N, C, H]

    t_slot = torch.arange(TOPK).expand(n_tokens, TOPK).contiguous()           # teacher = slots 0..7
    kept_slot = torch.arange(TOPK, C).expand(n_tokens, N_EXTRA).contiguous()
    kept_local = extra_local                                                  # already into keep
    return dict(cand_out=cand_out, t_slot=t_slot, t_w=t_w, hidden=hidden, router_w=w,
                router_b=b, keep=keep, kept_slot=kept_slot, kept_local=kept_local, top_k=TOPK)


def trial(fx, steps, lr, batch, label):
    r = RK.fit_layer(fx["cand_out"], fx["t_slot"], fx["t_w"], fx["hidden"], fx["router_w"],
                     fx["router_b"], fx["keep"], fx["kept_slot"], fx["kept_local"],
                     fx["top_k"], steps=steps, lr=lr, batch=batch)
    drift = (r["w"] - fx["router_w"][fx["keep"]].float()).abs().max()
    print(f"  {label:<34} base {r['baseline']:.4e} -> {r['last']:.4e}  "
          f"{r['improvement']:+8.2%}  |Δw|max {drift:.4f}  (w rms {W_RMS:.4f})")
    return r


def main():
    print(f"fixture: E={E} keep={KEEP} H={H} top_k={TOPK} C={C}, "
          f"router rms {W_RMS} / bias {B_MEAN}+-{B_STD} from the shipped checkpoint\n")

    print("[A] the shipped configuration, at the shipped token budget")
    fx = fixture(2048)
    trial(fx, 300, 1e-3, 512, "steps=300 lr=1e-3 n=2048")

    print("\n[B] hypothesis (a): is it sample starvation? hold the step, raise the tokens")
    for n in (4096, 8192):
        trial(fixture(n), 300, 1e-3, 512, f"steps=300 lr=1e-3 n={n}")

    print("\n[C] hypothesis (b): is it the step size? hold n=2048, lower the step")
    for lr in (1e-4, 1e-5, 1e-6):
        trial(fx, 300, lr, 512, f"steps=300 lr={lr:g} n=2048")

    print("\n[D] the keep-set REAP actually produces: mass-aligned, not random")
    fa = fixture(2048, aligned=True)
    t_kept = (fa["t_slot"] < TOPK).float().mean()
    import torch as T
    tid, _ = RK.gate_forward(fa["hidden"], fa["router_w"], fa["router_b"], TOPK)
    in_keep = (tid.unsqueeze(-1) == fa["keep"].view(1, 1, -1)).any(-1).float().mean()
    print(f"  teacher top-8 slots that survive the mask: {in_keep:.1%}")
    trial(fa, 300, 1e-3, 512, "ALIGNED steps=300 lr=1e-3 n=2048")
    trial(fa, 300, 1e-4, 512, "ALIGNED steps=300 lr=1e-4 n=2048")
    trial(fixture(8192, aligned=True), 300, 1e-3, 512, "ALIGNED steps=300 lr=1e-3 n=8192")


if __name__ == "__main__":
    main()
