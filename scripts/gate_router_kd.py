"""Gate router repair against the checkpoint's OWN gate module, on REAL router weights.

Three traps this is written against.

1. THE WRONG ROUTER. MiMo's noaux_tc gate uses the biased score to SELECT and the unbiased score
   to WEIGHT. An earlier version of router_kd folded the bias into the logits; measured here, it
   agrees with the real module on 1 expert of 8. So check 1 is `gate_forward` against the
   checkpoint's own `MiMoV2MoEGate`, with the checkpoint's own weights -- not a random fixture,
   because a random bias near zero would let the wrong formula pass.

2. NOTHING TO REPAIR. If the objective were routing-distribution matching, the sliced router
   already scores zero loss and a training loop would print a falling curve while learning
   nothing. So the untrained sliced router must have a NON-ZERO output loss once experts are
   actually removed -- and, the control whose sign is known in advance, EXACTLY ZERO when nothing
   is removed.

3. A CANDIDATE SET TOO TIGHT TO TRAIN IN. The student picks from a precomputed candidate set; if
   it is clipped by that boundary the loss curve still falls and the result is still wrong. The
   boundary rate is asserted small.
"""
from __future__ import annotations
import importlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import router_kd as RK      # noqa: E402
from mimo_shards import ShardReader  # noqa: E402

SRC = Path("/home/patrickd/models/MiMo-V2.6-Flash-RL")
FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(name)


def real_gate_module(cfg):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cls = get_class_from_dynamic_module("modeling_mimo_v2.MiMoV2MoEGate", str(SRC))
    return importlib.import_module(cls.__module__).MiMoV2MoEGate(cfg).eval()


def main():
    from transformers import AutoConfig
    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(SRC, trust_remote_code=True)
    E, H, K = cfg.n_routed_experts, cfg.hidden_size, cfg.num_experts_per_tok

    check("group routing is a no-op for this checkpoint", cfg.n_group == 1 and cfg.topk_group == 1,
          f"n_group={cfg.n_group} topk_group={cfg.topk_group}; groups would need a mask and "
          f"would go ragged under pruning")

    # ---- REAL router weights from the checkpoint (layer 1 is the first MoE layer) ----
    reader = ShardReader(SRC)
    W = reader.get("model.layers.1.mlp.gate.weight").float()
    B = reader.get("model.layers.1.mlp.gate.e_score_correction_bias").float()
    reader.release()
    check("real router weights loaded", W.shape == (E, H) and B.shape == (E,),
          f"W{tuple(W.shape)} B{tuple(B.shape)} bias|max|={B.abs().max():.3f}")

    # ---- 1. gate_forward vs the checkpoint's own module ----
    gate = real_gate_module(cfg)
    with torch.no_grad():
        gate.weight.copy_(W)
        gate.e_score_correction_bias.copy_(B)
    N = 256
    h = torch.randn(1, N, H) * 0.5
    with torch.no_grad():
        idx_o, w_o = gate(h)
        idx_m, w_m = RK.gate_forward(h.view(-1, H), W, B, K)

    def dense(i, w):
        return torch.zeros(N, E).scatter_(1, i, w.float())
    sel = float((torch.sort(idx_o, -1)[0] == torch.sort(idx_m, -1)[0]).float().mean())
    wdiff = float((dense(idx_o, w_o) - dense(idx_m, w_m)).abs().max())
    check("gate_forward selects what the real module selects", sel == 1.0, f"agreement {sel:.3f}")
    check("gate_forward weights match the real module", wdiff < 1e-6, f"max|diff| {wdiff:.2e}")

    # ---- build a candidate set, as the driver does ----
    # The candidate set is built PER KEEP SET, not once: the teacher's top-8 (so the teacher's
    # mixture is exact) plus a FIXED number of the highest-scoring KEPT experts (so the student
    # always has the same number of options). Choosing candidates before pruning, which is the
    # obvious way, leaves a random and sometimes tiny kept subset -- measured at 9 here, of which
    # the student must take 8, so it sits on the boundary by construction and cannot be trained.
    CK, top = 24, K
    torch.manual_seed(1)
    EO = torch.randn(E, H) * 0.1

    def candidates(keep_ids):
        """(cand_ids[N,C], kept_slot[N,CK], kept_local[N,CK]) -- kept count is exactly CK."""
        with torch.no_grad():
            choice = torch.sigmoid(h.view(-1, H) @ W.T) + B
        keep_sorted = torch.sort(keep_ids).values
        kept_top = keep_sorted[torch.topk(choice[:, keep_sorted], CK, dim=-1).indices]  # [N, CK]
        cand = torch.cat([idx_m, kept_top], dim=-1)                 # [N, K+CK], may repeat
        local_of = torch.full((E,), -1, dtype=torch.long)
        local_of[keep_sorted] = torch.arange(len(keep_sorted))
        kept_slot = torch.arange(K, K + CK).expand(N, CK).contiguous()
        return cand, kept_slot, local_of[kept_top]

    # f_e(x) is a FUNCTION of (token, expert). An earlier fixture drew fresh noise per candidate
    # SLOT, so the same expert appearing twice in the union got two different outputs and the
    # keep-everything control read 2.5e-05 instead of 0. The driver dedups the union for the same
    # reason -- computing an expert twice is both wasted work and a way to make the teacher's and
    # the student's lookups disagree about the same expert.
    tokscale = 1.0 + 0.1 * torch.randn(N, 1, 1)

    def outputs_for(cand):
        return EO[cand] * tokscale

    def slots_for(ids, cand):
        return (cand.unsqueeze(1) == ids.unsqueeze(-1)).float().argmax(-1)

    # ---- 2a. CONTROL, sign known in advance: keep everything -> loss must be exactly 0 ----
    keep_all = torch.arange(E)
    cand, ks, kl = candidates(keep_all)
    cand_out = outputs_for(cand)
    t_slot, t_w = slots_for(idx_m, cand), w_m
    l0 = RK.baseline_loss(cand_out, t_slot, t_w, h.view(-1, H), W, B, keep_all, ks, kl, top)
    check("keeping every expert gives exactly zero loss", l0 < 1e-12, f"loss {l0:.3e}")

    # ---- 2b. prune half -> there must be something to repair ----
    g = torch.Generator().manual_seed(7)
    keep = torch.sort(torch.randperm(E, generator=g)[:E // 2]).values
    cand, ks, kl = candidates(keep)
    cand_out = outputs_for(cand)
    t_slot, t_w = slots_for(idx_m, cand), w_m
    lb = RK.baseline_loss(cand_out, t_slot, t_w, h.view(-1, H), W, B, keep, ks, kl, top)
    check("pruning half leaves a non-zero loss to repair", lb > 1e-8, f"baseline {lb:.3e}")

    # ---- 3. training reduces it, without living on the candidate boundary ----
    r = RK.fit_layer(cand_out, t_slot, t_w, h.view(-1, H), W, B, keep, ks, kl, top,
                     steps=150, lr=1e-2)
    check("training reduces the output loss", r["last"] < r["baseline"],
          f"{r['baseline']:.4e} -> {r['last']:.4e}  ({r['improvement']:+.1%})")
    check("student is not clipped by the candidate boundary", r["boundary"] < 0.05,
          f"boundary rate {r['boundary']:.1%} over {kl.shape[1]} kept candidates for top-{top}")
    check("the bias slice is carried unchanged, not 'trained'",
          torch.equal(r["b"], B[keep].float()), "e_score_correction_bias has zero gradient")

    # ---- negative test: a wrong kept-order must break it ----
    bad = kl.flip(-1)
    lbad = RK.baseline_loss(cand_out, t_slot, t_w, h.view(-1, H), W, B, keep, ks, bad, top)
    check("a permuted kept-index map is detected", abs(lbad - lb) > 1e-9,
          f"{lb:.3e} vs permuted {lbad:.3e}")

    print(("GATE FAIL: " + ", ".join(FAIL)) if FAIL else "GATE PASS")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
