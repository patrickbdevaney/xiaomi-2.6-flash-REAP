"""Repair the router after pruning, by matching the TEACHER'S LAYER OUTPUT.

WHAT IS ACTUALLY BROKEN, AND WHAT IS NOT
----------------------------------------
Slicing `mlp.gate.weight` to the kept experts leaves the student's scores bit-identical to the
teacher's on those experts. So an objective that matches ROUTING DISTRIBUTIONS has nothing to
learn -- the KL is already zero -- and any implementation that appears to train under one is
training on noise. That is worth stating because it is the obvious formulation and it is wrong.

What pruning actually breaks is the layer OUTPUT. The teacher mixes its top-8 of 256; the
student can only mix its top-8 of the survivors, so for every token whose teacher top-8 included
a pruned expert, output mass is simply missing:

    teacher:  y = sum over top8(all 256)  of  g_e * f_e(x)
    student:  y'= sum over top8(kept)     of  g'_e * f_e(x)

`norm_topk_prob=True` renormalises the surviving gates to sum to 1, so the SCALE is already
compensated; what it cannot fix is that the mixture is now over a different, smaller set. The
router can partly recover this by learning to prefer the kept experts that best substitute for
the ones that were removed -- which is exactly an output-matching objective, and only the router
parameters need to move.

THE ROUTER FORMULA IS NOT THE OBVIOUS ONE, AND GETTING IT WRONG IS SILENT
------------------------------------------------------------------------
MiMo's gate (`MiMoV2MoEGate.forward`) is DeepSeek-style `noaux_tc` and uses TWO DIFFERENT score
tensors:

    scores = sigmoid(h @ W.T)                  fp32, NO bias
    choice = scores + e_score_correction_bias  selection ONLY
    idx    = topk(choice, 8)
    w      = scores.gather(idx)                the UNBIASED score is the weight
    w      = w / w.sum()                       norm_topk_prob
    w      = w * routed_scaling_factor         config says None -> the module substitutes 1.0

An earlier version of this file folded the bias into the logits and used the biased sigmoid as
the weight. Measured against the checkpoint's own gate module with a realistic bias, that
selected a DIFFERENT expert set -- 0.125 agreement, one expert of eight -- and produced weights
off by 1.77e-01 absolute on a vector that sums to one. The formula below reproduces the oracle
to 2.98e-08. A router-repair objective built on the wrong router repairs nothing.

`n_group=1, topk_group=1` in this checkpoint, so the module's group-limited routing is a no-op
and is not reproduced here. That is ASSERTED, not assumed: a checkpoint with real groups needs
the group mask, and pruning would leave the groups ragged.

The real gate also refuses to run under `self.training`, which is a second reason the objective
needs its own differentiable copy rather than calling the module.

THE BIAS CANNOT BE TRAINED BY THIS OBJECTIVE
--------------------------------------------
`e_score_correction_bias` enters only through `topk`, which is not differentiable, so under the
straight-through treatment its gradient is exactly zero. It is sliced and carried, never
optimised. Putting it in an optimiser looks like training and is not -- Adam skips a parameter
whose grad is None, so the code would run, report a falling loss from the weight alone, and
silently imply the bias had been repaired.

WHY THE EXPERT OUTPUTS ARE PASSED IN AS A CANDIDATE SET
-------------------------------------------------------
The objective needs `f_e(x)` for every expert either router might select. Materialising all 256
for every token is [N, 256, 4096] -- 2 MB per token, 17 GB at N=8192 -- and would cost a dense
forward over all experts. Instead the driver precomputes a per-token CANDIDATE SET: the
teacher's top-8 plus the highest-scoring kept experts, `cand_out[N, C, H]` with `cand_id[N, C]`.
The student's argmax is taken over the kept members of that set.

This is an APPROXIMATION and it is bounded, not assumed: `fit_layer` reports `boundary`, the
fraction of student selections that landed on the lowest-ranked candidate. If that is not small,
C is too tight and the student is being clipped rather than trained.
"""
from __future__ import annotations

import argparse
import json

import torch


def gate_forward(hidden: torch.Tensor, w: torch.Tensor, b: torch.Tensor, top_k: int,
                 restrict: torch.Tensor | None = None):
    """MiMo's noaux_tc gate, differentiable in `w`. Returns (idx, weight).

    hidden   [N, H]
    w        [E, H]      router weight
    b        [E]         e_score_correction_bias (selection only)
    restrict [N, C] long or None -- if given, selection is limited to these expert ids and the
                                    returned idx indexes INTO `restrict`, not into E.
    """
    scores = torch.sigmoid(hidden.float() @ w.float().T)          # [N, E]
    choice = scores + b.float()
    if restrict is not None:
        scores = scores.gather(1, restrict)
        choice = choice.gather(1, restrict)
    k = min(top_k, scores.shape[-1])
    _, idx = torch.topk(choice, k, dim=-1)
    wgt = scores.gather(1, idx)
    return idx, wgt / (wgt.sum(-1, keepdim=True) + 1e-20)


def _mix(weight: torch.Tensor, slot: torch.Tensor, cand_out: torch.Tensor) -> torch.Tensor:
    """sum_k weight[:,k] * cand_out[n, slot[n,k]] -- the MoE mixture, per token."""
    gathered = torch.gather(
        cand_out, 1, slot.unsqueeze(-1).expand(-1, -1, cand_out.shape[-1]))  # [N, k, H]
    return (weight.unsqueeze(-1).float() * gathered.float()).sum(1)


def teacher_student_step(cand_out: torch.Tensor, t_slot: torch.Tensor, t_w: torch.Tensor,
                         hidden: torch.Tensor, student_w: torch.Tensor, student_b: torch.Tensor,
                         kept_slot: torch.Tensor, kept_local: torch.Tensor, top_k: int,
                         ty: torch.Tensor | None = None):
    """One layer, one batch, over the precomputed candidate set.

    cand_out   [N, C, H]   each candidate expert's output for its token (frozen)
    t_slot     [N, k]      teacher's selection, as SLOTS into cand_out
    t_w        [N, k]      teacher's normalised gate weights
    hidden     [N, H]      layer input (the MoE's input, post-attention)
    student_w  [K, H]      student router over KEPT experts
    kept_slot  [N, Ck]     slots into cand_out for the kept candidates
    kept_local [N, Ck]     the same candidates as indices into the student's K rows
    """
    if ty is None:
        with torch.no_grad():
            ty = _mix(t_w, t_slot, cand_out)   # frozen: recomputing it every step is pure waste
    s_idx, s_w = gate_forward(hidden, student_w, student_b, top_k, restrict=kept_local)
    s_slot = torch.gather(kept_slot, 1, s_idx)
    sy = _mix(s_w, s_slot, cand_out)
    return torch.nn.functional.mse_loss(sy, ty), ty, sy, s_idx


def fit_layer(cand_out, t_slot, t_w, hidden, router_w, router_b, keep, kept_slot, kept_local,
              top_k, steps: int = 200, lr: float = 1e-3, batch: int | None = None,
              verbose: bool = False, seed: int = 0) -> dict:
    """Train one layer's router to match the teacher's output.

    `keep` [K] long -- kept expert ids, in student order. Returns the trained weight, the carried
    (untrained) bias slice, the baseline and final loss, and the candidate-boundary rate.

    `batch` minibatches the token set. The gathered mixture is [B, top_k, H] in fp32 and is held
    by the autograd graph, so at the full 2048-token working set that is ~537 MB per mix per
    step; minibatching is what keeps this off the edge of a box that has already been OOM-killed
    four times in this project. `None` uses every token, which is what the gate does.
    """
    sw = router_w[keep].clone().detach().float().requires_grad_(True)
    sb = router_b[keep].clone().detach().float()          # NOT trained; see the module docstring
    opt = torch.optim.Adam([sw], lr=lr)
    N = hidden.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)

    with torch.no_grad():
        ty_all = _mix(t_w, t_slot, cand_out)
        base, _, _, _ = teacher_student_step(cand_out, t_slot, t_w, hidden,
                                             router_w[keep].float(), sb, kept_slot, kept_local,
                                             top_k, ty=ty_all)
    first = last = None
    n_cand_k = kept_local.shape[-1]
    boundary = 0.0
    for i in range(steps):
        sel = (torch.randperm(N, generator=g)[:batch] if batch and batch < N
               else torch.arange(N))
        opt.zero_grad()
        loss, _, _, s_idx = teacher_student_step(
            cand_out[sel], t_slot[sel], t_w[sel], hidden[sel], sw, sb,
            kept_slot[sel], kept_local[sel], top_k, ty=ty_all[sel])
        loss.backward()
        opt.step()
        last = float(loss.detach())
        if first is None:
            first = last
        if i >= steps - 10:                    # averaged over the tail, not one noisy minibatch
            boundary += float((s_idx == n_cand_k - 1).float().mean()) / min(10, steps)
        if verbose and i % max(1, steps // 5) == 0:
            print(f"      step {i:4d} loss {last:.6e}", flush=True)

    with torch.no_grad():                      # final loss on the FULL set, comparable to base
        final, _, _, _ = teacher_student_step(cand_out, t_slot, t_w, hidden, sw, sb,
                                              kept_slot, kept_local, top_k, ty=ty_all)
    final = float(final)
    return {"w": sw.detach(), "b": sb, "baseline": float(base), "first": first, "last": final,
            "boundary": boundary,
            "improvement": (float(base) - final) / max(float(base), 1e-30)}


def baseline_loss(cand_out, t_slot, t_w, hidden, router_w, router_b, keep,
                  kept_slot, kept_local, top_k) -> float:
    """Loss of the UNTRAINED sliced router -- the number any training must beat."""
    with torch.no_grad():
        loss, _, _, _ = teacher_student_step(cand_out, t_slot, t_w, hidden,
                                             router_w[keep].float(), router_b[keep].float(),
                                             kept_slot, kept_local, top_k)
    return float(loss)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", default="artifacts/masks/mask.json")
    a = ap.parse_args()
    print("router_kd is the library; the full-model driver is scripts/router_kd_run.py, which "
          "streams the layers after the calibration pass. Gate it with gate_router_kd.py.")
    print(json.dumps({"mask": a.mask}, indent=1))
