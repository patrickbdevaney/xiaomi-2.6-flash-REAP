"""Repair the router after pruning, by matching the TEACHER'S LAYER OUTPUT.

WHAT IS ACTUALLY BROKEN, AND WHAT IS NOT
----------------------------------------
Slicing `mlp.gate.weight` to the kept experts leaves the student's logits bit-identical to the
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

WHAT TRAINS
-----------
Per MoE layer, `gate.weight` [n_kept, hidden] and `gate.e_score_correction_bias` [n_kept],
initialised from the teacher's slice. The experts themselves are frozen: they are MXFP4 and
retraining them would both destroy the quantisation and require a full-precision copy the box
cannot hold.

Top-k is not differentiable, so the gradient flows through the SOFT gate values of the selected
experts. The selection itself is treated as fixed within a step, which is the standard
straight-through treatment for MoE routers and is why this repairs the mixture weights rather
than rediscovering the mask.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def teacher_student_step(hidden: torch.Tensor, router_w: torch.Tensor, router_b: torch.Tensor,
                         expert_out: torch.Tensor, keep: torch.Tensor, top_k: int,
                         student_w: torch.Tensor, student_b: torch.Tensor):
    """One layer, one batch.

    hidden      [N, H]            layer input
    router_w    [E, H]            teacher router
    expert_out  [N, E, H]         each expert's output for each token (frozen)
    keep        [K] long          indices of kept experts, in student order
    Returns (loss, teacher_y, student_y).
    """
    tl = hidden @ router_w.T + router_b                      # [N, E]
    tg, ti = torch.topk(torch.sigmoid(tl), top_k, dim=-1)
    tg = tg / tg.sum(-1, keepdim=True).clamp(min=1e-9)       # norm_topk_prob
    ty = torch.zeros_like(hidden)
    for k in range(top_k):
        ty += tg[:, k:k + 1] * expert_out[torch.arange(hidden.shape[0]), ti[:, k]]

    sl = hidden @ student_w.T + student_b                    # [N, K]
    k_eff = min(top_k, sl.shape[-1])
    sg, si = torch.topk(torch.sigmoid(sl), k_eff, dim=-1)
    sg = sg / sg.sum(-1, keepdim=True).clamp(min=1e-9)
    sy = torch.zeros_like(hidden)
    for k in range(k_eff):
        sy = sy + sg[:, k:k + 1] * expert_out[torch.arange(hidden.shape[0]), keep[si[:, k]]]

    loss = torch.nn.functional.mse_loss(sy, ty)
    return loss, ty, sy


def fit_layer(hidden, router_w, router_b, expert_out, keep, top_k,
              steps: int = 200, lr: float = 1e-3, verbose: bool = False):
    """Train one layer's router to match the teacher's output. Returns (w, b, first, last)."""
    sw = router_w[keep].clone().detach().requires_grad_(True)
    sb = router_b[keep].clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([sw, sb], lr=lr)
    first = last = None
    for i in range(steps):
        opt.zero_grad()
        loss, _, _ = teacher_student_step(hidden, router_w, router_b, expert_out,
                                          keep, top_k, sw, sb)
        loss.backward()
        opt.step()
        last = float(loss)
        if first is None:
            first = last
        if verbose and i % max(1, steps // 5) == 0:
            print(f"      step {i:4d} loss {last:.6e}", flush=True)
    return sw.detach(), sb.detach(), first, last


def baseline_loss(hidden, router_w, router_b, expert_out, keep, top_k) -> float:
    """Loss of the UNTRAINED sliced router -- the number any training must beat."""
    with torch.no_grad():
        loss, _, _ = teacher_student_step(hidden, router_w, router_b, expert_out, keep,
                                          top_k, router_w[keep], router_b[keep])
    return float(loss)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", default="artifacts/masks/mask.json")
    ap.add_argument("--out", default="artifacts/masks/router_kd.json")
    ap.add_argument("--steps", type=int, default=200)
    a = ap.parse_args()
    print("router_kd is a library plus a driver; the full-model driver runs after the "
          "calibration pass, when the GPU is free. Gate it with gate_router_kd.py.")
    print(json.dumps({"mask": a.mask, "steps": a.steps}, indent=1))
