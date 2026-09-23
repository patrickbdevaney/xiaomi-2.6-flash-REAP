"""Saliency patcher for MiMo-V2.6-Flash: per-expert accumulators AND HOPE's F-matrix, one pass.

MiMo ships custom modelling code (`modeling_mimo_v2.py`, trust_remote_code), so this patches
`MiMoV2MoE.moe` rather than GLM's `Glm5NextTextExperts`. The arithmetic of the patched forward is
the upstream forward VERBATIM; the only additions are the statistics.

WHAT IS COLLECTED, AND WHY EACH IS HERE
---------------------------------------
Per expert, per corpus bucket (the bucket axis is what lets mixtures be re-weighted offline):

    cnt  sum 1              -> Frequency  (0,0,0)
    gat  sum g              -> SEER       (0,1,0)
    nrm  sum ||f||          -> EAN        (0,0,1)   and MAN (1,0,1) as nrm/cnt
    nsq  sum ||f||^2        -> MSAN       (1,0,2)   as nsq/cnt
    sum  sum g*||f||        -> REAP       (1,1,1)   as sum/cnt,  AND (0,1,1) as sum itself
    sq   sum (g*||f||)^2    -> (0,2,2)    as sq itself
    gsq  sum g^2            -> gate variance, for var-aware criteria

That covers every criterion in the unified family of arXiv 2606.15716 -- including the two that
win at 50% when calibration is capability-aligned, which is our case: (0,1,1) and (0,2,2).
All of them are re-derivable offline, for free, as many times as we like.

Per layer, the HOPE F-matrix (arXiv 2609.18916). This is the ONLY statistic here that cannot be
recovered afterwards, because it needs per-token CO-ACTIVATION structure that no per-expert
accumulator retains. See hope_fmatrix.py.

THE ONE SUBTLETY: the gate value.
`MiMoV2MoEGate.forward` renormalises topk_weight under `norm_topk_prob` (true for MiMo) and then
multiplies by `routed_scaling_factor` (None -> 1.0). We read that returned weight rather than
recomputing it from logits, which is why the upstream REAP logit-renormalisation fix of
2026-03-11 is a no-op for us. Recomputing the gate is exactly where that bug lived.
"""
from __future__ import annotations

import torch

from hope_fmatrix import FAccumulator

# Set by the driver before each batch.
CTX: dict = {"layer": None, "bucket": 0, "valid": None}
ACC: dict[str, dict] = {}
FACC: FAccumulator | None = None
LAYER_INDEX: dict[str, int] = {}
BUCKETS: list[str] = []


def configure(buckets, n_layers: int, n_experts: int) -> None:
    global FACC, BUCKETS
    BUCKETS = list(buckets)
    FACC = FAccumulator(n_layers, n_experts)


def _ensure(lname: str, n_experts: int, dev) -> dict:
    a = ACC.get(lname)
    if a is None:
        z = lambda: torch.zeros(len(BUCKETS), n_experts, dtype=torch.float64, device=dev)
        a = {k: z() for k in ("sum", "sq", "cnt", "nrm", "nsq", "gat", "gsq")}
        ACC[lname] = a
    return a


def patch(mod) -> None:
    """Replace MiMoV2MoE.moe with a copy that also accumulates. Idempotent."""
    MoE = mod.MiMoV2MoE
    if getattr(MoE, "_reap_patched", False):
        return

    def moe(self, hidden_states, topk_indices, topk_weights):
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=len(self.experts))
        expert_mask = expert_mask.permute(2, 0, 1)

        lname = CTX["layer"]
        acc = _ensure(lname, len(self.experts), hidden_states.device) if lname else None
        bkt = CTX["bucket"]
        valid = CTX["valid"]
        if valid is not None and valid.shape[0] != hidden_states.shape[0]:
            # The caller flattens with view(-1, H). If the mask ever stops being 1:1 with expert
            # rows, a silently misaligned mask corrupts every statistic in the run while still
            # looking plausible -- so refuse to guess how to broadcast it.
            raise RuntimeError(
                f"valid mask length {valid.shape[0]} != {hidden_states.shape[0]} expert rows in "
                f"{lname}; token flattening is not 1:1, fix the mask construction")
        # Per-token, per-SLOT gated magnitudes, filled in as the expert loop visits each slot.
        # `weight_indices` from torch.where IS the slot, which is what makes the F-matrix
        # outer product possible without a second pass over the experts.
        S = torch.zeros_like(topk_weights, dtype=torch.float64) if acc is not None else None

        for expert_idx, expert in enumerate(self.experts):
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)
            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)               # f_j, UNGATED -- what REAP is defined over
                final_hidden_states.index_add_(
                    0, token_indices, expert_output * expert_weights.unsqueeze(-1))
                if acc is not None:
                    with torch.no_grad():
                        keep = valid[token_indices] if valid is not None else None
                        f_v = expert_output if keep is None else expert_output[keep]
                        g_v = expert_weights if keep is None else expert_weights[keep]
                        if f_v.shape[0]:
                            nrm = f_v.to(torch.float32).norm(dim=-1).double()
                            g = g_v.to(torch.float64)
                            sv = g * nrm
                            acc["sum"][bkt, expert_idx] += sv.sum()
                            acc["sq"][bkt, expert_idx] += (sv * sv).sum()
                            acc["cnt"][bkt, expert_idx] += f_v.shape[0]
                            acc["nrm"][bkt, expert_idx] += nrm.sum()
                            acc["nsq"][bkt, expert_idx] += (nrm * nrm).sum()
                            acc["gat"][bkt, expert_idx] += g.sum()
                            acc["gsq"][bkt, expert_idx] += (g * g).sum()
                            # Same values into the per-slot scratch, so F sees EXACTLY the
                            # tokens the per-expert stats saw. gate_mimo_saliency asserts
                            # F[k,k] == sq/cnt, which is what catches any drift between them.
                            ti = token_indices if keep is None else token_indices[keep]
                            wi = weight_indices if keep is None else weight_indices[keep]
                            S[ti, wi] = sv

        if acc is not None and FACC is not None and lname in LAYER_INDEX:
            # ONLY the valid rows. F's denominator is |X_ij|, the number of tokens routing both
            # experts; feeding masked rows through would leave their values at zero but still
            # count them as co-activations, silently deflating every entry.
            ti_f = topk_indices if valid is None else topk_indices[valid]
            s_f = S if valid is None else S[valid]
            FACC.update(LAYER_INDEX[lname], ti_f, s_f)
        return final_hidden_states.type(hidden_states.dtype)

    MoE.moe = moe
    MoE._reap_patched = True
