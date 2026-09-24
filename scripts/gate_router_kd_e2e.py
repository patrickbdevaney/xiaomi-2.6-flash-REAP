"""End-to-end gate for STAGE 6 on REAL weights, on CPU, without touching the GPU.

Stage 6 sits ~31 hours into the run. Its library and pure pieces are gated, but until now the
chain that actually runs -- build a real MoE layer from MXFP4, pick candidates, evaluate the
experts, fit the router -- had never executed on a real layer. A failure there would burn the
restart budget at the worst possible moment.

THE LOAD-BEARING CHECK is the third one: the teacher mixture assembled from `candidates()` and
`expert_outputs()` must equal what the checkpoint's OWN MoE module returns for the same input.
That single equality validates the router formula, the candidate slots, the per-expert
evaluation and the gather all at once, against an oracle rather than against my expectations.

CPU and one layer on purpose: the calibration pass owns the GPU, and this must never contend
with it. One MoE layer is 12 GiB of bf16 experts, so the gate refuses to start if that would
eat into the running pass's headroom.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import router_kd as RK             # noqa: E402
import router_kd_run as D          # noqa: E402
from calib_pass import build_layer  # noqa: E402
from mimo_shards import ShardReader  # noqa: E402

SRC = Path("/home/patrickd/models/MiMo-V2.6-Flash-RL")
MIN_AVAIL_MB = 50_000
FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}", flush=True)
    if not cond:
        FAIL.append(name)


def avail_mb():
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) // 1024
    return 0


def main():
    a0 = avail_mb()
    if a0 < MIN_AVAIL_MB:
        print(f"REFUSING: MemAvailable {a0}MB < {MIN_AVAIL_MB}MB. One MoE layer is ~12 GiB of "
              f"bf16 experts and the calibration pass is running; this gate must never be the "
              f"reason that pass is OOM-killed.")
        return 2
    print(f"  (MemAvailable {a0}MB; building one real MoE layer on CPU)", flush=True)

    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(SRC, trust_remote_code=True)
    K, E = cfg.num_experts_per_tok, cfg.n_routed_experts
    reader = ShardReader(SRC)

    t0 = time.time()
    layer = build_layer(cfg, 1, reader, torch.bfloat16)      # layer 1 = first MoE layer
    check("a real MoE layer builds from MXFP4 on CPU", hasattr(layer.mlp, "experts"),
          f"{len(layer.mlp.experts)} experts in {time.time()-t0:.0f}s, "
          f"avail {a0}->{avail_mb()}MB")

    # Real hidden states: real token ids through the real embedding. Scale matters to the gate.
    ids = torch.load(sorted(Path("artifacts/chunks").glob("chunk_*.pt"))[0],
                     map_location="cpu", weights_only=False)[0]["ids"][0, :96]
    emb = reader.get("model.embed_tokens.weight")
    hid = torch.nn.functional.embedding(ids, emb).to(torch.bfloat16)
    del emb
    reader.release()
    check("real hidden states obtained", hid.shape == (96, cfg.hidden_size),
          f"{tuple(hid.shape)} from real token ids, |x| mean {hid.float().abs().mean():.3f}")

    W = layer.mlp.gate.weight.detach()
    B = layer.mlp.gate.e_score_correction_bias.detach()
    keep = torch.arange(E)        # keep everything: the teacher and student must then AGREE

    cand, t_slot, t_w, kept_slot, kept_local = D.candidates(hid, W, B, keep, K, 32)
    cand_out = D.expert_outputs(layer.mlp.experts, hid, cand, torch.bfloat16)

    # ---- THE ORACLE CHECK: our teacher mixture vs the checkpoint's own MoE forward ----
    with torch.no_grad():
        ref = layer.mlp(hid.unsqueeze(0)).squeeze(0).float()
    ours = RK._mix(t_w, t_slot, cand_out).float()
    num = (ours - ref).norm()
    den = ref.norm().clamp(min=1e-30)
    rel = float(num / den)
    cos = float(torch.nn.functional.cosine_similarity(ours.flatten(), ref.flatten(), dim=0))
    check("teacher mixture reproduces the real MoE module's output", rel < 2e-2,
          f"relative error {rel:.3e}, cosine {cos:.8f}")

    # a wrong router would still produce a plausible-looking vector; show this test can see it
    bad_w = D.candidates(hid, W, B, keep, K, 32)[2].flip(-1)      # permuted gate weights
    bad = RK._mix(bad_w, t_slot, cand_out).float()
    rel_bad = float((bad - ref).norm() / den)
    check("the oracle check can see a wrong mixture", rel_bad > 10 * max(rel, 1e-6),
          f"permuted weights give {rel_bad:.3e} vs {rel:.3e}")

    # ---- the fit itself, on a real 50% prune ----
    g = torch.Generator().manual_seed(0)
    keep50 = torch.sort(torch.randperm(E, generator=g)[:E // 2]).values
    cand, t_slot, t_w, kept_slot, kept_local = D.candidates(hid, W, B, keep50, K, 32)
    cand_out = D.expert_outputs(layer.mlp.experts, hid, cand, torch.bfloat16)
    t1 = time.time()
    r = RK.fit_layer(cand_out, t_slot, t_w, hid, W, B, keep50, kept_slot, kept_local, K,
                     steps=60, lr=1e-3, batch=48)
    check("router KD reduces real layer output error", r["last"] < r["baseline"],
          f"{r['baseline']:.4e} -> {r['last']:.4e} ({r['improvement']:+.1%}) "
          f"in {time.time()-t1:.0f}s")
    check("student is not clipped by the candidate boundary", r["boundary"] < 0.05,
          f"boundary {r['boundary']:.1%}")
    check("trained router has the right shape for apply_mask",
          r["w"].shape == (E // 2, cfg.hidden_size) and r["b"].shape == (E // 2,),
          f"w{tuple(r['w'].shape)} b{tuple(r['b'].shape)}")
    check("nothing became NaN or Inf", bool(torch.isfinite(r["w"]).all()), "")

    # ---- the stage-6 pre-flight: it must pass on a good model AND fail on a broken one ----
    from transformers import AutoConfig as _AC
    reader2 = ShardReader(SRC)
    keep_by = {1: torch.arange(E)[::2].clone()}
    pf = D.preflight(cfg, reader2, keep_by, "cpu", torch.bfloat16, ids[:96], 32, K)
    check("pre-flight passes on the real model", pf["rel"] < 2e-2,
          f"layer {pf['layer']}, teacher mixture matches to {pf['rel']:.2e} in {pf['secs']:.0f}s")
    check("pre-flight measures the candidate tensor rather than guessing",
          pf["mb_per_token"] > 0, f"{pf['mb_per_token']*2048:.0f}MB projected at 2048 tokens")

    # break the mixture and confirm the pre-flight refuses -- a check that cannot fail is not one
    orig = RK._mix
    RK._mix = lambda w, slot, out: orig(w, slot, out) * 0.5
    try:
        reader3 = ShardReader(SRC)
        D.preflight(cfg, reader3, keep_by, "cpu", torch.bfloat16, ids[:96], 32, K)
        check("pre-flight refuses a wrong teacher mixture", False, "it passed")
    except SystemExit as e:
        check("pre-flight refuses a wrong teacher mixture", "PREFLIGHT FAILED" in str(e),
              str(e)[:72])
    finally:
        RK._mix = orig

    del layer, cand_out
    print(("GATE FAIL: " + ", ".join(FAIL)) if FAIL else "GATE PASS")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
