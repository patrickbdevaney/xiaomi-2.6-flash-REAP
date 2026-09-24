"""Full-model driver for router repair. Streams the layers; trains one router per MoE layer.

WHY THIS IS A SECOND PASS AND NOT A HEAD ON THE FIRST
-----------------------------------------------------
The objective needs the MoE's INPUT (post-attention hidden states) and the outputs of experts
the teacher did NOT select -- the kept experts the student might substitute. The calibration
pass computes neither: it only ever evaluates the teacher's top-8. Folding this in would have
made a 33-hour pass materially longer and coupled the run's survival to code that had not been
gated yet, so it runs afterwards, over a small token budget, when the GPU is free.

THE TOKEN BUDGET IS SMALL ON PURPOSE
------------------------------------
Only `n_kept * hidden` parameters are fit per layer -- for a 50% prune, 128 x 4096. A few
thousand tokens is a large sample for that, and the memory cost is the binding constraint:
`cand_out` is [N, C, H] in bf16, which at N=2048, C=32 is 537 MB and at N=16384 would be 4.3 GB
on top of a resident layer, on a box that has been OOM-killed four times in this project.

The forward itself, however, CANNOT be subsampled: hidden states at layer L require the full
sequence through layers 0..L-1, because attention is not token-separable. So a whole chunk is
streamed and the subsample is drawn from it -- the cost is one chunk's forward, about the same
as one calibration chunk.

RESUMABLE, because it is 47 independent problems and there is no reason to lose 46 of them.
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import router_kd as RK                                    # noqa: E402
from calib_pass import build_layer, _modeling, masks_for, _rope  # noqa: E402
from mimo_shards import ShardReader                       # noqa: E402

CAP = {"hid": None, "valid": None, "budget": 0, "seen": 0, "gen": None}


def _reservoir_fast(h: torch.Tensor) -> None:
    """Reservoir-sample the MoE's input rows into CAP["hid"], vectorised.

    Uniform over the WHOLE chunk rather than the first N rows of the first batch -- which would
    be one domain, and the corpus is deliberately nine of them. Row i of a batch, with n rows
    already seen, keeps with probability B/(n+i+1), which is the standard reservoir rule applied
    to a whole batch at once instead of one row at a time.
    """
    buf, B, n = CAP["hid"], CAP["budget"], CAP["seen"]
    m = h.shape[0]
    if n < B:
        take = min(B - n, m)
        buf[n:n + take] = h[:take]
        CAP["seen"] = n + take
        h, n, m = h[take:], n + take, m - take
        if m == 0:
            return
    # For each remaining row i (0-based within this batch) the reservoir index is uniform over
    # [0, n+i]; keep it if it lands inside the buffer.
    idx = n + torch.arange(m, device=h.device)
    j = (torch.rand(m, generator=CAP["gen"], device="cpu").to(h.device) * (idx + 1).float()).long()
    hit = j < B
    if bool(hit.any()):
        buf[j[hit].cpu()] = h[hit].to(buf.dtype).cpu()
    CAP["seen"] = n + m


def candidates(hid, W, B, keep, top_k, n_kept_cand):
    """(cand[N,C], t_slot[N,k], t_w[N,k], kept_slot[N,CK], kept_local[N,CK]).

    cand = teacher's top-k, then the CK highest-scoring KEPT experts. The teacher's slots are
    therefore 0..k-1 by construction, and the kept block is a fixed CK wide so the student always
    has the same number of options -- see the gate, where building candidates before pruning left
    9 kept options for a top-8 choice and pinned the student to the boundary.
    """
    with torch.no_grad():
        scores = torch.sigmoid(hid.float() @ W.float().T)
        choice = scores + B.float()
    keep_sorted = torch.sort(keep).values
    _, t_idx = torch.topk(choice, top_k, dim=-1)
    t_w = scores.gather(1, t_idx)
    t_w = t_w / (t_w.sum(-1, keepdim=True) + 1e-20)
    ck = min(n_kept_cand, len(keep_sorted))
    kept_top = keep_sorted.to(hid.device)[torch.topk(choice[:, keep_sorted], ck, dim=-1).indices]
    cand = torch.cat([t_idx, kept_top], dim=-1)
    local_of = torch.full((W.shape[0],), -1, dtype=torch.long, device=hid.device)
    local_of[keep_sorted.to(hid.device)] = torch.arange(len(keep_sorted), device=hid.device)
    N = hid.shape[0]
    t_slot = torch.arange(top_k, device=hid.device).expand(N, top_k).contiguous()
    kept_slot = torch.arange(top_k, top_k + ck, device=hid.device).expand(N, ck).contiguous()
    return cand, t_slot, t_w, kept_slot, local_of[kept_top]


def expert_outputs(experts, hid, cand, dtype):
    """cand_out[N, C, H], computing each (token, expert) pair ONCE.

    An expert that appears at two candidate slots for the same token must produce the same value
    at both, or the teacher's lookup and the student's lookup disagree about the same expert --
    the failure the gate caught when its fixture drew fresh noise per slot.

    RUNS UNDER no_grad AND RETURNS A DETACHED TENSOR. The experts are frozen -- only the router
    trains -- but they are real nn.Modules whose weights carry requires_grad, so without this the
    returned tensor is attached to an autograd graph through all 256 of them. The first
    optimiser step then frees that graph and the second dies with "Trying to backward through
    the graph a second time". Measured on real layer-1 weights; the synthetic gate never saw it
    because its fixture was a plain tensor. It is also a memory bug: every step would otherwise
    retain a graph back through every expert it touched.
    """
    N, C = cand.shape
    out = torch.zeros(N, C, hid.shape[-1], dtype=dtype, device=hid.device)
    with torch.no_grad():
        for e in torch.unique(cand).tolist():
            m = cand == e
            toks = m.any(1).nonzero(as_tuple=True)[0]
            y = experts[e](hid[toks].to(dtype))
            pos = torch.full((N,), -1, dtype=torch.long, device=hid.device)
            pos[toks] = torch.arange(len(toks), device=hid.device)
            ns, cs = m.nonzero(as_tuple=True)
            out[ns, cs] = y[pos[ns]]
    return out.detach()


def _rss_mb() -> int:
    """This process's resident size. Reported per layer because stage 6 builds 47 layers of
    ~12 GiB each and frees them; if that free ever stops working the run dies at layer N, and a
    growth trend is the only thing that shows it before it does."""
    try:
        with open("/proc/self/status") as f:
            for ln in f:
                if ln.startswith("VmRSS:"):
                    return int(ln.split()[1]) // 1024
    except OSError:
        pass
    return 0



def _sample_states(chunks_dir: Path, per_domain: int, batches: int):
    """Batches drawn from chunks spanning EVERY domain, not one chunk.

    TWO THINGS WERE WRONG WITH TAKING A SINGLE CHUNK.

    First, the routers would be fitted on ONE DOMAIN. Chunks are bucket-homogeneous -- that is
    what the corpus blocking means -- so chunk 0 is agentic and nothing else. The mask being
    repaired is global across nine domains; fitting its routers on agentic alone repairs the
    wrong thing, and video, the domain with the least evidence already, would contribute nothing.

    Second, it was enormously wasteful. The driver streamed a whole 2 M-token chunk through 48
    layers -- one to two hours of GPU -- to keep a reservoir of 2048 rows. Hidden states at layer
    L require the full sequence through layers 0..L-1, so the forward cannot be subsampled by
    TOKEN, but it can be subsampled by BATCH, and a few batches per domain is a far better sample
    than hundreds from one.

    The reservoir still runs: it is what makes the kept rows uniform over the batches supplied.
    """
    files = sorted(chunks_dir.glob("chunk_*.pt"))
    if not files:
        raise SystemExit(f"no chunks in {chunks_dir}")
    cache = chunks_dir / "chunk_buckets.json"
    known = json.loads(cache.read_text()) if cache.exists() else {}
    by_bucket: dict[str, list[Path]] = {}
    for f in files:
        b = known.get(f.name)
        if b is None:                      # no cache: fall back to reading the chunk's own label
            st = torch.load(f, map_location="cpu", weights_only=False)
            b = st[0]["bucket"]; del st
        by_bucket.setdefault(b, []).append(f)
    picked, out = [], []
    for b in sorted(by_bucket):
        for f in by_bucket[b][:max(1, per_domain)]:
            st = torch.load(f, map_location="cpu", weights_only=False)
            take = st[:batches] if batches else st
            out.extend(take)
            picked.append(f"{b}:{len(take)}")
            del st
    print(f"router KD sample: {len(out)} batches from {len(by_bucket)} domains "
          f"({', '.join(picked)})", flush=True)
    return out, files[0]


def _avail_mb() -> int:
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) // 1024
    return 0


def preflight(cfg, reader, keep_by_layer, device, dtype, ids, kept_cand, top_k,
              tol: float = 2e-2) -> dict:
    """Run ONE MoE layer on the real device before committing to all 47.

    Stage 6 is ~31 hours into the run and its GPU path cannot be rehearsed beforehand without
    competing for memory with the calibration pass -- the contention that has OOM-killed this box
    four times. So the rehearsal happens here instead, at the moment the GPU is free and before
    any of the expensive work: build one real layer, assemble the teacher mixture, and check it
    against the layer's OWN MoE forward.

    That one equality exercises the router formula, the candidate slots, the per-expert
    evaluation and the gather together, on the real device in the real dtype. It also takes two
    optimiser steps, because the defect that actually bit here was an autograd graph retained
    through the frozen experts: it raised only on the SECOND step, and only on real weights.

    Two minutes, and the alternative is discovering it after the stage has run for hours.
    """
    li = min(keep_by_layer)
    t0 = time.time()
    layer = build_layer(cfg, li, reader, dtype).to(device)
    if not hasattr(layer.mlp, "experts"):
        del layer
        raise SystemExit(f"preflight: layer {li} has no experts; the mask indexes a dense layer")
    emb = reader.get("model.embed_tokens.weight")
    hid = torch.nn.functional.embedding(ids.to(emb.device), emb).to(device, dtype)
    del emb
    reader.release()

    W = layer.mlp.gate.weight.detach()
    B = layer.mlp.gate.e_score_correction_bias.detach()
    E = W.shape[0]
    all_keep = torch.arange(E, device=device)
    cand, t_slot, t_w, kept_slot, kept_local = candidates(hid, W, B, all_keep, top_k, kept_cand)
    cand_out = expert_outputs(layer.mlp.experts, hid, cand, dtype)
    with torch.no_grad():
        ref = layer.mlp(hid.unsqueeze(0)).squeeze(0).float()
    ours = RK._mix(t_w, t_slot, cand_out).float()
    rel = float((ours - ref).norm() / ref.norm().clamp(min=1e-30))
    if not (rel < tol):
        del layer, cand_out
        raise SystemExit(
            f"PREFLIGHT FAILED: the teacher mixture disagrees with layer {li}'s own MoE output "
            f"by {rel:.3e} (tolerance {tol:.0e}). The router formula, the candidate slots or the "
            f"expert gather is wrong on this device, and every layer would be trained against a "
            f"teacher that is not the model. Nothing has been written.")

    keep = keep_by_layer[li].to(device)
    cand, t_slot, t_w, kept_slot, kept_local = candidates(hid, W, B, keep, top_k, kept_cand)
    cand_out = expert_outputs(layer.mlp.experts, hid, cand, dtype)
    try:
        RK.fit_layer(cand_out, t_slot, t_w, hid, W, B, keep, kept_slot, kept_local, top_k,
                     steps=2, lr=1e-4, batch=None)
    except RuntimeError as e:
        del layer, cand_out
        raise SystemExit(
            f"PREFLIGHT FAILED: the optimiser could not take two steps on real weights on "
            f"{device} ({e}). This is the shape of a retained autograd graph through the frozen "
            f"experts, which raises only on the second step. Nothing has been written.")
    mb_per_token = cand_out.numel() * cand_out.element_size() / 2 ** 20 / hid.shape[0]
    del layer, cand_out, hid
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"layer": li, "rel": rel, "mb_per_token": mb_per_token, "secs": time.time() - t0}


def load_keep(mask_path: Path, n_exp: int) -> dict[int, torch.Tensor]:
    d = json.loads(Path(mask_path).read_text())
    mask = d.get("mask", d)
    keep = {}
    for k, v in mask.items():
        li = int(k.split(".")[2])
        drop = set(int(i) for i in v)
        keep[li] = torch.tensor([e for e in range(n_exp) if e not in drop], dtype=torch.long)
    return keep


def _busy() -> str | None:
    try:
        r = subprocess.run(["systemctl", "--user", "is-active", "reap_run.service"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.stdout.strip() == "active" else None
    except Exception:
        return None


def run(src, chunks_dir, mask_path, out_dir, device="cuda", dtype=torch.bfloat16,
        tokens=2048, kept_cand=32, steps=300, lr=1e-3, batch=512, chunk_index=0, seed=0,
        max_layers=None, max_batches=None, max_seq=None, skip_preflight=False,
        chunks_per_domain=1, batches_per_chunk=8):
    from transformers import AutoConfig
    src, out_dir = Path(src), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = AutoConfig.from_pretrained(src, trust_remote_code=True)
    assert cfg.n_group == 1 and cfg.topk_group == 1, (
        f"n_group={cfg.n_group} topk_group={cfg.topk_group}: this checkpoint uses group-limited "
        f"routing, which pruning leaves ragged. gate_forward does not implement the group mask.")
    n_layers, n_exp, K = cfg.num_hidden_layers, cfg.n_routed_experts, cfg.num_experts_per_tok
    keep_by_layer = load_keep(Path(mask_path), n_exp)
    reader = ShardReader(src)

    states, cf = _sample_states(Path(chunks_dir), chunks_per_domain, batches_per_chunk)
    # REHEARSAL LIMITS. Both default to None. They exist so the full driver -- hook, reservoir,
    # candidate build, expert evaluation, fit, checkpoint, resume -- can be executed end to end
    # on CPU against real weights before it runs for real at hour 31. A stage that has never
    # been executed is not a stage that works.
    if max_batches:
        states = states[:max_batches]
    if max_seq:
        # Truncating the sequence is what makes a CPU rehearsal possible at all: eager attention
        # on a 4096-token sequence is refused upstream for good reason, and the GPU belongs to
        # the calibration pass. It changes what the routers see, never whether the code runs.
        states = [{**st, "ids": st["ids"][:, :max_seq], "valid": st["valid"][:, :max_seq]}
                  for st in states]
    S = states[0]["ids"].shape[1]
    masks = masks_for(S, cfg.sliding_window, device, dtype)
    pos = torch.arange(S, device=device)[None]

    state_path = out_dir / "router_kd_state.json"
    res = json.loads(state_path.read_text()) if state_path.exists() else {"layers": {}}
    trained_path = out_dir / "router_kd.pt"
    trained = torch.load(trained_path, map_location="cpu") if trained_path.exists() else {}

    if not skip_preflight:
        pf = preflight(cfg, reader, keep_by_layer, device, dtype,
                       states[0]["ids"][0, :96], kept_cand, K)
        # The candidate tensor is the one allocation in this stage that scales with the token
        # budget, and it is measured here rather than predicted: [tokens, K+kept_cand, hidden].
        # Refusing now costs nothing; discovering it at token 2048 costs the calibration pass,
        # because on this box an over-allocation is charged to no cgroup and killed by nobody.
        proj_mb = pf["mb_per_token"] * tokens
        avail_mb = _avail_mb()
        if proj_mb > 0.25 * avail_mb:
            raise SystemExit(
                f"PREFLIGHT REFUSED: the candidate tensor would be {proj_mb:.0f}MB at "
                f"--tokens {tokens}, more than a quarter of the {avail_mb}MB available. "
                f"Re-run with --tokens {int(tokens * 0.25 * avail_mb / max(proj_mb, 1)):d} "
                f"or fewer, or --kept-candidates below {kept_cand}. Nothing has been written.")
        print(f"preflight OK on layer {pf['layer']}: teacher mixture matches the real MoE to "
              f"{pf['rel']:.2e}, two optimiser steps taken; candidate tensor projects to "
              f"{proj_mb:.0f}MB at {tokens} tokens ({100*proj_mb/max(avail_mb,1):.1f}% of "
              f"{avail_mb}MB available) [{pf['secs']:.0f}s]", flush=True)

    mod = _modeling(cfg)
    emb = None
    print(f"router KD: {len(states)} batches x {S} tokens, "
          f"budget {tokens} rows, {kept_cand} kept candidates, {steps} steps", flush=True)

    t0 = time.time()
    for li in range(min(n_layers, max_layers or n_layers)):
        layer = build_layer(cfg, li, reader, dtype).to(device)
        is_moe = hasattr(layer.mlp, "experts")
        name = f"model.layers.{li}.mlp"
        want = is_moe and li in keep_by_layer and name not in res["layers"]

        CAP.update({"budget": tokens if want else 0, "seen": 0,
                    "gen": torch.Generator().manual_seed(seed + li),
                    "hid": torch.zeros(tokens, cfg.hidden_size, dtype=torch.float32)
                           if want else None})
        handle = None
        if want:
            handle = layer.mlp.register_forward_pre_hook(
                lambda m, a: _reservoir_fast(
                    a[0].detach().reshape(-1, a[0].shape[-1])[CAP["valid"]]
                    if CAP["valid"] is not None
                    else a[0].detach().reshape(-1, a[0].shape[-1])))
        with torch.no_grad():
            for st in states:
                CAP["valid"] = st["valid"].to(device).reshape(-1) if want else None
                if li == 0:
                    if emb is None:
                        emb = reader.get("model.embed_tokens.weight").to(device, dtype)
                    hs = torch.nn.functional.embedding(st["ids"].to(device), emb)
                else:
                    hs = st["hs"].to(device, dtype)
                pe = _rope(cfg, layer.attention_type, hs, pos, device, dtype)
                out = layer(hs, attention_mask=masks[layer.attention_type],
                            position_ids=pos, position_embeddings=pe)
                st["hs"] = out.to("cpu", torch.bfloat16)
                del hs, out
        if handle is not None:
            handle.remove()
        CAP["valid"] = None

        if want:
            n = min(CAP["seen"], tokens)
            hid = CAP["hid"][:n].to(device, dtype)
            W = layer.mlp.gate.weight.detach().to(device)
            B = layer.mlp.gate.e_score_correction_bias.detach().to(device)
            keep = keep_by_layer[li].to(device)
            cand, t_slot, t_w, kept_slot, kept_local = candidates(hid, W, B, keep, K, kept_cand)
            cand_out = expert_outputs(layer.mlp.experts, hid, cand, dtype)
            r = RK.fit_layer(cand_out, t_slot, t_w, hid, W, B, keep, kept_slot, kept_local, K,
                             steps=steps, lr=lr, batch=batch, seed=seed + li)
            # NEVER SHIP A ROUTER WORSE THAN THE ONE WE STARTED FROM. The teacher's sliced router
            # is a valid, measured baseline; training is only justified where it beats it. This
            # is not hypothetical: when the mask prunes only experts the corpus never routed to,
            # the baseline loss is exactly zero and every gradient step can only move away from
            # it. Falling back costs nothing and removes a whole class of silent regression.
            if not (r["last"] < r["baseline"]):
                # Record what training WOULD have done before discarding it: on a near-zero
                # baseline the optimiser can be catastrophically worse (measured: -743770390%
                # at layer 4), and that number is the justification for this fallback existing.
                r["rejected_last"] = r["last"]
                r["rejected_improvement"] = r["improvement"]
                r["w"], r["kept_teacher"] = W[keep].float().cpu(), True
                r["last"] = r["baseline"]
                r["improvement"] = 0.0      # the shipped router IS the baseline, by definition
            trained[name] = {"weight": r["w"].cpu(), "bias": r["b"].cpu(), "keep": keep.cpu()}
            res["layers"][name] = {k: r[k] for k in
                                   ("baseline", "first", "last", "boundary", "improvement")}
            res["layers"][name]["tokens"] = n
            res["layers"][name]["kept_teacher"] = bool(r.get("kept_teacher", False))
            if r.get("kept_teacher"):
                res["layers"][name]["rejected_improvement"] = r["rejected_improvement"]
            tmp = trained_path.with_suffix(".tmp")
            torch.save(trained, tmp); tmp.replace(trained_path)
            tmp2 = state_path.with_suffix(".tmp")
            tmp2.write_text(json.dumps(res, indent=1)); tmp2.replace(state_path)
            print(f"  layer {li:>2} kept {len(keep):>3}/{n_exp}  loss {r['baseline']:.4e} -> "
                  f"{r['last']:.4e} ({r['improvement']:+6.1%})  boundary {r['boundary']:.1%}  "
                  f"n={n}{'  [kept teacher]' if r.get('kept_teacher') else ''}  "
                  f"rss {_rss_mb()/1024:.1f}G  [{(time.time()-t0)/60:.1f}m]", flush=True)
            del hid, cand, cand_out
        del layer
        CAP["hid"] = None
        reader.release(); gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    improved = [v["improvement"] for v in res["layers"].values()]
    fellback = [k for k, v in res["layers"].items() if v.get("kept_teacher")]
    if fellback:
        print(f"note: {len(fellback)} of {len(res['layers'])} layers kept the teacher's sliced "
              f"router because training did not beat it. That is the expected outcome wherever "
              f"the mask prunes only experts the corpus never routed to.", flush=True)
    bad = [k for k, v in res["layers"].items() if v["boundary"] > 0.05]
    print(f"router KD done: {len(improved)} layers, mean improvement "
          f"{sum(improved)/max(len(improved),1):+.1%}, min {min(improved, default=0):+.1%}",
          flush=True)
    if bad:
        print(f"WARNING: {len(bad)} layers sat on the candidate boundary (>5%); raise "
              f"--kept-candidates and rerun them: {bad[:6]}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/patrickd/models/MiMo-V2.6-Flash-RL")
    ap.add_argument("--chunks", default="artifacts/chunks")
    ap.add_argument("--mask", default="artifacts/masks/mask.json")
    ap.add_argument("--out", default="artifacts/masks")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--kept-candidates", type=int, default=32)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--chunk-index", type=int, default=0, help="(unused; kept for scripts)")
    ap.add_argument("--chunks-per-domain", type=int, default=1)
    ap.add_argument("--batches-per-chunk", type=int, default=8,
                    help="batches taken from each sampled chunk; 0 takes the whole chunk")
    ap.add_argument("--max-layers", type=int, default=None,
                    help="rehearsal only: stop after this many layers")
    ap.add_argument("--max-batches", type=int, default=None,
                    help="rehearsal only: stream only this many batches of the chunk")
    ap.add_argument("--max-seq", type=int, default=None,
                    help="rehearsal only: truncate each sequence to this length")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="skip the one-layer device check (not recommended)")
    ap.add_argument("--force", action="store_true",
                    help="run even while the calibration pass holds the GPU")
    a = ap.parse_args()
    if _busy() and not a.force:
        print("REFUSING: reap_run.service is active. Router KD streams all 48 layers and would "
              "contend for memory with the calibration pass -- which has been OOM-killed four "
              "times in this project. Wait for it, or pass --force.")
        return 2
    run(a.src, a.chunks, a.mask, a.out, device=a.device, tokens=a.tokens,
        kept_cand=a.kept_candidates, steps=a.steps, lr=a.lr, batch=a.batch,
        chunk_index=a.chunk_index, max_layers=a.max_layers, max_batches=a.max_batches,
        chunks_per_domain=a.chunks_per_domain, batches_per_chunk=a.batches_per_chunk,
        max_seq=a.max_seq, skip_preflight=a.skip_preflight)
    return 0


if __name__ == "__main__":
    sys.exit(main())
