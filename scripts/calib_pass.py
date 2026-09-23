"""The calibration pass: sweep the unpruned MiMo layer by layer, accumulating saliency.

THE SHAPE OF THE PROBLEM. The model is 177.8 GB and Thor has ~117 GiB, so it is never all
resident. The pass is layer-sequential: hold ONE decoder layer at a time, push every sample's
hidden states through it, replace them with its output, free it, move on. The same structure the
GLM pipeline used to calibrate a 311 GB model on this box.

MODALITY-AGNOSTIC BY CONSTRUCTION. This consumes chunks of `inputs_embeds`, not token ids, so
the vision / audio / video towers live in the chunk builder and never here. That is deliberate:
the driver cannot accidentally drop a modality, because it never knows which modality a row came
from -- only which BUCKET, which is what the accumulators are keyed on. It also means the towers
are run exactly once rather than once per layer.

WHY CHUNKS, AND WHAT THE TRADE IS. Hidden states for the whole corpus are ~450 GB at bf16, so
they cannot all be resident. Chunking bounds them, at the cost of re-reading the 177.8 GB of
weights once per chunk. That is the right way round: weights stream from NVMe at GB/s, whereas
holding hidden states costs RAM we do not have. Make chunks as large as memory allows to keep
the re-read count down.

RESUMABILITY IS NOT OPTIONAL. This is a multi-hour pass whose F-matrix cannot be recovered from
a partial result, so accumulators are checkpointed after every chunk and a restart skips the
chunks already folded in. A crash at hour six must not cost hour one.
"""
from __future__ import annotations

import argparse
import gc
import os
import json
import time
from pathlib import Path

import torch

import mimo_saliency as MS
import verify_pass as VP
from hope_fmatrix import FAccumulator
from mimo_shards import ShardReader


def build_layer(cfg, li: int, reader: ShardReader, dtype=torch.bfloat16):
    """One decoder layer, weights streamed in and assigned without a second copy."""
    from accelerate import init_empty_weights
    import transformers.models.auto.modeling_auto  # noqa: F401  (registers auto classes)
    mod = _modeling(cfg)
    with init_empty_weights():
        layer = mod.MiMoV2DecoderLayer(
            cfg, li, attention_projection_layout=getattr(cfg, "attention_projection_layout", None))
    sd = reader.load_module(f"model.layers.{li}.", dtype)
    missing, unexpected = layer.load_state_dict(sd, strict=False, assign=True)
    # STRICT IN EFFECT, but reported rather than raised on `missing`, because buffers (rotary
    # inv_freq and friends) legitimately are not in the checkpoint. An UNEXPECTED key means the
    # prefix matched something that is not this layer, which is never benign.
    assert not unexpected, f"layer {li}: unexpected keys {unexpected[:5]}"
    real_missing = [k for k in missing if not k.endswith(("inv_freq", "attention_scaling"))]
    assert not real_missing, f"layer {li}: missing weights {real_missing[:5]}"
    return layer.to(dtype).eval()


_MOD_CACHE = {}


def _modeling(cfg):
    """Import the checkpoint's own modelling module (trust_remote_code), once."""
    if "m" not in _MOD_CACHE:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        cls = get_class_from_dynamic_module(
            "modeling_mimo_v2.MiMoV2DecoderLayer", str(cfg._name_or_path))
        _MOD_CACHE["m"] = __import__(cls.__module__, fromlist=["*"])
    return _MOD_CACHE["m"]


def masks_for(seq: int, window: int, device, dtype):
    """Additive masks: causal for the 9 full-attention layers, causal+banded for the 39 SWA ones.

    MiMo alternates them by `hybrid_layer_pattern`, and the two are NOT interchangeable -- a full
    causal mask on an SWA layer silently gives that layer global attention and changes every
    downstream hidden state.
    """
    i = torch.arange(seq, device=device)
    causal = i[:, None] >= i[None, :]
    band = causal & ((i[:, None] - i[None, :]) < window)
    neg = torch.finfo(dtype).min
    def to_add(m):
        return torch.where(m, torch.zeros((), device=device, dtype=dtype),
                           torch.full((), neg, device=device, dtype=dtype))[None, None]
    return {"full_attention": to_add(causal), "sliding_window_attention": to_add(band)}


def run(src, chunks_dir, out_dir, device="cuda", dtype=torch.bfloat16,
        smoke_layers: int = 0, smoke_chunks: int = 0):
    src, chunks_dir, out_dir = Path(src), Path(chunks_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(src, trust_remote_code=True)
    cfg._name_or_path = str(src)
    # ATTENTION BACKEND IS NOT A FREE CHOICE HERE, and "sdpa" is the trap.
    #
    # `add_swa_attention_sink_bias` is true and `add_full_attention_sink_bias` is false, so 39 of
    # the 48 layers carry a sink bias and 9 do not. Under sdpa the model silently falls back to
    # EAGER on exactly those 39 -- correct, but it materialises a full [B, H, S, S] score matrix
    # despite the window being 128 wide. At S=16,384 that is 64 heads x 16384^2 x 2 B = 34 GB per
    # forward, per layer, for a band that holds 0.8% of it.
    #
    # flex_attention takes the other branch, receiving `s_aux` and `sliding_window` directly, and
    # is the only available backend that handles the sink bias without densifying (flash_attn is
    # not installed). It requires CUDA -- "Attention sinks cannot be run on CPU with flex
    # attention" -- so CPU smokes use eager, which is fine at smoke sequence lengths and
    # catastrophic at real ones.
    cfg._attn_implementation = "flex_attention" if str(device).startswith("cuda") else "eager"

    n_layers = smoke_layers or cfg.num_hidden_layers
    n_exp = cfg.n_routed_experts
    # DISCOVERED, not assumed -- the same lesson as the dsv4 N_ROUTED constant.
    reader = ShardReader(src)
    seen = 0
    while f"model.layers.1.mlp.experts.{seen}.gate_proj.weight" in reader.map:
        seen += 1
    assert seen == n_exp, f"config says {n_exp} experts, checkpoint has {seen}"

    # The embedding table, and only the table: chunks carry token ids plus already-towered media
    # embeddings, so the pass never loads the vision or audio towers. 1.25 GB resident.
    emb_w = ShardReader(src).load_module("model.embed_tokens.", dtype)["weight"].to(device)

    def to_embeds(st):
        e = torch.nn.functional.embedding(st["ids"].to(device).long(), emb_w)
        me = st.get("media_embeds")
        if me is not None:
            mask = st["ids"].to(device).eq(st["media_token_id"])
            assert int(mask.sum()) == me.shape[0], (
                f"{int(mask.sum())} placeholders but {me.shape[0]} media embeddings in "
                f"{st['bucket']} -- refusing to splice a misaligned row")
            e[mask] = me.to(device, dtype)
        return e

    buckets = json.loads((chunks_dir / "manifest.json").read_text())["buckets"]
    chunk_files = sorted(chunks_dir.glob("chunk_*.pt"))[: smoke_chunks or None]
    assert chunk_files, f"no chunks in {chunks_dir}"

    MS.configure(buckets, n_layers=n_layers, n_experts=n_exp, device=device)
    MS.LAYER_INDEX.update({f"model.layers.{i}.mlp": i for i in range(n_layers)})
    MS.patch(_modeling(cfg))

    prev_snap = None
    state_path = out_dir / "pass_state.json"
    done = set(json.loads(state_path.read_text())["done"]) if state_path.exists() else set()
    if done:
        _load_acc(out_dir, device)
        print(f"resuming: {len(done)} chunks already folded in", flush=True)

    for cf in chunk_files:
        if cf.name in done:
            continue
        t0 = time.time()
        states = torch.load(cf, map_location="cpu")
        S = states[0]["ids"].shape[1]
        if cfg._attn_implementation == "eager" and S > 2048:
            raise RuntimeError(
                f"refusing to run eager attention at S={S}: the score matrix is "
                f"{cfg.num_attention_heads * S * S * 2 / 2**30:.1f} GiB per forward. Run on CUDA "
                f"so flex_attention handles the sink bias, or shorten the chunk.")
        masks = masks_for(S, cfg.sliding_window, device, dtype)
        for li in range(n_layers):
            layer = build_layer(cfg, li, reader, dtype).to(device)
            atype = layer.attention_type
            pos = torch.arange(S, device=device)[None]
            with torch.no_grad():
                for st in states:
                    MS.CTX.update({"layer": f"model.layers.{li}.mlp",
                                   "bucket": buckets.index(st["bucket"]),
                                   "valid": st["valid"].to(device).reshape(-1)})
                    # Layer 0 embeds from ids; every later layer consumes the previous
                    # layer's output, which is carried in `hs`.
                    hs = (to_embeds(st) if li == 0 else st["hs"].to(device, dtype))
                    pe = _rope(cfg, atype, hs, pos, device, dtype)
                    out = layer(hs, attention_mask=masks[atype], position_ids=pos,
                                position_embeddings=pe)
                    st["hs"] = out.to("cpu", torch.bfloat16)
                    MS.CTX["valid"] = None
                    del hs, out
            del layer
            reader.release(); gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        for st in states:
            st.pop("hs", None)          # free the activations before the next chunk loads
        # VERIFY *BEFORE* CHECKPOINTING -- and it means before, which the previous ordering did
        # not do: it dumped the accumulators and marked the chunk done, and only then verified.
        # A corrupted accumulator was therefore persisted and its chunk recorded as folded in
        # before anything checked it, so the resume path would load the corruption and SKIP the
        # chunk that produced it, baking the fault in permanently and silently. Verifying first
        # means a bad chunk leaves the last good checkpoint untouched and gets reprocessed.
        # HOPE's F cannot be recovered from a partial result, so this ordering is the difference
        # between losing one chunk and losing the run.
        status = VP.verify(MS.ACC, MS.FACC.sum, MS.FACC.cnt, buckets, cfg.num_experts_per_tok,
                           prev=prev_snap)
        prev_snap = VP.snapshot(MS.ACC)
        # EXPERT COVERAGE is the number that says whether the corpus is big enough. An expert
        # never routed to has zero saliency and zero F mass, so REAP and HOPE both prune it
        # ARBITRARILY rather than on evidence -- and with 256 experts and top-8 routing, a short
        # corpus leaves most of them dark. Reported every chunk so a thin corpus is visible early
        # instead of at the end of a 34-hour pass.
        _dump_acc(out_dir)
        done.add(cf.name)
        _atomic_write(state_path, json.dumps({"done": sorted(done)}, indent=1))
        _atomic_write(out_dir / "status.json", json.dumps(
            {"chunks_done": len(done), "chunk": cf.name, **status}, indent=1))
        # MiMo layer 0 is DENSE, so a run restricted to it accumulates nothing and the
        # coverage reduction is over an empty set. Never let the progress line be the thing
        # that crashes a 34-hour pass.
        cov = [float((v["cnt"].sum(0) > 0).float().mean()) for v in MS.ACC.values()]
        covtxt = (f"min {min(cov):.1%} mean {sum(cov)/len(cov):.1%}" if cov
                  else "n/a (no MoE layer in range)")
        print(f"chunk {cf.name}: {len(states)} batches x {n_layers} layers "
              f"in {time.time()-t0:.0f}s | expert coverage {covtxt}", flush=True)
        del states; gc.collect()
    return len(done)


_ROPE_CACHE = {}


def _rope(cfg, atype, hs, pos, device, dtype):
    key = (atype, hs.shape[1])
    if key not in _ROPE_CACHE:
        mod = _modeling(cfg)
        emb = mod.MiMoV2RotaryEmbedding(config=cfg,
                                        is_swa=(atype == "sliding_window_attention")).to(device)
        _ROPE_CACHE[key] = tuple(t.to(dtype) for t in emb(hs, pos))
    return _ROPE_CACHE[key]


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _dump_acc(out_dir: Path) -> None:
    """Write the accumulators ATOMICALLY.

    A 34-hour unattended run will eventually be interrupted mid-write. torch.save straight onto
    the live path leaves a truncated accumulators.pt that pass_state.json still points at, and
    the resume path then either throws or -- worse -- loads a partial tensor set. Writing to a
    temp file and renaming makes the swap atomic on the same filesystem, so a crash at any
    instant leaves either the previous complete checkpoint or the new one, never a torn file.
    """
    dst = out_dir / "accumulators.pt"
    tmp = dst.with_suffix(".pt.tmp")
    # .cpu() the F tensors too. Saving them as CUDA tensors and reloading with
    # map_location="cpu" put FACC.sum on the host while update() kept feeding it device
    # indices, so the FIRST chunk after any resume died in index_add_ on a device mismatch --
    # the resume path was written but never once executed end to end.
    torch.save({"acc": {k: {kk: vv.cpu() for kk, vv in v.items()} for k, v in MS.ACC.items()},
                "f_sum": MS.FACC.sum.cpu(), "f_cnt": MS.FACC.cnt.cpu(),
                "buckets": MS.BUCKETS}, tmp)
    os.replace(tmp, dst)


def _load_acc(out_dir: Path, device) -> None:
    d = torch.load(out_dir / "accumulators.pt", map_location="cpu")
    MS.ACC.clear()
    for k, v in d["acc"].items():
        MS.ACC[k] = {kk: vv.to(device) for kk, vv in v.items()}
    MS.FACC.sum = d["f_sum"].to(MS.FACC.device)
    MS.FACC.cnt = d["f_cnt"].to(MS.FACC.device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path.home() / "models" / "MiMo-V2.6-Flash-RL"))
    ap.add_argument("--chunks", default="artifacts/chunks")
    ap.add_argument("--out", default="artifacts/saliency")
    ap.add_argument("--smoke-layers", type=int, default=0)
    ap.add_argument("--smoke-chunks", type=int, default=0)
    a = ap.parse_args()
    n = run(a.src, a.chunks, a.out, smoke_layers=a.smoke_layers, smoke_chunks=a.smoke_chunks)
    print(f"pass complete over {n} chunks")
