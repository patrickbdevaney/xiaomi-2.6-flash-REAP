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
import json
import time
from pathlib import Path

import torch

import mimo_saliency as MS
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

    buckets = json.loads((chunks_dir / "manifest.json").read_text())["buckets"]
    chunk_files = sorted(chunks_dir.glob("chunk_*.pt"))[: smoke_chunks or None]
    assert chunk_files, f"no chunks in {chunks_dir}"

    MS.configure(buckets, n_layers=n_layers, n_experts=n_exp)
    MS.LAYER_INDEX.update({f"model.layers.{i}.mlp": i for i in range(n_layers)})
    MS.patch(_modeling(cfg))

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
        S = states[0]["embeds"].shape[1]
        if cfg._attn_implementation == "eager" and S > 2048:
            raise RuntimeError(
                f"refusing to run eager attention at S={S}: the score matrix is "
                f"{cfg.num_attention_heads * S * S * 2 / 2**30:.1f} GiB per forward. Run on CUDA "
                f"so flex_attention handles the sink bias, or shorten the chunk.")
        masks = masks_for(S, cfg.sliding_window, device, dtype)
        for li in range(n_layers):
            layer = build_layer(cfg, li, reader, dtype).to(device)
            atype = layer.attention_type
            pos = torch.arange(states[0]["embeds"].shape[1], device=device)[None]
            with torch.no_grad():
                for st in states:
                    MS.CTX.update({"layer": f"model.layers.{li}.mlp",
                                   "bucket": buckets.index(st["bucket"]),
                                   "valid": st["valid"].to(device).reshape(-1)})
                    hs = st["embeds"].to(device, dtype)
                    pe = _rope(cfg, atype, hs, pos, device, dtype)
                    out = layer(hs, attention_mask=masks[atype], position_ids=pos,
                                position_embeddings=pe)
                    st["embeds"] = out.to("cpu", torch.bfloat16)
                    MS.CTX["valid"] = None
                    del hs, out
            del layer
            reader.release(); gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        done.add(cf.name)
        _dump_acc(out_dir)
        state_path.write_text(json.dumps({"done": sorted(done)}, indent=1))
        print(f"chunk {cf.name}: {len(states)} batches x {n_layers} layers "
              f"in {time.time()-t0:.0f}s", flush=True)
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


def _dump_acc(out_dir: Path) -> None:
    torch.save({"acc": {k: {kk: vv.cpu() for kk, vv in v.items()} for k, v in MS.ACC.items()},
                "f_sum": MS.FACC.sum, "f_cnt": MS.FACC.cnt, "buckets": MS.BUCKETS},
               out_dir / "accumulators.pt")


def _load_acc(out_dir: Path, device) -> None:
    d = torch.load(out_dir / "accumulators.pt", map_location="cpu")
    MS.ACC.clear()
    for k, v in d["acc"].items():
        MS.ACC[k] = {kk: vv.to(device) for kk, vv in v.items()}
    MS.FACC.sum, MS.FACC.cnt = d["f_sum"], d["f_cnt"]


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
