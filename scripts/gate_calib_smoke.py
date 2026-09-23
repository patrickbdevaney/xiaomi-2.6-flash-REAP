"""End-to-end smoke of the calibration pass on ONE real layer. CPU, no GPU needed.

This is the first thing in the project that touches the actual model, and it is deliberately the
cheapest possible version of the expensive pass: build one decoder layer from the real 177.8 GB
checkpoint, push a tiny batch through it, and assert that the statistics the whole REAP depends
on actually filled. Everything before this was gated against synthetic fixtures or another
implementation; this is the first gate against the model itself.
"""
import sys, time
from pathlib import Path

import torch

sys.path.insert(0, "scripts")
import mimo_saliency as MS
import calib_pass as CP
from mimo_shards import ShardReader

SRC = Path.home() / "models" / "MiMo-V2.6-Flash-RL"
fail = 0


def check(name, ok, extra=""):
    global fail
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{('  ' + extra) if extra else ''}", flush=True)
    fail += (not ok)


if not (SRC / "model.safetensors.index.json").exists():
    sys.exit(f"checkpoint not staged at {SRC} -- refusing to gate against nothing")

from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(SRC, trust_remote_code=True)
cfg._name_or_path = str(SRC)
cfg._attn_implementation = "eager"   # CPU smoke; the real pass uses flex_attention on CUDA (see calib_pass)
check("config is MiMoV2 with the expected shape",
      cfg.num_hidden_layers == 48 and cfg.n_routed_experts == 256
      and cfg.num_experts_per_tok == 8 and cfg.hidden_size == 4096,
      f"{cfg.num_hidden_layers}L {cfg.n_routed_experts}E top-{cfg.num_experts_per_tok}")

LI = 1                                   # an SWA MoE layer (hybrid_layer_pattern[1] == 1)
reader = ShardReader(SRC)
t0 = time.time()
layer = CP.build_layer(cfg, LI, reader, torch.float32)
print(f"  (built layer {LI} from real shards in {time.time()-t0:.0f}s)", flush=True)
check("layer is SWA as the hybrid pattern says",
      layer.attention_type == "sliding_window_attention", layer.attention_type)
check("MoE has all 256 experts materialised", len(layer.mlp.experts) == 256)
w = layer.mlp.experts[0].gate_proj.weight
check("expert weights dequantised to a real dtype, finite, non-trivial",
      w.dtype == torch.float32 and torch.isfinite(w).all() and w.std().item() > 1e-4,
      f"{tuple(w.shape)} std {w.std().item():.5f}")

MS.configure(["code", "ballast"], n_layers=cfg.num_hidden_layers, n_experts=cfg.n_routed_experts)
MS.LAYER_INDEX[f"model.layers.{LI}.mlp"] = LI
MS.patch(CP._modeling(cfg))

B, S = 2, 24
hs = torch.randn(B, S, cfg.hidden_size, dtype=torch.float32) * 0.02
pos = torch.arange(S)[None]
masks = CP.masks_for(S, cfg.sliding_window, "cpu", torch.float32)
valid = torch.ones(B, S, dtype=torch.bool); valid[1, -6:] = False     # exercise the mask path

MS.CTX.update({"layer": f"model.layers.{LI}.mlp", "bucket": 0, "valid": valid.reshape(-1)})
pe = CP._rope(cfg, layer.attention_type, hs, pos, "cpu", torch.float32)
t0 = time.time()
with torch.no_grad():
    out = layer(hs, attention_mask=masks[layer.attention_type], position_ids=pos,
                position_embeddings=pe)
print(f"  (forward {B}x{S} in {time.time()-t0:.1f}s)", flush=True)

check("output shape and finiteness", out.shape == hs.shape and torch.isfinite(out).all(),
      str(tuple(out.shape)))
check("the layer actually changed the hidden states", not torch.allclose(out, hs))

acc = MS.ACC[f"model.layers.{LI}.mlp"]
routed = acc["cnt"].sum().item()
check("routed-token count is exactly valid_tokens * top_k",
      routed == valid.sum().item() * cfg.num_experts_per_tok,
      f"{routed:.0f} == {valid.sum().item()} * {cfg.num_experts_per_tok}")
check("masked tokens were excluded", routed < B * S * cfg.num_experts_per_tok,
      f"{routed:.0f} < {B*S*cfg.num_experts_per_tok}")
check("more than one expert fired", int((acc["cnt"].sum(0) > 0).sum()) > 1,
      f"{int((acc['cnt'].sum(0) > 0).sum())} of 256 experts hit")

F = MS.FACC.finalize(LI)
import numpy as np
diag_acc = (acc["sq"].sum(0) / acc["cnt"].sum(0).clamp(min=1)).cpu().numpy()
live = acc["cnt"].sum(0).cpu().numpy() > 0
check("F diagonal == sq/cnt on real weights",
      np.allclose(np.diag(F)[live], diag_acc[live], rtol=1e-9),
      f"max|d| {np.abs(np.diag(F)[live]-diag_acc[live]).max():.2e}")
check("F has real off-diagonal structure", np.abs(F - np.diag(np.diag(F))).max() > 0,
      f"max off-diag {np.abs(F - np.diag(np.diag(F))).max():.4f}")

reader.release()
print("\nGATE " + ("PASS" if not fail else f"FAIL ({fail})"))
sys.exit(1 if fail else 0)
