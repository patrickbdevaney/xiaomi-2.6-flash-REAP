"""Gate the MiMo dequantisers against an INDEPENDENT implementation, on REAL checkpoint weights.

CLAUDE.md §2: a kernel is not believed because its comment is convincing. The MXFP4 nibble order,
the sign-bit position and the E8M0 bias are conventions -- get any of them wrong and the weights
are silently garbage while every shape still matches and nothing raises. So this compares against
`transformers.integrations.mxfp4.convert_moe_packed_tensors`, which was written by someone else
for a different model, on tensors read out of the actual 177.8 GB checkpoint.

A gate that passes against an absent file is worse than no gate, so the checkpoint's presence and
the tensors' dtypes are asserted before anything is compared.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "scripts")
from mimo_shards import ShardReader, dequant_mxfp4, dequant_fp8_block, MXFP4_BLOCK

SRC = Path.home() / "models" / "MiMo-V2.6-Flash-RL"
fail = 0


def check(name, ok, extra=""):
    global fail
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{('  ' + extra) if extra else ''}")
    fail += (not ok)


if not (SRC / "model.safetensors.index.json").exists():
    sys.exit(f"checkpoint not staged at {SRC} -- refusing to run a gate against nothing")

r = ShardReader(SRC)
EP = "model.layers.1.mlp.experts.0."

# ---- real tensors, real dtypes -------------------------------------------------------------
packed = r.get(EP + "gate_proj.weight")
scale = r.get(EP + "gate_proj.weight_scale")
check("expert weight is packed uint8", packed.dtype == torch.uint8, str(tuple(packed.shape)))
check("expert scale is uint8 (E8M0)", scale.dtype == torch.uint8, str(tuple(scale.shape)))
check("scale blocks match a 32-wide group",
      scale.shape[1] * MXFP4_BLOCK == packed.shape[1] * 2,
      f"{scale.shape[1]}*32 == {packed.shape[1]}*2")

mine = dequant_mxfp4(packed, scale, dtype=torch.float32)

# ---- the independent oracle -----------------------------------------------------------------
from transformers.integrations.mxfp4 import convert_moe_packed_tensors

R, Chalf = packed.shape
G, B = scale.shape[1], MXFP4_BLOCK // 2            # bytes per 32-value block
# The reference takes [..., G, B] and ends with a transpose written for GPT-OSS's layout, so
# give it a leading singleton and undo that transpose to get back to [R, C].
ref = convert_moe_packed_tensors(packed.view(1, R, G, B), scale.view(1, R, G),
                                 dtype=torch.float32).squeeze(0).transpose(0, 1).contiguous()
check("MXFP4 matches transformers' implementation exactly",
      torch.equal(mine, ref),
      f"max|d| {(mine - ref).abs().max().item():.3e}  shape {tuple(mine.shape)}")

# A wrong nibble order still produces a plausible-looking matrix, so prove the test discriminates.
swapped = torch.empty_like(mine)
lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
swapped[:, 0::2] = lut[(packed >> 4).long()]
swapped[:, 1::2] = lut[(packed & 0x0F).long()]
swapped = torch.ldexp(swapped.view(R, G, MXFP4_BLOCK),
                      (scale.to(torch.int32) - 127)[:, :, None]).view(R, Chalf * 2)
check("the gate would CATCH a swapped nibble order", not torch.equal(swapped, ref),
      f"differs in {(swapped != ref).float().mean().item()*100:.1f}% of elements")

# ---- sanity that the result is a plausible weight matrix -------------------------------------
check("no NaN/Inf after dequant", torch.isfinite(mine).all())
check("weights are centred and non-trivial",
      abs(mine.mean().item()) < 0.01 and 1e-4 < mine.std().item() < 1.0,
      f"mean {mine.mean().item():+.5f}  std {mine.std().item():.5f}")

# ---- FP8 path, on the real fused qkv ----------------------------------------------------------
qw = r.get("model.layers.1.self_attn.qkv_proj.weight")
qs = r.get("model.layers.1.self_attn.qkv_proj.weight_scale_inv")
check("qkv is F8_E4M3 with F32 block scales",
      qw.dtype == torch.float8_e4m3fn and qs.dtype == torch.float32,
      f"{tuple(qw.shape)} / {tuple(qs.shape)}")
q = dequant_fp8_block(qw, qs, dtype=torch.float32)
check("FP8 block scale tiles to the full matrix", q.shape == qw.shape, str(tuple(q.shape)))
check("FP8 dequant is finite and non-trivial",
      torch.isfinite(q).all() and q.std().item() > 1e-5, f"std {q.std().item():.5f}")

# ---- o_proj must arrive untouched -------------------------------------------------------------
o = r.get("model.layers.1.self_attn.o_proj.weight")
check("o_proj is BF16 and unquantised (it is in ignored_layers)",
      o.dtype == torch.bfloat16 and tuple(o.shape) == (4096, 8192), str(tuple(o.shape)))

# ---- load_module wires all three together ------------------------------------------------------
m = r.load_module("model.layers.1.self_attn")
check("load_module resolves every attention tensor",
      all(v.dtype == torch.bfloat16 for v in m.values()) and "o_proj.weight" in m
      and "qkv_proj.weight" in m and not any("scale" in k for k in m),
      f"{sorted(m)}")
r.release()

print("\nGATE " + ("PASS" if not fail else f"FAIL ({fail})"))
sys.exit(1 if fail else 0)
