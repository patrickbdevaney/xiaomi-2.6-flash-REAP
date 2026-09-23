"""Streaming reader for the MiMo-V2.6-Flash checkpoint: one module resident at a time.

The unpruned model is 177.8 GB against Thor's ~117 GiB, so it is never all resident. The
calibration pass sweeps layer by layer, and this hands back one layer's weights at a time,
dequantised, then lets them go. Same shape as the GLM pipeline's ShardReader, with MiMo's two
storage formats instead of GLM's one.

TWO DEQUANTISERS, because MiMo mixes formats within a single layer:

  routed experts   MXFP4 -- U8 packed E2M1 nibbles + U8 E8M0 per-32 scales
                   `gate_proj.weight` U8 [2048, 2048] with `.weight_scale` U8 [2048, 128]
                   (logical [2048, 4096]: two nibbles per byte, 4096/32 = 128 scale blocks)
  qkv_proj         FP8 E4M3 + F32 per-128x128 RECIPROCAL scale (`weight_scale_inv`)
  o_proj           BF16, untouched -- it is in `ignored_layers`, which is why it is 3.2 GB of
                   every token's read and why the NVFP4 overlay targets it

The MXFP4 convention is NOT guessed. It matches transformers' own
`integrations/mxfp4.py`: low nibble first into even positions, sign in bit 3 of the nibble, and
`ldexp` by (scale_byte - 127). gate_mimo_shards.py asserts equality against that implementation
on REAL checkpoint weights, which is the only reason to believe any of it.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

# E2M1: magnitude in bits 0-2, sign in bit 3. Index 8 is -0.0, which decodes to 0.0 anyway.
FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)
MXFP4_BLOCK = 32


def dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor,
                  dtype=torch.bfloat16) -> torch.Tensor:
    """packed U8 [R, C/2], scale U8 [R, C/32]  ->  [R, C].

    Done in float32 before the cast: the E8M0 exponent spans 2^-127..2^127 and intermediate
    values can leave bf16's range even when the result does not.
    """
    assert packed.dtype == torch.uint8 and scale.dtype == torch.uint8, \
        f"expected packed/scale uint8, got {packed.dtype}/{scale.dtype}"
    R, Chalf = packed.shape
    C = Chalf * 2
    G = C // MXFP4_BLOCK
    assert scale.shape == (R, G), f"scale {tuple(scale.shape)} != expected {(R, G)}"
    lut = FP4_VALUES.to(packed.device)
    out = torch.empty(R, C, dtype=torch.float32, device=packed.device)
    out[:, 0::2] = lut[(packed & 0x0F).long()]      # LOW nibble -> even positions
    out[:, 1::2] = lut[(packed >> 4).long()]
    exp = (scale.to(torch.int32) - 127)             # E8M0 bias
    out = torch.ldexp(out.view(R, G, MXFP4_BLOCK), exp[:, :, None]).view(R, C)
    return out.to(dtype)


def dequant_fp8_block(w: torch.Tensor, scale_inv: torch.Tensor, block: int = 128,
                      dtype=torch.bfloat16) -> torch.Tensor:
    """FP8 E4M3 + per-block F32 scale -> dtype. The stored value is the scale itself here;
    `_inv` is the checkpoint's naming, and the GLM pipeline established the same multiply."""
    out_f, in_f = w.shape
    s = scale_inv.to(torch.float32)
    s = s.repeat_interleave(block, 0)[:out_f].repeat_interleave(block, 1)[:, :in_f]
    return (w.to(torch.float32) * s).to(dtype)


class ShardReader:
    """Lazily mmap the shards and hand back one module's tensors at a time."""

    def __init__(self, src):
        self.src = Path(src)
        self.map: dict[str, str] = json.loads(
            (self.src / "model.safetensors.index.json").read_text())["weight_map"]
        self._open: dict[str, object] = {}

    def _f(self, shard: str):
        from safetensors import safe_open
        if shard not in self._open:
            self._open[shard] = safe_open(str(self.src / shard), framework="pt", device="cpu")
        return self._open[shard]

    def get(self, name: str) -> torch.Tensor:
        return self._f(self.map[name]).get_tensor(name)

    @staticmethod
    def _norm(prefix: str) -> str:
        """Force a trailing dot. `model.layers.1` is a prefix of `model.layers.11`, so a bare
        prefix would silently fold eleven other layers' tensors into layer 1's state dict --
        shapes would still match, load_state_dict would still succeed, and every statistic in the
        run would be wrong."""
        return prefix if prefix.endswith(".") else prefix + "."

    def names_for(self, prefix: str) -> list[str]:
        return [k for k in self.map if k.startswith(self._norm(prefix))]

    def release(self) -> None:
        """Close every shard handle.

        safe_open holds a live mmap. While the mapping exists its faulted-in pages CANNOT be
        reclaimed -- drop_caches is a no-op against them -- so holding all 65 handles open across
        a layer sweep accumulates the whole 177.8 GB as unreclaimable page cache. Callers must
        ensure no returned tensor still views the mapping; every path below copies.
        """
        self._open.clear()

    def load_module(self, prefix: str, dtype=torch.bfloat16) -> dict[str, torch.Tensor]:
        """{relative_name: tensor}, with both quantised formats resolved."""
        out: dict[str, torch.Tensor] = {}
        prefix = self._norm(prefix)
        names = self.names_for(prefix)
        aux = {n for n in names if n.endswith(("weight_scale", "weight_scale_inv"))}
        for n in names:
            if n in aux:
                continue
            rel = n[len(prefix):].lstrip(".")
            t = self.get(n)
            mx, fp8 = n + "_scale", n + "_scale_inv"
            if mx in aux:
                t = dequant_mxfp4(t, self.get(mx), dtype=dtype)
            elif fp8 in aux:
                t = dequant_fp8_block(t, self.get(fp8), dtype=dtype)
            else:
                # copy=True matters: .to(dtype) on a tensor ALREADY in that dtype returns self,
                # which is a view into the mmap. Releasing the handle under a live view is a
                # use-after-unmap; keeping it open is what pins the cache.
                t = t.to(dtype, copy=True) if t.is_floating_point() else t.clone()
            out[rel] = t
        return out
