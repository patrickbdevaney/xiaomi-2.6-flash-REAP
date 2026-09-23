"""Gate the chunk builder on the REAL embedding table and the REAL vision tower.

The assertion that matters is the last one: an image row must produce embeddings that DIFFER
from what the same token ids would give without the tower. That is the difference between
calibrating the vision experts and calibrating nothing, and it is not visible in any shape,
count or dtype -- which is precisely why the corpus spec's rule ("text descriptions of images
route like text and protect nothing") needs a test rather than a comment.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, "scripts")
from chunk_builder import Embedder, ChunkWriter

SRC = Path.home() / "models" / "MiMo-V2.6-Flash-RL"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
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
cfg._attn_implementation = "flex_attention" if DEV == "cuda" else "eager"

E = Embedder(SRC, cfg, device=DEV)
check("embedding table loaded at the config's vocab size",
      E.embed.weight.shape == (cfg.vocab_size, cfg.hidden_size),
      str(tuple(E.embed.weight.shape)))
check("modality token ids all present",
      all(v is not None for v in E.ids.values()), str(E.ids))

ids = torch.tensor([[100, 200, 300, 400]])
emb = E.embed_batch(ids)
check("text embedding matches a direct table lookup",
      torch.equal(emb.cpu().float(), E.embed.weight[ids.flatten()].reshape(1, 4, -1).cpu().float()),
      str(tuple(emb.shape)))

# ---- the real vision tower on a real image --------------------------------------------------
from transformers import AutoProcessor
from PIL import Image
import numpy as np
proc = AutoProcessor.from_pretrained(SRC, trust_remote_code=True)
img = Image.fromarray((np.random.default_rng(0).random((224, 224, 3)) * 255).astype("uint8"))
vis = proc.image_processor(images=[img], return_tensors="pt")
merge = cfg.vision_config["spatial_merge_size"]
npatch = vis["image_grid_thw"].prod(-1).sum().item() // (merge ** 2)
ids_img = torch.tensor([[100] + [E.ids["image"]] * npatch + [200]])
check("image placeholders sized from the real grid", npatch > 0, f"{npatch} image tokens")

emb_img = E.embed_batch(ids_img, media={"image": {"pixel_values": vis["pixel_values"],
                                                  "grid_thw": vis["image_grid_thw"]}})
plain = E.embed(ids_img.to(DEV)).to(emb_img.dtype)
mask = ids_img.to(DEV).eq(E.ids["image"])
check("vision tower output was actually spliced in",
      not torch.allclose(emb_img[mask].float(), plain[mask].float()),
      f"mean|d| {(emb_img[mask]-plain[mask]).abs().float().mean().item():.4f}")
check("non-media positions are untouched",
      torch.equal(emb_img[~mask].float(), plain[~mask].float()))
check("image embeddings are finite and non-degenerate",
      torch.isfinite(emb_img).all() and emb_img[mask].float().std().item() > 1e-3,
      f"std {emb_img[mask].float().std().item():.4f}")

# supplying media with no placeholder must RAISE, not silently drop the tower output
try:
    E.embed_batch(torch.tensor([[100, 200]]),
                  media={"image": {"pixel_values": vis["pixel_values"],
                                   "grid_thw": vis["image_grid_thw"]}})
    check("media without placeholders raises", False)
except AssertionError as e:
    check("media without placeholders raises", "silently discarded" in str(e))

# ---- the guard against a text-only media bucket -----------------------------------------------
import tempfile
with tempfile.TemporaryDirectory() as td:
    w = ChunkWriter(td, ["code", "image"], tokens_per_chunk=10**9)
    w.add(torch.zeros(1, 4, dtype=torch.long), torch.ones(1, 4, dtype=torch.bool), "image")
    try:
        w.close(); check("a text-only image bucket is refused", False)
    except RuntimeError as e:
        check("a text-only image bucket is refused", "NONE came from real media" in str(e))

with tempfile.TemporaryDirectory() as td:
    w = ChunkWriter(td, ["code", "image"], tokens_per_chunk=10**9)
    IMG = E.ids["image"]
    w.add(torch.full((1, 4), IMG, dtype=torch.long), torch.ones(1, 4, dtype=torch.bool), "image",
          media_embeds=torch.zeros(4, cfg.hidden_size), media_token_id=IMG)
    w.add(torch.zeros(1, 6, dtype=torch.long), torch.ones(1, 6, dtype=torch.bool), "code")
    w.close()
    man = __import__("json").loads((Path(td) / "manifest.json").read_text())
    check("manifest records per-bucket token and media counts",
          man["tokens_by_bucket"] == {"code": 6, "image": 4}
          and man["media_tokens_by_bucket"]["image"] == 4, str(man["tokens_by_bucket"]))
    rt = torch.load(Path(td) / "chunk_00000.pt", map_location="cpu")
    check("chunk round-trips for the pass",
          len(rt) == 2 and rt[0]["ids"].dtype == torch.int32
          and rt[0]["media_embeds"].shape[0] == 4 and rt[1]["media_embeds"] is None,
          "ids + media_embeds, text rows carry none")
    # A mismatched placeholder count must RAISE: splicing a misaligned row would attach every
    # later media embedding to the wrong token, silently.
    w2 = ChunkWriter(td, ["image"], tokens_per_chunk=10**9)
    try:
        w2.add(torch.full((1, 4), IMG, dtype=torch.long), torch.ones(1, 4, dtype=torch.bool),
               "image", media_embeds=torch.zeros(3, cfg.hidden_size), media_token_id=IMG)
        check("placeholder/embedding count mismatch raises", False)
    except AssertionError as e:
        check("placeholder/embedding count mismatch raises", "placeholders" in str(e))

print("\nGATE " + ("PASS" if not fail else f"FAIL ({fail})"))
sys.exit(1 if fail else 0)
