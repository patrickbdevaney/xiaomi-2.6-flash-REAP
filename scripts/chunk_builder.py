"""Build calibration chunks: text and REAL media, embedded once, tagged by bucket.

This owns the towers, and therefore owns the omni capability. `calib_pass` consumes
`inputs_embeds` and never learns which modality a row came from -- which is exactly why the
guarantee has to live here, and why it is asserted rather than assumed.

THE FAILURE THIS FILE EXISTS TO MAKE IMPOSSIBLE. MiMo routes audio- and video-derived tokens
through the same 256 experts as text. An expert specialised in those distributions shows exactly
zero gated mass against a corpus that contains none, so it is not under-weighted -- it is
guaranteed pruned, and pruned first, silently, while every text benchmark still passes. The
corpus spec states the rule for images already: "rows must carry real images, which are pushed
through the real processor. Text descriptions of images route like text and protect nothing."
A transcript protects the audio path no better than a caption protects the vision path.

So: a chunk declared `audio` MUST contain audio-token placeholders that were replaced by real
audio-encoder output, and `emit` refuses to write one that does not. A silently text-only audio
bucket would poison the whole REAP and leave no trace.

HOW EMBEDDING WORKS HERE. The model builds `inputs_embeds` from `embed_tokens`, then overwrites
the rows at each modality's placeholder token with tower output
(`_replace_modal_embeddings_inplace`). We do the same, with only the embedding table and the
towers resident -- never the 48 decoder layers, which is what keeps this inside the envelope.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import torch

from mimo_shards import ShardReader



def _build_tower(factory, sd: dict, device, dtype):
    """Construct a tower with REAL init, load the checkpoint over it, zero what is absent.

    init_empty_weights leaves anything the checkpoint lacks on the meta device, and `.to(device)`
    then dies with "Cannot copy out of meta tensor" -- the same trap channel_saliency hit. Real
    init avoids that, but then a tensor the checkpoint does not carry would silently keep its
    RANDOM initialisation, which is worse than crashing.

    So: real init, load, then explicitly zero every PARAMETER the checkpoint did not supply, and
    leave BUFFERS alone because they are derived (rotary inv_freq recomputes correctly at init).
    A missing bias means the shipped model has none, and zero is exactly that. A missing weight
    MATRIX means something is wrong with our prefix or the checkpoint, so that is fatal rather
    than zeroed -- zeroing it would produce a tower that runs, returns finite numbers, and is
    meaningless.
    """
    tower = factory()
    missing, unexpected = tower.load_state_dict(sd, strict=False, assign=True)
    assert not unexpected, f"unexpected keys {unexpected[:4]}"
    buffers = set(dict(tower.named_buffers()))
    params = dict(tower.named_parameters())
    bad = [k for k in missing if k not in buffers and not k.endswith(".bias")]
    assert not bad, f"checkpoint is missing non-bias weights {bad[:4]} -- refusing to fabricate them"
    with torch.no_grad():
        for k in missing:
            if k in params:
                params[k].zero_()
    return tower.to(device=device, dtype=dtype).eval()


class Embedder:
    """embed_tokens + the vision and audio towers. ~3.5 GB resident, no decoder layers."""

    def __init__(self, src, cfg, device="cuda", dtype=torch.bfloat16):
        self.cfg, self.device, self.dtype = cfg, device, dtype
        self.reader = ShardReader(src)
        w = self.reader.load_module("model.embed_tokens.", dtype)["weight"]
        self.embed = torch.nn.Embedding.from_pretrained(w.to(device), freeze=True)
        self._visual = None
        self._audio = None
        self._speech = None
        self.ids = {
            "image": getattr(cfg, "image_token_id", None),
            "video": getattr(cfg, "video_token_id", None),
            "audio": getattr(cfg, "audio_token_id", None),
        }

    def _lazy(self, which):
        """Towers are built on first use: a text-only run should not pay 2.3 GB for them."""
        import calib_pass as CP
        mod = CP._modeling(self.cfg)
        # The sub-configs are plain dicts in config.json; the model wraps them in a namespace
        # before constructing the towers (`_as_namespace`). Passing the dict straight through
        # fails on the first getattr, so use the model's own helper rather than a local copy.
        if which == "visual" and self._visual is None:
            self._visual = _build_tower(
                lambda: mod.MiMoVisionTransformer(mod._as_namespace(self.cfg.vision_config)),
                self.reader.load_module("visual.", self.dtype), self.device, self.dtype)
        if which == "audio" and self._audio is None:
            acfg = mod._as_namespace(self.cfg.audio_config)
            self._audio = _build_tower(
                lambda: mod.MiMoAudioEncoder(acfg),
                self.reader.load_module("audio_encoder.", self.dtype), self.device, self.dtype)
            # The audio path needs the speech embedding table the model builds alongside it.
            sp = self.reader.load_module("speech_embeddings.", self.dtype)
            self._speech = _build_tower(lambda: mod._build_speech_embeddings(acfg),
                                        sp, self.device, self.dtype) if sp else None
        return self._visual if which == "visual" else self._audio

    @torch.no_grad()
    def embed_batch(self, input_ids: torch.Tensor, media: dict | None = None) -> torch.Tensor:
        """input_ids [B, S] -> inputs_embeds [B, S, H], with real tower output spliced in."""
        ids = input_ids.to(self.device)
        emb = self.embed(ids).to(self.dtype)
        media = media or {}
        for kind, key in (("image", "image"), ("video", "video")):
            if key in media:
                tok = self.ids[kind]
                assert tok is not None, f"config has no {kind}_token_id"
                out = self._lazy("visual")(pixel_values=media[key]["pixel_values"].to(self.device),
                                           grid_thw=media[key]["grid_thw"].to(self.device))
                self._splice(ids, emb, tok, out, kind)
        if "audio" in media:
            a = media["audio"]
            out = self._lazy("audio")(speech_embeddings=self._speech,
                                      audio_codes=a.get("audio_codes"),
                                      audio_embeds=a.get("audio_embeds"))
            self._splice(ids, emb, self.ids["audio"], out, "audio")
        return emb

    @staticmethod
    def _splice(ids, emb, token_id, modal, kind):
        mask = ids.eq(token_id)
        n = int(mask.sum().item())
        assert n > 0, (f"{kind} media was supplied but the prompt contains no {kind} placeholder "
                       f"tokens -- the tower output would be silently discarded")
        assert modal.shape[0] == n, (
            f"{kind}: {n} placeholders but {modal.shape[0]} embeddings")
        emb[mask] = modal.to(emb.dtype)


class ChunkWriter:
    """Accumulate batches until the token budget is hit, then write one chunk file."""

    def __init__(self, out_dir, buckets, tokens_per_chunk: int = 2_000_000):
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.buckets = list(buckets)
        self.budget = tokens_per_chunk
        self._buf: list[dict] = []
        self._tok = 0
        self._n = 0
        self._seen: dict[str, int] = {b: 0 for b in self.buckets}
        self._media_seen: dict[str, int] = {b: 0 for b in self.buckets}

    def add(self, embeds: torch.Tensor, valid: torch.Tensor, bucket: str,
            had_media: bool = False) -> None:
        assert bucket in self.buckets, f"unknown bucket {bucket}"
        assert embeds.shape[:2] == valid.shape, f"{embeds.shape} vs {valid.shape}"
        self._buf.append({"embeds": embeds.to("cpu", torch.bfloat16),
                          "valid": valid.cpu(), "bucket": bucket})
        n = int(valid.sum().item())
        self._tok += n
        self._seen[bucket] += n
        self._media_seen[bucket] += n if had_media else 0
        if self._tok >= self.budget:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        torch.save(self._buf, self.out / f"chunk_{self._n:05d}.pt")
        self._n += 1
        self._buf, self._tok = [], 0
        gc.collect()

    def close(self) -> None:
        self.flush()
        # THE GUARD. A media bucket with no media in it is the silent failure this file exists
        # to prevent, so refuse to finish rather than write a manifest that lies.
        for b in ("image", "audio", "video"):
            if b in self.buckets and self._seen.get(b, 0) and not self._media_seen.get(b, 0):
                raise RuntimeError(
                    f"bucket '{b}' collected {self._seen[b]} tokens but NONE came from real "
                    f"media -- it would calibrate the {b} experts on text and guarantee they are "
                    f"pruned first. Refusing to write the manifest.")
        (self.out / "manifest.json").write_text(json.dumps(
            {"buckets": self.buckets, "chunks": self._n,
             "tokens_by_bucket": self._seen,
             "media_tokens_by_bucket": self._media_seen}, indent=1))

    @property
    def n_chunks(self) -> int:
        return self._n
