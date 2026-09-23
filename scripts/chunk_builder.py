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
import os
import json
from pathlib import Path

import torch

from mimo_shards import ShardReader



def _build_tower(factory, sd: dict, device, dtype, allow_missing=()):
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
    bad = [k for k in missing if k not in buffers and not k.endswith(".bias")
           and k not in allow_missing]
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
        del w
        # Release the shard handles: safe_open keeps a live mmap, and pages faulted through a
        # live mapping cannot be reclaimed, so a prefix scan over 65 shards pins cache the
        # kernel may not evict. calib_pass releases after every layer; this path never did.
        # MEASURED, and smaller than it looks: an A/B over the tower load (gate_shard_release)
        # put the pinned amount at 7.2 GiB without the release vs 6.2 GiB with it, so this is
        # worth ~1 GiB and is NOT the cause of the image-bucket OOM -- that hypothesis was
        # tested here and refuted. Kept because it is correct and free, not because it is the fix.
        # Safe because load_module copies every tensor out (copy=True / .clone()).
        self.reader.release()
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
                self.reader.load_module("audio_encoder.", self.dtype), self.device, self.dtype,
                # STRUCTURALLY DEAD, not missing. `input_local_transformer` is only ever called
                # as `inputs_embeds=...` (_apply_input_local_transformer), so its embedding table
                # is never reached and the checkpoint rightly omits it. Allowlisted by name so
                # the guard keeps its teeth for every other tensor -- a missing weight matrix
                # stays fatal.
                allow_missing={"input_local_transformer.embed_tokens.weight"})
            # The audio path needs the speech embedding table the model builds alongside it.
            sp = self.reader.load_module("speech_embeddings.", self.dtype)
            self._speech = _build_tower(lambda: mod._build_speech_embeddings(acfg),
                                        sp, self.device, self.dtype) if sp else None
        # Same reasoning as the constructor: the tower weights are copied into a real module,
        # so the mappings that the prefix scan faulted in must not outlive this call.
        self.reader.release()
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
        self._done: list[str] = []

    def add(self, ids: torch.Tensor, valid: torch.Tensor, bucket: str,
            media_embeds: torch.Tensor | None = None, media_token_id: int | None = None) -> None:
        """Store TOKEN IDS plus already-towered media embeddings, not inputs_embeds.

        50M tokens of bf16 inputs_embeds is 410 GB against 324 GB free -- the corpus would not
        fit on the box. Ids are 4 bytes a token instead of 8,192, and only the media rows need
        their embeddings carried (49 GB at a 12% media share). The towers still run exactly
        ONCE, here, which was the reason for pre-embedding in the first place; the pass does the
        embedding-table lookup, which is a gather against a 1.25 GB table.
        """
        assert bucket in self.buckets, f"unknown bucket {bucket}"
        assert ids.shape == valid.shape, f"{ids.shape} vs {valid.shape}"
        had_media = media_embeds is not None and media_embeds.numel() > 0
        if had_media:
            n_slot = int((ids == media_token_id).sum())
            assert n_slot == media_embeds.shape[0], (
                f"{n_slot} media placeholders but {media_embeds.shape[0]} embeddings")
        self._buf.append({"ids": ids.to(torch.int32).cpu(), "valid": valid.cpu(),
                          "bucket": bucket,
                          "media_embeds": media_embeds.to("cpu", torch.bfloat16)
                          if had_media else None,
                          "media_token_id": media_token_id if had_media else None})
        n = int(valid.sum().item())
        self._tok += n
        self._seen[bucket] += n
        self._media_seen[bucket] += n if had_media else 0
        if self._tok >= self.budget:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        # Atomic, for the same reason the pass checkpoints atomically: a torn chunk_000NN.pt is
        # indistinguishable from a good one until torch.load fails 20 hours into the pass.
        dst = self.out / f"chunk_{self._n:05d}.pt"
        tmp = dst.with_suffix(".pt.tmp")
        torch.save(self._buf, tmp)
        os.replace(tmp, dst)
        self._n += 1
        self._buf, self._tok = [], 0
        gc.collect()

    # ---- resume -------------------------------------------------------------------------
    # The corpus stage streams from nine source families and runs the towers over every image,
    # audio clip and video clip. That is hours, and a transient network fault at hour three
    # previously meant redoing hour one. Bucket boundaries are the natural checkpoint: a bucket
    # is either fully collected or not started, so resuming at one cannot half-count a bucket.

    def checkpoint(self, bucket: str) -> None:
        """Flush at a bucket boundary and record enough to resume after it."""
        self.flush()
        state = {"next_chunk": self._n, "seen": self._seen,
                 "media_seen": self._media_seen, "done_buckets": self._done}
        tmp = self.out / "corpus_state.json.tmp"
        tmp.write_text(json.dumps(state, indent=1))
        os.replace(tmp, self.out / "corpus_state.json")

    def mark_done(self, bucket: str) -> None:
        if bucket not in self._done:
            self._done.append(bucket)

    def resume(self) -> list:
        """Restore counters from corpus_state.json; return the buckets already finished."""
        f = self.out / "corpus_state.json"
        if not f.exists():
            # No state, but chunks present: they are orphans from a build that died before its
            # first checkpoint. Starting again at index 0 overwrites only as far as this run
            # reaches, and anything past that would be read by the pass as real data.
            orphans = sorted(self.out.glob("chunk_*.pt"))
            for o in orphans:
                o.rename(o.with_suffix(".pt.orphan"))
            if orphans:
                print(f"    set aside {len(orphans)} orphaned chunk(s) from an aborted build",
                      flush=True)
            return []
        st = json.loads(f.read_text())
        self._n = int(st["next_chunk"])
        self._seen.update(st.get("seen", {}))
        self._media_seen.update(st.get("media_seen", {}))
        self._done = list(st.get("done_buckets", []))
        # Any chunk index at or beyond the checkpoint is from an aborted bucket that will now be
        # recollected. Leaving it would feed the pass duplicate, half-written data that every
        # invariant in verify_pass would happily accept.
        for stale in sorted(self.out.glob("chunk_*.pt")):
            if int(stale.stem.split("_")[1]) >= self._n:
                stale.rename(stale.with_suffix(".pt.superseded"))
        for t in self.out.glob("*.tmp"):
            t.unlink()
        return list(self._done)

    def guard_bucket(self, bucket: str) -> None:
        """Fail fast on a media bucket that collected no media.

        close() already refuses this, but close() runs after every bucket -- i.e. hours after
        the fault, having already paid for the rest of the corpus. Checking at the boundary
        surfaces it as soon as it is knowable.
        """
        if bucket in ("image", "audio", "video") and self._seen.get(bucket, 0) \
                and not self._media_seen.get(bucket, 0):
            raise RuntimeError(
                f"bucket '{bucket}' collected {self._seen[bucket]} tokens but NONE came from "
                f"real media -- it would calibrate the {bucket} experts on text and guarantee "
                f"they are pruned first. Refusing to continue.")

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
