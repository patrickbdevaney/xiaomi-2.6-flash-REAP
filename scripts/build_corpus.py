"""Wire the calibration sources into chunks of inputs_embeds, bucket by bucket.

Fills every bucket to its share of TOKEN_TARGET, packs samples into fixed-length sequences,
embeds them once (text through the table, media through the towers), and hands them to
ChunkWriter. The pass then never touches a dataset, a tokenizer or a tower.

THE RULE THIS FILE ENFORCES. A media bucket must be fed real media. Every adapter here either
produces decoded media or RAISES -- none of them falls back to the row's text. That is not
defensive style, it is the whole point: audio and video tokens route through the same 256 experts
as text, so an audio bucket quietly filled with transcripts would guarantee the audio experts are
pruned first while every text benchmark still passed. ChunkWriter.close() re-checks the same
invariant from the other side.

SOURCE NOTES, from probing the actual rows on 2026-09-23:
  * `ShareGPTVideo/train_video_and_instruction` is a WEBDATASET OF FRAMES, not clips: keys look
    like `./v_<id>-Scene-001/c01_0007`. Frames are grouped back into clips by that prefix, which
    is the only reason this source is usable at all.
  * `nvidia/Nemotron-VLM-Dataset-v2` references images BY PATH inside `messages`
    (`{"type": "image", "image": "train/png/two_col_2954.png"}`) rather than embedding bytes, so
    streaming a row yields no image. Image sources are therefore ordered to prefer collections
    that embed their images, and a path-only row raises rather than silently contributing text.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

import mimo_corpus_spec as SPEC
import media_loaders as ML
from chunk_builder import Embedder, ChunkWriter


# ---------------------------------------------------------------- row adapters

def _pil(v):
    """Return a PIL image from whatever shape a row carries, or None."""
    from PIL import Image
    if isinstance(v, Image.Image):
        return v
    if isinstance(v, dict) and "bytes" in v and v["bytes"]:
        import io
        return Image.open(io.BytesIO(v["bytes"])).convert("RGB")
    return None


def row_image(row):
    """-> (PIL image, prompt text). Raises if the row carries no decodable image."""
    img = None
    for v in row.values():
        img = _pil(v)
        if img is not None:
            break
        if isinstance(v, list) and v:
            img = _pil(v[0])
            if img is not None:
                break
    if img is None:
        raise ValueError("row carries no embedded image (path-only sources need a resolver); "
                         "refusing to contribute it as text")
    return img.convert("RGB"), _row_text(row)


def row_audio(row):
    """-> (waveform float32 @ 24 kHz, prompt text). Raises if no decodable audio."""
    for v in row.values():
        a = sr = None
        if hasattr(v, "get_all_samples"):                 # torchcodec AudioDecoder
            s = v.get_all_samples()
            a = s.data.to(torch.float32)
            a = (a.mean(0) if a.ndim > 1 else a).numpy()
            sr = int(s.sample_rate)
        elif isinstance(v, dict) and "array" in v:        # older datasets form
            a = np.asarray(v["array"], dtype=np.float32)
            sr = int(v.get("sampling_rate", ML.AUDIO_SR))
        if a is None or a.size < ML.AUDIO_SR // 4:
            continue
        if sr != ML.AUDIO_SR:
            n = int(len(a) * ML.AUDIO_SR / sr)
            a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype("float32")
        pk = float(np.abs(a).max()) or 1.0
        clip = ML.CLIP_SECONDS_AUDIO * ML.AUDIO_SR
        return (a[:clip] / pk * 0.7).astype("float32"), _row_text(row)
    raise ValueError("row carries no decodable audio; refusing to contribute it as text")


def _row_text(row):
    for k in ("question", "text", "prompt", "instruction", "caption"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()[:2000]
    return ""


def iter_video_clips(frames_per_clip: int):
    """Regroup the frame webdataset into clips by `__key__` prefix.

    Frames arrive in order within a scene, so a clip is simply a run of consecutive rows sharing
    the directory part of the key. A scene shorter than `frames_per_clip` is skipped rather than
    padded -- a clip of repeated frames would teach the temporal path nothing.
    """
    from datasets import load_dataset
    hid, cfgn, split, _ = SPEC.VIDEO_SOURCES[0]
    ds = load_dataset(hid, cfgn, split=split, streaming=True)
    cur_key, buf = None, []
    for row in ds:
        key = row["__key__"].rsplit("/", 1)[0]
        if key != cur_key:
            cur_key, buf = key, []
        buf.append(row["jpeg"].convert("RGB"))
        if len(buf) == frames_per_clip:
            yield buf, ""
            buf = []


def iter_rows(hid, cfgn, split):
    from datasets import load_dataset
    return load_dataset(hid, cfgn, split=split, streaming=True)


# ---------------------------------------------------------------- packing

class Packer:
    """Pack samples into fixed-length sequences, carrying their media along.

    Media stays with its own placeholders: pixel/grid rows are appended in the SAME ORDER the
    placeholder tokens appear in the packed sequence, because the tower returns one embedding row
    per patch in grid order and the splice assigns them positionally. Get that order wrong and
    every media row lands on the wrong token, silently.
    """

    def __init__(self, embedder, seq_len: int, eos: int):
        self.E, self.S, self.eos = embedder, seq_len, eos
        self.reset()

    def reset(self):
        self.ids: list[int] = []
        self.pix: list[torch.Tensor] = []
        self.grid: list[torch.Tensor] = []
        self.audio: list[torch.Tensor] = []
        self.kind = None

    def room(self) -> int:
        return self.S - len(self.ids)

    def add(self, ids, pix=None, grid=None, audio=None, kind=None):
        self.ids.extend(ids)
        self.ids.append(self.eos)
        if pix is not None:
            self.pix.append(pix); self.grid.append(grid)
        if audio is not None:
            self.audio.append(audio)
        if kind:
            self.kind = kind

    def flush(self):
        """-> (embeds [1,S,H], valid [1,S], had_media) or None."""
        if not self.ids:
            return None
        n = min(len(self.ids), self.S)
        ids = torch.tensor([self.ids[:n] + [self.eos] * (self.S - n)], dtype=torch.long)
        valid = torch.zeros(1, self.S, dtype=torch.bool); valid[0, :n] = True
        media = {}
        if self.pix:
            media[self.kind] = {"pixel_values": torch.cat(self.pix, 0),
                                "grid_thw": torch.cat(self.grid, 0)}
        if self.audio:
            media["audio"] = {"audio_embeds": torch.cat(self.audio, 0)}
        had = bool(media)
        emb = self.E.embed_batch(ids, media=media or None)
        self.reset()
        return emb, valid, had


# ---------------------------------------------------------------- the build

def build(src, out_dir, total_tokens: int, seq_len: int = 4096,
          tokens_per_chunk: int = 2_000_000, device: str = "cuda", limit_per_source: int = 0):
    """Fill every bucket to its TOKEN_TARGET share and write chunks."""
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer
    cfg = AutoConfig.from_pretrained(src, trust_remote_code=True)
    cfg._name_or_path = str(src)
    cfg._attn_implementation = "flex_attention" if device.startswith("cuda") else "eager"
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(src, trust_remote_code=True)
    E = Embedder(src, cfg, device=device)
    V = ML.VideoLoader(E, proc)
    A = ML.AudioLoader(E, src)

    buckets = list(SPEC.TOKEN_TARGET)
    want = {b: int(total_tokens * SPEC.TOKEN_TARGET[b]) for b in buckets}
    W = ChunkWriter(out_dir, buckets, tokens_per_chunk=tokens_per_chunk)
    eos = cfg.eos_token_id
    print(f"target {total_tokens:,} tokens, seq_len {seq_len}", flush=True)
    for b in buckets:
        print(f"  {b:11} {want[b]:>12,}", flush=True)

    for bucket in buckets:
        got = 0
        P = Packer(E, seq_len, eos)

        def emit(force=False):
            nonlocal got
            if P.room() > seq_len // 8 and not force:
                return
            r = P.flush()
            if r is None:
                return
            emb, valid, had = r
            W.add(emb, valid, bucket, had_media=had)
            got += int(valid.sum())

        if bucket in ("image", "audio", "video"):
            src_list = {"image": SPEC.IMAGE_SOURCES, "audio": SPEC.AUDIO_SOURCES,
                        "video": SPEC.VIDEO_SOURCES}[bucket]
            if bucket == "video":
                stream = iter_video_clips(ML.CLIP_FRAMES_VIDEO)
                for frames, text in stream:
                    if got >= want[bucket]:
                        break
                    prep = V.prepare(frames)
                    n = V.n_tokens(prep)
                    ids = tok(text or "Describe this video.", add_special_tokens=False)["input_ids"]
                    ids = ids + [E.ids["video"]] * n
                    if len(ids) + 1 > P.room():
                        emit(force=True)
                    P.add(ids, pix=prep["pixel_values"], grid=prep["grid_thw"], kind="video")
                    emit()
            else:
                for (hid, cfgn, split, _w) in src_list:
                    if got >= want[bucket]:
                        break
                    rows = iter_rows(hid, cfgn, split)
                    if limit_per_source:
                        rows = itertools.islice(rows, limit_per_source)
                    for row in rows:
                        if got >= want[bucket]:
                            break
                        try:
                            if bucket == "image":
                                img, text = row_image(row)
                                prep = proc.image_processor(images=[img], return_tensors="pt")
                                n = int(prep["image_grid_thw"].prod(-1).sum()) // (
                                    cfg.vision_config["spatial_merge_size"] ** 2)
                                ids = tok(text or "Describe this image.",
                                          add_special_tokens=False)["input_ids"]
                                ids = ids + [E.ids["image"]] * n
                                if len(ids) + 1 > P.room():
                                    emit(force=True)
                                P.add(ids, pix=prep["pixel_values"],
                                      grid=prep["image_grid_thw"], kind="image")
                            else:
                                wave, text = row_audio(row)
                                ae = A.embed([wave])
                                ids = tok(text or "Transcribe and answer.",
                                          add_special_tokens=False)["input_ids"]
                                ids = ids + [E.ids["audio"]] * ae.shape[0]
                                if len(ids) + 1 > P.room():
                                    emit(force=True)
                                P.add(ids, audio=ae, kind="audio")
                        except ValueError:
                            continue          # a row with no usable media is SKIPPED, never
                                              # downgraded to text -- see the module docstring
                        emit()
        else:
            for (hid, cfgn, split, _w, text_fn) in SPEC.SOURCES[bucket]:
                if got >= want[bucket]:
                    break
                rows = iter_rows(hid, cfgn, split)
                if limit_per_source:
                    rows = itertools.islice(rows, limit_per_source)
                for row in rows:
                    if got >= want[bucket]:
                        break
                    try:
                        t = text_fn(row)
                    except Exception:
                        t = None
                    if not t:
                        continue
                    ids = tok(t, add_special_tokens=False,
                              truncation=True, max_length=seq_len - 1)["input_ids"]
                    if len(ids) + 1 > P.room():
                        emit(force=True)
                    P.add(ids)
                    emit()
        emit(force=True)
        print(f"  {bucket:11} collected {got:,} / {want[bucket]:,} tokens", flush=True)

    W.close()
    print(f"wrote {W.n_chunks} chunks to {out_dir}", flush=True)
    return W.n_chunks


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path.home() / "models" / "MiMo-V2.6-Flash-RL"))
    ap.add_argument("--out", default="artifacts/chunks")
    ap.add_argument("--total-tokens", type=int, default=50_000_000)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--tokens-per-chunk", type=int, default=2_000_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-per-source", type=int, default=0, help="smoke: rows per source")
    a = ap.parse_args()
    n = build(a.src, a.out, a.total_tokens, a.seq_len, a.tokens_per_chunk,
              a.device, a.limit_per_source)
    import os
    sys.stdout.flush(); os._exit(0 if n else 1)
