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
import time
import sys
from pathlib import Path

import numpy as np
import torch

import mimo_corpus_spec as SPEC
import media_loaders as ML
from chunk_builder import Embedder, ChunkWriter
from media_worker import MediaWorker


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


MIN_BUCKET_FRAC = 0.50      # below this a bucket is a failed run, not a thin one
WARN_BUCKET_FRAC = 0.95     # below this it is worth saying out loud
SOURCE_RETRIES = 4


def iter_rows(hid, cfgn, split):
    from datasets import load_dataset
    return load_dataset(hid, cfgn, split=split, streaming=True)


def live_sources(kind, src_list, probe_rows: int = 12):
    """Drop sources that cannot actually contribute, and renormalise the survivors' weights.

    MEASURED 2026-09-23: nine of the fifteen configured image sources yielded ZERO usable
    images -- every `nvidia/Nemotron-VLM-Dataset-v2` config plus `xlangai/aguvis-stage2` and
    `ServiceNow/BigDocs-Bench`, together 52% of the declared image weight, all of them
    path-only collections whose rows reference an image file rather than embedding one. The
    build loop's `except ValueError: continue` swallowed every one of them silently, so the
    bucket would have been served entirely by whichever source happened to be listed next.

    Datasets rot, get gated, and change schema. Hard-coding the list that happened to work on
    one afternoon is not robustness; probing at startup is. A dead source costs a few seconds
    here instead of hours of a bucket that silently collects nothing.
    """
    live, dead = [], []
    for entry in src_list:
        hid, cfgn, split, w = entry[0], entry[1], entry[2], entry[3]
        ok = 0
        try:
            for row in robust_rows(hid, cfgn, split, retries=1):
                try:
                    if kind == "image":
                        row_image(row)
                    elif kind == "audio":
                        row_audio(row)
                    ok += 1
                    break
                except ValueError:
                    ok += 0
                except Exception:
                    ok += 0
                probe_rows -= 1
                if probe_rows <= 0:
                    break
        except Exception:
            pass
        (live if ok else dead).append(entry)
        probe_rows = 12
    for entry in dead:
        print(f"    DEAD SOURCE ({kind}): {entry[0]}:{entry[1]} -- yields no usable media",
              flush=True)
    if dead:
        lost = sum(e[3] for e in dead)
        print(f"    {len(dead)}/{len(src_list)} {kind} sources dead, {lost:.0%} of declared "
              f"weight; renormalising over the {len(live)} live ones", flush=True)
    if not live:
        raise RuntimeError(f"every {kind} source is dead -- cannot build the {kind} bucket")
    return live


def source_targets(src_list, want_total: int) -> list:
    """Split a bucket's token target across its sources BY WEIGHT.

    The weights in mimo_corpus_spec were being ignored entirely: the loop consumed sources in
    order until the bucket target was met, so the first source supplied essentially the whole
    bucket and every later one contributed nothing. That is the corpus-imbalance failure this
    project has already measured once -- a bucket dominated by one source calibrates the router
    on one distribution and the experts that serve the rest look dark.
    """
    tot = sum(e[3] for e in src_list) or 1.0
    return [(e, max(1, int(want_total * e[3] / tot))) for e in src_list]


def robust_rows(hid, cfgn, split, retries: int = SOURCE_RETRIES):
    """Iterate a streaming source, surviving transient faults.

    EVERY source here is an HTTP stream. Over the hours this stage runs, a connection reset, a
    503 from the Hub or a decode error on one malformed row is not unlikely -- it is expected --
    and any of them propagating out of the row loop kills a stage that has no partial credit.
    The failure is retried from the start of the source with backoff; rows already consumed are
    re-yielded, which costs duplicates but never loses the bucket. After `retries` attempts the
    source is abandoned and the next one in the list takes over, because one dead source must
    not be able to end the run either.

    A source that dies is reported, not swallowed: a bucket quietly served by two of its five
    sources is exactly the corpus-imbalance failure this project already paid for once.
    """
    for attempt in range(retries):
        n = 0
        try:
            for row in iter_rows(hid, cfgn, split):
                n += 1
                yield row
            return
        except Exception as e:                      # noqa: BLE001 - any stream fault
            wait = min(60, 5 * 2 ** attempt)
            print(f"    ! {hid}:{cfgn or '-'} failed after {n:,} rows "
                  f"({type(e).__name__}: {str(e)[:120]}); retry {attempt+1}/{retries} in {wait}s",
                  flush=True)
            if attempt == retries - 1:
                print(f"    ! ABANDONING source {hid}:{cfgn or '-'}", flush=True)
                return
            time.sleep(wait)


# ---------------------------------------------------------------- packing

class Packer:
    """Pack samples into fixed-length sequences, carrying their media along.

    Media stays with its own placeholders: pixel/grid rows are appended in the SAME ORDER the
    placeholder tokens appear in the packed sequence, because the tower returns one embedding row
    per patch in grid order and the splice assigns them positionally. Get that order wrong and
    every media row lands on the wrong token, silently.
    """

    def __init__(self, embedder, seq_len: int, eos: int, worker=None):
        # `worker` runs the towers in a disposable subprocess (see media_worker). When it is
        # set the Packer never touches a tower itself, so nothing in THIS process can allocate
        # the box away.
        self.E, self.S, self.eos, self.worker = embedder, seq_len, eos, worker
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
        """-> (ids [1,S], valid [1,S], media_embeds or None, media_token_id) or None.

        The TOWERS RUN HERE and their output is returned; the pass only ever does a table
        lookup and a positional splice. That keeps the towers a once-per-corpus cost while
        storing 4 bytes a text token instead of 8,192.
        """
        if not self.ids:
            return None
        n = min(len(self.ids), self.S)
        ids = torch.tensor([self.ids[:n] + [self.eos] * (self.S - n)], dtype=torch.long)
        valid = torch.zeros(1, self.S, dtype=torch.bool); valid[0, :n] = True
        me, tokid = None, None
        if self.pix:
            pv, gr = torch.cat(self.pix, 0), torch.cat(self.grid, 0)
            if self.worker is not None:
                me = self.worker.visual(pv, gr)
                if me is None:
                    # The worker died or refused. Drop the whole packed sequence rather than
                    # splice media onto the wrong tokens -- losing one sequence is free, a
                    # misaligned splice is silent corruption.
                    self.reset()
                    return None
            else:
                me = self.E._lazy("visual")(pixel_values=pv.to(self.E.device),
                                            grid_thw=gr.to(self.E.device))
            tokid = self.E.ids[self.kind]
        elif self.audio:
            me = torch.cat(self.audio, 0)
            tokid = self.E.ids["audio"]
        if me is not None:
            n_slot = int((ids == tokid).sum())
            # Truncation can cut a media row's placeholders; drop the orphaned tail rather than
            # splice a misaligned one, which would silently attach every later patch to the
            # wrong token.
            me = me[:n_slot]
            assert me.shape[0] == n_slot, f"{n_slot} placeholders, {me.shape[0]} embeddings"
        self.reset()
        return ids, valid, me, tokid


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
    # The PARENT holds no CUDA context and no towers. Everything that can allocate at scale
    # lives in the worker, which is expendable; this process only streams rows, tokenises, and
    # writes chunks. That is what makes an unforeseen allocation cost a sample instead of the
    # run -- see media_worker for why no kernel-side limit is available on this machine.
    E = Embedder(src, cfg, device="cpu")
    V = ML.VideoLoader(E, proc)
    MW = MediaWorker(src, device=device)
    lim = ML.configure_media_limits(proc, cfg)
    print(f"media limits: <= {lim['max_patch_rows']} patch rows per tower chunk "
          f"(image <= {lim['image_max_pixels']:,} px, video <= "
          f"{lim['video_max_pixels_per_frame']:,} px/frame at T={lim['t_grid']})", flush=True)

    buckets = list(SPEC.TOKEN_TARGET)
    want = {b: int(total_tokens * SPEC.TOKEN_TARGET[b]) for b in buckets}
    W = ChunkWriter(out_dir, buckets, tokens_per_chunk=tokens_per_chunk)
    eos = cfg.eos_token_id
    print(f"target {total_tokens:,} tokens, seq_len {seq_len}", flush=True)
    for b in buckets:
        print(f"  {b:11} {want[b]:>12,}", flush=True)

    oversize: dict[str, int] = {}
    already = W.resume()
    if already:
        print(f"resuming corpus build: {len(already)} buckets already collected "
              f"({', '.join(already)})", flush=True)

    for bucket in buckets:
        if bucket in already:
            print(f"  {bucket:11} SKIPPED (checkpointed)", flush=True)
            continue
        got = 0
        P = Packer(E, seq_len, eos, worker=MW)

        def emit(force=False):
            nonlocal got
            if P.room() > seq_len // 8 and not force:
                return
            r = P.flush()
            if r is None:
                return
            ids, valid, me, tokid = r
            W.add(ids, valid, bucket, media_embeds=me, media_token_id=tokid)
            got += int(valid.sum())

        if bucket in ("image", "audio", "video"):
            src_list = {"image": SPEC.IMAGE_SOURCES, "audio": SPEC.AUDIO_SOURCES,
                        "video": SPEC.VIDEO_SOURCES}[bucket]
            if bucket == "video":
                stream = iter_video_clips(ML.CLIP_FRAMES_VIDEO)
                for frames, text in stream:
                    if got >= want[bucket]:
                        break
                    frames = ML.fit_frames(frames, cfg)
                    prep = V.prepare(frames)
                    # A clip is ONE attention chunk (grid is T,h,w and L = T*h*w), so the whole
                    # clip must fit the budget, not each frame.
                    L = ML.patch_rows(prep["grid_thw"])
                    if L > ML.MAX_PATCH_ROWS:
                        oversize["video"] = oversize.get("video", 0) + 1
                        continue
                    n = V.n_tokens(prep)
                    ids = tok(text or "Describe this video.", add_special_tokens=False)["input_ids"]
                    ids = ids + [E.ids["video"]] * n
                    if len(ids) + 1 > P.room():
                        emit(force=True)
                    P.add(ids, pix=prep["pixel_values"], grid=prep["grid_thw"], kind="video")
                    emit()
            else:
                src_list = live_sources(bucket, src_list)
                for (entry, sub) in source_targets(src_list, want[bucket]):
                    hid, cfgn, split = entry[0], entry[1], entry[2]
                    if got >= want[bucket]:
                        break
                    stop_at = min(want[bucket], got + sub)
                    rows = robust_rows(hid, cfgn, split)
                    if limit_per_source:
                        rows = itertools.islice(rows, limit_per_source)
                    for row in rows:
                        if got >= stop_at:
                            break
                        try:
                            if bucket == "image":
                                img, text = row_image(row)
                                img = ML.fit_image(img, cfg)
                                prep = proc.image_processor(images=[img], return_tensors="pt")
                                # THE HARD STOP. The cap above is a hint to the processor; this
                                # is the guarantee. One sample whose attention length exceeds
                                # the budget allocates tens of GB in a single burst and the box
                                # is gone -- no retry, no resume, just SIGKILL. Skipping the
                                # sample costs nothing; there are millions more.
                                L = ML.patch_rows(prep["image_grid_thw"])
                                if L > ML.MAX_PATCH_ROWS:
                                    oversize[bucket] = oversize.get(bucket, 0) + 1
                                    continue
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
                                ae = MW.audio([wave])
                                if ae is None:
                                    continue
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
            for (entry, sub) in source_targets(SPEC.SOURCES[bucket], want[bucket]):
                hid, cfgn, split, text_fn = entry[0], entry[1], entry[2], entry[4]
                if got >= want[bucket]:
                    break
                stop_at = min(want[bucket], got + sub)
                rows = robust_rows(hid, cfgn, split)
                if limit_per_source:
                    rows = itertools.islice(rows, limit_per_source)
                for row in rows:
                    if got >= stop_at:
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
        if oversize.get(bucket):
            print(f"    skipped {oversize[bucket]:,} oversized {bucket} sample(s) over the "
                  f"{ML.MAX_PATCH_ROWS}-patch-row budget", flush=True)
        frac = got / max(1, want[bucket])
        print(f"  {bucket:11} collected {got:,} / {want[bucket]:,} tokens ({frac:.0%})",
              flush=True)
        # UNDERFILL IS A QUALITY FAILURE, NOT A COSMETIC ONE. An under-served bucket leaves its
        # experts under-routed, and an expert that is never routed to gets pruned arbitrarily
        # rather than on evidence. Catching it here costs one bucket; catching it after the
        # 34-hour pass costs the run.
        if frac < MIN_BUCKET_FRAC:
            raise RuntimeError(
                f"bucket '{bucket}' collected {got:,} of {want[bucket]:,} tokens ({frac:.0%}), "
                f"below the {MIN_BUCKET_FRAC:.0%} floor -- its sources are exhausted or failing. "
                f"Calibrating on this corpus would prune the {bucket} experts on no evidence.")
        if frac < WARN_BUCKET_FRAC:
            print(f"    WARNING: {bucket} under target at {frac:.0%}", flush=True)
        W.guard_bucket(bucket)
        W.mark_done(bucket)
        W.checkpoint(bucket)

    W.close()
    if MW.deaths or MW.skipped:
        print(f"media worker: {MW.deaths} restart(s), {MW.skipped} sample(s) dropped",
              flush=True)
    MW.close()
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
