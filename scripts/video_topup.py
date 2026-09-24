"""Append ONE properly-captioned video chunk to a finished corpus.

WHY THIS EXISTS. The video frames in `ShareGPTVideo/train_video_and_instruction` carry no text
at all -- a row is exactly ['jpeg', '__key__', '__url__'] -- so every clip in the built corpus
was paired with the same literal fallback prompt. Measured on the calibration pass: video
reached 35.4% expert coverage (6.2% in its worst layer, 16 of 256) against audio's 59.2% on an
IDENTICAL token count. The captions exist in the same repo, keyed by scene id; build_corpus now
joins them.

WHY APPEND RATHER THAN REBUILD. The accumulators are running sums over chunks already folded in,
and a sum cannot be un-folded. Rebuilding the bad chunk would mean discarding the whole pass.
Appending doubles the video budget and gives half of it real text diversity, which is the best
available outcome that does not cost 30 hours.

It writes a NEW chunk file and rewrites manifest.json; it never edits or removes an existing
chunk, and it refuses to run while the calibration pass holds the GPU.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build_corpus as BC          # noqa: E402
import media_loaders as ML         # noqa: E402
from chunk_builder import ChunkWriter   # noqa: E402
from media_worker import MediaWorker    # noqa: E402


def busy() -> bool:
    try:
        r = subprocess.run(["systemctl", "--user", "is-active", "reap_run.service"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() == "active"
    except Exception:
        return False


def run(src, out_dir, tokens: int, seq_len: int = 4096, device: str = "cuda") -> dict:
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer
    out_dir = Path(out_dir)
    state_f = out_dir / "corpus_state.json"
    man_f = out_dir / "manifest.json"
    if not state_f.exists() or not man_f.exists():
        raise SystemExit(f"{out_dir} has no finished corpus (need corpus_state.json and "
                         f"manifest.json). This appends to a corpus; it does not build one.")
    st = json.loads(state_f.read_text())
    man = json.loads(man_f.read_text())
    next_chunk = int(st["next_chunk"])
    # resume() deletes any chunk at or beyond next_chunk as a half-written orphan. That is right
    # during a build and wrong here, so refuse rather than let it delete finished work.
    ahead = [p.name for p in sorted(out_dir.glob("chunk_*.pt"))
             if int(p.stem.split("_")[1]) >= next_chunk]
    if ahead:
        raise SystemExit(f"chunk(s) at or beyond the checkpoint index {next_chunk}: {ahead}. "
                         f"Refusing -- ChunkWriter.resume() would delete them as orphans.")
    on_disk = len(list(out_dir.glob("chunk_*.pt")))
    if on_disk != int(man["chunks"]):
        raise SystemExit(f"{on_disk} chunks on disk but the manifest declares {man['chunks']}; "
                         f"refusing to append to a corpus that does not agree with itself.")

    cfg = AutoConfig.from_pretrained(src, trust_remote_code=True)
    cfg._name_or_path = str(src)
    cfg._attn_implementation = "flex_attention" if device.startswith("cuda") else "eager"
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(src, trust_remote_code=True)
    E = BC.Embedder(src, cfg, device="cpu")       # the parent holds no CUDA and no towers
    V = ML.VideoLoader(E, proc)
    MW = MediaWorker(src, device=device)
    ML.configure_media_limits(proc, cfg)

    W = ChunkWriter(out_dir, man["buckets"], tokens_per_chunk=tokens)
    W.resume()
    P = BC.Packer(E, seq_len, cfg.eos_token_id, worker=MW)
    got = 0
    oversize: dict[str, int] = {}
    print(f"appending video from chunk index {next_chunk}, target {tokens:,} tokens", flush=True)

    def emit(force=False):
        nonlocal got
        if P.room() > seq_len // 8 and not force:
            return
        r = P.flush()
        if r is None:
            return
        ids, valid, me, tokid = r
        W.add(ids, valid, "video", media_embeds=me, media_token_id=tokid)
        got += int(valid.sum())

    last = 0
    for ids, prep in BC.video_samples(cfg, V, tok, E, oversize):
        if got >= tokens:
            break
        if len(ids) + 1 > P.room():
            emit(force=True)
        P.add(ids, pix=prep["pixel_values"], grid=prep["grid_thw"], kind="video")
        emit()
        if got - last >= 200_000:
            last = got
            print(f"    video {got:,} / {tokens:,} ({got/tokens:.0%})", flush=True)
    emit(force=True)
    W.close()                      # rewrites manifest.json with the merged counters
    MW.stop()

    man2 = json.loads(man_f.read_text())
    new_chunks = int(man2["chunks"]) - int(man["chunks"])
    # chunk_order caches bucket labels; a stale cache would sort the new chunk as unknown
    cache = out_dir / "chunk_buckets.json"
    if cache.exists():
        c = json.loads(cache.read_text())
        for i in range(next_chunk, int(man2["chunks"])):
            c[f"chunk_{i:05d}.pt"] = "video"
        cache.write_text(json.dumps(c, indent=1))
    res = {"new_chunks": new_chunks, "video_tokens_added": got,
           "video_tokens_total": man2["tokens_by_bucket"]["video"],
           "chunks_total": man2["chunks"], "oversize_skipped": oversize.get("video", 0)}
    print(json.dumps(res, indent=1), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/patrickd/models/MiMo-V2.6-Flash-RL")
    ap.add_argument("--out", default="artifacts/chunks")
    ap.add_argument("--tokens", type=int, default=1_500_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if busy() and not a.force:
        print("REFUSING: reap_run.service is active. Building a video chunk runs the vision "
              "tower on the GPU, and competing with the calibration pass for it already cost "
              "that pass a 228 s layer once tonight. Wait for it, or pass --force.")
        return 2
    run(a.src, a.out, a.tokens, device=a.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
