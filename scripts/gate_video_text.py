"""Gate the video caption join against the REAL Hub data.

The defect: `ShareGPTVideo/train_video_and_instruction` frames carry no text at all -- a row is
exactly ['jpeg', '__key__', '__url__'] -- so every video clip in the corpus fell back to the same
literal prompt. Measured consequence on the calibration pass: video reached 35.4% expert coverage
(6.2% in its worst layer) against audio's 59.2% on an IDENTICAL token count.

The fix is a join, so the join is what is tested: real caption ids against real frame keys, not a
fixture. A fixture would have agreed with itself and shipped the same bug.
"""
from __future__ import annotations
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build_corpus as BC   # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}", flush=True)
    if not cond:
        FAIL.append(name)


def main():
    caps = BC.video_captions()
    check("captions load from the repo's separate instruction files", len(caps) > 100_000,
          f"{len(caps):,} scenes")
    lens = [len(v) for v in itertools.islice(caps.values(), 5000)]
    check("captions carry real text, not a stub", sum(lens) / len(lens) > 100,
          f"mean {sum(lens)/len(lens):.0f} chars over {len(lens)} sampled")
    check("the <video> placeholder is stripped",
          not any("<video>" in v for v in itertools.islice(caps.values(), 2000)))

    # THE JOIN, against real frame keys streamed from the same repo
    from datasets import load_dataset
    hid, cfgn, split, _ = BC.SPEC.VIDEO_SOURCES[0]
    ds = load_dataset(hid, cfgn, split=split, streaming=True)
    keys, seen = [], set()
    for row in itertools.islice(ds, 400):
        k = row["__key__"].rsplit("/", 1)[0].lstrip("./")
        if k not in seen:
            seen.add(k); keys.append(k)
        if len(keys) >= 12:
            break
    hit = [k for k in keys if k in caps]
    check("real frame keys join to real captions", len(hit) == len(keys),
          f"{len(hit)}/{len(keys)} scenes matched, e.g. {keys[0]!r}")
    check("a joined caption is substantial", len(caps[hit[0]]) > 200 if hit else False,
          f"{len(caps[hit[0]])} chars" if hit else "no join")

    # negative: the bug this replaces returned "" for every clip, and a stripped id must NOT match
    check("the join is not trivially permissive", ("./" + keys[0]) not in caps,
          "a key that still carries its './' prefix does not match")

    # diversity is the whole point -- the old path gave every clip the SAME text
    txts = [caps[k] for k in hit]
    check("captions differ between clips", len(set(txts)) == len(txts),
          f"{len(set(txts))} distinct texts across {len(txts)} clips (the old path gave 1)")

    # ---- the retry loop: a dropped stream must resume, not end the top-up ----
    # Documented Hub behaviour for long reads is HTTP 429 and mid-stream disconnects, and
    # streaming=True has no resume -- an exception ends the iterator. Drive that path directly.
    import datasets as _ds
    real_load = _ds.load_dataset
    state = {"opens": 0, "skipped": None}

    class Flaky:
        """Yields 5 rows, then raises once; on reopen, honours .skip() and yields normally."""
        def __init__(self, rows, start=0):
            self.rows, self.start = rows, start
        def skip(self, n):
            state["skipped"] = n
            return Flaky(self.rows, self.start + n)
        def __iter__(self):
            for i in range(self.start, len(self.rows)):
                if state["opens"] == 1 and i == self.start + 5:
                    raise ConnectionError("simulated mid-stream drop")
                yield self.rows[i]

    rows = [{"__key__": f"./scene{i//2:03d}/f{i%2}", "jpeg": None} for i in range(40)]

    def fake_load(*a, **k):
        state["opens"] += 1
        return Flaky(rows)

    class _Img:
        def convert(self, _):
            return "frame"
    for r in rows:
        r["jpeg"] = _Img()

    _ds.load_dataset = fake_load
    real_caps = BC.video_captions
    BC.video_captions = lambda *a, **k: {}
    try:
        got = list(itertools.islice(BC.iter_video_clips(2), 12))
    finally:
        _ds.load_dataset = real_load
        BC.video_captions = real_caps

    check("a dropped stream is retried rather than ending the top-up", state["opens"] >= 2,
          f"{state['opens']} stream opens")
    check("the retry resumes instead of re-reading from row 0", state["skipped"] == 5,
          f"skipped {state['skipped']} rows on reopen")
    check("clips still come out after the drop", len(got) >= 10, f"{len(got)} clips yielded")

    print(("GATE FAIL: " + ", ".join(FAIL)) if FAIL else "GATE PASS")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
