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

    print(("GATE FAIL: " + ", ".join(FAIL)) if FAIL else "GATE PASS")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
