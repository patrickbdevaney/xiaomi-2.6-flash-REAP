"""Gate the crash-resilience added for the unattended REAP run.

Every check here is a NEGATIVE test: it proves the guard fires, not merely that the happy path
still runs. A guard that has never been seen to fire is the memguard failure again -- it logged
a successful reclaim of zero bytes for nine hours because nothing ever forced its trigger.
"""
from __future__ import annotations
import json, sys, tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from chunk_builder import ChunkWriter          # noqa: E402
import build_corpus as BC                      # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(name)


def _row(bucket, n=8, media=False):
    ids = torch.arange(n, dtype=torch.long)[None]
    valid = torch.ones(1, n, dtype=torch.bool)
    me = torch.ones(3, 4, dtype=torch.bfloat16) if media else None
    if media:
        ids = ids.clone(); ids[0, :3] = 999
    return ids, valid, bucket, me, (999 if media else None)


# ---------------------------------------------------------------- 1. resume equivalence
print("[1] corpus resume produces the same chunks as an uninterrupted build")
with tempfile.TemporaryDirectory() as td:
    a, b = Path(td) / "straight", Path(td) / "resumed"
    w = ChunkWriter(a, ["code", "math"], tokens_per_chunk=16)
    for bucket in ("code", "math"):
        for _ in range(4):
            i, v, bk, me, mt = _row(bucket)
            w.add(i, v, bk, media_embeds=me, media_token_id=mt)
        w.mark_done(bucket); w.checkpoint(bucket)
    w.close()

    # interrupted after 'code', then a fresh writer resumes
    w1 = ChunkWriter(b, ["code", "math"], tokens_per_chunk=16)
    w1.resume()
    for _ in range(4):
        i, v, bk, me, mt = _row("code")
        w1.add(i, v, bk, media_embeds=me, media_token_id=mt)
    w1.mark_done("code"); w1.checkpoint("code")
    # simulate the crash: a half-written chunk of the NEXT bucket lands on disk
    i, v, bk, me, mt = _row("math")
    w1.add(i, v, bk, media_embeds=me, media_token_id=mt)
    w1.flush()
    del w1

    w2 = ChunkWriter(b, ["code", "math"], tokens_per_chunk=16)
    done = w2.resume()
    check("resume reports the finished bucket", done == ["code"], f"got {done}")
    for _ in range(4):
        i, v, bk, me, mt = _row("math")
        w2.add(i, v, bk, media_embeds=me, media_token_id=mt)
    w2.mark_done("math"); w2.checkpoint("math")
    w2.close()

    ca = sorted(p.name for p in a.glob("chunk_*.pt"))
    cb = sorted(p.name for p in b.glob("chunk_*.pt"))
    check("same chunk set", ca == cb, f"{ca} vs {cb}")
    sup = list(b.glob("*.superseded"))
    check("the torn post-checkpoint chunk was set aside", len(sup) == 1, f"{[p.name for p in sup]}")
    ta = [torch.load(a / n, weights_only=False) for n in ca]
    tb = [torch.load(b / n, weights_only=False) for n in cb]
    same = (len(ta) == len(tb) and all(
        len(x) == len(y) and all(torch.equal(p["ids"], q["ids"]) and p["bucket"] == q["bucket"]
                                 for p, q in zip(x, y)) for x, y in zip(ta, tb)))
    check("chunk CONTENTS identical after resume", same)
    ma = json.loads((a / "manifest.json").read_text())
    mb = json.loads((b / "manifest.json").read_text())
    check("manifest token counts identical",
          ma["tokens_by_bucket"] == mb["tokens_by_bucket"], f"{ma['tokens_by_bucket']} vs {mb['tokens_by_bucket']}")

# ---------------------------------------------------------------- 2. media guard fires early
print("[2] a media bucket with no media is refused AT THE BUCKET BOUNDARY")
with tempfile.TemporaryDirectory() as td:
    w = ChunkWriter(td, ["image"], tokens_per_chunk=10**9)
    i, v, bk, _, _ = _row("image")
    w.add(i, v, "image")                       # text only, no media
    try:
        w.guard_bucket("image"); check("guard_bucket refuses text-only image bucket", False)
    except RuntimeError as e:
        check("guard_bucket refuses text-only image bucket", "NONE came from real media" in str(e))
    w2 = ChunkWriter(td, ["image"], tokens_per_chunk=10**9)
    i, v, bk, me, mt = _row("image", media=True)
    w2.add(i, v, "image", media_embeds=me, media_token_id=mt)
    try:
        w2.guard_bucket("image"); check("guard_bucket accepts a real media bucket", True)
    except RuntimeError as e:
        check("guard_bucket accepts a real media bucket", False, str(e))

# ---------------------------------------------------------------- 3. source retry / abandon
print("[3] a failing stream is retried, then abandoned without killing the run")
calls = {"n": 0}


def flaky(hid, cfgn, split):
    calls["n"] += 1
    def g():
        yield {"text": "ok"}
        raise ConnectionError("simulated reset")
    return g()


BC.iter_rows, _orig = flaky, BC.iter_rows
BC.time.sleep = lambda s: None
rows = list(BC.robust_rows("fake/ds", None, "train", retries=3))
check("retried the configured number of times", calls["n"] == 3, f"called {calls['n']}x")
check("yielded rows from each attempt rather than dying", len(rows) == 3, f"{len(rows)} rows")
calls["n"] = 0


def clean(hid, cfgn, split):
    calls["n"] += 1
    return iter([{"text": "a"}, {"text": "b"}])


BC.iter_rows = clean
check("a healthy source is consumed once", len(list(BC.robust_rows("x", None, "train"))) == 2
      and calls["n"] == 1, f"called {calls['n']}x")
BC.iter_rows = _orig

print("\n" + ("GATE FAIL: " + ", ".join(FAIL) if FAIL else "GATE PASS: all resilience guards fire"))
sys.exit(1 if FAIL else 0)
