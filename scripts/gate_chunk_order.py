"""Gate the chunk interleaving: it must be a permutation, it must front-load every domain, and
it must not change the answer.

The last one is the whole licence for touching a running 33-hour pass. Every accumulator is a
sum of per-chunk contributions, so reordering can only move floating-point rounding; if it moved
anything else, the reordering would be a correctness change rather than a scheduling one.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from calib_pass import chunk_order   # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(name)


def main():
    real = Path("artifacts/chunks")
    files = sorted(real.glob("chunk_*.pt"))
    order = chunk_order(real, files)
    buckets = json.loads((real / "chunk_buckets.json").read_text())

    check("order is a permutation of the input", sorted(order) == sorted(files),
          f"{len(order)} chunks, no loss and no duplication")

    nb = len(set(buckets.values()))
    first_nb = {buckets[f.name] for f in order[:nb]}
    check("every domain appears within the first n_buckets chunks", len(first_nb) == nb,
          f"{len(first_nb)} of {nb} domains in the first {nb}")

    # where the media domains land, old vs new -- the number this change exists for
    def frac(seq, b):
        return min(i for i, f in enumerate(seq) if buckets[f.name] == b) / len(seq)
    media = ["image", "audio", "video"]
    was = {b: frac(files, b) for b in media}
    now = {b: frac(order, b) for b in media}
    check("media evidence arrives in the first third of the run",
          all(v < 0.34 for v in now.values()),
          "first chunk at " + ", ".join(f"{b} {was[b]:.0%}->{now[b]:.0%}" for b in media))

    # a bucket's own chunks must stay in their original relative order: resume reads a name set,
    # so this is not correctness, but it keeps the log readable and the ETA monotone.
    bad = [b for b in set(buckets.values())
           if [f.name for f in order if buckets[f.name] == b]
           != sorted(f.name for f in files if buckets[f.name] == b)]
    check("within-bucket order is preserved", not bad, f"{len(bad)} buckets reordered internally")

    # ---- the claim that licenses the change: summation is order-independent ----
    g = torch.Generator().manual_seed(0)
    parts = [torch.rand(64, 256, generator=g, dtype=torch.float64) * 10 ** torch.randint(
        -3, 4, (1,), generator=g).item() for _ in range(len(files))]
    idx = [files.index(f) for f in order]
    a = torch.zeros(64, 256, dtype=torch.float64)
    for p in parts:
        a += p
    b = torch.zeros(64, 256, dtype=torch.float64)
    for i in idx:
        b += parts[i]
    rel = float(((a - b).abs() / a.abs().clamp(min=1e-30)).max())
    check("reordering changes the sum only by float rounding", rel < 1e-12,
          f"max relative difference {rel:.2e} over {len(parts)} float64 partial sums")

    # negative control: a sum that is NOT order-independent must be visible to this test
    c = torch.zeros(64, 256, dtype=torch.float64)
    for j, i in enumerate(idx):
        c += parts[i] * (1.0 + 1e-6 * j)          # an order-DEPENDENT accumulation
    rel_bad = float(((a - c).abs() / a.abs().clamp(min=1e-30)).max())
    check("the order-independence test can see an order-dependent sum", rel_bad > 1e-9,
          f"{rel_bad:.2e}")

    # ---- the stage 2B seam: a chunk appended AFTER the cache was written ----
    # video_topup adds chunk 27 and relabels the cache. If chunk_order could not place a chunk
    # the cache does not know, the new chunk would sort as an unknown bucket and the fold would
    # either skip it or order it last -- and this seam runs exactly once, unattended, 30 hours in.
    import tempfile, pathlib, torch as _t
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        for i, b in enumerate(["agentic", "code", "video"]):
            _t.save([{"bucket": b, "ids": _t.zeros(1, 4, dtype=_t.long),
                      "valid": _t.ones(1, 4, dtype=_t.bool)}], td / f"chunk_{i:05d}.pt")
        (td / "chunk_buckets.json").write_text(json.dumps(
            {"chunk_00000.pt": "agentic", "chunk_00001.pt": "code", "chunk_00002.pt": "video"}))
        # the appended chunk, absent from the cache
        _t.save([{"bucket": "video", "ids": _t.zeros(1, 4, dtype=_t.long),
                  "valid": _t.ones(1, 4, dtype=_t.bool)}], td / "chunk_00003.pt")
        files2 = sorted(td.glob("chunk_*.pt"))
        order2 = chunk_order(td, files2)
        cache2 = json.loads((td / "chunk_buckets.json").read_text())
        check("an appended chunk is placed even though the cache predates it",
              len(order2) == 4 and set(order2) == set(files2), f"{len(order2)} chunks ordered")
        check("its bucket is read from the chunk itself and cached",
              cache2.get("chunk_00003.pt") == "video", f"cached as {cache2.get('chunk_00003.pt')!r}")
        seq = [cache2[f.name] for f in order2]
        check("the appended video chunk sorts as a second video chunk, not first",
              seq.index("video") < len(seq) - 1 and seq[-1] == "video", f"order {seq}")

    print(("GATE FAIL: " + ", ".join(FAIL)) if FAIL else "GATE PASS")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
