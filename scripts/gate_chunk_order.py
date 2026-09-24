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

    print(("GATE FAIL: " + ", ".join(FAIL)) if FAIL else "GATE PASS")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
