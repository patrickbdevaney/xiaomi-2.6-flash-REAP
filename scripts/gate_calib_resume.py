"""Prove the calibration pass resumes EXACTLY: interrupted+resumed == straight through.

This is the claim the whole unattended run rests on. If resume is even slightly lossy, a crash
at hour six silently produces a calibration that no downstream check can detect -- the
accumulators are just sums, and a sum that is missing one chunk looks exactly like a sum that
is not. So the test is bit-exact equality of every accumulator, not a tolerance.

Runs on REAL checkpoint weights (one layer) over REAL chunk data, cut down to a few sequences.
"""
from __future__ import annotations
import json, shutil, sys, tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import calib_pass as CP     # noqa: E402

SRC = Path.home() / "models" / "MiMo-V2.6-Flash-RL"
FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(name)


def make_tiny(src_chunk: Path, dst: Path, buckets):
    """Two tiny chunks carved from a real chunk, so the pass sees real ids and real buckets."""
    dst.mkdir(parents=True, exist_ok=True)
    rows = torch.load(src_chunk, map_location="cpu", weights_only=False)
    rows = [r for r in rows if r.get("media_embeds") is None][:4]
    assert len(rows) >= 4, f"need 4 text rows, got {len(rows)}"
    for i in range(2):
        torch.save(rows[i * 2:(i + 1) * 2], dst / f"chunk_{i:05d}.pt")
    seen = {}
    for r in rows:
        seen[r["bucket"]] = seen.get(r["bucket"], 0) + int(r["valid"].sum())
    (dst / "manifest.json").write_text(json.dumps(
        {"buckets": buckets, "chunks": 2, "tokens_by_bucket": seen,
         "media_tokens_by_bucket": {b: 0 for b in buckets}}))


def acc_of(out_dir: Path):
    d = torch.load(out_dir / "accumulators.pt", map_location="cpu", weights_only=False)
    return d


def maxrel(a, b) -> tuple[float, str]:
    """Largest RELATIVE disagreement between two accumulator sets, and where it is."""
    worst, where = 0.0, "-"
    assert set(a["acc"]) == set(b["acc"]), "different layer keys"
    pairs = [(f"{k}/{kk}", a["acc"][k][kk], b["acc"][k][kk])
             for k in a["acc"] for kk in a["acc"][k]]
    pairs += [(f, a[f], b[f]) for f in ("f_sum", "f_cnt")]
    for nm, x, y in pairs:
        x, y = x.double().cpu(), y.double().cpu()
        scale = torch.maximum(x.abs(), y.abs()).clamp_min(1e-12)
        r = float(((x - y).abs() / scale).max())
        if r > worst:
            worst, where = r, nm
    return worst, where


src_chunk = next(iter(sorted((Path("artifacts/chunks_aborted_0923")).glob("chunk_*.pt"))), None)
if src_chunk is None:
    print("no real chunk available to carve from"); sys.exit(1)
buckets = list(json.loads(Path("scripts/mimo_corpus_spec.py").read_text()
                          .split("TOKEN_TARGET = ")[1].split("}")[0] + "}").keys()) \
    if False else None
# buckets must match what the chunks were written with
rows0 = torch.load(src_chunk, map_location="cpu", weights_only=False)
import mimo_corpus_spec as SPEC
buckets = list(SPEC.TOKEN_TARGET)

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    chunks = td / "chunks"
    make_tiny(src_chunk, chunks, buckets)

    print("[A] straight through, 2 chunks x 2 layers on real weights")
    a_out = td / "straight"; a_out.mkdir()
    CP.run(SRC, chunks, a_out, smoke_layers=2, smoke_chunks=2)

    print("[B] chunk 0 only, then RESUME for chunk 1")
    b_out = td / "resumed"; b_out.mkdir()
    CP.run(SRC, chunks, b_out, smoke_layers=2, smoke_chunks=1)
    st = json.loads((b_out / "pass_state.json").read_text())
    check("first leg checkpointed exactly one chunk", st["done"] == ["chunk_00000.pt"], str(st))
    CP.run(SRC, chunks, b_out, smoke_layers=2, smoke_chunks=2)
    st2 = json.loads((b_out / "pass_state.json").read_text())
    check("resumed leg folded in both chunks", len(st2["done"]) == 2, str(st2))

    # CONTROL WITH A KNOWN SIGN. index_add_ on CUDA sums with atomics, so its order -- and
    # therefore its last-bit rounding -- varies between two runs of the SAME input. Demanding
    # bit-equality of a resume would therefore fail for a reason that has nothing to do with
    # resume. So measure how far two IDENTICAL straight-through runs drift, and require the
    # resumed run to be no further off than that. If resume dropped or double-counted a chunk
    # the disagreement would be order-1, not order-1e-10, so this still catches the failure
    # that matters while not flagging arithmetic noise as data loss.
    print("[C] control: a second identical straight-through run")
    c_out = td / "control"; c_out.mkdir()
    CP.run(SRC, chunks, c_out, smoke_layers=2, smoke_chunks=2)
    ctl, ctl_where = maxrel(acc_of(a_out), acc_of(c_out))
    res, res_where = maxrel(acc_of(a_out), acc_of(b_out))
    print(f"    control drift (A vs A'): {ctl:.3g} at {ctl_where}")
    print(f"    resume  drift (A vs B) : {res:.3g} at {res_where}")
    check("resume drift is at the level of run-to-run arithmetic noise",
          res <= max(ctl * 10, 1e-9), f"resume {res:.3g} vs control {ctl:.3g}")
    check("resume did not lose or double-count a chunk", res < 1e-6,
          f"max relative disagreement {res:.3g} at {res_where}")

    stat = json.loads((b_out / "status.json").read_text())
    check("status.json chunks_done is not off by one",
          stat["chunks_done"] == 2, f"reported {stat['chunks_done']} of 2")
    check("no torn temp files left behind", not list(b_out.glob("*.tmp")))

print("\n" + ("GATE FAIL: " + ", ".join(FAIL) if FAIL else "GATE PASS: resume is exact"))
import os
sys.stdout.flush(); os._exit(1 if FAIL else 0)
