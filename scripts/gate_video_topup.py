"""Gate the video top-up's refusals. It appends to a FINISHED corpus, so its failure modes are
destructive in a way the rest of the pipeline's are not.

The specific hazard: ChunkWriter.resume() deletes any chunk at or beyond the checkpoint index,
treating it as a half-written orphan. That is correct during a build and catastrophic here --
it would delete finished chunks the pass has already folded in. So the top-up must refuse before
resume() is ever reached, and that refusal is what this gate drives.
"""
from __future__ import annotations
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import video_topup as VT   # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(name)


def corpus(td: Path, n_chunks: int, next_chunk: int, declared: int | None = None):
    td.mkdir(parents=True, exist_ok=True)
    for i in range(n_chunks):
        (td / f"chunk_{i:05d}.pt").write_bytes(b"x")
    (td / "corpus_state.json").write_text(json.dumps(
        {"next_chunk": next_chunk, "seen": {"video": 10}, "media_seen": {"video": 10},
         "done_buckets": ["video"]}))
    (td / "manifest.json").write_text(json.dumps(
        {"buckets": ["video"], "chunks": declared if declared is not None else n_chunks,
         "tokens_by_bucket": {"video": 10}, "media_tokens_by_bucket": {"video": 10}}))


def expect_exit(fn, needle, name):
    try:
        fn()
        check(name, False, "it proceeded")
    except SystemExit as e:
        check(name, needle in str(e), str(e)[:76])
    except Exception as e:                      # must fail LOUDLY, never silently proceed
        check(name, False, f"{type(e).__name__}: {str(e)[:60]}")


def main():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)

        d = t / "no_corpus"; d.mkdir()
        expect_exit(lambda: VT.run("/nonexistent", d, 1000),
                    "does not build one", "refuses a directory with no finished corpus")

        d = t / "ahead"; corpus(d, 5, next_chunk=3)
        expect_exit(lambda: VT.run("/nonexistent", d, 1000),
                    "Refusing", "refuses when chunks exist at or beyond the checkpoint index")
        check("the chunks it refused over still exist",
              len(list(d.glob("chunk_*.pt"))) == 5, "nothing was deleted")

        d = t / "mismatch"; corpus(d, 5, next_chunk=5, declared=7)
        expect_exit(lambda: VT.run("/nonexistent", d, 1000),
                    "does not agree with itself", "refuses a manifest that disagrees with disk")

        # the healthy shape must NOT be refused -- it fails later, on the missing model
        d = t / "ok"; corpus(d, 5, next_chunk=5)
        try:
            VT.run("/nonexistent", d, 1000)
            check("a consistent corpus is not refused", False, "it returned")
        except SystemExit as e:
            check("a consistent corpus is not refused", False, f"refused: {str(e)[:60]}")
        except Exception as e:
            check("a consistent corpus is not refused", True,
                  f"got past the guards and failed on the model, as expected "
                  f"({type(e).__name__})")

    print(("GATE FAIL: " + ", ".join(FAIL)) if FAIL else "GATE PASS")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
