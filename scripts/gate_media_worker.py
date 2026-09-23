"""Prove the media worker contains the blast radius.

The design claim is narrow and testable: when the tower process dies for ANY reason -- including
the SIGKILL the OOM killer sends -- the parent loses one sample and keeps going. If that is not
true the isolation buys nothing, because the whole point is to survive an allocation nobody
predicted.

SIGKILL is used deliberately: it is precisely what the kernel OOM killer delivers, it is
unmaskable, and unlike a real OOM it does not risk the machine.
"""
from __future__ import annotations
import os
import signal
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from media_worker import MediaWorker      # noqa: E402

SRC = Path.home() / "models" / "MiMo-V2.6-Flash-RL"
FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(name)


def tiny_image(patch=16, tpatch=2, h=32, w=32):
    pv = torch.randn(h * w, 3 * patch * patch * tpatch, dtype=torch.float32)
    return pv, torch.tensor([[1, h, w]])


def main():
    print("[1] the worker computes real embeddings")
    t0 = time.time()
    MW = MediaWorker(SRC, device="cuda")
    print(f"    worker ready in {time.time()-t0:.0f}s, pid {MW._p.pid}")
    pv, grid = tiny_image()
    out = MW.visual(pv, grid)
    check("visual() returns embeddings", out is not None and out.ndim == 2,
          f"{None if out is None else tuple(out.shape)}")
    expect = (32 * 32) // 4          # spatial_merge_size 2 -> /4 rows
    check("one embedding row per merged patch", out is not None and out.shape[0] == expect,
          f"{None if out is None else out.shape[0]} vs {expect}")

    print("\n[2] the worker volunteers as the OOM victim")
    adj = Path(f"/proc/{MW._p.pid}/oom_score_adj").read_text().strip()
    parent_adj = Path("/proc/self/oom_score_adj").read_text().strip()
    print(f"    worker oom_score_adj={adj}, parent={parent_adj}")
    check("worker is the most killable process", adj == "1000", f"got {adj}")
    check("worker is more killable than its parent", int(adj) > int(parent_adj),
          f"{adj} vs {parent_adj}")

    print("\n[3] SIGKILL mid-run -- exactly what the OOM killer sends")
    old_pid = MW._p.pid
    MW.call_timeout = 25                      # keep the gate quick; production waits far longer
    os.kill(old_pid, signal.SIGKILL)
    time.sleep(2)
    check("worker is dead", not MW._p.is_alive() or MW._p.pid != old_pid)
    res = MW.visual(pv, grid)
    check("the call fails SOFTLY, returning None rather than raising", res is None,
          f"got {type(res).__name__}")
    check("parent is still running", True)

    print("\n[4] the run continues: worker restarted, next sample succeeds")
    check("worker was restarted with a new pid", MW._p.pid != old_pid,
          f"{old_pid} -> {MW._p.pid}")
    out2 = MW.visual(pv, grid)
    check("embeddings again after the restart", out2 is not None and out2.shape[0] == expect,
          f"{None if out2 is None else tuple(out2.shape)}")
    check("the loss is accounted, not silent", MW.deaths >= 1 and MW.skipped >= 1,
          f"deaths={MW.deaths} skipped={MW.skipped}")

    MW.close()
    print("\n" + ("GATE FAIL: " + ", ".join(FAIL) if FAIL else
                  "GATE PASS: a killed tower costs one sample, not the run"))
    sys.stdout.flush()
    os._exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
