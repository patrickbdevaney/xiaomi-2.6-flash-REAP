"""Measure that releasing shard handles actually returns the page cache.

The claim under test: pages faulted in through a live safe_open mmap cannot be reclaimed, so a
prefix scan over 65 shards pins an unbounded slice of the 177.8 GB checkpoint and the next large
allocation kills the box. Run as `A` (release suppressed) or `B` (release as shipped); the
caller compares. Separate processes because a mapping is a property of the process.
"""
from __future__ import annotations
import sys, gc
from pathlib import Path
import subprocess

sys.path.insert(0, str(Path(__file__).parent))


def avail_mb() -> int:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return -1


def reclaim():
    subprocess.run(["sync"], check=False)
    subprocess.run(["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], check=False)


def main(arm: str):
    import torch
    from transformers import AutoConfig
    import chunk_builder as CB
    from mimo_shards import ShardReader

    SRC = Path.home() / "models" / "MiMo-V2.6-Flash-RL"
    if arm == "A":
        ShardReader.release = lambda self: None      # the pre-fix behaviour, exactly

    reclaim()
    base = avail_mb()
    cfg = AutoConfig.from_pretrained(SRC, trust_remote_code=True)
    cfg._name_or_path = str(SRC)
    cfg._attn_implementation = "flex_attention"
    E = CB.Embedder(SRC, cfg, device="cuda")
    E._lazy("visual")                                 # the allocation that killed two runs
    gc.collect()
    held = avail_mb()
    reclaim()                                         # can the kernel get it back?
    after = avail_mb()
    print(f"ARM {arm} base={base} after_tower={held} after_reclaim={after} "
          f"unreclaimable={base - after}", flush=True)
    import os
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("A", "B"):
        main(sys.argv[1])
    else:
        PY = sys.executable
        res = {}
        for arm in ("A", "B"):
            out = subprocess.run([PY, __file__, arm], capture_output=True, text=True)
            line = [l for l in out.stdout.splitlines() if l.startswith("ARM")]
            if not line:
                print(out.stdout[-2000:]); print(out.stderr[-2000:]); sys.exit(1)
            print("  " + line[0])
            res[arm] = int(line[0].split("unreclaimable=")[1])
        print()
        print(f"  unreclaimable WITHOUT release : {res['A']:,} MB")
        print(f"  unreclaimable WITH release    : {res['B']:,} MB")
        ok = res["B"] < res["A"] - 2000
        print("\n" + ("GATE PASS: releasing the handles returns the cache "
                      f"({res['A'] - res['B']:,} MB recovered)" if ok else
                      f"GATE FAIL: release changed nothing (A={res['A']} B={res['B']})"))
        sys.exit(0 if ok else 1)
