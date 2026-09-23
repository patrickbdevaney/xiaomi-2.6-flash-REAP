#!/usr/bin/env bash
# Detached pull of MiMo-V2.6-Flash-RL (CLAUDE.md §1: never a child of a Claude Code Bash call).
#
# EVERYTHING, deliberately: the routed experts, the vision (681M) and audio (308M tokenizer +
# 127M patch encoder) towers, the 3 MTP nextn layers, AND the separate 5-layer DFlash drafter in
# dflash/. The towers are the entire reason to build this -- a Thor-runnable omnimodal coding
# agent is the deliverable, and no published artifact has them at a size that fits 128 GB. The
# drafter is 2.94 GB and stays resident unless measurement says it does not earn its place.
#
# 177.8 GB against 491 GB free. --local-dir writes real files (hub v1.x), not symlinks into the
# cache, so this does NOT cost a second copy.
set -u
DEST="$HOME/models/MiMo-V2.6-Flash-RL"
REPO="XiaomiMiMo/MiMo-V2.6-Flash-RL"
LOG="$(cd "$(dirname "$0")/.." && pwd)/logs/pull_mimo.log"
HF="${HF_BIN:-$HOME/glm-5.3-reap/.venv/bin/hf}"
mkdir -p "$DEST" "$(dirname "$LOG")"
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

say "pull start -> $DEST (free: $(df -h / | awk 'NR==2{print $4}'))"
# A 178 GB transfer WILL hit transient 5xx/reset. snapshot_download resumes per-file, so the
# retry is cheap; what it must not do is spin forever on a hard failure.
for attempt in $(seq 1 40); do
  say "attempt $attempt"
  if "$HF" download "$REPO" --local-dir "$DEST" >> "$LOG" 2>&1; then
    say "download reported success on attempt $attempt"; break
  fi
  say "attempt $attempt failed; sleeping 60s"
  sleep 60
done

# VERIFY, because rc=0 from a downloader has lied before (llama-imatrix --save-frequency).
# Compare the on-disk byte count against the repo's own usedStorage, and assert the pieces that
# the whole plan depends on are actually present.
say "verifying"
"${PY_BIN:-$HOME/glm-5.3-reap/.venv/bin/python}" - "$DEST" "$REPO" >> "$LOG" 2>&1 <<'PY'
import json, sys, os, urllib.request
dest, repo = sys.argv[1], sys.argv[2]
want = json.load(urllib.request.urlopen(
    f"https://huggingface.co/api/models/{repo}", timeout=60)).get("usedStorage")
have = sum(os.path.getsize(os.path.join(r, f))
           for r, _, fs in os.walk(dest) for f in fs if not r.endswith(".cache"))
print(f"on disk {have/1e9:.1f} GB vs usedStorage {want/1e9:.1f} GB  delta {(have-want)/1e9:+.2f} GB")
idx = json.load(open(os.path.join(dest, "model.safetensors.index.json")))
shards = set(idx["weight_map"].values())
missing = [s for s in shards if not os.path.exists(os.path.join(dest, s))]
must = ["config.json", "dflash/config.json", "dflash/mask_embedding.pt",
        "tokenizer.json", "preprocessor_config.json", "modeling_mimo_v2.py"]
absent = [m for m in must if not os.path.exists(os.path.join(dest, m))]
print(f"shards {len(shards)-len(missing)}/{len(shards)} present")
if missing: print("MISSING SHARDS:", missing[:10])
if absent:  print("MISSING FILES:", absent)
ok = not missing and not absent and abs(have-want) < 2e9
print("VERIFY PASS" if ok else "VERIFY FAIL")
sys.exit(0 if ok else 1)
PY
say "verify rc=$?  free now: $(df -h / | awk 'NR==2{print $4}')"
say "pull done"
