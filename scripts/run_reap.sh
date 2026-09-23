#!/usr/bin/env bash
# The REAP calibration run. Detached (CLAUDE.md §1): build the corpus, then sweep all 48 layers.
#
# Resumable at both stages. The pass records every folded-in chunk in pass_state.json and skips
# them on restart, and verify_pass runs after each chunk and ABORTS on a broken invariant --
# HOPE's F cannot be recovered from a partial result, so a corrupted accumulator means starting
# over regardless. Better to stop at chunk 3 than to discover it after 34 hours.
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY_BIN:-$HOME/glm-5.3-reap/.venv/bin/python}"
LOG=logs/reap_run.log
TOTAL=${TOTAL_TOKENS:-50000000}
SEQ=${SEQ_LEN:-4096}
mkdir -p logs artifacts
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

say "REAP run start: $TOTAL tokens, seq_len $SEQ, free $(df -h / | awk 'NR==2{print $4}')"

if [ ! -f artifacts/chunks/manifest.json ]; then
  say "STAGE 1 build corpus"
  "$PY" scripts/build_corpus.py --out artifacts/chunks --total-tokens "$TOTAL" \
      --seq-len "$SEQ" --tokens-per-chunk 2000000 >> "$LOG" 2>&1 \
    || { say "STAGE 1 FAILED"; exit 1; }
  say "STAGE 1 done, $(du -sh artifacts/chunks | cut -f1) of chunks"
else
  say "STAGE 1 SKIPPED -- artifacts/chunks/manifest.json already exists"
fi

say "STAGE 2 calibration pass (48 layers)"
"$PY" scripts/calib_pass.py --chunks artifacts/chunks --out artifacts/saliency >> "$LOG" 2>&1
RC=$?
say "STAGE 2 rc=$RC"
[ "$RC" -eq 0 ] || { say "PASS FAILED -- accumulators left intact for resume"; exit 1; }

"$PY" scripts/verify_pass.py --out artifacts/saliency >> "$LOG" 2>&1 \
  && say "FINAL VERIFY PASS" || { say "FINAL VERIFY FAILED"; exit 1; }
say "REAP calibration COMPLETE -- next: HOPE QP, per-layer budget search, Router KD"
