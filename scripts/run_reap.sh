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

# MEMORY PRE-FLIGHT. The first launch of this run was OOM-killed 7 minutes in, at a cgroup peak
# of 7.0 GiB on a 122 GiB box -- because ~115 GiB was already held by the nvmap driver pool,
# which appears in NO process's RSS, in Cached, or in Slab. The kernel OOM killer scores by RSS,
# so it selected us. `echo 3 > drop_caches` returned the box to 119 GiB instantly; `echo 1`,
# which is all the memguard did at the time, reclaimed nothing. So: reclaim with the full
# shrinker, then REFUSE TO START if the box is still short. Starting a 34-hour run into a
# poisoned allocator only buys another 7-minute failure, and the corpus stage is not resumable.
# VOLUNTEER AS THE OOM VICTIM. On this box there is no kernel-side limit that can bound a CUDA
# allocation: unified memory means the driver pins ordinary system RAM, which is charged to no
# cgroup and appears in no Rss counter, so `MemoryMax` cannot constrain it and the OOM killer --
# which ranks by RSS -- has repeatedly picked the wrong process. The only remaining lever is to
# make OURSELVES the preferred victim, so that when the box is cornered it takes this run (which
# resumes exactly, per bucket in stage 1 and per chunk in stage 2, and is restarted automatically)
# instead of the session or the desktop. The media worker sets 1000 so it is taken before us.
echo 500 > /proc/self/oom_score_adj 2>/dev/null || say "WARN: could not set oom_score_adj"
MIN_AVAIL_MB=${MIN_AVAIL_MB:-90000}
sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || say "WARN: drop_caches unavailable"
AVAIL=$(awk '/MemAvailable:/{print int($2/1024)}' /proc/meminfo)
say "pre-flight MemAvailable ${AVAIL}MB (floor ${MIN_AVAIL_MB}MB)"
if [ "$AVAIL" -lt "$MIN_AVAIL_MB" ]; then
  say "ABORT: only ${AVAIL}MB available after reclaim -- something live holds the box."
  say "  check: ps -eo pid,rss,cmd --sort=-rss | head; systemctl --user list-units --state=running"
  exit 1
fi

# MEMORY TRACE. Two runs were OOM-killed with nothing in the log to say where the memory went,
# because the kernel's own accounting does not see the driver pool and systemd's cgroup peak
# (7.0 then 10.7 GiB on a 122 GiB box) is not the number that matters. A 5-second sample costs
# nothing and turns the next failure into a trajectory instead of a guess.
( while true; do
    echo "$(date +%H:%M:%S) avail=$(awk '/MemAvailable:/{print int($2/1024)}' /proc/meminfo)MB" \
         "cached=$(awk '/^Cached:/{print int($2/1024)}' /proc/meminfo)MB" \
         "stage=$(cat logs/.stage 2>/dev/null)"
    sleep 5
  done ) >> logs/reap_mem.log 2>&1 &
MEMPID=$!
trap 'kill $MEMPID 2>/dev/null' EXIT

if [ ! -f artifacts/chunks/manifest.json ]; then
  echo stage1-corpus > logs/.stage
  say "STAGE 1 build corpus"
  "$PY" scripts/build_corpus.py --out artifacts/chunks --total-tokens "$TOTAL" \
      --seq-len "$SEQ" --tokens-per-chunk 2000000 >> "$LOG" 2>&1 \
    || { say "STAGE 1 FAILED"; exit 1; }
  say "STAGE 1 done, $(du -sh artifacts/chunks | cut -f1) of chunks"
else
  say "STAGE 1 SKIPPED -- artifacts/chunks/manifest.json already exists"
fi

echo stage2-pass > logs/.stage
say "STAGE 2 calibration pass (48 layers)"
"$PY" scripts/calib_pass.py --chunks artifacts/chunks --out artifacts/saliency >> "$LOG" 2>&1
RC=$?
say "STAGE 2 rc=$RC"
[ "$RC" -eq 0 ] || { say "PASS FAILED -- accumulators left intact for resume"; exit 1; }

"$PY" scripts/verify_pass.py --out artifacts/saliency >> "$LOG" 2>&1 \
  && say "FINAL VERIFY PASS" || { say "FINAL VERIFY FAILED"; exit 1; }
say "REAP calibration COMPLETE"

# ---------------------------------------------------------------------------------------------
# STAGES 3-7. Each one skips if its output already exists, so a restart anywhere past the pass
# costs only the stage it died in. None of them touch the source checkpoint.
# ---------------------------------------------------------------------------------------------
RATIO=${PRUNE_RATIO:-0.50}
MASKS=artifacts/masks
mkdir -p "$MASKS"

echo stage3-criterion > logs/.stage
if [ ! -f "$MASKS/comparison.json" ]; then
  say "STAGE 3 criterion comparison at ratio $RATIO"
  "$PY" scripts/criterion_compare.py --acc artifacts/saliency/accumulators.pt \
        --out-dir "$MASKS" --ratio "$RATIO" >> "$LOG" 2>&1 \
    || { say "STAGE 3 FAILED"; exit 1; }
else
  say "STAGE 3 SKIPPED -- $MASKS/comparison.json exists"
fi
# The winner is chosen by WORST-DOMAIN retention, never by the mean: averaging is how a criterion
# that destroys one capability outscores one that preserves all of them (arXiv 2606.03328 measured
# 2.85 averaged points hiding 51.9 points of code retention).
CRIT=$("$PY" -c "import json;r=json.load(open('$MASKS/comparison.json'));print(r[0]['criterion'])")
MODE=$("$PY" -c "import json;r=json.load(open('$MASKS/comparison.json'));print(r[0]['mode'])")
say "STAGE 3 winner: $CRIT / $MODE"

echo stage4-budget > logs/.stage
if [ ! -f "$MASKS/layer_budget.json" ]; then
  say "STAGE 4 per-layer budget search (EvoESAP), criterion $CRIT"
  "$PY" scripts/layer_budget.py --acc artifacts/saliency/accumulators.pt \
        --out "$MASKS/layer_budget.json" --ratio "$RATIO" --criterion "$CRIT" \
    >> "$LOG" 2>&1 || { say "STAGE 4 FAILED"; exit 1; }
else
  say "STAGE 4 SKIPPED -- $MASKS/layer_budget.json exists"
fi
# REPORTED, NOT APPLIED. `n_routed_experts` is a global scalar in the modelling code and llama.cpp
# reads a single `n_expert`, so a ragged per-layer budget cannot be loaded by transformers, vLLM
# or any GGUF without patched modelling. The search runs because its GAIN is the number that says
# whether non-portability would be worth it; acting on it is a deliberate choice, not a default.
say "STAGE 4 gain over uniform: $("$PY" -c "import json;d=json.load(open('$MASKS/layer_budget.json'));print(f\"{d['uniform_worst']:.4f} -> {d['searched_worst']:.4f} ({d['gain']:+.4f})\")")"
say "STAGE 4 NOT APPLIED -- uniform budget keeps the checkpoint loadable; see layer_budget.py"

echo stage5-select > logs/.stage
if [ ! -f "$MASKS/mask.json" ]; then
  say "STAGE 5 expert selection: $CRIT / $MODE at $RATIO (uniform per-layer)"
  "$PY" scripts/reap_select.py --acc artifacts/saliency/accumulators.pt \
        --out "$MASKS/mask.json" --ratio "$RATIO" --mode "$MODE" --criterion "$CRIT" \
    >> "$LOG" 2>&1 || { say "STAGE 5 FAILED"; exit 1; }
else
  say "STAGE 5 SKIPPED -- $MASKS/mask.json exists"
fi

echo stage6-routerkd > logs/.stage
if [ ! -f "$MASKS/router_kd.pt" ] || [ "${FORCE_KD:-0}" = "1" ]; then
  # --force because this script IS reap_run.service: router_kd_run refuses to start
  # while that unit is active, which would otherwise make it refuse itself.
  say "STAGE 6 router KD (output matching, routers only)"
  "$PY" scripts/router_kd_run.py --chunks artifacts/chunks --mask "$MASKS/mask.json" \
        --out "$MASKS" --force >> "$LOG" 2>&1 || { say "STAGE 6 FAILED"; exit 1; }
else
  say "STAGE 6 SKIPPED -- $MASKS/router_kd.pt exists"
fi

echo stage7-apply > logs/.stage
DST=${REAP_DST:-$HOME/models/MiMo-V2.6-Flash-REAP$(printf '%.0f' "$(echo "$RATIO*100" | bc)")}
if [ ! -f "$DST/config.json" ]; then
  # The pruned checkpoint is written, never edited in place, so the source survives a failure --
  # and there must be room for it before a multi-hour copy discovers there is not.
  NEED=$(du -sm "$HOME/models/MiMo-V2.6-Flash-RL" | cut -f1)
  FREE=$(df -Pm "$(dirname "$DST")" | awk 'NR==2{print $4}')
  say "STAGE 7 apply mask -> $DST (need <= ${NEED}MB, free ${FREE}MB)"
  if [ "$FREE" -lt "$NEED" ]; then
    say "ABORT: not enough free space for the pruned checkpoint. Nothing was deleted."
    exit 1
  fi
  "$PY" scripts/apply_mask.py --dst "$DST" --mask "$MASKS/mask.json" >> "$LOG" 2>&1 \
    || { say "STAGE 7 FAILED"; exit 1; }
else
  say "STAGE 7 SKIPPED -- $DST/config.json exists"
fi

echo done > logs/.stage
say "REAP COMPLETE: $DST  (criterion $CRIT / $MODE at $RATIO)"
