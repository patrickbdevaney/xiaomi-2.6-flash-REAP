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
START_STAGE=${START_STAGE:-1}
DEAD_FLAG=""; [ "${ALLOW_DEAD:-0}" = "1" ] && DEAD_FLAG="--allow-dead-domains"
DRY_FLAG="";  [ "${DRY_RUN:-0}" = "1" ]    && DRY_FLAG="--dry-run"
ACC=${ACC_PATH:-artifacts/saliency/accumulators.pt}
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
# The floor is what STAGE 2 needs: it streams 48 layers of a 173 GB model. Entering at a later
# stage does not, and enforcing stage 2's floor there would refuse a legitimate re-selection --
# stages 3-5 are numpy over a 56 MB accumulator file.
if [ "$START_STAGE" -le 2 ]; then MIN_AVAIL_MB=${MIN_AVAIL_MB:-90000}
else MIN_AVAIL_MB=${MIN_AVAIL_MB:-20000}; fi
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

if [ "$START_STAGE" -le 2 ]; then
echo stage2-pass > logs/.stage
say "STAGE 2 calibration pass (48 layers)"
"$PY" scripts/calib_pass.py --chunks artifacts/chunks --out artifacts/saliency >> "$LOG" 2>&1
RC=$?
say "STAGE 2 rc=$RC"
[ "$RC" -eq 0 ] || { say "PASS FAILED -- accumulators left intact for resume"; exit 1; }

"$PY" scripts/verify_pass.py --out artifacts/saliency >> "$LOG" 2>&1 \
  && say "FINAL VERIFY PASS" || { say "FINAL VERIFY FAILED"; exit 1; }
say "REAP calibration COMPLETE"
else
  say "STAGES 1-2 SKIPPED -- START_STAGE=$START_STAGE"
fi

# ---------------------------------------------------------------------------------------------
# STAGES 3-7. Each one skips if its output already exists, so a restart anywhere past the pass
# costs only the stage it died in. None of them touch the source checkpoint.
# ---------------------------------------------------------------------------------------------
RATIO=${PRUNE_RATIO:-0.50}
MASKS=${MASKS_DIR:-artifacts/masks}
mkdir -p "$MASKS"
# OPERATIONAL KNOBS, all off by default.
#   START_STAGE  enter the pipeline at a later stage (re-select at a different ratio without
#                re-running a 33-hour pass; also what makes the 3->7 chain rehearsable at all)
#   ALLOW_DEAD   score only the domains that have routed mass (a deliberately partial pass)
#   DRY_RUN      stage 7 plans the checkpoint instead of writing it

echo stage3-criterion > logs/.stage
if [ "$START_STAGE" -le 3 ] && [ ! -f "$MASKS/comparison.json" ]; then
  say "STAGE 3 criterion comparison at ratio $RATIO"
  "$PY" scripts/criterion_compare.py --acc "$ACC" \
        --out-dir "$MASKS" --ratio "$RATIO" $DEAD_FLAG >> "$LOG" 2>&1 \
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
if [ "$START_STAGE" -le 4 ] && [ ! -f "$MASKS/layer_budget.json" ]; then
  say "STAGE 4 per-layer budget search (EvoESAP), criterion $CRIT"
  "$PY" scripts/layer_budget.py --acc "$ACC" \
        --out "$MASKS/layer_budget.json" --ratio "$RATIO" --criterion "$CRIT" \
        --generations "${GENERATIONS:-300}" $DEAD_FLAG \
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
if [ "$START_STAGE" -le 5 ] && [ ! -f "$MASKS/mask.json" ]; then
  say "STAGE 5 expert selection: $CRIT / $MODE at $RATIO (uniform per-layer)"
  # PROTECT_FRAC protects the top slice of EVERY DOMAIN separately. It is the only guard against
  # a global ranking dropping an expert that only one thin domain uses. Measured at 5 chunks:
  # video reached 35.4% expert coverage (6.2% in its worst layer, 16 of 256) against agentic's
  # 73.6%, on the same token count as audio, because its clips carry no text diversity -- and
  # video has exactly ONE chunk in the corpus, so that number is final. Default 0 until the full
  # accumulators exist, because the right value depends on how much the domains overlap.
  "$PY" scripts/reap_select.py --acc "$ACC" \
        --out "$MASKS/mask.json" --ratio "$RATIO" --mode "$MODE" --criterion "$CRIT" $DEAD_FLAG \
        --protect-frac "${PROTECT_FRAC:-0}" \
    >> "$LOG" 2>&1 || { say "STAGE 5 FAILED"; exit 1; }
else
  say "STAGE 5 SKIPPED -- $MASKS/mask.json exists"
fi

echo stage6-routerkd > logs/.stage
if [ "$START_STAGE" -le 6 ] && { [ ! -f "$MASKS/router_kd.pt" ] || [ "${FORCE_KD:-0}" = "1" ]; }; then
  # --force because this script IS reap_run.service: router_kd_run refuses to start
  # while that unit is active, which would otherwise make it refuse itself.
  say "STAGE 6 router KD (output matching, routers only)"
  # BOUNDED. Left unbounded this streams every batch of a 2 M-token chunk through 48 layers on
  # the GPU. Rehearsing it that way against the live calibration pass cost the pass a 228 s
  # layer (normal ~90 s) and took MemAvailable to 28 GB -- so the knobs exist, and stage 6 is
  # never to be exercised while stage 2 holds the device.
  "$PY" scripts/router_kd_run.py --chunks artifacts/chunks --mask "$MASKS/mask.json" \
        --out "$MASKS" --device "${KD_DEVICE:-cuda}" --tokens "${KD_TOKENS:-2048}" \
        --steps "${KD_STEPS:-300}" --batch "${KD_BATCH:-512}" \
        ${KD_MAX_LAYERS:+--max-layers $KD_MAX_LAYERS} \
        ${KD_MAX_BATCHES:+--max-batches $KD_MAX_BATCHES} \
        ${KD_MAX_SEQ:+--max-seq $KD_MAX_SEQ} \
        --force >> "$LOG" 2>&1 || { say "STAGE 6 FAILED"; exit 1; }
else
  say "STAGE 6 SKIPPED -- $MASKS/router_kd.pt exists"
fi

echo stage7-apply > logs/.stage
DST=${REAP_DST:-$HOME/models/MiMo-V2.6-Flash-REAP$(printf '%.0f' "$(echo "$RATIO*100" | bc)")}
if [ "$START_STAGE" -le 7 ] && [ ! -f "$DST/config.json" ]; then
  # The pruned checkpoint is written, never edited in place, so the source survives a failure --
  # and there must be room for it before a multi-hour copy discovers there is not.
  NEED=$(du -sm "$HOME/models/MiMo-V2.6-Flash-RL" | cut -f1)
  FREE=$(df -Pm "$(dirname "$DST")" | awk 'NR==2{print $4}')
  say "STAGE 7 apply mask -> $DST (need <= ${NEED}MB, free ${FREE}MB)"
  if [ "$FREE" -lt "$NEED" ]; then
    say "ABORT: not enough free space for the pruned checkpoint. Nothing was deleted."
    exit 1
  fi
  "$PY" scripts/apply_mask.py --dst "$DST" --mask "$MASKS/mask.json" \
        --router-kd "$MASKS/router_kd.pt" $DRY_FLAG >> "$LOG" 2>&1 \
    || { say "STAGE 7 FAILED"; exit 1; }
else
  say "STAGE 7 SKIPPED -- $DST/config.json exists"
fi

echo done > logs/.stage
say "REAP COMPLETE: $DST  (criterion $CRIT / $MODE at $RATIO)"
