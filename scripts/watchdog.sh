#!/usr/bin/env bash
# Detached supervisor for the REAP run. Survives the session that started it.
#
# WHAT THIS IS FOR. reap_run.service already restarts on failure, and memguard already contains
# an OOM. What neither does is notice that the run has STOPPED FOR GOOD: StartLimitBurst=6 over
# 6h means a deterministic failure -- a VerifyError, a bad chunk, a bug in a late stage -- burns
# every restart in about nine minutes and then the unit sits in `failed` indefinitely with
# nobody told. Both VerifyErrors in this project would have ended that way unattended.
#
# THE DISTINCTION THAT MATTERS, and the reason this is not just `systemctl restart` on a timer:
#
#   TRANSIENT   the run made progress since the last restart (a chunk was checkpointed).
#               An OOM kill looks like this. Resetting the limit and restarting is correct.
#   DETERMINISTIC  the run failed again having folded in no new chunk. Restarting replays the
#               same failure forever and turns a diagnosable stop into an invisible loop.
#
# So a reset is only ever granted against PROGRESS. Without progress the watchdog stops trying,
# writes why, and leaves the failure intact for a human to read. It never deletes anything and
# never edits the pipeline's state -- its only actions are `reset-failed` and `start`.
set -u
ROOT=${ROOT:-/home/patrickd/xiaomi-2.6-flash-REAP}
UNIT=${UNIT:-reap_run.service}
STATUS="$ROOT/logs/WATCHDOG_STATUS.txt"
LOG="$ROOT/logs/watchdog.log"
STATE="$ROOT/logs/.watchdog_state"
POLL=${POLL:-300}
MAX_REVIVALS=${MAX_REVIVALS:-12}

say() { echo "[$(date -Is)] $*" >> "$LOG"; }

progress() {   # number of chunks folded in, or -1
  "${PYBIN:-$ROOT/../glm-5.3-reap/.venv/bin/python}" -c \
    "import json;print(len(json.load(open('$ROOT/artifacts/saliency/pass_state.json'))['done']))" \
    2>/dev/null || echo -1
}

[ -f "$STATE" ] && . "$STATE"
LAST_PROGRESS=${LAST_PROGRESS:--1}
REVIVALS=${REVIVALS:-0}

say "watchdog start (poll ${POLL}s, max revivals $MAX_REVIVALS)"
while true; do
  ACTIVE=$(systemctl --user is-active "$UNIT" 2>/dev/null)
  NOW=$(progress)
  STAGE=$(cat "$ROOT/logs/.stage" 2>/dev/null)
  AVAIL=$(awk '/MemAvailable:/{print int($2/1024)}' /proc/meminfo)

  {
    echo "REAP watchdog -- $(date -Is)"
    echo "unit          : $ACTIVE"
    echo "stage         : ${STAGE:-?}"
    echo "chunks folded : $NOW / 27"
    echo "MemAvailable  : ${AVAIL}MB"
    echo "revivals used : $REVIVALS / $MAX_REVIVALS"
    echo
    echo "last log lines:"
    tail -4 "$ROOT/logs/reap_run.log" 2>/dev/null
  } > "$STATUS.tmp" && mv "$STATUS.tmp" "$STATUS"

  if [ "$ACTIVE" = "failed" ] || [ "$ACTIVE" = "inactive" ]; then
    if [ "$STAGE" = "done" ]; then
      say "run COMPLETE (stage=done); watchdog exiting"
      echo "REAP COMPLETE at $(date -Is)" >> "$STATUS"
      exit 0
    fi
    if [ "$NOW" -gt "$LAST_PROGRESS" ] && [ "$REVIVALS" -lt "$MAX_REVIVALS" ]; then
      REVIVALS=$((REVIVALS + 1))
      say "unit $ACTIVE but progress advanced ($LAST_PROGRESS -> $NOW): TRANSIENT, reviving ($REVIVALS/$MAX_REVIVALS)"
      systemctl --user reset-failed "$UNIT" 2>/dev/null
      systemctl --user start "$UNIT" 2>/dev/null
      LAST_PROGRESS=$NOW
    else
      if [ "$NOW" -le "$LAST_PROGRESS" ]; then
        why="no chunk folded in since the last revival ($NOW <= $LAST_PROGRESS) -- the failure repeats"
      else
        why="revival budget exhausted ($REVIVALS/$MAX_REVIVALS)"
      fi
      say "unit $ACTIVE and NOT reviving: $why"
      {
        echo
        echo "*** STOPPED, NOT REVIVED ***"
        echo "$why"
        echo "Nothing was deleted; accumulators and pass_state.json are intact and resume exactly."
        echo "Inspect:  journalctl --user -u $UNIT -n 50 --no-pager"
        echo "          tail -40 $ROOT/logs/reap_run.log"
        echo "Resume :  systemctl --user reset-failed $UNIT && systemctl --user start $UNIT"
      } >> "$STATUS"
      say "watchdog exiting; human intervention required"
      exit 1
    fi
  else
    [ "$NOW" -gt "$LAST_PROGRESS" ] && { LAST_PROGRESS=$NOW; REVIVALS=0
      say "progress: $NOW/27 chunks, stage=$STAGE, avail=${AVAIL}MB (revival budget reset)"; }
  fi

  printf 'LAST_PROGRESS=%s\nREVIVALS=%s\n' "$LAST_PROGRESS" "$REVIVALS" > "$STATE"
  sleep "$POLL"
done
