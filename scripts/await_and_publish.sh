#!/usr/bin/env bash
# Wait for the REAP to finish, then publish the pruned checkpoint to the Hub.
#
# This is a separate unit, not a stage inside run_reap.sh, for a mechanical reason: bash reads a
# script incrementally from the inode it started with, so run_reap.sh cannot be given a new stage
# while it is running -- the edit would shift every byte the live shell has not read yet.
#
# It publishes ONLY on `logs/.stage == done`. A failed or abandoned run leaves the stage marker
# at whatever stage died, and this exits without touching the network.
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY_BIN:-$HOME/glm-5.3-reap/.venv/bin/python}"
LOG=logs/publish.log
STATUS=logs/PUBLISH_STATUS.txt
POLL=${POLL:-300}
MAX_WAIT_H=${MAX_WAIT_H:-72}
MAX_TRIES=${MAX_TRIES:-5}
DST=${REAP_DST:-$HOME/models/MiMo-V2.6-Flash-REAP50}
REPO=${HF_REPO:-patrickbdevaney/MiMo-V2.6-Flash-REAP50}
PUBLIC_FLAG=""; [ "${HF_PUBLIC:-0}" = "1" ] && PUBLIC_FLAG="--public"
mkdir -p logs

say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
status() { printf 'REAP publish -- %s\n%s\n' "$(date -Is)" "$1" > "$STATUS"; }

say "publish watcher start (poll ${POLL}s, give up after ${MAX_WAIT_H}h, repo $REPO)"
deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))

while :; do
  stage=$(cat logs/.stage 2>/dev/null || echo absent)
  [ "$stage" = "done" ] && break
  if [ "$(date +%s)" -ge "$deadline" ]; then
    say "GIVING UP: still at stage '$stage' after ${MAX_WAIT_H}h. Nothing was uploaded."
    status "gave up waiting; stage=$stage"
    exit 0
  fi
  # A run that has stopped and is not done is a failed run. Do not publish it, and do not sit
  # here for three days pretending it might still finish.
  if ! systemctl --user is-active --quiet reap_run.service; then
    say "reap_run is not active and stage is '$stage' -- the run did not complete."
    say "  Nothing was uploaded. Re-run the pipeline, then: systemctl --user restart reap-publish"
    status "run did not complete (stage=$stage); nothing uploaded"
    exit 0
  fi
  status "waiting for the run; stage=$stage"
  sleep "$POLL"
done

say "stage is done -- preflighting $DST"
status "preflighting the checkpoint"
if ! "$PY" scripts/publish_hf.py --dst "$DST" --repo "$REPO" --dry-run >> "$LOG" 2>&1; then
  say "PREFLIGHT REFUSED -- see $LOG. Nothing was uploaded."
  status "preflight refused; nothing uploaded"
  exit 1
fi

for try in $(seq 1 "$MAX_TRIES"); do
  say "upload attempt $try/$MAX_TRIES -> $REPO"
  status "uploading (attempt $try/$MAX_TRIES)"
  if "$PY" scripts/publish_hf.py --dst "$DST" --repo "$REPO" $PUBLIC_FLAG >> "$LOG" 2>&1; then
    say "PUBLISHED: https://huggingface.co/$REPO"
    [ -z "$PUBLIC_FLAG" ] && say "  it is PRIVATE. To make it public, deliberately:" \
      && say "  hf repo settings $REPO --private=false"
    status "published https://huggingface.co/$REPO $([ -z "$PUBLIC_FLAG" ] && echo '(private)')"
    exit 0
  fi
  # upload_large_folder resumes from what already landed, so a retry is cheap and a dropped
  # connection 80 GB in does not start over.
  say "attempt $try failed; retrying in 300s (the upload resumes, it does not restart)"
  sleep 300
done
say "UPLOAD FAILED after $MAX_TRIES attempts -- see $LOG"
status "upload failed after $MAX_TRIES attempts"
exit 1
