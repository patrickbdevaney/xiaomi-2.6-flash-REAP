#!/usr/bin/env bash
# Gate the watchdog's one real decision: revive on progress, refuse without it.
#
# A watchdog that restarts unconditionally is worse than none -- it converts a diagnosable stop
# into an invisible loop that replays the same failure for thirty hours. Both VerifyErrors in
# this project were exactly that shape. So both branches are driven here against a REAL failing
# systemd unit, not a mock.
set -u
FAIL=()
check() { if [ "$2" = 1 ]; then echo "  PASS  $1${3:+ -- $3}"; else echo "  FAIL  $1${3:+ -- $3}"; FAIL+=("$1"); fi; }

TD=$(mktemp -d); trap 'rm -rf "$TD"; systemctl --user stop wdtest.service 2>/dev/null; rm -f ~/.config/systemd/user/wdtest.service; systemctl --user daemon-reload' EXIT
mkdir -p "$TD/logs" "$TD/artifacts/saliency"
echo '{"done":["a.pt"]}' > "$TD/artifacts/saliency/pass_state.json"
echo stage2-pass > "$TD/logs/.stage"
touch "$TD/logs/reap_run.log"

# a unit that always fails immediately, with no restart of its own
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/wdtest.service <<'U'
[Service]
Type=oneshot
ExecStart=/bin/false
U
systemctl --user daemon-reload
systemctl --user start wdtest.service 2>/dev/null
sleep 1
st=$(systemctl --user is-active wdtest.service)
check "test unit is in a failed state" "$([ "$st" = failed ] && echo 1 || echo 0)" "is-active=$st"

run_wd() { ROOT="$TD" UNIT=wdtest.service PYBIN=/usr/bin/python3 POLL=1 MAX_REVIVALS=${1:-12} \
             timeout 12 bash scripts/watchdog.sh; echo "rc=$?"; }

# ---- 1. progress since last look -> TRANSIENT, revive ----
printf 'LAST_PROGRESS=0\nREVIVALS=0\n' > "$TD/logs/.watchdog_state"
out=$(run_wd)
grep -q "TRANSIENT, reviving" "$TD/logs/watchdog.log" && r=1 || r=0
check "revives when a chunk was folded in since the last restart" "$r" \
      "$(grep -o 'TRANSIENT[^\"]*' "$TD/logs/watchdog.log" | head -1)"

# ---- 2. NO progress -> DETERMINISTIC, refuse and explain ----
rm -f "$TD/logs/watchdog.log"
printf 'LAST_PROGRESS=1\nREVIVALS=0\n' > "$TD/logs/.watchdog_state"
out=$(run_wd)
grep -q "NOT reviving" "$TD/logs/watchdog.log" && r=1 || r=0
check "refuses to revive when no chunk was folded in" "$r" "$(grep -o 'NOT reviving.*' "$TD/logs/watchdog.log" | head -1 | cut -c1-90)"
grep -q "STOPPED, NOT REVIVED" "$TD/logs/WATCHDOG_STATUS.txt" && r=1 || r=0
check "writes a status file a human can act on" "$r" \
      "$(grep -c . "$TD/logs/WATCHDOG_STATUS.txt") lines, names the resume command"
grep -q "reset-failed" "$TD/logs/WATCHDOG_STATUS.txt" && r=1 || r=0
check "the status file says how to resume" "$r" ""
echo "$out" | grep -q "rc=1" && r=1 || r=0
check "exits non-zero when it gives up" "$r" "$(echo "$out" | tail -1)"

# ---- 3. revival budget is finite ----
rm -f "$TD/logs/watchdog.log"
printf 'LAST_PROGRESS=0\nREVIVALS=12\n' > "$TD/logs/.watchdog_state"
out=$(run_wd 12)
grep -q "revival budget exhausted" "$TD/logs/watchdog.log" && r=1 || r=0
check "stops after the revival budget is exhausted" "$r" "even though progress advanced"

# ---- 4. a completed run is not treated as a failure ----
rm -f "$TD/logs/watchdog.log"
echo done > "$TD/logs/.stage"
printf 'LAST_PROGRESS=0\nREVIVALS=0\n' > "$TD/logs/.watchdog_state"
out=$(run_wd)
grep -q "run COMPLETE" "$TD/logs/watchdog.log" && r=1 || r=0
check "treats stage=done as success, not as a dead unit" "$r" "$(echo "$out" | tail -1)"

if [ ${#FAIL[@]} -eq 0 ]; then echo "GATE PASS"; else echo "GATE FAIL: ${FAIL[*]}"; exit 1; fi
