#!/usr/bin/env bash
# GPU supervisor daemon — NEVER lets the GPU idle.
#
# Runs forever (launch once with nohup; survives session close/dormancy).
# Loop: if no cohort.runner is running AND the queue has a wrapper,
# run the next wrapper (blocking until all its cells finish), then move
# it to done/. If the queue is empty, wait and re-check. Stops cleanly
# if a STOP sentinel file appears.
#
# Add work: drop an executable round-wrapper .sh into the queue dir.
# They run in lexical order, so name them 01_*.sh, 02_*.sh, etc.
#
# Launch:  nohup bash tools/gpu_supervisor.sh > /dev/null 2>&1 & disown
# Stop:    touch plans/recipe-expansion-and-robustness/queue/STOP

set -u
REPO=/c/Users/jsmit/source/repos/rl-betfair
cd "$REPO"
QUEUE="plans/recipe-expansion-and-robustness/queue"
DONE="$QUEUE/done"
LOG="registry/_gpu_supervisor.log"
mkdir -p "$DONE"

echo "[$(date -Iseconds)] gpu_supervisor started" >> "$LOG"

cohort_running () {
  ps -ef 2>/dev/null | grep "cohort.runner" | grep -v grep | grep -q . && return 0 || return 1
}

while true; do
  if [[ -f "$QUEUE/STOP" ]]; then
    echo "[$(date -Iseconds)] STOP sentinel found; supervisor exiting" >> "$LOG"
    rm -f "$QUEUE/STOP"
    exit 0
  fi
  if cohort_running; then
    sleep 60; continue
  fi
  # GPU idle. Pull the next queued wrapper (lexical order).
  next=$(ls "$QUEUE"/*.sh 2>/dev/null | sort | head -1)
  if [[ -z "$next" ]]; then
    echo "[$(date -Iseconds)] queue empty; GPU idle; waiting 5min" >> "$LOG"
    sleep 300; continue
  fi
  echo "[$(date -Iseconds)] launching queued round: $next" >> "$LOG"
  bash "$next"   # blocking — runs every cell in the wrapper
  rc=$?
  echo "[$(date -Iseconds)] finished $next rc=$rc; moving to done/" >> "$LOG"
  mv "$next" "$DONE/" 2>/dev/null || true
  sleep 5
done
