#!/usr/bin/env bash
# Chain: wait for Round M, then run Round M2. Survives session close.
set -u
cd /c/Users/jsmit/source/repos/rl-betfair
exec >> registry/_chain_M_M2.log 2>&1
echo "[$(date -Iseconds)] chain M->M2 started; waiting on round M"
while true; do
  if grep -q "round M fan-out complete" registry/_roundM_wrapper.log 2>/dev/null; then
    echo "[$(date -Iseconds)] round M done; starting M2"; break
  fi
  sleep 60
done
bash plans/recipe-expansion-and-robustness/run_roundM2.sh
echo "[$(date -Iseconds)] chain M->M2 complete"
