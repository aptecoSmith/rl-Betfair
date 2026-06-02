#!/usr/bin/env bash
# Chain: wait for roundQ to complete, then launch roundR.
cd /c/Users/jsmit/source/repos/rl-betfair
exec >> registry/_chain_QR.log 2>&1
echo "[$(date -Iseconds)] chain Q->R started; waiting on roundQ"
while true; do
  if grep -q "roundQ fan-out complete" registry/_roundQ_wrapper.log 2>/dev/null; then
    # also confirm no cohort running (be sure Q is fully done)
    if ! ps -ef 2>/dev/null | grep "cohort.runner" | grep -v grep | grep -q . ; then
      echo "[$(date -Iseconds)] roundQ done; starting roundR"; break
    fi
  fi
  sleep 60
done
bash plans/recipe-expansion-and-robustness/run_roundR.sh
echo "[$(date -Iseconds)] chain Q->R complete"
