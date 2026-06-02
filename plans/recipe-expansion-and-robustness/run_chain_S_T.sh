#!/usr/bin/env bash
# Chain: wait for roundS (which chains after R, which chains after Q),
# then run roundT (Path C mature_prob open-gate probe).
cd /c/Users/jsmit/source/repos/rl-betfair
exec >> registry/_chain_ST.log 2>&1
echo "[$(date -Iseconds)] chain S->T started; waiting on roundS"
while true; do
  if grep -q "roundS fan-out complete" registry/_roundS_wrapper.log 2>/dev/null; then
    if ! ps -ef 2>/dev/null | grep "cohort.runner" | grep -v grep | grep -q . ; then
      echo "[$(date -Iseconds)] roundS done; starting roundT"; break
    fi
  fi
  sleep 60
done
bash plans/recipe-expansion-and-robustness/run_roundT.sh
echo "[$(date -Iseconds)] chain S->T complete"
