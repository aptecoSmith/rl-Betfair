#!/usr/bin/env bash
# Chain: wait for roundT (Path C mature-gate, last in Q->R->S->T),
# then run roundW (close-walk A/B). Keeps the close-walk experiment
# off the GPU until the current chain finishes — no contention.
cd /c/Users/jsmit/source/repos/rl-betfair
exec >> registry/_chain_TW.log 2>&1
echo "[$(date -Iseconds)] chain T->W started; waiting on roundT"
while true; do
  if grep -q "roundT fan-out complete" registry/_roundT_wrapper.log 2>/dev/null; then
    if ! ps -ef 2>/dev/null | grep "cohort.runner" | grep -v grep | grep -q . ; then
      echo "[$(date -Iseconds)] roundT done; starting roundW"; break
    fi
  fi
  sleep 60
done
bash plans/recipe-expansion-and-robustness/run_roundW.sh
echo "[$(date -Iseconds)] chain T->W complete"
