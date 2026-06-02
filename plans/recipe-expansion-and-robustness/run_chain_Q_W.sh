#!/usr/bin/env bash
# Chain: wait for roundQ (Q4 finishing) then run roundW (close-walk
# broad rerun). The old Q->R->S->T->W chain was killed 2026-05-30 to
# redirect GPU to the close-matching work; this is the replacement.
cd /c/Users/jsmit/source/repos/rl-betfair
exec >> registry/_chain_QW.log 2>&1
echo "[$(date -Iseconds)] chain Q->W started; waiting on roundQ to finish"
while true; do
  if grep -q "roundQ fan-out complete" registry/_roundQ_wrapper.log 2>/dev/null; then
    if ! ps -ef 2>/dev/null | grep "cohort.runner" | grep -v grep | grep -q . ; then
      echo "[$(date -Iseconds)] roundQ done; starting roundW"; break
    fi
  fi
  sleep 60
done
bash plans/recipe-expansion-and-robustness/run_roundW.sh
echo "[$(date -Iseconds)] chain Q->W complete"
