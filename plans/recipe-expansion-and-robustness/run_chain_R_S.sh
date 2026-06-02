#!/usr/bin/env bash
# Chain: wait for roundR (which itself chains after Q), then run roundS.
cd /c/Users/jsmit/source/repos/rl-betfair
exec >> registry/_chain_RS.log 2>&1
echo "[$(date -Iseconds)] chain R->S started; waiting on roundR"
while true; do
  if grep -q "roundR fan-out complete" registry/_roundR_wrapper.log 2>/dev/null; then
    if ! ps -ef 2>/dev/null | grep "cohort.runner" | grep -v grep | grep -q . ; then
      echo "[$(date -Iseconds)] roundR done; starting roundS"; break
    fi
  fi
  sleep 60
done
bash plans/recipe-expansion-and-robustness/run_roundS.sh
echo "[$(date -Iseconds)] chain R->S complete"
