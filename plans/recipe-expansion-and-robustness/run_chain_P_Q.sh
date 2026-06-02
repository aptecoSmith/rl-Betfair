#!/usr/bin/env bash
cd /c/Users/jsmit/source/repos/rl-betfair
exec >> registry/_chain_PQ.log 2>&1
echo "[$(date -Iseconds)] chain P->Q started; waiting on roundP"
while true; do
  # Wait until roundP wrapper log says complete AND no cohort.runner alive
  if grep -q "roundP fan-out complete" registry/_roundP_wrapper.log 2>/dev/null; then
    if ! ps -ef 2>/dev/null | grep "cohort.runner" | grep -v grep | grep -q . ; then
      echo "[$(date -Iseconds)] roundP done; starting roundQ"; break
    fi
  fi
  sleep 60
done
bash plans/recipe-expansion-and-robustness/queue/03_roundQ.sh
echo "[$(date -Iseconds)] chain P->Q complete"
