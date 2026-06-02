#!/usr/bin/env bash
# Chain after Round 9.8 — waits for the combo round to finish, then
# runs Round 11 (reward-shape) and Round 8 (scale-up patterns).
# Round 10's remaining cells were dropped to prioritise the combo;
# the 2 completed M1 8-agent cells are kept in registry.

set -u
REPO=/c/Users/jsmit/source/repos/rl-betfair
cd "$REPO"
CHAIN_LOG="registry/_chain_after_round9_8_wrapper.log"
exec >> "$CHAIN_LOG" 2>&1
echo "[$(date -Iseconds)] chain-after-9.8 started; waiting on round 9.8"

while true; do
    if grep -q "round 9.8 fan-out complete" registry/_round9_8_wrapper.log 2>/dev/null; then
        echo "[$(date -Iseconds)] round 9.8 complete; starting round 11"
        break
    fi
    sleep 60
done

bash plans/recipe-expansion-and-robustness/run_round11.sh
echo "[$(date -Iseconds)] round 11 complete; starting round 10 (full scale-up rerun)"

bash plans/recipe-expansion-and-robustness/run_round10.sh
echo "[$(date -Iseconds)] round 10 complete; starting round 8"

bash plans/recipe-expansion-and-robustness/run_round8.sh
echo "[$(date -Iseconds)] round 8 complete; chain fan-out complete"
