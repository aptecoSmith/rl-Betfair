#!/usr/bin/env bash
# Chain wrapper v3 (2026-05-27 21:15 BST) — Round 9 → 9.5 → 9.75 → 10 → 11 → 8.
#
# Round 9.75 inserted after Round 9 EV-by-pwin analysis discovered
# the 0.30-0.35 band has +£9.49/pair vs 0.40-0.50's -£0.19/pair.
# Round 9.75 sweeps pwin BANDS using the new
# --predictor-p-win-back-max-threshold flag.

set -u
REPO=/c/Users/jsmit/source/repos/rl-betfair
cd "$REPO"

CHAIN_LOG="registry/_chain_after_round9_wrapper.log"
exec >> "$CHAIN_LOG" 2>&1

echo "[$(date -Iseconds)] chain wrapper v3 started; waiting on round 9"

while true; do
    if grep -q "round 9 fan-out complete" registry/_round9_wrapper.log 2>/dev/null; then
        echo "[$(date -Iseconds)] round 9 complete; starting round 9.5"
        break
    fi
    sleep 60
done

bash plans/recipe-expansion-and-robustness/run_round9_5.sh
echo "[$(date -Iseconds)] round 9.5 complete; starting round 9.75 (pwin band sweep)"

bash plans/recipe-expansion-and-robustness/run_round9_75.sh
echo "[$(date -Iseconds)] round 9.75 complete; starting round 10"

bash plans/recipe-expansion-and-robustness/run_round10.sh
echo "[$(date -Iseconds)] round 10 complete; starting round 11"

bash plans/recipe-expansion-and-robustness/run_round11.sh
echo "[$(date -Iseconds)] round 11 complete; starting round 8"

bash plans/recipe-expansion-and-robustness/run_round8.sh
echo "[$(date -Iseconds)] round 8 complete; chain fan-out complete"
