#!/bin/bash
# Durable bash polling chain (EXPERIMENTS.md: bash chains are the durable
# mechanism; smart daemons fail). Waits for Cohort 1 to finish, then launches
# complementary Cohorts 2 and 3 to keep the GPU busy + search the lever space.
# Cohort 1 (already running): locked_weight=9, gate 0.30, locked_per_std.
# Cohort 2: locked_weight=0 (NO amplification — does the 10x locked help?) +
#           reward_clip gene. seed 43.
# Cohort 3: locked_weight=20 (STRONG amp) + gate 0.20 (more opens) +
#           matured_arb_bonus_weight gene. seed 44.
# All: 30 agents x 5 gens batched, full obs + predictors + input_norm,
# per_pair_reward_at_resolution pinned, holdout May 20-29 EXCLUDED, select on
# locked_per_std, argmax eval.
set -u
cd "C:/Users/jsmit/source/repos/rl-betfair"
PY=".venv/Scripts/python.exe"
CHAMP="C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json"
RANK="C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json"
DIRM="C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json"
HOLDOUT=(2026-05-20 2026-05-21 2026-05-22 2026-05-25 2026-05-27 2026-05-28 2026-05-29)

wait_for_complete () {  # $1 = log file
  local log="$1"
  while ! grep -q "Cohort complete" "$log" 2>/dev/null; do
    # bail if the python process died without writing the marker (crash)
    if ! grep -q "Cohort complete" "$log" 2>/dev/null \
       && [ "$(powershell -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*cohort.runner*' } | Measure-Object).Count" 2>/dev/null)" = "0" ]; then
      echo "chain: no cohort.runner process AND no complete marker in $log — assuming crash/done, proceeding."
      return 0
    fi
    sleep 120
  done
}

# 1. Wait for Cohort 1.
C1LOG=$(ls -t registry/scalping_ga_c1_*.log 2>/dev/null | head -1)
echo "chain: waiting for Cohort 1 ($C1LOG)..."
wait_for_complete "$C1LOG"
echo "chain: Cohort 1 done."

# 2. Cohort 2 — locked_weight=0 + reward_clip gene + day_pnl-ish via locked_per_std.
TS2=$(date +%s); OUT2="registry/scalping_ga_c2_${TS2}"
echo "chain: launching Cohort 2 -> $OUT2"
$PY -m training_v2.cohort.runner --n-agents 30 --generations 5 --batched \
  --days 27 --n-eval-days 2 --exclude-days "${HOLDOUT[@]}" --device cuda --seed 43 \
  --output-dir "$OUT2" --strategy-mode arb \
  --predictor-bundle-manifests "$CHAMP" "$RANK" "$DIRM" \
  --use-race-outcome-predictor --use-direction-predictor \
  --mature-prob-open-threshold 0.30 \
  --enable-gene open_cost --enable-gene stop_loss_pnl_threshold \
  --enable-gene mature_prob_loss_weight --enable-gene reward_clip \
  --reward-overrides per_pair_reward_at_resolution=true \
  --bc-pretrain-steps 500 --composite-score-mode locked_per_std --argmax-eval \
  > "${OUT2}.log" 2>&1
echo "chain: Cohort 2 done."

# 3. Cohort 3 — locked_weight=20 + gate 0.20 + matured_arb_bonus_weight gene.
TS3=$(date +%s); OUT3="registry/scalping_ga_c3_${TS3}"
echo "chain: launching Cohort 3 -> $OUT3"
$PY -m training_v2.cohort.runner --n-agents 30 --generations 5 --batched \
  --days 27 --n-eval-days 2 --exclude-days "${HOLDOUT[@]}" --device cuda --seed 44 \
  --output-dir "$OUT3" --strategy-mode arb \
  --predictor-bundle-manifests "$CHAMP" "$RANK" "$DIRM" \
  --use-race-outcome-predictor --use-direction-predictor \
  --mature-prob-open-threshold 0.20 \
  --enable-gene open_cost --enable-gene stop_loss_pnl_threshold \
  --enable-gene mature_prob_loss_weight --enable-gene matured_arb_bonus_weight \
  --reward-overrides per_pair_reward_at_resolution=true \
  --reward-overrides locked_pnl_reward_weight=20.0 \
  --bc-pretrain-steps 500 --composite-score-mode locked_per_std --argmax-eval \
  > "${OUT3}.log" 2>&1
echo "chain: Cohort 3 done. CHAIN COMPLETE."
