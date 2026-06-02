#!/usr/bin/env bash
# Round 9 — Force-close-off variations.
#
# Assumes Round 6.5 confirms force_close=0 unlocks positive day_pnl.
# Explores the parameter space around fc=0: lay-price caps, pwin
# thresholds, seed replicates.
#
# 12 cells × ~25 min = ~5h wall.

set -u
REPO=/c/Users/jsmit/source/repos/rl-betfair
cd "$REPO"

WRAPPER_LOG="registry/_round9_wrapper.log"
exec >> "$WRAPPER_LOG" 2>&1
echo "[$(date -Iseconds)] round 9 wrapper started"

TRAIN_DAYS=( 2026-04-06 2026-04-08 2026-04-09 )
EVAL_DAYS=( 2026-04-10 2026-04-17 2026-04-21 2026-05-03 2026-05-06 )
PREDICTORS=(
  "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json"
  "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json"
  "C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json"
)

# Base: E7 + fc=0 (the hypothesised winning recipe).
BASE=(
  --n-agents 4
  --generations 1
  --device cuda
  --strategy-mode arb
  --training-days-explicit "${TRAIN_DAYS[@]}"
  --cohort-eval-days "${EVAL_DAYS[@]}"
  --rotating-eval-sample 0
  --direction-head-manifest models/direction_head/sweep_c11
  --predictor-lean-obs
  --use-race-outcome-predictor
  --use-direction-predictor
  --predictor-bundle-manifests "${PREDICTORS[@]}"
  --reward-overrides force_close_before_off_seconds=0
  --reward-overrides close_feasibility_max_spread_pct=0.05
  --reward-overrides matured_arb_expected_random=0.0
  --bc-pretrain-steps 500
  --predictor-p-win-back-threshold 0.20
)

run_cell () {
    local cell="$1"; shift
    local seed="${R9_SEED:-42}"
    local ts; ts=$(date +%s)
    local outdir="registry/_round9_${cell}_${ts}"
    local logfile="${outdir}.log"
    echo "[$(date -Iseconds)] starting cell ${cell} (seed=${seed}) -> ${outdir}"
    python -m training_v2.cohort.runner "${BASE[@]}" --seed "${seed}" --output-dir "${outdir}" "$@" > "${logfile}" 2>&1
    local rc=$?
    echo "[$(date -Iseconds)] cell ${cell} finished rc=${rc}"
}

# Group L1 — lay_price_max sweep at fc=0 (4 cells)
run_cell "L1_lay_max15"  --lay-price-max 15
run_cell "L1_lay_max20"  --lay-price-max 20
run_cell "L1_lay_max25"  --lay-price-max 25
run_cell "L1_lay_max30"  --lay-price-max 30

# Group L2 — pwin_back sweep at fc=0 + lay_max=20 (4 cells)
run_cell "L2_pwin018" --lay-price-max 20 --predictor-p-win-back-threshold 0.18
run_cell "L2_pwin022" --lay-price-max 20 --predictor-p-win-back-threshold 0.22
run_cell "L2_pwin025" --lay-price-max 20 --predictor-p-win-back-threshold 0.25
run_cell "L2_pwin030" --lay-price-max 20 --predictor-p-win-back-threshold 0.30

# Group L3 — seed replicates of the most-likely winner (4 cells)
# This is the variance-confirmation. If lay_max=20 + fc=0 wins at
# seed=42, does it generalise to 43/44/45/46?
R9_SEED=43 run_cell "L3_seed43_lay_max20" --lay-price-max 20
R9_SEED=44 run_cell "L3_seed44_lay_max20" --lay-price-max 20
R9_SEED=45 run_cell "L3_seed45_lay_max20" --lay-price-max 20
R9_SEED=46 run_cell "L3_seed46_lay_max20" --lay-price-max 20

echo "[$(date -Iseconds)] round 9 fan-out complete"
