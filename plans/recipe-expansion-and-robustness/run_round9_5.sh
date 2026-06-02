#!/usr/bin/env bash
# Round 9.5 — Extreme lay_price_max sweep at fc=0.
#
# Round 9 L1 showed lay_max=30 (+£259) beats lay_max=20 (+£202).
# Test if higher caps (50, 100) continue the trend.
#
# 6 cells × ~25 min = ~2.5h wall.

set -u
REPO=/c/Users/jsmit/source/repos/rl-betfair
cd "$REPO"

WRAPPER_LOG="registry/_round9_5_wrapper.log"
exec >> "$WRAPPER_LOG" 2>&1
echo "[$(date -Iseconds)] round 9.5 wrapper started"

TRAIN_DAYS=( 2026-04-06 2026-04-08 2026-04-09 )
EVAL_DAYS=( 2026-04-10 2026-04-17 2026-04-21 2026-05-03 2026-05-06 )
PREDICTORS=(
  "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json"
  "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json"
  "C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json"
)

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
    local seed="${R95_SEED:-42}"
    local ts; ts=$(date +%s)
    local outdir="registry/_round9_5_${cell}_${ts}"
    local logfile="${outdir}.log"
    echo "[$(date -Iseconds)] starting cell ${cell} (seed=${seed}) -> ${outdir}"
    python -m training_v2.cohort.runner "${BASE[@]}" --seed "${seed}" --output-dir "${outdir}" "$@" > "${logfile}" 2>&1
    local rc=$?
    echo "[$(date -Iseconds)] cell ${cell} finished rc=${rc}"
}

# Group L4 — extreme lay_price_max at seed=42 (3 cells)
run_cell "L4_lay_max40"  --lay-price-max 40
run_cell "L4_lay_max50"  --lay-price-max 50
run_cell "L4_lay_max100" --lay-price-max 100

# Group L5 — replicate top candidate (lay_max=100 or whichever wins)
# across 3 more seeds for robustness (3 cells)
R95_SEED=43 run_cell "L5_seed43_lay_max100" --lay-price-max 100
R95_SEED=44 run_cell "L5_seed44_lay_max100" --lay-price-max 100
R95_SEED=46 run_cell "L5_seed46_lay_max100" --lay-price-max 100

echo "[$(date -Iseconds)] round 9.5 fan-out complete"
