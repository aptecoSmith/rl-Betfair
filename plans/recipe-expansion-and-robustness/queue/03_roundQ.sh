#!/usr/bin/env bash
export PATH="/c/Python314:/c/Python314/Scripts:$PATH"
set -u
cd /c/Users/jsmit/source/repos/rl-betfair
exec >> registry/_roundQ_wrapper.log 2>&1
echo "[$(date -Iseconds)] roundQ started"
TRAIN=( 2026-04-06 2026-04-08 2026-04-09 )
EVAL=( 2026-05-07 2026-05-09 2026-05-11 2026-05-13 2026-05-15 2026-05-17 2026-05-19 )
PRED=( "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json" "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json" "C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json" )
BASE=( --n-agents 4 --generations 1 --device cuda --strategy-mode arb --training-days-explicit "${TRAIN[@]}" --cohort-eval-days "${EVAL[@]}" --rotating-eval-sample 0 --direction-head-manifest models/direction_head/sweep_c11 --predictor-lean-obs --use-race-outcome-predictor --use-direction-predictor --predictor-bundle-manifests "${PRED[@]}" --reward-overrides close_feasibility_max_spread_pct=0.05 --reward-overrides matured_arb_expected_random=0.0 --bc-pretrain-steps 500 --bc-include-negative-samples --bc-include-close-hold-samples --predictor-p-win-back-threshold 0.25 )
run_cell () { local cell="$1"; shift; local ts; ts=$(date +%s); local o="registry/_roundQ_${cell}_${ts}"; echo "[$(date -Iseconds)] starting ${cell}"; python -m training_v2.cohort.runner "${BASE[@]}" --seed "${SD:-42}" --output-dir "$o" "$@" > "$o.log" 2>&1; echo "[$(date -Iseconds)] ${cell} rc=$?"; }
run_cell Q1_tight0030_band050 --reward-overrides force_close_before_off_seconds=120 --arb-spread-target-lock-pct 0.003 --predictor-p-win-back-max-threshold 0.50
run_cell Q2_tight0020_band050 --reward-overrides force_close_before_off_seconds=120 --arb-spread-target-lock-pct 0.002 --predictor-p-win-back-max-threshold 0.50
SD=43 run_cell Q3_tight0030_s43 --reward-overrides force_close_before_off_seconds=120 --arb-spread-target-lock-pct 0.003
SD=44 run_cell Q4_tight0030_s44 --reward-overrides force_close_before_off_seconds=120 --arb-spread-target-lock-pct 0.003
echo "[$(date -Iseconds)] roundQ fan-out complete"
