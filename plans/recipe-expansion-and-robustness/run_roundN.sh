#!/usr/bin/env bash
# Round N — selectivity push around the M6 winner (full-aug + pwin025,
# held-out -£98). Test tighter pwin floors + replicate across seeds.
# Held-out eval, fc=120.
set -u
cd /c/Users/jsmit/source/repos/rl-betfair
exec >> registry/_roundN_wrapper.log 2>&1
echo "[$(date -Iseconds)] round N wrapper started — selectivity push"
TRAIN=( 2026-04-06 2026-04-08 2026-04-09 )
EVAL=( 2026-05-07 2026-05-09 2026-05-11 2026-05-13 2026-05-15 2026-05-17 2026-05-19 )
PRED=( "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json" "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json" "C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json" )
BASE=( --n-agents 4 --generations 1 --device cuda --strategy-mode arb --training-days-explicit "${TRAIN[@]}" --cohort-eval-days "${EVAL[@]}" --rotating-eval-sample 0 --direction-head-manifest models/direction_head/sweep_c11 --predictor-lean-obs --use-race-outcome-predictor --use-direction-predictor --predictor-bundle-manifests "${PRED[@]}" --reward-overrides force_close_before_off_seconds=120 --reward-overrides close_feasibility_max_spread_pct=0.05 --reward-overrides matured_arb_expected_random=0.0 --bc-pretrain-steps 500 --bc-include-negative-samples --bc-include-close-hold-samples )
run_cell () { local cell="$1"; shift; local ts; ts=$(date +%s); local o="registry/_roundN_${cell}_${ts}"; echo "[$(date -Iseconds)] starting ${cell}"; python -m training_v2.cohort.runner "${BASE[@]}" --seed "${NSEED:-42}" --output-dir "$o" "$@" > "$o.log" 2>&1; echo "[$(date -Iseconds)] ${cell} rc=$?"; }
run_cell "N1_pwin030"            --predictor-p-win-back-threshold 0.30
run_cell "N2_pwin035"            --predictor-p-win-back-threshold 0.35
run_cell "N3_pwin030_tight0030"  --predictor-p-win-back-threshold 0.30 --arb-spread-target-lock-pct 0.003
run_cell "N4_pwin025_band050"    --predictor-p-win-back-threshold 0.25 --predictor-p-win-back-max-threshold 0.50
NSEED=43 run_cell "N5_pwin025_s43" --predictor-p-win-back-threshold 0.25
NSEED=44 run_cell "N6_pwin025_s44" --predictor-p-win-back-threshold 0.25
echo "[$(date -Iseconds)] round N fan-out complete"
