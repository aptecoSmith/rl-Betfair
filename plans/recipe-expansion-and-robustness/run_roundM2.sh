#!/usr/bin/env bash
# Round M2 — blind continuation of the mat%-push (pre-designed so GPU
# stays busy overnight). Pushes spread tighter still + replicates the
# full-aug+tight recipe across seeds. Held-out eval.
set -u
REPO=/c/Users/jsmit/source/repos/rl-betfair
cd "$REPO"
exec >> registry/_roundM2_wrapper.log 2>&1
echo "[$(date -Iseconds)] round M2 wrapper started"
TRAIN=( 2026-04-06 2026-04-08 2026-04-09 )
EVAL=( 2026-05-07 2026-05-09 2026-05-11 2026-05-13 2026-05-15 2026-05-17 2026-05-19 )
PRED=( "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json" "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json" "C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json" )
BASE=( --n-agents 4 --generations 1 --device cuda --strategy-mode arb --training-days-explicit "${TRAIN[@]}" --cohort-eval-days "${EVAL[@]}" --rotating-eval-sample 0 --direction-head-manifest models/direction_head/sweep_c11 --predictor-lean-obs --use-race-outcome-predictor --use-direction-predictor --predictor-bundle-manifests "${PRED[@]}" --reward-overrides force_close_before_off_seconds=120 --reward-overrides close_feasibility_max_spread_pct=0.05 --reward-overrides matured_arb_expected_random=0.0 --bc-pretrain-steps 500 --predictor-p-win-back-threshold 0.20 --bc-include-negative-samples --bc-include-close-hold-samples )
run_cell () { local cell="$1"; shift; local ts; ts=$(date +%s); local o="registry/_roundM2_${cell}_${ts}"; echo "[$(date -Iseconds)] starting ${cell}"; python -m training_v2.cohort.runner "${BASE[@]}" --seed "${R2SEED:-42}" --output-dir "$o" "$@" > "$o.log" 2>&1; echo "[$(date -Iseconds)] ${cell} rc=$?"; }
# Even tighter spreads
run_cell "P1_tight0015" --arb-spread-target-lock-pct 0.0015
run_cell "P2_tight0010" --arb-spread-target-lock-pct 0.0010
# Best-guess recipe (full-aug + tight0020) replicated across seeds
R2SEED=43 run_cell "P3_fullaug_tight0020_s43" --arb-spread-target-lock-pct 0.002
R2SEED=44 run_cell "P4_fullaug_tight0020_s44" --arb-spread-target-lock-pct 0.002
# Lower BC (BC makes selection worse) + tight
run_cell "P5_bc100_tight0020" --bc-pretrain-steps 100 --arb-spread-target-lock-pct 0.002
run_cell "P6_bc0_tight0020" --bc-pretrain-steps 0 --arb-spread-target-lock-pct 0.002
echo "[$(date -Iseconds)] round M2 fan-out complete"
