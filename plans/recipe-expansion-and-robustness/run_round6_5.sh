#!/usr/bin/env bash
# Round 6.5 — Force-close hypothesis test.
#
# Critical insight from Round 6: naked P&L is consistently positive
# (+£44 to +£95/day across 7 no-augmentation cells). The agent's
# back-leg selection is structurally EV-positive on the surviving
# nakeds. Yet force_close=120s eats most of this by force-closing
# 60-70% of pairs that never matured naturally.
#
# Hypothesis: disabling force-close lets those would-be force-closes
# settle naked instead. If naked is EV-positive, day_pnl turns
# positive overall.
#
# 6 cells × ~25 min = ~2.5h wall. CRITICAL TEST — these 6 may
# unlock the positive-day_pnl regime if the hypothesis holds.

set -u

REPO=/c/Users/jsmit/source/repos/rl-betfair
cd "$REPO"

WRAPPER_LOG="registry/_round6_5_wrapper.log"
exec >> "$WRAPPER_LOG" 2>&1

echo "[$(date -Iseconds)] round 6.5 wrapper started — force-close hypothesis test"

TRAIN_DAYS=( 2026-04-06 2026-04-08 2026-04-09 )
EVAL_DAYS=( 2026-04-10 2026-04-17 2026-04-21 2026-05-03 2026-05-06 )
PREDICTORS=(
  "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json"
  "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json"
  "C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json"
)

# E7-base minus force_close_before_off_seconds — each cell sets its own.
BASE=(
  --n-agents 4
  --generations 1
  --device cuda
  --seed 42
  --strategy-mode arb
  --training-days-explicit "${TRAIN_DAYS[@]}"
  --cohort-eval-days "${EVAL_DAYS[@]}"
  --rotating-eval-sample 0
  --direction-head-manifest models/direction_head/sweep_c11
  --predictor-lean-obs
  --use-race-outcome-predictor
  --use-direction-predictor
  --predictor-bundle-manifests "${PREDICTORS[@]}"
  --reward-overrides close_feasibility_max_spread_pct=0.05
  --reward-overrides matured_arb_expected_random=0.0
  --bc-pretrain-steps 500
  --predictor-p-win-back-threshold 0.20
)

run_cell () {
    local cell="$1"; shift
    local ts; ts=$(date +%s)
    local outdir="registry/_round6_5_${cell}_${ts}"
    local logfile="${outdir}.log"
    echo "[$(date -Iseconds)] starting cell ${cell} -> ${outdir}"
    python -m training_v2.cohort.runner \
        "${BASE[@]}" \
        --output-dir "${outdir}" \
        "$@" \
        > "${logfile}" 2>&1
    local rc=$?
    echo "[$(date -Iseconds)] cell ${cell} finished rc=${rc}"
}

# K1 — disable force-close on E7 base.
# Expected: opens still ~138, mat ~5%, cls similar, fc% drops to ~0,
# naked% rises to ~70%. Day_pnl turns positive if hypothesis is right.
run_cell "K1_e7_fc_off" \
    --reward-overrides force_close_before_off_seconds=0

# K2 — disable force-close + lay_price_max=20 (current day_pnl leader).
# Expected: best of both — lay-side selection + naked harvest.
run_cell "K2_e7_lay_max_fc_off" \
    --reward-overrides force_close_before_off_seconds=0 \
    --lay-price-max 20

# K3 — disable force-close + bc1000 (variance-discipline winner from Round 6).
run_cell "K3_e7_bc1000_fc_off" \
    --reward-overrides force_close_before_off_seconds=0 \
    --bc-pretrain-steps 1000

# K4 — short force-close window (T-30s) instead of disable. Tests
# whether a SHORTER window also captures most of the naked EV
# while still bounding pre-off variance.
run_cell "K4_e7_fc30" \
    --reward-overrides force_close_before_off_seconds=30

# K5 — disable force-close + stack lay_max AND bc1000 (the kitchen
# sink of Round 6 winners).
run_cell "K5_e7_lay_max_bc1000_fc_off" \
    --reward-overrides force_close_before_off_seconds=0 \
    --lay-price-max 20 \
    --bc-pretrain-steps 1000

# K6 — disable force-close + seed=43 (lucky-seed L/σN winner).
run_cell "K6_e7_seed43_fc_off" \
    --reward-overrides force_close_before_off_seconds=0 \
    --seed 43

echo "[$(date -Iseconds)] round 6.5 fan-out complete"
