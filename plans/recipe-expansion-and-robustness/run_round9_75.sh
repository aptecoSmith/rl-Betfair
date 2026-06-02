#!/usr/bin/env bash
# Round 9.75 — pwin_back BAND sweep at fc=0.
#
# Motivated by Round 9 EV-by-pwin analysis (10,157 pairs):
#   p_win 0.20-0.25: +£1.72/pair
#   p_win 0.25-0.30: +£4.86/pair
#   p_win 0.30-0.35: +£9.49/pair  ← peak
#   p_win 0.35-0.40: +£4.08/pair
#   p_win 0.40-0.50: -£0.19/pair  ← dead zone
#   p_win 0.50+   : +£1.35/pair
#
# Tests whether tightening the band lifts mean day_pnl by removing
# the 0.40-0.50 dead zone and/or focusing on the peak.
#
# 8 cells × ~25 min = ~3.3h wall.

set -u
REPO=/c/Users/jsmit/source/repos/rl-betfair
cd "$REPO"

WRAPPER_LOG="registry/_round9_75_wrapper.log"
exec >> "$WRAPPER_LOG" 2>&1
echo "[$(date -Iseconds)] round 9.75 wrapper started — pwin band sweep"

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
  --reward-overrides force_close_before_off_seconds=0
  --reward-overrides close_feasibility_max_spread_pct=0.05
  --reward-overrides matured_arb_expected_random=0.0
  --bc-pretrain-steps 500
  --lay-price-max 30
)

run_cell () {
    local cell="$1"; shift
    local ts; ts=$(date +%s)
    local outdir="registry/_round9_75_${cell}_${ts}"
    local logfile="${outdir}.log"
    echo "[$(date -Iseconds)] starting cell ${cell} -> ${outdir}"
    python -m training_v2.cohort.runner "${BASE[@]}" --output-dir "${outdir}" "$@" > "${logfile}" 2>&1
    local rc=$?
    echo "[$(date -Iseconds)] cell ${cell} finished rc=${rc}"
}

# Group P1 — explore bands with current floor 0.20 (lift the ceiling).
# Tests whether dropping favourites helps without losing volume.
run_cell "P1_band_020_040" --predictor-p-win-back-threshold 0.20 --predictor-p-win-back-max-threshold 0.40
run_cell "P1_band_020_050" --predictor-p-win-back-threshold 0.20 --predictor-p-win-back-max-threshold 0.50

# Group P2 — narrow peak band (start at 0.25-0.30).
run_cell "P2_band_025_040" --predictor-p-win-back-threshold 0.25 --predictor-p-win-back-max-threshold 0.40
run_cell "P2_band_030_040" --predictor-p-win-back-threshold 0.30 --predictor-p-win-back-max-threshold 0.40

# Group P3 — peak only.
run_cell "P3_band_030_035" --predictor-p-win-back-threshold 0.30 --predictor-p-win-back-max-threshold 0.35

# Group P4 — favourites-only (negative control: should be marginal).
run_cell "P4_band_040_100" --predictor-p-win-back-threshold 0.40 --predictor-p-win-back-max-threshold 1.00

# Group P5 — outsiders-only (negative control: should LOSE money).
run_cell "P5_band_020_025" --predictor-p-win-back-threshold 0.20 --predictor-p-win-back-max-threshold 0.25

# Group P6 — the "complete sweet spot": 0.20-0.40 broad capture.
# Different from P1_020_040 only in that this is the explicit "all
# the positive-EV trades" cell. Replicate for variance check.
run_cell "P6_band_020_040_seed43" --seed 43 --predictor-p-win-back-threshold 0.20 --predictor-p-win-back-max-threshold 0.40

echo "[$(date -Iseconds)] round 9.75 fan-out complete"
