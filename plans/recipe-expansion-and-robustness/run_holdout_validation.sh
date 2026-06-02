#!/usr/bin/env bash
# HELD-OUT VALIDATION — the honest deployment number.
#
# Every cell in the campaign so far evaluated on the SAME 5 days
# (2026-04-10, -17, -21, 2026-05-03, -06). This tests the deploy
# candidate on 14 days NEVER seen in training or eval
# (2026-05-07 .. 05-20). If day_pnl stays positive here, the edge
# is real; if it collapses, we overfit the 5-day eval window.
#
# Recipe: fc=0 + pwin_back>=0.20 (no cap) + lay_max=40 + BC=500.
# Tested across 4 seeds. Also a lay_max=100 variant for comparison.
#
# Eval is on 14 days (vs 5) so each cell is slower (~35-40 min).
# 6 cells ≈ 3.5-4h wall.

set -u
REPO=/c/Users/jsmit/source/repos/rl-betfair
cd "$REPO"

WRAPPER_LOG="registry/_holdout_validation_wrapper.log"
exec >> "$WRAPPER_LOG" 2>&1
echo "[$(date -Iseconds)] held-out validation wrapper started"

TRAIN_DAYS=( 2026-04-06 2026-04-08 2026-04-09 )
# 14 held-out days, never in training or prior eval.
HOLDOUT_DAYS=(
  2026-05-07 2026-05-08 2026-05-09 2026-05-10 2026-05-11
  2026-05-12 2026-05-13 2026-05-14 2026-05-15 2026-05-16
  2026-05-17 2026-05-18 2026-05-19 2026-05-20
)
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
  --cohort-eval-days "${HOLDOUT_DAYS[@]}"
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
    local seed="${HV_SEED:-42}"
    local ts; ts=$(date +%s)
    local outdir="registry/_holdout_${cell}_${ts}"
    local logfile="${outdir}.log"
    echo "[$(date -Iseconds)] starting cell ${cell} (seed=${seed}) -> ${outdir}"
    python -m training_v2.cohort.runner "${BASE[@]}" --seed "${seed}" --output-dir "${outdir}" "$@" > "${logfile}" 2>&1
    local rc=$?
    echo "[$(date -Iseconds)] cell ${cell} finished rc=${rc}"
}

# Deploy candidate (lay_max=40) across 4 seeds on held-out days.
HV_SEED=42 run_cell "HV_laymax40_seed42" --lay-price-max 40
HV_SEED=43 run_cell "HV_laymax40_seed43" --lay-price-max 40
HV_SEED=44 run_cell "HV_laymax40_seed44" --lay-price-max 40
HV_SEED=45 run_cell "HV_laymax40_seed45" --lay-price-max 40

# lay_max=100 variant (higher mean in-sample, wider variance) on
# held-out — does the higher cap survive out-of-sample?
HV_SEED=42 run_cell "HV_laymax100_seed42" --lay-price-max 100
HV_SEED=43 run_cell "HV_laymax100_seed43" --lay-price-max 100

echo "[$(date -Iseconds)] held-out validation fan-out complete"
