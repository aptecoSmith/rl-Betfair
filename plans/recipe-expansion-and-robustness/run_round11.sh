#!/usr/bin/env bash
# Round 11 — Reward-shape variations + alternate exit mechanisms.
#
# Even if force_close=0 works, naked variance may still be too large
# in edge cases. Tests alternate exits: agent-only close, naked_loss_scale
# variations, close_signal_bonus +ve to incentivise agent-close.

set -u
REPO=/c/Users/jsmit/source/repos/rl-betfair
cd "$REPO"

WRAPPER_LOG="registry/_round11_wrapper.log"
exec >> "$WRAPPER_LOG" 2>&1
echo "[$(date -Iseconds)] round 11 wrapper started"

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
  --reward-overrides close_feasibility_max_spread_pct=0.05
  --reward-overrides matured_arb_expected_random=0.0
  --reward-overrides force_close_before_off_seconds=0
  --bc-pretrain-steps 500
  --predictor-p-win-back-threshold 0.20
  --lay-price-max 20
)

run_cell () {
    local cell="$1"; shift
    local ts; ts=$(date +%s)
    local outdir="registry/_round11_${cell}_${ts}"
    local logfile="${outdir}.log"
    echo "[$(date -Iseconds)] starting cell ${cell} -> ${outdir}"
    python -m training_v2.cohort.runner "${BASE[@]}" --output-dir "${outdir}" "$@" > "${logfile}" 2>&1
    local rc=$?
    echo "[$(date -Iseconds)] cell ${cell} finished rc=${rc}"
}

# Group N1 — naked_loss_scale variations (4 cells)
# Lower scale = less penalty on naked losses = policy more naked-friendly
run_cell "N1_naked_loss_scale_05" --reward-overrides naked_loss_scale=0.5
run_cell "N1_naked_loss_scale_07" --reward-overrides naked_loss_scale=0.7
run_cell "N1_naked_loss_scale_03" --reward-overrides naked_loss_scale=0.3
run_cell "N1_naked_loss_scale_15" --reward-overrides naked_loss_scale=1.5

# Group N2 — close_signal_bonus variations (3 cells)
# Positive bonus encourages agent-close instead of force-close
run_cell "N2_close_bonus_+2" --reward-overrides close_signal_bonus=2.0
run_cell "N2_close_bonus_+5" --reward-overrides close_signal_bonus=5.0
run_cell "N2_close_bonus_-1" --reward-overrides close_signal_bonus=-1.0

# Group N3 — matured_arb_bonus + force_close_off (3 cells)
# Encourage natural maturation when force-close isn't catching pairs
run_cell "N3_mat_bonus_5" --reward-overrides matured_arb_bonus_weight=5.0
run_cell "N3_mat_bonus_2" --reward-overrides matured_arb_bonus_weight=2.0
run_cell "N3_mat_bonus_10" --reward-overrides matured_arb_bonus_weight=10.0

# Group N4 — close_feasibility tighter (2 cells)
# Smaller spread requirement for close action
run_cell "N4_close_spread_003" --reward-overrides close_feasibility_max_spread_pct=0.03
run_cell "N4_close_spread_002" --reward-overrides close_feasibility_max_spread_pct=0.02

echo "[$(date -Iseconds)] round 11 fan-out complete"
