#!/usr/bin/env bash
# Round 9.8 — COMBO of the two winners: lay_max=100 + pwin band 0.20-0.50.
#
# Round 9.5: lay_max=100 best mean (+£287) but wide seed variance.
# Round 9.75: pwin band 0.20-0.50 best floor (+£214).
# Hypothesis: combining them yields high mean AND tight floor —
# the candidate DEPLOY recipe. Tested across 4 seeds + bc1000 variant.
#
# 8 cells × ~25 min = ~3.3h wall.

set -u
REPO=/c/Users/jsmit/source/repos/rl-betfair
cd "$REPO"

WRAPPER_LOG="registry/_round9_8_wrapper.log"
exec >> "$WRAPPER_LOG" 2>&1
echo "[$(date -Iseconds)] round 9.8 wrapper started — combo lay_max=100 + band 0.20-0.50"

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
  --predictor-p-win-back-max-threshold 0.50
  --lay-price-max 100
)

run_cell () {
    local cell="$1"; shift
    local seed="${R98_SEED:-42}"
    local ts; ts=$(date +%s)
    local outdir="registry/_round9_8_${cell}_${ts}"
    local logfile="${outdir}.log"
    echo "[$(date -Iseconds)] starting cell ${cell} (seed=${seed}) -> ${outdir}"
    python -m training_v2.cohort.runner "${BASE[@]}" --seed "${seed}" --output-dir "${outdir}" "$@" > "${logfile}" 2>&1
    local rc=$?
    echo "[$(date -Iseconds)] cell ${cell} finished rc=${rc}"
}

# Group Q1 — the combo across 4 seeds (robustness of the candidate
# deploy recipe).
R98_SEED=42 run_cell "Q1_combo_seed42"
R98_SEED=43 run_cell "Q1_combo_seed43"
R98_SEED=44 run_cell "Q1_combo_seed44"
R98_SEED=45 run_cell "Q1_combo_seed45"

# Group Q2 — combo + bc1000 (variance-discipline) across 2 seeds.
R98_SEED=42 run_cell "Q2_combo_bc1000_seed42" --bc-pretrain-steps 1000
R98_SEED=43 run_cell "Q2_combo_bc1000_seed43" --bc-pretrain-steps 1000

# Group Q3 — combo at lay_max=50 (lower variance than 100) for
# comparison.
R98_SEED=42 run_cell "Q3_combo_laymax50_seed42" --lay-price-max 50
R98_SEED=43 run_cell "Q3_combo_laymax50_seed43" --lay-price-max 50

echo "[$(date -Iseconds)] round 9.8 fan-out complete"
