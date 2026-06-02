#!/usr/bin/env bash
# Round 7 — explore Round 6's most informative axes more deeply.
#
# Hypothesis-driven design assuming Round 6 shows:
# - E7 has £40-60 seed variance (R1-like)
# - One or two env-side levers (Group B) improve E7
# - Partial augmentation (Group C) lands closer to in-band than full
# - BC dose around 500 is approximately right
#
# 12 cells, ~5h wall.

set -u

REPO=/c/Users/jsmit/source/repos/rl-betfair
cd "$REPO"

WRAPPER_LOG="registry/_round7_wrapper.log"
exec >> "$WRAPPER_LOG" 2>&1

echo "[$(date -Iseconds)] round 7 wrapper started"

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
  --reward-overrides force_close_before_off_seconds=120
  --reward-overrides close_feasibility_max_spread_pct=0.05
  --reward-overrides matured_arb_expected_random=0.0
  --bc-pretrain-steps 500
  --predictor-p-win-back-threshold 0.20
)

run_cell () {
    local cell="$1"; shift
    local seed="${R7_SEED:-42}"
    local ts; ts=$(date +%s)
    local outdir="registry/_round7_${cell}_${ts}"
    local logfile="${outdir}.log"
    echo "[$(date -Iseconds)] starting cell ${cell} (seed=${seed}) -> ${outdir}"
    python -m training_v2.cohort.runner \
        "${BASE[@]}" \
        --seed "${seed}" \
        --output-dir "${outdir}" \
        "$@" \
        > "${logfile}" 2>&1
    local rc=$?
    echo "[$(date -Iseconds)] cell ${cell} finished rc=${rc}"
}

# Group A — pwin_back fine-sweep above 0.20 (where R3 showed
# day_pnl improved). 4 cells.
run_cell "H1_pwin022" --predictor-p-win-back-threshold 0.22
run_cell "H1_pwin028" --predictor-p-win-back-threshold 0.28
run_cell "H1_pwin032" --predictor-p-win-back-threshold 0.32
run_cell "H1_pwin040" --predictor-p-win-back-threshold 0.40

# Group B — stack two env-side levers on E7. 4 cells.
run_cell "H2_e7_tight_lock_pwin025" --arb-spread-target-lock-pct 0.005 --predictor-p-win-back-threshold 0.25
run_cell "H2_e7_lay_max_race_conf" --lay-price-max 20 --race-confidence-threshold 0.35
run_cell "H2_e7_tight_lock_race_conf" --arb-spread-target-lock-pct 0.005 --race-confidence-threshold 0.35
run_cell "H2_e7_tight_lock_lay_max" --arb-spread-target-lock-pct 0.005 --lay-price-max 20

# Group C — Partial augmentation refinements. 2 cells.
run_cell "H3_l34_only_tight_lock" --bc-include-close-hold-samples --arb-spread-target-lock-pct 0.005
run_cell "H3_l34_only_pwin025" --bc-include-close-hold-samples --predictor-p-win-back-threshold 0.25

# Group D — larger BC + augmentation. 2 cells.
run_cell "H4_bc1500" --bc-pretrain-steps 1500
run_cell "H4_bc1500_l34" --bc-pretrain-steps 1500 --bc-include-close-hold-samples

echo "[$(date -Iseconds)] round 7 fan-out complete"
