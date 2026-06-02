#!/usr/bin/env bash
# Round 6 — E7-anchored exploration after Round 5 plateau.
#
# Strategy: Round 5 confirmed full L2+L3a+L4 augmentation regressed
# vs E7 on day_pnl. E7 remains the 3/5 leader. Round 6 tests
# E7-anchored variations to break the plateau.
#
# 12 cells × ~25 min = ~5h wall.

set -u

REPO=/c/Users/jsmit/source/repos/rl-betfair
cd "$REPO"

WRAPPER_LOG="registry/_round6_wrapper.log"
exec >> "$WRAPPER_LOG" 2>&1

echo "[$(date -Iseconds)] round 6 wrapper started"

TRAIN_DAYS=( 2026-04-06 2026-04-08 2026-04-09 )
EVAL_DAYS=( 2026-04-10 2026-04-17 2026-04-21 2026-05-03 2026-05-06 )
PREDICTORS=(
  "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json"
  "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json"
  "C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json"
)

# E7-base flags: BC=500 + pwin_back=0.20, NO augmentation. This is
# the recipe that won 3/5 acceptance in Round 4.
E7_BASE=(
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
    local seed="${R6_SEED:-42}"
    local ts; ts=$(date +%s)
    local outdir="registry/_round6_${cell}_${ts}"
    local logfile="${outdir}.log"
    echo "[$(date -Iseconds)] starting cell ${cell} (seed=${seed}) -> ${outdir}"
    python -m training_v2.cohort.runner \
        "${E7_BASE[@]}" \
        --seed "${seed}" \
        --output-dir "${outdir}" \
        "$@" \
        > "${logfile}" 2>&1
    local rc=$?
    echo "[$(date -Iseconds)] cell ${cell} finished rc=${rc}"
}

# ---------------------------------------------------------------
# Group A — E7 baseline replicates (3 cells)
# Establish E7's seed variance to know if E7's -£66 was lucky.
# ---------------------------------------------------------------
R6_SEED=43 run_cell "G1_e7_seed43"
R6_SEED=44 run_cell "G1_e7_seed44"
R6_SEED=46 run_cell "G1_e7_seed46"

# ---------------------------------------------------------------
# Group B — E7 + single env-side lever (4 cells)
# Tests whether stacking known levers on E7 lifts metrics.
# ---------------------------------------------------------------
run_cell "G2_e7_tight_lock" --arb-spread-target-lock-pct 0.005
run_cell "G2_e7_race_conf035" --race-confidence-threshold 0.35
run_cell "G2_e7_lay_price_max20" --lay-price-max 20
run_cell "G2_e7_pwin_back025" --predictor-p-win-back-threshold 0.25

# ---------------------------------------------------------------
# Group C — Partial augmentation on E7 (3 cells)
# Round 5 used full L2+L3a+L4 stack. Test if PARTIAL augmentation
# (L2 alone or L3a+L4 alone) lands closer to the goal.
# ---------------------------------------------------------------
run_cell "G3_e7_l2_only" --bc-include-negative-samples
run_cell "G3_e7_l34_only" --bc-include-close-hold-samples
run_cell "G3_e7_l2_lowweight" --bc-include-negative-samples --bc-positive-weight 0.5

# ---------------------------------------------------------------
# Group D — BC dose on E7 (2 cells)
# ---------------------------------------------------------------
run_cell "G4_e7_bc1000" --bc-pretrain-steps 1000
run_cell "G4_e7_bc250"  --bc-pretrain-steps 250

echo "[$(date -Iseconds)] round 6 fan-out complete"
