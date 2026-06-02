#!/usr/bin/env bash
# Phase A validation probe — runs after BOTH preconditions land:
#   1. Round 3 wrapper completes ("round 3 fan-out complete" in
#      registry/_round3_wrapper.log).
#   2. Phase A oracle re-scan completes (marker file
#      registry/_bc_label_aug_phase_a_ready exists).
#
# Three cells × ~25 min each = ~1.3h wall.

set -u

REPO=/c/Users/jsmit/source/repos/rl-betfair
cd "$REPO"

WRAPPER_LOG="registry/_phase_a_wrapper.log"
exec >> "$WRAPPER_LOG" 2>&1

echo "[$(date -Iseconds)] phase-a wrapper started; waiting on round3 + oracle re-scan"

# Poll for both preconditions every 60s.
while true; do
    round3_done=false
    rescan_done=false
    if grep -q "round 3 fan-out complete" registry/_round3_wrapper.log 2>/dev/null; then
        round3_done=true
    fi
    if [[ -f registry/_bc_label_aug_phase_a_ready ]]; then
        rescan_done=true
    fi
    if [[ "$round3_done" == "true" && "$rescan_done" == "true" ]]; then
        echo "[$(date -Iseconds)] both prereqs satisfied; launching phase-a cells"
        break
    fi
    sleep 60
done

# Settle.
sleep 15

TRAIN_DAYS=( 2026-04-06 2026-04-08 2026-04-09 )
EVAL_DAYS=( 2026-04-10 2026-04-17 2026-04-21 2026-05-03 2026-05-06 )
PREDICTORS=(
  "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json"
  "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json"
  "C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json"
)

COMMON_FLAGS=(
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
  --reward-overrides force_close_before_off_seconds=120
  --reward-overrides close_feasibility_max_spread_pct=0.05
  --reward-overrides matured_arb_expected_random=0.0
  --predictor-p-win-back-threshold 0.20
  --bc-pretrain-steps 500
)

run_cell () {
    local cell="$1"; shift
    local ts; ts=$(date +%s)
    local outdir="registry/_phase_a_${cell}_${ts}"
    local logfile="${outdir}.log"
    echo "[$(date -Iseconds)] starting cell ${cell} -> ${outdir}"
    python -m training_v2.cohort.runner \
        "${COMMON_FLAGS[@]}" \
        --output-dir "${outdir}" \
        "$@" \
        > "${logfile}" 2>&1
    local rc=$?
    echo "[$(date -Iseconds)] cell ${cell} finished rc=${rc}"
}

# F0: E7 repeat sanity. NO NOOP augmentation. Confirms the augmented
# code path is byte-identical to pre-plan when the augmentation flag
# is off.
run_cell "F0_e7_repeat"

# F1: NOOP augmentation at 2:1 negative:positive ratio (default).
run_cell "F1_noop_aug" \
    --bc-include-negative-samples

# F1b: NOOP augmentation but positives weighted 2× in the CE loss
# (defensive — if F1 over-suppresses opens, F1b balances back).
run_cell "F1b_noop_aug_pos2x" \
    --bc-include-negative-samples \
    --bc-positive-weight 2.0

echo "[$(date -Iseconds)] phase-a fan-out complete"
