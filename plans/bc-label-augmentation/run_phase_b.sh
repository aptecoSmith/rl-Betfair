#!/usr/bin/env bash
# Phase B validation probe — runs after BOTH preconditions land:
#   1. Phase A wrapper completes ("phase-a fan-out complete" in
#      registry/_phase_a_wrapper.log).  (Already done at 21:03.)
#   2. Phase B oracle re-scan completes (marker file
#      registry/_bc_label_aug_phase_b_ready exists).
#
# Three cells × ~25 min each = ~1.3h wall.

set -u

REPO=/c/Users/jsmit/source/repos/rl-betfair
cd "$REPO"

WRAPPER_LOG="registry/_phase_b_wrapper.log"
exec >> "$WRAPPER_LOG" 2>&1

echo "[$(date -Iseconds)] phase-b wrapper started; waiting on phase-a + oracle re-scan"

# Poll for both preconditions every 60s.
while true; do
    phase_a_done=false
    rescan_done=false
    if grep -q "phase-a fan-out complete" registry/_phase_a_wrapper.log 2>/dev/null; then
        phase_a_done=true
    fi
    if [[ -f registry/_bc_label_aug_phase_b_ready ]]; then
        rescan_done=true
    fi
    if [[ "$phase_a_done" == "true" && "$rescan_done" == "true" ]]; then
        echo "[$(date -Iseconds)] both prereqs satisfied; launching phase-b cells"
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
    local outdir="registry/_phase_b_${cell}_${ts}"
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

# F2: Close-positive (L3a) + hold-positive (L4) only. NO NOOP-at-
# oracle-negative (L2). Isolates whether close labels alone rescue
# fc% — Phase A confirmed L2 alone makes the close problem WORSE,
# so this tests whether L3a+L4 alone is sufficient.
run_cell "F2_close_hold_aug" \
    --bc-include-close-hold-samples

# F3: Full stack — L2 + L3a + L4. The deploy-candidate hopeful.
# L2 keeps opens in band; L3a+L4 ground the close decision.
run_cell "F3_full_aug" \
    --bc-include-negative-samples \
    --bc-include-close-hold-samples

# F3b: F3 with positive weight 2× — the F1b setting that gave the
# best per-agent variance discipline in Phase A. Hedges against
# the augmentation pool ratios being wrong.
run_cell "F3b_full_aug_pos2x" \
    --bc-include-negative-samples \
    --bc-include-close-hold-samples \
    --bc-positive-weight 2.0

echo "[$(date -Iseconds)] phase-b fan-out complete"
