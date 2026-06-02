#!/usr/bin/env bash
# Wait for the close-penalty wrapper to complete, then run the
# D-cells (direction-predictor mechanism: gate-on vs gate-off
# vs no-direction-at-all).
#
# This wrapper assumes the DIRECTION_GATE_THRESHOLD_MIN/MAX clamp
# fix has been deployed (2026-05-25). With the fix:
#   - MIN = 0.10 (was 0.5)
#   - MAX = 0.60 (was 0.95)
# The cells below set thresholds 0.30 and 0.45 — both will now pass
# through the clamp instead of being silently rounded up to 0.5.

set -u

REPO=/c/Users/jsmit/source/repos/rl-betfair
cd "$REPO"

UPSTREAM_LOG="registry/_close_penalty_wrapper.log"
WRAPPER_LOG="registry/_d_cells_wrapper.log"
exec >> "$WRAPPER_LOG" 2>&1

echo "[$(date -Iseconds)] D-cells wrapper started; waiting on close-penalty wrapper"

while true; do
    if grep -q "close-penalty fan-out complete" "$UPSTREAM_LOG" 2>/dev/null; then
        echo "[$(date -Iseconds)] close-penalty complete; starting D-cells"
        break
    fi
    sleep 60
done

sleep 30

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
  --bc-pretrain-steps 0
  --direction-head-manifest models/direction_head/sweep_c11
  --predictor-lean-obs
  --use-race-outcome-predictor
  --use-direction-predictor
  --predictor-bundle-manifests "${PREDICTORS[@]}"
)

run_cell () {
    local cell="$1"; shift
    local ts; ts=$(date +%s)
    local outdir="registry/_d_cells_${cell}_${ts}"
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

# D0 control: gate completely OFF. Direction signals in obs + C11
# in actor_head, but no policy-side gate masking.
run_cell "D0_gate_off" \
    --reward-overrides force_close_before_off_seconds=120 \
    --reward-overrides close_feasibility_max_spread_pct=0.05 \
    --reward-overrides matured_arb_expected_random=0.0

# D2 policy-side gate at threshold 0.30. With the clamp fix this
# is the TRUE 0.30 instead of being silently clamped to 0.5.
run_cell "D2_gate_t030" \
    --direction-gate-enabled \
    --reward-overrides force_close_before_off_seconds=120 \
    --reward-overrides close_feasibility_max_spread_pct=0.05 \
    --reward-overrides matured_arb_expected_random=0.0 \
    --reward-overrides direction_gate_threshold=0.30

# D2b policy-side gate at threshold 0.45 — second point on the
# response curve to confirm gradient between thresholds.
run_cell "D2b_gate_t045" \
    --direction-gate-enabled \
    --reward-overrides force_close_before_off_seconds=120 \
    --reward-overrides close_feasibility_max_spread_pct=0.05 \
    --reward-overrides matured_arb_expected_random=0.0 \
    --reward-overrides direction_gate_threshold=0.45

# D3 gate at threshold 0.20 — near-no-op (refuses only the tail).
# Should produce results very close to D0 if the threshold is
# meaningful.
run_cell "D3_gate_t020" \
    --direction-gate-enabled \
    --reward-overrides force_close_before_off_seconds=120 \
    --reward-overrides close_feasibility_max_spread_pct=0.05 \
    --reward-overrides matured_arb_expected_random=0.0 \
    --reward-overrides direction_gate_threshold=0.20

echo "[$(date -Iseconds)] D-cells fan-out complete"
