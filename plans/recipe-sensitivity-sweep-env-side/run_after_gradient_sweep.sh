#!/usr/bin/env bash
# Wait for the gradient sweep to complete, then run the 6-cell env-side
# fan-out. Logs every step so we can see when each cohort starts /
# finishes if the operator checks back in mid-run.
#
# Cohort completion signal: "Cohort complete" appears in the main log.
# (training_v2/cohort/runner.py logs this at end of run.)

set -u

REPO=/c/Users/jsmit/source/repos/rl-betfair
cd "$REPO"

GRAD_LOG="registry/_recipe_sensitivity_sweep_1779662659.log"
WRAPPER_LOG="registry/_env_sweep_wrapper.log"
exec >> "$WRAPPER_LOG" 2>&1

echo "[$(date -Iseconds)] wrapper started; waiting on $GRAD_LOG"

# Poll for completion of the gradient sweep.
# We watch for either "Cohort complete" (success) or an exception
# block (failure). On failure we abort the env-side sweep — no point
# running it if the gradient sweep died.
while true; do
    if grep -q "Cohort complete" "$GRAD_LOG" 2>/dev/null; then
        echo "[$(date -Iseconds)] gradient sweep complete; starting env-side fan-out"
        break
    fi
    if grep -qE "Traceback \(most recent call last\):" "$GRAD_LOG" 2>/dev/null; then
        if ! grep -q "Cohort complete" "$GRAD_LOG" 2>/dev/null; then
            echo "[$(date -Iseconds)] gradient sweep died (traceback in log); aborting"
            exit 1
        fi
    fi
    sleep 60
done

# Brief settle so file writes are flushed.
sleep 30

TRAIN_DAYS=( 2026-04-06 2026-04-08 2026-04-09 )
EVAL_DAYS=( 2026-04-10 2026-04-17 2026-04-21 2026-05-03 2026-05-06 )
PREDICTORS=(
  "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json"
  "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json"
  "C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json"
)

# Common flags every cell shares. NOTE: --direction-gate-enabled is
# NOT in COMMON_FLAGS — cells that test the policy-side direction
# gate (C5, C6, C7) add it explicitly. C0 through C4 have the gate
# completely off for a clean baseline.
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

# Run one cell. Args: cell_name <extra_cli_args...>
run_cell () {
    local cell="$1"; shift
    local ts; ts=$(date +%s)
    local outdir="registry/_env_sweep_${cell}_${ts}"
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

# Each cell's env-side knob delta is appended via extra args. The
# common reward-overrides + the cell's own get layered.

# C0 baseline — same as gradient sweep's structural defaults.
run_cell "C0_baseline" \
    --reward-overrides force_close_before_off_seconds=120 \
    --reward-overrides close_feasibility_max_spread_pct=0.05 \
    --reward-overrides matured_arb_expected_random=0.0

# C1 tight force-close
run_cell "C1_fc60" \
    --reward-overrides force_close_before_off_seconds=60 \
    --reward-overrides close_feasibility_max_spread_pct=0.05 \
    --reward-overrides matured_arb_expected_random=0.0

# C2 back-side pwin gate
run_cell "C2_pwin_back_020" \
    --reward-overrides force_close_before_off_seconds=120 \
    --reward-overrides close_feasibility_max_spread_pct=0.05 \
    --reward-overrides matured_arb_expected_random=0.0 \
    --predictor-p-win-back-threshold 0.20

# C3 lay-side pwin gate
run_cell "C3_pwin_lay_050" \
    --reward-overrides force_close_before_off_seconds=120 \
    --reward-overrides close_feasibility_max_spread_pct=0.05 \
    --reward-overrides matured_arb_expected_random=0.0 \
    --predictor-p-win-lay-threshold 0.50

# C4 race-level confidence gate
run_cell "C4_race_conf_035" \
    --reward-overrides force_close_before_off_seconds=120 \
    --reward-overrides close_feasibility_max_spread_pct=0.05 \
    --reward-overrides matured_arb_expected_random=0.0 \
    --race-confidence-threshold 0.35

# C5 direction gate moderate (threshold=0.30). Gate ENABLED via the
# CLI flag. The gate refuses OPEN_BACK/LAY where max(C11 head's
# back_prob, lay_prob) is below 0.30. Calibrated to the C11 head's
# actual output distribution: observed `max(back, lay)` per-runner
# mean ~0.32, so 0.30 ≈ refuse ~40-50 % of opens.
run_cell "C5_dir_gate_030" \
    --direction-gate-enabled \
    --reward-overrides force_close_before_off_seconds=120 \
    --reward-overrides close_feasibility_max_spread_pct=0.05 \
    --reward-overrides matured_arb_expected_random=0.0 \
    --reward-overrides direction_gate_threshold=0.30

# C6 direction gate aggressive (threshold=0.45). Refuses ~80-85 %
# of opens given the C11 head's distribution.
run_cell "C6_dir_gate_045" \
    --direction-gate-enabled \
    --reward-overrides force_close_before_off_seconds=120 \
    --reward-overrides close_feasibility_max_spread_pct=0.05 \
    --reward-overrides matured_arb_expected_random=0.0 \
    --reward-overrides direction_gate_threshold=0.45

# C7 stack everything on (tighter FC + both pwin gates + race conf +
# lay cap + direction gate at 0.35).
run_cell "C7_all_on" \
    --direction-gate-enabled \
    --reward-overrides force_close_before_off_seconds=60 \
    --reward-overrides close_feasibility_max_spread_pct=0.05 \
    --reward-overrides matured_arb_expected_random=0.0 \
    --reward-overrides direction_gate_threshold=0.35 \
    --predictor-p-win-back-threshold 0.20 \
    --predictor-p-win-lay-threshold 0.50 \
    --race-confidence-threshold 0.35 \
    --lay-price-max 20

echo "[$(date -Iseconds)] env-side fan-out complete"
