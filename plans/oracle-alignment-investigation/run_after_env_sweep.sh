#!/usr/bin/env bash
# Wait for the env-side sweep to complete, then run 3 close_signal
# discouragement probes + 1 BC-pretrain probe.
#
# Mechanism: ``close_signal_bonus`` is already a --reward-overrides
# key (env/betfair_env.py:182). Setting it negative shaped-penalises
# every successful close_signal action. The bigger the magnitude,
# the harder the policy is pushed away from close_signal as a
# stop-loss substitute.
#
# Why this matters: the recipe-sensitivity-sweep behavioural deep
# dive (plans/recipe-sensitivity-sweep/behavioural_findings.md)
# found close_signal kills 69 % of opens before maturation; 84 %
# of agent_closed back opens are at adverse drift (= stop-loss
# behaviour). The 7-9% of opens that DO mature have an upside of
# +£0.31/pair vs the -£2.01 cost of agent_closed. Forcing the
# policy to hold instead of close should shift the distribution
# substantially.

set -u

REPO=/c/Users/jsmit/source/repos/rl-betfair
cd "$REPO"

ENV_WRAPPER_LOG="registry/_env_sweep_wrapper.log"
WRAPPER_LOG="registry/_close_penalty_wrapper.log"
exec >> "$WRAPPER_LOG" 2>&1

echo "[$(date -Iseconds)] wrapper started; waiting on env-side sweep completion"

# Poll for "env-side fan-out complete" in the env-side wrapper log.
while true; do
    if grep -q "env-side fan-out complete" "$ENV_WRAPPER_LOG" 2>/dev/null; then
        echo "[$(date -Iseconds)] env-side sweep complete; starting close-penalty fan-out"
        break
    fi
    sleep 60
done

# Brief settle.
sleep 30

TRAIN_DAYS=( 2026-04-06 2026-04-08 2026-04-09 )
EVAL_DAYS=( 2026-04-10 2026-04-17 2026-04-21 2026-05-03 2026-05-06 )
PREDICTORS=(
  "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json"
  "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json"
  "C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json"
)

# Common flags every cell shares.
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
)

run_cell () {
    local cell="$1"; shift
    local ts; ts=$(date +%s)
    local outdir="registry/_close_penalty_${cell}_${ts}"
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

# PC0: baseline reference (same as env-side C0 — replicate for
# direct comparability since cells were run different days).
run_cell "PC0_baseline_bc0" \
    --bc-pretrain-steps 0 \
    --reward-overrides force_close_before_off_seconds=120 \
    --reward-overrides close_feasibility_max_spread_pct=0.05 \
    --reward-overrides matured_arb_expected_random=0.0

# PC1: moderate close penalty
run_cell "PC1_close_penalty_-2_bc0" \
    --bc-pretrain-steps 0 \
    --reward-overrides force_close_before_off_seconds=120 \
    --reward-overrides close_feasibility_max_spread_pct=0.05 \
    --reward-overrides matured_arb_expected_random=0.0 \
    --reward-overrides close_signal_bonus=-2.0

# PC2: strong close penalty
run_cell "PC2_close_penalty_-5_bc0" \
    --bc-pretrain-steps 0 \
    --reward-overrides force_close_before_off_seconds=120 \
    --reward-overrides close_feasibility_max_spread_pct=0.05 \
    --reward-overrides matured_arb_expected_random=0.0 \
    --reward-overrides close_signal_bonus=-5.0

# PC3: BC pretrain on (1000 steps), close penalty off — isolates
# the selection-fix hypothesis. BC initialises the policy in the
# oracle's distribution (price 5-30, low champion_p_win) so the
# starting open-band should shift right.
run_cell "PC3_bc1000_no_penalty" \
    --bc-pretrain-steps 1000 \
    --reward-overrides force_close_before_off_seconds=120 \
    --reward-overrides close_feasibility_max_spread_pct=0.05 \
    --reward-overrides matured_arb_expected_random=0.0

# PC4: BC + close penalty — joint fix (selection + holding).
# If the headline finding is right, this should give the largest
# mat% bump.
run_cell "PC4_bc1000_close_penalty_-2" \
    --bc-pretrain-steps 1000 \
    --reward-overrides force_close_before_off_seconds=120 \
    --reward-overrides close_feasibility_max_spread_pct=0.05 \
    --reward-overrides matured_arb_expected_random=0.0 \
    --reward-overrides close_signal_bonus=-2.0

echo "[$(date -Iseconds)] close-penalty fan-out complete"
