#!/usr/bin/env bash
# Round 8 — step changes if Rounds 6/7 still don't crack 4/5.
#
# Tests structural changes: bigger cohort (8 agents), more training
# data (5 train days), force-close window variations, and a "kitchen
# sink" stack of the most promising levers from Rounds 1-7.
#
# 10 cells, ~5-7h wall (some cells larger).

set -u

REPO=/c/Users/jsmit/source/repos/rl-betfair
cd "$REPO"

WRAPPER_LOG="registry/_round8_wrapper.log"
exec >> "$WRAPPER_LOG" 2>&1

echo "[$(date -Iseconds)] round 8 wrapper started"

TRAIN_DAYS_3=( 2026-04-06 2026-04-08 2026-04-09 )
TRAIN_DAYS_5=( 2026-04-06 2026-04-07 2026-04-08 2026-04-09 2026-04-10 )
EVAL_DAYS=( 2026-04-17 2026-04-21 2026-05-03 2026-05-06 )
PREDICTORS=(
  "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json"
  "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json"
  "C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json"
)

BASE=(
  --device cuda
  --strategy-mode arb
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
    local n_agents="${R8_AGENTS:-4}"
    local generations="${R8_GENS:-1}"
    local seed="${R8_SEED:-42}"
    # Default to 3-day train; --training-days-explicit override allowed via "$@"
    local train_flag=( --training-days-explicit "${TRAIN_DAYS_3[@]}" )
    if [[ -n "${R8_5DAY:-}" ]]; then
        train_flag=( --training-days-explicit "${TRAIN_DAYS_5[@]}" )
    fi
    local ts; ts=$(date +%s)
    local outdir="registry/_round8_${cell}_${ts}"
    local logfile="${outdir}.log"
    echo "[$(date -Iseconds)] starting cell ${cell} (n=${n_agents} gen=${generations} seed=${seed}) -> ${outdir}"
    python -m training_v2.cohort.runner \
        "${BASE[@]}" \
        --n-agents "${n_agents}" \
        --generations "${generations}" \
        --seed "${seed}" \
        "${train_flag[@]}" \
        --cohort-eval-days "${EVAL_DAYS[@]}" \
        --output-dir "${outdir}" \
        "$@" \
        > "${logfile}" 2>&1
    local rc=$?
    echo "[$(date -Iseconds)] cell ${cell} finished rc=${rc}"
}

# Group A — Scale-up cohort (8 agents). 2 cells, ~80 min wall each.
R8_AGENTS=8 run_cell "I1_8agents_e7"
R8_AGENTS=8 run_cell "I1_8agents_e7_tight_lock" --arb-spread-target-lock-pct 0.005

# Group B — Force-close window variations. 3 cells.
run_cell "I2_fc90"  --reward-overrides force_close_before_off_seconds=90
run_cell "I2_fc45"  --reward-overrides force_close_before_off_seconds=45
run_cell "I2_fc30"  --reward-overrides force_close_before_off_seconds=30

# Group C — 5-day training. 2 cells.
R8_5DAY=1 run_cell "I3_5day_e7"
R8_5DAY=1 run_cell "I3_5day_e7_tight_lock" --arb-spread-target-lock-pct 0.005

# Group D — Kitchen sink (best-of-best levers stacked). 3 cells.
run_cell "I4_kitchen_sink" \
    --predictor-p-win-back-threshold 0.25 \
    --arb-spread-target-lock-pct 0.005 \
    --lay-price-max 20
run_cell "I4_kitchen_sink_l34" \
    --predictor-p-win-back-threshold 0.25 \
    --arb-spread-target-lock-pct 0.005 \
    --lay-price-max 20 \
    --bc-include-close-hold-samples
R8_AGENTS=8 run_cell "I4_kitchen_sink_8agents" \
    --predictor-p-win-back-threshold 0.25 \
    --arb-spread-target-lock-pct 0.005 \
    --lay-price-max 20

echo "[$(date -Iseconds)] round 8 fan-out complete"
