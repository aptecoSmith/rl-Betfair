#!/usr/bin/env bash
# Round 10 — Scale-up of presumed winners + multi-gen.
#
# 12 cells, ~7h wall (includes multi-gen cells).

set -u
REPO=/c/Users/jsmit/source/repos/rl-betfair
cd "$REPO"

WRAPPER_LOG="registry/_round10_wrapper.log"
exec >> "$WRAPPER_LOG" 2>&1
echo "[$(date -Iseconds)] round 10 wrapper started"

TRAIN_DAYS=( 2026-04-06 2026-04-08 2026-04-09 )
EVAL_DAYS=( 2026-04-10 2026-04-17 2026-04-21 2026-05-03 2026-05-06 )
PREDICTORS=(
  "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json"
  "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json"
  "C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json"
)

BASE=(
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
  --lay-price-max 20
)

run_cell () {
    local cell="$1"; shift
    local n_agents="${R10_AGENTS:-4}"
    local generations="${R10_GENS:-1}"
    local seed="${R10_SEED:-42}"
    local ts; ts=$(date +%s)
    local outdir="registry/_round10_${cell}_${ts}"
    local logfile="${outdir}.log"
    echo "[$(date -Iseconds)] starting cell ${cell} (n=${n_agents} gen=${generations} seed=${seed}) -> ${outdir}"
    python -m training_v2.cohort.runner "${BASE[@]}" --n-agents "${n_agents}" --generations "${generations}" --seed "${seed}" --output-dir "${outdir}" "$@" > "${logfile}" 2>&1
    local rc=$?
    echo "[$(date -Iseconds)] cell ${cell} finished rc=${rc}"
}

# Group M1 — 8-agent scale-up (3 cells, ~50 min each)
R10_AGENTS=8 run_cell "M1_8agents_base"
R10_AGENTS=8 R10_SEED=43 run_cell "M1_8agents_seed43"
R10_AGENTS=8 run_cell "M1_8agents_pwin025" --predictor-p-win-back-threshold 0.25

# Group M2 — Multi-gen (3 cells, ~75-125 min)
R10_GENS=2 run_cell "M2_2gen"
R10_GENS=3 run_cell "M2_3gen"
R10_GENS=5 run_cell "M2_5gen"

# Group M3 — BC dose at presumed winner (3 cells)
run_cell "M3_bc1000" --bc-pretrain-steps 1000
run_cell "M3_bc250" --bc-pretrain-steps 250
run_cell "M3_bc1500" --bc-pretrain-steps 1500

# Group M4 — Larger seed sample for robustness (3 cells)
R10_SEED=47 run_cell "M4_seed47"
R10_SEED=48 run_cell "M4_seed48"
R10_SEED=49 run_cell "M4_seed49"

echo "[$(date -Iseconds)] round 10 fan-out complete"
