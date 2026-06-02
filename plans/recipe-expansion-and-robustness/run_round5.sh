#!/usr/bin/env bash
# Round 5 — Recipe Expansion and Robustness.
#
# Polls for Phase B completion ("phase-b fan-out complete" in
# registry/_phase_b_wrapper.log), then fires ~25 cells in 7 groups.
#
# BASE_RECIPE is filled in below AFTER Phase B results are
# interpreted (F3 vs F3b winner sets the positive_weight). The
# wrapper picks up whichever version of this file is current at
# the moment Phase B finishes — so the operator can edit the
# BASE_RECIPE between the wrapper launching and Phase B completing
# without restarting the wrapper. Polling is shell-state-only.

set -u

REPO=/c/Users/jsmit/source/repos/rl-betfair
cd "$REPO"

WRAPPER_LOG="registry/_round5_wrapper.log"
exec >> "$WRAPPER_LOG" 2>&1

echo "[$(date -Iseconds)] round 5 wrapper started; waiting on phase-b completion"

# Wait for Phase B to land. Re-reads the script on every poll so
# operator edits to BASE_RECIPE / cell list take effect before the
# first cell fires.
while true; do
    if grep -q "phase-b fan-out complete" registry/_phase_b_wrapper.log 2>/dev/null; then
        echo "[$(date -Iseconds)] phase-b complete; reloading wrapper to pick up BASE_RECIPE edits"
        # Exec ourselves with a magic env var to skip the wait next time.
        if [[ -z "${R5_REENTERED:-}" ]]; then
            export R5_REENTERED=1
            exec bash "$0"
        fi
        break
    fi
    sleep 60
done

# Brief settle.
sleep 15

# ---------------------------------------------------------------
# BASE_RECIPE — EDIT THIS based on Phase B's winning cell.
# Plausible values (placeholder is F3 = pos_weight 1.0 + both augs).
# After F3/F3b lands, change positive_weight here if F3b won.
# ---------------------------------------------------------------
TRAIN_DAYS=( 2026-04-06 2026-04-08 2026-04-09 )
EVAL_DAYS=( 2026-04-10 2026-04-17 2026-04-21 2026-05-03 2026-05-06 )
PREDICTORS=(
  "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json"
  "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json"
  "C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json"
)
BASE_BC_STEPS=500
BASE_PWIN_BACK=0.20
BASE_POSITIVE_WEIGHT=1.0   # set to 2.0 if F3b beat F3

# Common flags shared by every cell in the round.
BASE_FLAGS=(
  --n-agents 4
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
  --bc-include-negative-samples
  --bc-include-close-hold-samples
  --bc-pretrain-steps "${BASE_BC_STEPS}"
  --predictor-p-win-back-threshold "${BASE_PWIN_BACK}"
  --bc-positive-weight "${BASE_POSITIVE_WEIGHT}"
)

run_cell () {
    local cell="$1"; shift
    local generations="${R5_GENS:-1}"
    local seed="${R5_SEED:-42}"
    local ts; ts=$(date +%s)
    local outdir="registry/_round5_${cell}_${ts}"
    local logfile="${outdir}.log"
    echo "[$(date -Iseconds)] starting cell ${cell} (seed=${seed} gens=${generations}) -> ${outdir}"
    python -m training_v2.cohort.runner \
        "${BASE_FLAGS[@]}" \
        --seed "${seed}" \
        --generations "${generations}" \
        --output-dir "${outdir}" \
        "$@" \
        > "${logfile}" 2>&1
    local rc=$?
    echo "[$(date -Iseconds)] cell ${cell} finished rc=${rc}"
}

# ---------------------------------------------------------------
# Group 1 — Robustness across seeds (5 cells)
# ---------------------------------------------------------------
R5_SEED=42 run_cell "R1_seed42_repeat"
R5_SEED=43 run_cell "R1_seed43"
R5_SEED=44 run_cell "R1_seed44"
R5_SEED=45 run_cell "R1_seed45"
R5_SEED=46 run_cell "R1_seed46"

# ---------------------------------------------------------------
# Group 2 — BC dose-response on augmented pool (4 cells)
# ---------------------------------------------------------------
run_cell "R2_bc100"  --bc-pretrain-steps 100
run_cell "R2_bc200"  --bc-pretrain-steps 200
run_cell "R2_bc1000" --bc-pretrain-steps 1000
run_cell "R2_bc2000" --bc-pretrain-steps 2000

# ---------------------------------------------------------------
# Group 3 — pwin_back threshold sweep (4 cells)
# ---------------------------------------------------------------
run_cell "R3_pwin015" --predictor-p-win-back-threshold 0.15
run_cell "R3_pwin025" --predictor-p-win-back-threshold 0.25
run_cell "R3_pwin030" --predictor-p-win-back-threshold 0.30
run_cell "R3_pwin035" --predictor-p-win-back-threshold 0.35

# ---------------------------------------------------------------
# Group 4 — Multi-generation training (3 cells, longer wall)
# ---------------------------------------------------------------
R5_GENS=2 run_cell "R4_2gen"
R5_GENS=3 run_cell "R4_3gen"
R5_GENS=5 run_cell "R4_5gen"

# ---------------------------------------------------------------
# Group 5 — Direction signal value (2 cells)
# Skip R5_dir_gain_zero — needs new CLI flag not yet implemented.
# R5_no_direction is a structural change that requires removing
# the predictor; we approximate by zeroing the predictor weight.
# (Operator decision: defer the dir_gain probe to a follow-on
# unless wrapper is amended.)
# ---------------------------------------------------------------
# (intentionally empty — gated on follow-on engineering)

# ---------------------------------------------------------------
# Group 6 — Re-test dropped lay-side levers (3 cells)
# ---------------------------------------------------------------
run_cell "R6_pwin_lay050"    --predictor-p-win-lay-threshold 0.50
run_cell "R6_race_conf035"   --race-confidence-threshold 0.35
run_cell "R6_lay_price_max20" --lay-price-max 20

# ---------------------------------------------------------------
# Group 7 — Deploy-time safety probes (2 cells)
# ---------------------------------------------------------------
run_cell "R7_fc60" --reward-overrides force_close_before_off_seconds=60
run_cell "R7_tight_lock" --arb-spread-target-lock-pct 0.005

echo "[$(date -Iseconds)] round 5 fan-out complete"
