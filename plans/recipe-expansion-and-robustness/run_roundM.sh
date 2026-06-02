#!/usr/bin/env bash
# Round M — mat%-push on HELD-OUT eval.
#
# H1 held-out found two mat% levers: tight_lock (mat 4.3%) and
# full-aug (mat 4.2%, best day_pnl -£145 via fewer/more-selective
# opens). Stack them + push tight_lock harder. Goal: lift held-out
# mat% toward 10%+ while keeping locked-per-matured positive.
#
# Train 2026-04-06/08/09; EVAL on 7 held-out odd days. fc=120 ON.
# 8 cells × ~35 min = ~4.7h.

set -u
REPO=/c/Users/jsmit/source/repos/rl-betfair
cd "$REPO"
WRAPPER_LOG="registry/_roundM_wrapper.log"
exec >> "$WRAPPER_LOG" 2>&1
echo "[$(date -Iseconds)] round M wrapper started — mat%-push on held-out"

TRAIN_DAYS=( 2026-04-06 2026-04-08 2026-04-09 )
EVAL_DAYS=( 2026-05-07 2026-05-09 2026-05-11 2026-05-13 2026-05-15 2026-05-17 2026-05-19 )
PRED=(
  "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json"
  "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json"
  "C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json"
)
BASE=(
  --n-agents 4 --generations 1 --device cuda --seed 42 --strategy-mode arb
  --training-days-explicit "${TRAIN_DAYS[@]}"
  --cohort-eval-days "${EVAL_DAYS[@]}"
  --rotating-eval-sample 0
  --direction-head-manifest models/direction_head/sweep_c11
  --predictor-lean-obs --use-race-outcome-predictor --use-direction-predictor
  --predictor-bundle-manifests "${PRED[@]}"
  --reward-overrides force_close_before_off_seconds=120
  --reward-overrides close_feasibility_max_spread_pct=0.05
  --reward-overrides matured_arb_expected_random=0.0
  --bc-pretrain-steps 500
  --predictor-p-win-back-threshold 0.20
)
# full-aug = the two BC label augmentations
AUG=( --bc-include-negative-samples --bc-include-close-hold-samples )

run_cell () {
    local cell="$1"; shift
    local ts; ts=$(date +%s)
    local outdir="registry/_roundM_${cell}_${ts}"
    echo "[$(date -Iseconds)] starting cell ${cell} -> ${outdir}"
    python -m training_v2.cohort.runner "${BASE[@]}" --output-dir "${outdir}" "$@" > "${outdir}.log" 2>&1
    echo "[$(date -Iseconds)] cell ${cell} finished rc=$?"
}

# M1-M3: full-aug + progressively tighter target spread.
run_cell "M1_fullaug_tight0050" "${AUG[@]}" --arb-spread-target-lock-pct 0.005
run_cell "M2_fullaug_tight0030" "${AUG[@]}" --arb-spread-target-lock-pct 0.003
run_cell "M3_fullaug_tight0020" "${AUG[@]}" --arb-spread-target-lock-pct 0.002

# M4-M5: tight_lock alone (no aug) at the tighter values — isolate
# the spread lever.
run_cell "M4_tight0030" --arb-spread-target-lock-pct 0.003
run_cell "M5_tight0020" --arb-spread-target-lock-pct 0.002

# M6: full-aug + more selective opens (tighter pwin floor).
run_cell "M6_fullaug_pwin025" "${AUG[@]}" --predictor-p-win-back-threshold 0.25

# M7: stack everything — full-aug + tight0030 + pwin025.
run_cell "M7_fullaug_tight0030_pwin025" "${AUG[@]}" --arb-spread-target-lock-pct 0.003 --predictor-p-win-back-threshold 0.25

# M8: full-aug replicate at seed 43 (variance check on the H1 winner).
run_cell "M8_fullaug_seed43" "${AUG[@]}" --seed 43

echo "[$(date -Iseconds)] round M fan-out complete"
