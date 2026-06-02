#!/usr/bin/env bash
# Round H1 — HONEST RE-BASELINE on held-out eval.
#
# The fc=0 "+£287 breakthrough" was eval-window overfitting: it
# collapsed to -£175 on held-out days. Naked P&L is zero-EV
# directional variance. The REAL structural edge is LOCKED P&L
# (scalping spread capture), which was only +£4-20/day.
#
# New methodology (locked in 2026-05-28):
#   - Train: 2026-04-06/08/09 (unchanged)
#   - Iteration eval: 7 held-out days (ODD: 05-07,09,11,13,15,17,19)
#   - Final test: 7 held-out days (EVEN: 05-08,10,...,20) — reserved,
#     looked at ONCE at the end. NOT used here.
#   - fc=120 ON (bounds naked variance — it's a safety rail, not a
#     cost to remove).
#   - SELECT ON LOCKED P&L + mat%, not total day_pnl.
#
# 6 cells × ~30 min (7 eval days) = ~3h wall.

set -u
REPO=/c/Users/jsmit/source/repos/rl-betfair
cd "$REPO"

WRAPPER_LOG="registry/_roundH1_wrapper.log"
exec >> "$WRAPPER_LOG" 2>&1
echo "[$(date -Iseconds)] round H1 wrapper started — honest held-out re-baseline"

TRAIN_DAYS=( 2026-04-06 2026-04-08 2026-04-09 )
# Iteration eval = 7 odd-dated held-out days. Even days reserved for final test.
EVAL_DAYS=( 2026-05-07 2026-05-09 2026-05-11 2026-05-13 2026-05-15 2026-05-17 2026-05-19 )
PREDICTORS=(
  "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json"
  "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json"
  "C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json"
)

# fc=120 ON — bounds naked variance. This is the honest baseline.
BASE=(
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
)

run_cell () {
    local cell="$1"; shift
    local ts; ts=$(date +%s)
    local outdir="registry/_roundH1_${cell}_${ts}"
    local logfile="${outdir}.log"
    echo "[$(date -Iseconds)] starting cell ${cell} -> ${outdir}"
    python -m training_v2.cohort.runner "${BASE[@]}" --output-dir "${outdir}" "$@" > "${logfile}" 2>&1
    local rc=$?
    echo "[$(date -Iseconds)] cell ${cell} finished rc=${rc}"
}

# HB0 — pure baseline (no gates, no BC). Floor reference.
run_cell "HB0_baseline" --bc-pretrain-steps 0

# HB1 — C2: env pwin_back gate only.
run_cell "HB1_pwinback" --bc-pretrain-steps 0 --predictor-p-win-back-threshold 0.20

# HB2 — E7: pwin_back + BC=500 (the in-sample 3/5 leader).
run_cell "HB2_e7" --bc-pretrain-steps 500 --predictor-p-win-back-threshold 0.20

# HB3 — E7 + matured_arb_bonus (reward natural maturation → lift locked).
run_cell "HB3_e7_matbonus5" --bc-pretrain-steps 500 --predictor-p-win-back-threshold 0.20 --reward-overrides matured_arb_bonus_weight=5.0

# HB4 — E7 + tight_lock (tighter passive → more fills → more locked).
run_cell "HB4_e7_tight_lock" --bc-pretrain-steps 500 --predictor-p-win-back-threshold 0.20 --arb-spread-target-lock-pct 0.005

# HB5 — full augmentation (pwin + BC + L2 + L3a + L4): restores cls%,
# should maximise matured+closed lifecycle = locked.
run_cell "HB5_full_aug" --bc-pretrain-steps 500 --predictor-p-win-back-threshold 0.20 --bc-include-negative-samples --bc-include-close-hold-samples

echo "[$(date -Iseconds)] round H1 fan-out complete"
