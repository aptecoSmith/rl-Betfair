#!/usr/bin/env bash
# Round 3 — BC + Exit Recovery (plans/bc-exit-recovery/purpose.md).
#
# REVISED 2026-05-25 16:35 after E1 (BC=200) showed BC has a sharp
# saturation transition between 0 and 200 steps — cls% collapsed to
# 2.2 %, fc% rose to 88.9 %, opens to 239. E2 (BC=500) and E3
# (BC=2000) would be redundant. Replaced with no-BC partner cells
# building on C2 (pwin_back, the current winner at -£102/day) to
# test whether the shaped partners can lift mat% WITHOUT the BC
# headwind. E4 dropped too (weakest of the BC partners).
#
# Remaining sequence (7 cells, ~3.2h):
#   N1   pwin_back + matbonus5            (no BC)
#   N2   pwin_back + open_cost05          (no BC)
#   N3   pwin_back + matbonus + opencost  (no BC, triple-shaped)
#   E5   bc500 + matbonus5
#   E6   bc500 + open_cost05
#   E7   pwin_back + bc500
#   E8   pwin_back + bc500 + matbonus5
#
# E1 (already done) provides the BC saturation reference point.

set -u

REPO=/c/Users/jsmit/source/repos/rl-betfair
cd "$REPO"

WRAPPER_LOG="registry/_round3_wrapper.log"
exec >> "$WRAPPER_LOG" 2>&1

echo "[$(date -Iseconds)] round 3 wrapper RESTARTED with revised cell list"

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
)

run_cell () {
    local cell="$1"; shift
    local ts; ts=$(date +%s)
    local outdir="registry/_round3_${cell}_${ts}"
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

# ---------------------------------------------------------------
# Group B' — No-BC + shaped partner stack (3 cells)
# Builds on C2 (pwin_back -£102/d, opens 112, mat 1.6%). Tests
# whether matured_arb_bonus and/or open_cost can lift mat% from
# C2's 1.6 % toward the 5 % target without BC's exit catastrophe.
# ---------------------------------------------------------------

# N1: pwin_back + matured_arb_bonus. Direct positive shaped gradient
# on natural maturation, on top of the confirmed selection prior.
run_cell "N1_pwinback_matbonus5" \
    --predictor-p-win-back-threshold 0.20 \
    --reward-overrides matured_arb_bonus_weight=5.0

# N2: pwin_back + open_cost. Penalises any open that doesn't
# resolve favourably (matured or agent-closed). Force-closes and
# nakeds do NOT refund. Directly punishes the over-opening
# pathology BC creates and is symmetric with the matbonus.
run_cell "N2_pwinback_opencost05" \
    --predictor-p-win-back-threshold 0.20 \
    --reward-overrides open_cost=0.5

# N3: pwin_back + matbonus + open_cost. Triple shaped stack —
# tests whether layering both shaped pressures composes vs
# interferes.
run_cell "N3_pwinback_matbonus5_opencost05" \
    --predictor-p-win-back-threshold 0.20 \
    --reward-overrides matured_arb_bonus_weight=5.0 \
    --reward-overrides open_cost=0.5

# ---------------------------------------------------------------
# Group B — BC=500 + shaped partner (2 cells, originally E5/E6)
# Tests whether the partners can rescue BC's broken exit. BC=500
# alone is assumed to look like E1 (BC=200 → mat 5.5%, fc 89%);
# the partner must drag fc% down and opens back into 100-180 band.
# ---------------------------------------------------------------

run_cell "E5_bc500_matbonus5" \
    --bc-pretrain-steps 500 \
    --reward-overrides matured_arb_bonus_weight=5.0

run_cell "E6_bc500_opencost05" \
    --bc-pretrain-steps 500 \
    --reward-overrides open_cost=0.5

# ---------------------------------------------------------------
# Group C — pwin_back + BC combinations (2 cells, originally E7/E8)
# The deploy-candidate hopefuls.
# ---------------------------------------------------------------

run_cell "E7_pwinback_bc500" \
    --bc-pretrain-steps 500 \
    --predictor-p-win-back-threshold 0.20

run_cell "E8_pwinback_bc500_matbonus5" \
    --bc-pretrain-steps 500 \
    --predictor-p-win-back-threshold 0.20 \
    --reward-overrides matured_arb_bonus_weight=5.0

echo "[$(date -Iseconds)] round 3 fan-out complete"
