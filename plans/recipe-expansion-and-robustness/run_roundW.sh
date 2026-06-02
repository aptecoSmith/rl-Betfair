#!/usr/bin/env bash
# Round W — CLOSE-WALK BROAD RERUN (2026-05-30, findings.md KEY FINDING #2).
#
# The held-out leaderboard was measured under the BROKEN single-level
# close matcher, whose under-hedging cost ~£339/agent/7d of avoidable
# directional loss. This round re-measures every fc=120 leaderboard
# recipe with the new bounded close-walk ON vs OFF, to see whether the
# fix reshuffles the board / flips marginal recipes positive.
#
# Within-round A/B: each recipe at close_walk_ticks 0 (control — should
# reproduce its old held-out number) and 10 (walk on). Walk depth is
# effectively binary given the 3-level book-depth cap (see monitoring
# notes 2026-05-30), so {0,10} captures the effect. fc=120 except O1.
# Base = Round N full-aug base (bc500 + neg + close-hold + predictor
# stack + direction head). Seed 42 (the leaderboard seed).
export PATH="/c/Python314:/c/Python314/Scripts:$PATH"
set -u
cd /c/Users/jsmit/source/repos/rl-betfair
exec >> registry/_roundW_wrapper.log 2>&1
echo "[$(date -Iseconds)] roundW started — close-walk broad rerun"
TRAIN=( 2026-04-06 2026-04-08 2026-04-09 )
EVAL=( 2026-05-07 2026-05-09 2026-05-11 2026-05-13 2026-05-15 2026-05-17 2026-05-19 )
PRED=( "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json" "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json" "C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json" )
BASE=( --n-agents 4 --generations 1 --device cuda --strategy-mode arb --training-days-explicit "${TRAIN[@]}" --cohort-eval-days "${EVAL[@]}" --rotating-eval-sample 0 --direction-head-manifest models/direction_head/sweep_c11 --predictor-lean-obs --use-race-outcome-predictor --use-direction-predictor --predictor-bundle-manifests "${PRED[@]}" --reward-overrides force_close_before_off_seconds=120 --reward-overrides close_feasibility_max_spread_pct=0.05 --reward-overrides matured_arb_expected_random=0.0 --bc-pretrain-steps 500 --bc-include-negative-samples --bc-include-close-hold-samples )
run_cell () { local cell="$1"; shift; local ts; ts=$(date +%s); local o="registry/_roundW_${cell}_${ts}"; echo "[$(date -Iseconds)] starting ${cell}"; python -m training_v2.cohort.runner "${BASE[@]}" --seed "${SD:-42}" --output-dir "$o" "$@" > "$o.log" 2>&1; echo "[$(date -Iseconds)] ${cell} rc=$?"; }

# ── N4: full-aug + pwin band 0.25–0.50 (leader, old -£78) ────────────
run_cell N4_walk0  --predictor-p-win-back-threshold 0.25 --predictor-p-win-back-max-threshold 0.50 --reward-overrides close_walk_ticks=0
run_cell N4_walk10 --predictor-p-win-back-threshold 0.25 --predictor-p-win-back-max-threshold 0.50 --reward-overrides close_walk_ticks=10
# ── N2: full-aug + pwin 0.35 (old -£98) ──────────────────────────────
run_cell N2_walk0  --predictor-p-win-back-threshold 0.35 --reward-overrides close_walk_ticks=0
run_cell N2_walk10 --predictor-p-win-back-threshold 0.35 --reward-overrides close_walk_ticks=10
# ── M6: full-aug + pwin 0.25 (old -£98) ──────────────────────────────
run_cell M6_walk0  --predictor-p-win-back-threshold 0.25 --reward-overrides close_walk_ticks=0
run_cell M6_walk10 --predictor-p-win-back-threshold 0.25 --reward-overrides close_walk_ticks=10
# ── M7: full-aug + tight0030 + pwin 0.25 (old -£128) ─────────────────
run_cell M7_walk0  --arb-spread-target-lock-pct 0.003 --predictor-p-win-back-threshold 0.25 --reward-overrides close_walk_ticks=0
run_cell M7_walk10 --arb-spread-target-lock-pct 0.003 --predictor-p-win-back-threshold 0.25 --reward-overrides close_walk_ticks=10
# ── O1: full-aug + fc=60 + pwin 0.25 (old -£114). fc override last-wins. ──
run_cell O1_walk0  --predictor-p-win-back-threshold 0.25 --reward-overrides force_close_before_off_seconds=60 --reward-overrides close_walk_ticks=0
run_cell O1_walk10 --predictor-p-win-back-threshold 0.25 --reward-overrides force_close_before_off_seconds=60 --reward-overrides close_walk_ticks=10
echo "[$(date -Iseconds)] roundW fan-out complete"
