#!/usr/bin/env bash
# Round T (resurrected 2026-05-30) — Path C: mature_prob OPEN-gate.
#
# THE EXISTENTIAL TEST of open-selection: can a LEARNED maturation
# signal pick opens better than the ~5% base rate? mat% has sat at
# 4-7% across every recipe (≈ base rate = no selection edge). This
# gates OPEN_BACK/OPEN_LAY per runner where the policy's own
# mature_prob_head sigmoid < threshold (NOOP/CLOSE never gated;
# effective threshold anneals 0→gene over warmup to avoid cold-start
# collapse). Naive DIRECTION-gating already failed (findings.md "what
# doesn't work" #3); mature_prob is trained on the actual maturation
# outcome, so it's a more complete signal than raw direction.
#
# CRITICAL: every prior round ran mature_prob_loss_weight=0 → the head
# was UNTRAINED (~const 0.5), so gating on it would be degenerate. ALL
# cells here pin mature_prob_loss_weight=2.0 so the head learns the
# strict-maturation label inline.
#
# Base = Round N full-aug + pwin 0.25 (= the M6 recipe, broad enough
# that the gate has room to select; old M6 held-out -£98, mat 5.2%,
# fc 69.5%). close_walk OFF (mat%/fc% — the metrics that matter here —
# are pair-lifecycle, independent of close matching; keeps comparison
# clean vs old M6). fc=120. Seed 42.
#
# Cells isolate two effects:
#   T1_nogate (thr 0)      : head TRAINED, gate OFF. vs old M6 (weight 0)
#                            = the actor-input effect of a trained head.
#   T2/T3/T4 (thr .3/.4/.5): head trained + gate ON. vs T1 = the GATE
#                            effect (does masking low-mature opens lift
#                            mat% / cut fc%?).
#   T5 (thr .3, seed 43)   : variance replicate of the mid threshold.
#
# VERIFY on first cell's day-1 log: train_mean_mature_prob_bce > 0
# (head training) AND eval_direction_gate_refusals > 0 in gated cells
# (gate firing — it reuses that refusal counter).
export PATH="/c/Python314:/c/Python314/Scripts:$PATH"
set -u
cd /c/Users/jsmit/source/repos/rl-betfair
exec >> registry/_roundT_wrapper.log 2>&1
echo "[$(date -Iseconds)] roundT started — mature_prob open-gate (resurrected)"
TRAIN=( 2026-04-06 2026-04-08 2026-04-09 )
EVAL=( 2026-05-07 2026-05-09 2026-05-11 2026-05-13 2026-05-15 2026-05-17 2026-05-19 )
PRED=( "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json" "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json" "C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json" )
BASE=( --n-agents 4 --generations 1 --device cuda --strategy-mode arb --training-days-explicit "${TRAIN[@]}" --cohort-eval-days "${EVAL[@]}" --rotating-eval-sample 0 --direction-head-manifest models/direction_head/sweep_c11 --predictor-lean-obs --use-race-outcome-predictor --use-direction-predictor --predictor-bundle-manifests "${PRED[@]}" --reward-overrides force_close_before_off_seconds=120 --reward-overrides close_feasibility_max_spread_pct=0.05 --reward-overrides matured_arb_expected_random=0.0 --reward-overrides mature_prob_loss_weight=2.0 --bc-pretrain-steps 500 --bc-include-negative-samples --bc-include-close-hold-samples --predictor-p-win-back-threshold 0.25 )
run_cell () { local cell="$1"; shift; local ts; ts=$(date +%s); local o="registry/_roundT_${cell}_${ts}"; echo "[$(date -Iseconds)] starting ${cell}"; python -m training_v2.cohort.runner "${BASE[@]}" --seed "${SD:-42}" --output-dir "$o" "$@" > "$o.log" 2>&1; echo "[$(date -Iseconds)] ${cell} rc=$?"; }

run_cell T1_nogate     --mature-prob-open-threshold 0.0
run_cell T2_mgate030   --mature-prob-open-threshold 0.30
run_cell T3_mgate040   --mature-prob-open-threshold 0.40
run_cell T4_mgate050   --mature-prob-open-threshold 0.50
SD=43 run_cell T5_mgate030_s43 --mature-prob-open-threshold 0.30
echo "[$(date -Iseconds)] roundT fan-out complete"
