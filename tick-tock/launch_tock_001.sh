#!/usr/bin/env bash
# Tick-Tock — MANUAL first tock (hypothesis_001: direction machinery + BC-on
# maturation), on the NEW rotation structure (operator-revised 2026-06-07).
#
# Mirrors the first Tick's predictor/BC/gene flags PLUS the rotation-rework:
#   * --pbt-rotation-mode chronological — old-anchored date-folds; R1..R(n-1)
#     stay FIXED as data accumulates; the top tier trains on the freshest data.
#   * --holdout-recent 7 — slide the held-out judge to the NEWEST 7 racing days
#     (currently 2026-05-29..06-05); all older cached days become the training
#     pool (currently 48 days, 2026-04-06..05-28). Replaces the old fixed
#     sealed list + --days/--exclude-days.
#   * 4-tier ladder (R4 live): --pbt-tier-sizes 6,4,3 (R2,R3,R4) +
#     --pbt-promote-counts 3,2,2 (R1->R2,R2->R3,R3->R4) + --pbt-freeze-top 2
#     => --pbt-rotations 4 derived; R1 absorbs slack (3 fresh).
#   * 12-day rotations split 6 train / 6 eval (was 6/4) — more held-out per
#     tier for a lower-variance, less-overfit-prone selection signal under
#     fixed folds. 48 training days = 4 x 12 exactly.
#   * --maturation-gens 2 (self-healing budget): generations = n_tiers + 2, so
#     the top tier always matures K+1=3 gens no matter how deep the ladder
#     grows (now 4 tiers => 6 gens). No fixed --generations to remember to bump.
#   * the 8 --seed-gene seeds (seeds/seed_args_001.txt), incl.
#     predictor_lean_obs=false so BC (full-obs oracle) runs era-wide, and
#     bc_pretrain_steps=500 carried by the SEED (no --bc-pretrain-steps flag).
#   * era tags --era-type tock --hypothesis-id hypothesis_001 --era-id tt_tock_001.
#
# This tock is intentionally an "oranges" one-off: it is the first era on the
# new structure, so it is a valid comparison ANCHOR for future eras (the old
# Tick used random folds on the 05-19 pool and is not comparable).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
DIR=registry/tt_tock_001
PRED=../betfair-predictors/production
mkdir -p "$DIR"

"$PY" -m training_v2.cohort.runner \
  --breeding pbt --n-agents 16 --maturation-gens 2 \
  --holdout-recent 7 --pbt-rotation-mode chronological \
  --seed 2001 --parallel-agents 16 --device cpu \
  --composite-score-mode locked_per_std --force-close-rate-penalty-weight 20 \
  --big-model-threads 1 --gpu-policy-lane --gpu-lane-max-concurrent 2 \
  --use-race-outcome-predictor --use-direction-predictor \
  --predictor-bundle-manifests \
      "$PRED/race-outcome/manifest.json" \
      "$PRED/race-outcome-ranker/manifest.json" \
      "$PRED/direction-predictor/manifest.json" \
  --enable-all-genes \
  --pbt-train-per-rotation 6 --pbt-eval-per-rotation 6 \
  --pbt-tier-sizes 6,4,3 --pbt-promote-counts 3,2,2 --pbt-freeze-top 2 \
  --era-type tock --hypothesis-id hypothesis_001 --era-id tt_tock_001 \
  --seed-gene use_direction_predictor=true \
  --seed-gene direction_gate_enabled=true \
  --seed-gene predictor_lean_obs=false \
  --seed-gene direction_gate_threshold=0.25:0.40 \
  --seed-gene direction_gate_warmup_eps=8:16 \
  --seed-gene stop_loss_pnl_threshold=0.18:0.26 \
  --seed-gene bc_pretrain_steps=500 \
  --seed-gene bc_learning_rate=5e-4:1e-3 \
  --output-dir "$DIR"
