#!/usr/bin/env bash
# Tick-Tock — Tock 002 (hypothesis_002: LSTM dominance + optimization levers)
#
# Following the success of Tock 001 (+9.79 locked_pnl), this era consolidates
# on the winning architecture (LSTM) and pushes the clip_range/batch_size knobs.
#
#   * --breeding pbt --n-agents 16 --maturation-gens 2
#   * --holdout-recent 7 --pbt-rotation-mode chronological
#   * --seed-gene architecture=lstm (structural pin)
#   * --seed-gene clip_range=0.25:0.35 (strongest locked driver +0.35)
#   * --seed-gene mini_batch_size=64:128 (locked driver +0.29)
#   * Inherited from H001: direction gate on, predictor on, BC on (500 steps, high LR).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
DIR=registry/tt_tock_002
PRED=../betfair-predictors/production
mkdir -p "$DIR"

"$PY" -m training_v2.cohort.runner \
  --breeding pbt --n-agents 16 --maturation-gens 2 \
  --holdout-recent 7 --pbt-rotation-mode chronological \
  --seed 2002 --parallel-agents 16 --device cpu \
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
  --era-type tock --hypothesis-id hypothesis_002 --era-id tt_tock_002 \
  --seed-gene architecture=lstm \
  --seed-gene use_direction_predictor=true \
  --seed-gene direction_gate_enabled=true \
  --seed-gene predictor_lean_obs=false \
  --seed-gene clip_range=0.25:0.35 \
  --seed-gene mini_batch_size=64:128 \
  --seed-gene bc_pretrain_steps=500 \
  --seed-gene bc_learning_rate=5e-4:1e-3 \
  --output-dir "$DIR"
