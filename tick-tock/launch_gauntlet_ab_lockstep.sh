#!/usr/bin/env bash
# Phase 6 A/B — arm B: the CURRENT lockstep path (plans/lockstep-cohort/).
# Matched to launch_gauntlet_ab_gauntlet.sh except for --breeding.
# See plans/gauntlet-pipeline/ab_runbook.md. Multi-hour; launch detached+logged.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
DIR=registry/gauntlet_ab_lockstep
PRED=../betfair-predictors/production
mkdir -p "$DIR"

"$PY" -m training_v2.cohort.runner \
  --breeding lockstep --n-agents 16 \
  --holdout-recent 7 --validation-holdout-recent 5 \
  --pbt-train-per-rotation 6 --pbt-eval-per-rotation 4 \
  --seed 6006 --parallel-agents 16 --device cpu \
  --gpu-policy-lane --gpu-lane-max-concurrent 2 \
  --composite-score-mode locked_weighted \
  --survivor-fraction 0.5 --pbt-perturb-frac 0.20 \
  --use-race-outcome-predictor --use-direction-predictor \
  --predictor-bundle-manifests \
      "$PRED/race-outcome/manifest.json" \
      "$PRED/race-outcome-ranker/manifest.json" \
      "$PRED/direction-predictor/manifest.json" \
  --enable-all-genes \
  --era-type lockstep --era-id gauntlet_ab_lockstep \
  --output-dir "$DIR"
