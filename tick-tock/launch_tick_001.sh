#!/usr/bin/env bash
# Tick-Tock — full-width TICK on the new rotation structure (2026-06-07).
#
# The cold-exploration baseline the held-out compare needs: same structure as
# the tock (chronological folds, --holdout-recent 7, 4-tier, --maturation-gens
# 2, 12-day rotations 6/6, predictor bundle, BC-on) BUT **full-width** —
# --enable-all-genes, NO --seed-gene, NO --hypothesis-id. predictor_lean_obs is
# sampled (~50/50) as in a normal Tick; lean-obs agents skip BC (no lean
# oracle), exactly like the original pbt_genes_v2 Tick — that's correct for a
# cold explore era. Writes to its OWN dir (tt_tick_001); never touches the
# tock's dir. era_type=tick so the phenotype tool's --tick-only can use it.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
DIR=registry/tt_tick_001
PRED=../betfair-predictors/production
rm -rf "$DIR"; mkdir -p "$DIR"

"$PY" -m training_v2.cohort.runner \
  --breeding pbt --n-agents 16 --maturation-gens 2 \
  --holdout-recent 7 --pbt-rotation-mode chronological \
  --seed 3001 --parallel-agents 16 --device cpu \
  --composite-score-mode locked_per_std --force-close-rate-penalty-weight 20 \
  --big-model-threads 1 --gpu-policy-lane --gpu-lane-max-concurrent 2 \
  --use-race-outcome-predictor --use-direction-predictor --bc-pretrain-steps 500 \
  --predictor-bundle-manifests \
      "$PRED/race-outcome/manifest.json" \
      "$PRED/race-outcome-ranker/manifest.json" \
      "$PRED/direction-predictor/manifest.json" \
  --enable-all-genes \
  --pbt-train-per-rotation 6 --pbt-eval-per-rotation 6 \
  --pbt-tier-sizes 6,4,3 --pbt-promote-counts 3,2,2 --pbt-freeze-top 2 \
  --era-type tick --era-id tt_tick_001 \
  --output-dir "$DIR" > "$DIR/tick.console.log" 2>&1
