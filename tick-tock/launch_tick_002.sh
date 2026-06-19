#!/usr/bin/env bash
# Tick-Tock — full-width TICK on the GAUNTLET pipeline (2026-06-18).
#
# Same intent as launch_tick_001.sh (cold full-width explore: --enable-all-genes,
# no --seed-gene, predictor bundle, BC-on, gpu-policy-lane) but on the new
# --breeding gauntlet pipeline. Writes to its OWN dir (tt_tick_002).
#
# CULL-EARLY (--gauntlet-cull per_tranche, operator 2026-06-19 — "the tick"):
# a pool of 32 climbs, the bottom half is culled AFTER EACH tranche, survivors
# are mutated and the mutants re-climb T1..TK to rejoin. Duds die after T1;
# compute concentrates on winners. (--n-agents 32 → 16-survivor pool; the
# 32 run 16-at-a-time over 2 waves.) Replaces the rejected full-fair-shot
# climb (which spent compute climbing duds through every tranche).
#
# composite-score-mode = locked_per_std (tnv2 = mean_locked/(1+sigma_naked),
# NEVER reads naked-sign). The earlier locked_weighted run (locked+0.25*naked)
# bred toward naked LUCK and culled the genuine scalpers at the frontier
# (e.g. b8fd6650: 10 matured/£10 locked ranked LAST). tnv2 selects on the
# structural locked floor discounted by naked volatility; needs >=2 eval days
# (we run 10 validation days, so it does NOT hit the locked_weighted fallback).
#
# MUST be launched DETACHED via tools/launch_detached.py — a long run hosted as
# a child of an ephemeral shell (Claude Code background bash) gets TerminateProcess'd
# when that shell's Windows job object closes (the 2026-06-18 silent-collapse bug;
# heartbeat stacks showed 16 healthy workers killed simultaneously mid-training).
# CREATE_BREAKAWAY_FROM_JOB lets this tree outlive the launching shell.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
DIR=registry/tt_tick_002
PRED=../betfair-predictors/production
rm -rf "$DIR"; mkdir -p "$DIR"

"$PY" -m training_v2.cohort.runner \
  --breeding gauntlet --gauntlet-cull per_tranche \
  --generations 5 --n-agents 32 --parallel-agents 16 \
  --device cpu --seed 3002 \
  --survivor-fraction 0.5 --pbt-train-per-rotation 6 --pbt-eval-per-rotation 4 \
  --holdout-recent 7 --validation-holdout-recent 10 --validation-holdout-mode sampled \
  --composite-score-mode locked_per_std --force-close-rate-penalty-weight 20 \
  --big-model-threads 1 --gpu-policy-lane --gpu-lane-max-concurrent 2 \
  --use-race-outcome-predictor --use-direction-predictor --bc-pretrain-steps 500 \
  --predictor-bundle-manifests \
      "$PRED/race-outcome/manifest.json" \
      "$PRED/race-outcome-ranker/manifest.json" \
      "$PRED/direction-predictor/manifest.json" \
  --enable-all-genes \
  --era-type tick --era-id tt_tick_002 \
  --output-dir "$DIR" > "$DIR/tick.console.log" 2>&1
