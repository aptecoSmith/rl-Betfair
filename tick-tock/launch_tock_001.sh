#!/usr/bin/env bash
# Tick-Tock — MANUAL first tock (hypothesis_001: direction machinery + BC-on
# maturation). One-shot launcher for the Phase-1 de-risk cycle (operator
# signed off 2026-06-06). Mirrors the first Tick's exact flags
# (plans/pbt-gpu-forward/_scripts/run_genes_campaign.ps1) PLUS:
#   * the 7 --seed-gene band/point seeds (seeds/seed_args_001.txt),
#   * era tags --era-type tock --hypothesis-id hypothesis_001 --era-id tt_tock_001,
#   * a NEW output dir (clean tagging + a fresh hall-of-fame; a tock is its own
#     5-gen era, not a continuation of the Tick),
#   * --bc-pretrain-steps is DROPPED on purpose: the seed bc_pretrain_steps=500
#     carries it, so the tock's rows PROVE the structural seed landed era-wide.
#   * predictor_lean_obs=false is seeded (structural, era-wide) so EVERY agent
#     uses the full-obs layout. BC clones a layout-specific oracle and only the
#     FULL-obs oracle is scanned (obs_dim 2254); a lean-obs agent (574) finds no
#     matching oracle and SKIPS BC. Pinning full obs makes "BC on era-wide"
#     actually true. (Lean-vs-full WITH BC on both is deferred until the
#     per-layout oracle cache lands — see the spawned task.)
#   * --exclude-days also drops 2026-05-30..06-05 (the 6 non-sealed parquets
#     added AFTER the first Tick ran). select_days takes the LAST n_days, so
#     those newer days shifted the window into dates with NO direction-label
#     cache (cache stops at 05-19) — the first launch CRASHED at gen-1 on a
#     missing label cache. Excluding them pins the tock to the Tick's EXACT
#     pool (ending 05-19): all label+oracle caches present AND a fair same-pool
#     warm-start comparison. (Using the newer pool would need scanning labels +
#     oracle for those 6 days AND re-running the Tick on it — a bigger job.)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
DIR=registry/tt_tock_001
PRED=../betfair-predictors/production
mkdir -p "$DIR"

"$PY" -m training_v2.cohort.runner \
  --breeding pbt --n-agents 16 --generations 5 --days 30 \
  --exclude-days 2026-05-20 2026-05-21 2026-05-22 2026-05-23 2026-05-24 \
                 2026-05-25 2026-05-26 2026-05-27 2026-05-28 2026-05-29 \
                 2026-05-30 2026-05-31 2026-06-02 2026-06-03 2026-06-04 2026-06-05 \
  --seed 2001 --parallel-agents 16 --device cpu \
  --composite-score-mode locked_weighted --force-close-rate-penalty-weight 20 \
  --big-model-threads 1 --gpu-policy-lane --gpu-lane-max-concurrent 2 \
  --use-race-outcome-predictor --use-direction-predictor \
  --predictor-bundle-manifests \
      "$PRED/race-outcome/manifest.json" \
      "$PRED/race-outcome-ranker/manifest.json" \
      "$PRED/direction-predictor/manifest.json" \
  --enable-all-genes \
  --pbt-rotations 3 --pbt-train-per-rotation 6 --pbt-eval-per-rotation 4 \
  --pbt-r2-size 6 --pbt-r3-size 4 --pbt-promote-from-r1 3 \
  --pbt-promote-from-r2 2 --pbt-freeze-top-r3 2 \
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
