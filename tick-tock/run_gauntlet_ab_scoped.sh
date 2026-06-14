#!/usr/bin/env bash
# Phase 6 A/B — SCOPED (fast cutover-gate validation). plans/gauntlet-pipeline/.
# Runs arm A (gauntlet) -> arm B (lockstep) -> the held-out judge, sequentially.
# Both arms identical except --breeding. Scoped for ~half-day wall:
#   * pool capped to ~20 days (2 tranches of 10) via --exclude-days (oldest 23)
#   * --n-agents 8, gauntlet --generations 2 (1 breed round); lockstep auto = 2 tranches
#   * seed architecture=lstm + predictor_lean_obs=true => fast + identical both arms
#     (this A/B validates ORCHESTRATION quality, not the arch tournament)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
PRED=../betfair-predictors/production
LOG=C:/tmp

EXCL="2026-04-06 2026-04-07 2026-04-08 2026-04-09 2026-04-10 2026-04-11 2026-04-12 2026-04-13 2026-04-14 2026-04-15 2026-04-16 2026-04-17 2026-04-19 2026-04-20 2026-04-21 2026-04-22 2026-04-23 2026-04-24 2026-04-25 2026-04-26 2026-04-28 2026-04-29 2026-04-30"

common_args() {
  echo --n-agents 8 --parallel-agents 8 --device cpu \
    --holdout-recent 7 --validation-holdout-recent 5 \
    --pbt-train-per-rotation 6 --pbt-eval-per-rotation 4 \
    --seed 6006 --composite-score-mode locked_weighted \
    --survivor-fraction 0.5 --pbt-perturb-frac 0.20 \
    --use-race-outcome-predictor --use-direction-predictor \
    --predictor-bundle-manifests \
      "$PRED/race-outcome/manifest.json" \
      "$PRED/race-outcome-ranker/manifest.json" \
      "$PRED/direction-predictor/manifest.json" \
    --enable-all-genes \
    --seed-gene architecture=lstm --seed-gene predictor_lean_obs=true \
    --exclude-days $EXCL
}

echo "=== [$(date)] ARM A: gauntlet ===" | tee -a "$LOG/ab_scoped.log"
rm -rf registry/gauntlet_ab_gauntlet; mkdir -p registry/gauntlet_ab_gauntlet
"$PY" -m training_v2.cohort.runner --breeding gauntlet --generations 2 \
  $(common_args) --era-id gauntlet_ab_gauntlet \
  --output-dir registry/gauntlet_ab_gauntlet > "$LOG/ab_gauntlet.log" 2>&1
echo "ARM A exit=$?" | tee -a "$LOG/ab_scoped.log"

echo "=== [$(date)] ARM B: lockstep ===" | tee -a "$LOG/ab_scoped.log"
rm -rf registry/gauntlet_ab_lockstep; mkdir -p registry/gauntlet_ab_lockstep
"$PY" -m training_v2.cohort.runner --breeding lockstep \
  $(common_args) --era-id gauntlet_ab_lockstep \
  --output-dir registry/gauntlet_ab_lockstep > "$LOG/ab_lockstep.log" 2>&1
echo "ARM B exit=$?" | tee -a "$LOG/ab_scoped.log"

echo "=== [$(date)] JUDGE: cross-era held-out board (sealed-7) ===" | tee -a "$LOG/ab_scoped.log"
"$PY" -m tools.cross_era_holdout_board \
  --era-dir registry/gauntlet_ab_gauntlet \
  --era-dir registry/gauntlet_ab_lockstep \
  --holdout-recent 7 --top-n 8 --rank-by locked_over_sigma \
  --device cpu --argmax-eval \
  --reeval-arg=--use-race-outcome-predictor \
  --reeval-arg=--use-direction-predictor \
  --reeval-arg=--predictor-bundle-manifests \
  --reeval-arg="$PRED/race-outcome/manifest.json" \
  --reeval-arg="$PRED/race-outcome-ranker/manifest.json" \
  --reeval-arg="$PRED/direction-predictor/manifest.json" \
  --output registry/gauntlet_ab_board > "$LOG/ab_judge.log" 2>&1
echo "JUDGE exit=$?" | tee -a "$LOG/ab_scoped.log"
echo "=== [$(date)] A/B SCOPED COMPLETE ===" | tee -a "$LOG/ab_scoped.log"
