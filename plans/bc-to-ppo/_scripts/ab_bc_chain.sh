#!/bin/bash
# Two-arm BC A/B. 30 agents x 5 gens via --parallel-agents (fast multiprocess
# path, ~8x, predictors-ON wired), 7 eval days (stable locked_per_std σ),
# holdout May 20-29 sealed. Arm A = BC OFF, Arm B = BC ON (500 steps). SAME
# seed 42 -> clean paired A/B; the only difference is BC. Per-arm
# show_cohort_status --watch writes <dir>/status.txt every 60s.
# CRITICAL: NO --batched (it would override --parallel-agents and silently drop
# predictors/BC/input_norm — the bug that wrecked c1/c2).
set -u
cd "C:/Users/jsmit/source/repos/rl-betfair"
PY=".venv/Scripts/python.exe"
CHAMP="C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json"
RANK="C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json"
DIRM="C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json"
HOLDOUT="2026-05-20 2026-05-21 2026-05-22 2026-05-25 2026-05-27 2026-05-28 2026-05-29"
TS=$(date +%s)
echo "bcoff=registry/ab_bcoff_${TS}" > registry/ab_dirs.txt
echo "bcon=registry/ab_bcon_${TS}" >> registry/ab_dirs.txt
cat registry/ab_dirs.txt

run_arm () {  # $1=outdir  $2=bcsteps (0 = OFF)
  local OUT="$1" BC="$2" BCARG=""
  if [ "$BC" -gt 0 ]; then BCARG="--bc-pretrain-steps $BC"; fi
  echo "=== $(date '+%H:%M') launching arm $OUT (bc_steps=$BC) ==="
  $PY -m training_v2.cohort.runner --n-agents 30 --generations 5 \
    --parallel-agents 16 --days 32 --n-eval-days 7 \
    --exclude-days $HOLDOUT --device cuda --seed 42 \
    --output-dir "$OUT" --strategy-mode arb \
    --predictor-bundle-manifests "$CHAMP" "$RANK" "$DIRM" \
    --use-race-outcome-predictor --use-direction-predictor \
    --mature-prob-open-threshold 0.30 \
    --enable-gene open_cost --enable-gene stop_loss_pnl_threshold \
    --enable-gene mature_prob_loss_weight --enable-gene arb_spread_target_lock_pct \
    --reward-overrides per_pair_reward_at_resolution=true \
    --reward-overrides locked_pnl_reward_weight=9.0 \
    --composite-score-mode locked_per_std --argmax-eval \
    $BCARG > "${OUT}.log" 2>&1 &
  local TPID=$!
  # wait for the cohort dir to materialise, then start the status.txt watcher
  while [ ! -d "$OUT" ] && kill -0 "$TPID" 2>/dev/null; do sleep 5; done
  if [ -d "$OUT" ]; then
    $PY -m tools.show_cohort_status "$OUT" --watch 60 > "${OUT}_watch.log" 2>&1 &
    local WPID=$!
  fi
  wait "$TPID"
  [ -n "${WPID:-}" ] && kill "$WPID" 2>/dev/null
  echo "=== $(date '+%H:%M') arm $OUT done ==="
}

run_arm "registry/ab_bcoff_${TS}" 0
run_arm "registry/ab_bcon_${TS}" 500
echo "=== $(date '+%H:%M') A/B CHAIN COMPLETE ==="
