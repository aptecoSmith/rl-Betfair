#!/bin/bash
# Autonomous ~18h BC A/B — runs on the NEW shared-memory-day-cache setup.
#
# Same config as plans/bc-to-ppo/_scripts/ab_bc_chain.sh (the behavioural-
# cloning session) EXCEPT:
#   * --parallel-agents 16 (band-aid 4 retired; the static_obs shared-memory
#     path makes N=16 fit at ~30 GB — measured 2026-06-02). At N=16 the
#     predictors-ON multiprocess path runs 9.1x (calibrated), so 30 agents x
#     5 gens x 2 arms ~= 16h, fitting the 18h window.
#   * predictors-ON automatically uses the shared static_obs memmap (workers
#     skip per-tick inference) — no flag needed, it's the runner default for
#     --use-race-outcome-predictor multiprocess.
#   * system python (C:\Python314) — the env every shared-memory-day-cache
#     gate (golden parity, e2e cohort, the optimal-N calibration) was
#     validated on.
#
# Arm A = BC OFF, Arm B = BC ON (500 steps); SAME seed 42 -> clean paired A/B.
# Holdout May 20-29 sealed (final-test set, NOT trained/selected on).
# NO --batched (it silently drops predictors/BC/input_norm).
set -u
cd "C:/Users/jsmit/source/repos/rl-betfair"
PY="python"
CHAMP="C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json"
RANK="C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json"
DIRM="C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json"
HOLDOUT="2026-05-20 2026-05-21 2026-05-22 2026-05-25 2026-05-27 2026-05-28 2026-05-29"
GENS=5
TS=$(date +%s)
OUTOFF="registry/smdc_bcoff_${TS}"
OUTON="registry/smdc_bcon_${TS}"
printf 'bcoff=%s\nbcon=%s\n' "$OUTOFF" "$OUTON" > registry/smdc_18h_dirs.txt
echo "=== $(date '+%F %H:%M:%S') 18h BC A/B start; dirs:" && cat registry/smdc_18h_dirs.txt

run_arm () {  # $1=outdir  $2=bcsteps (0 = OFF)
  local OUT="$1" BC="$2" BCARG=""
  if [ "$BC" -gt 0 ]; then BCARG="--bc-pretrain-steps $BC"; fi
  echo "=== $(date '+%F %H:%M:%S') launching arm $OUT (bc_steps=$BC, gens=$GENS, N=16) ==="
  PYTHONIOENCODING=utf-8 $PY -m training_v2.cohort.runner --n-agents 30 --generations $GENS \
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
    $BCARG > "${OUT}.log" 2>&1
  echo "=== $(date '+%F %H:%M:%S') arm $OUT FINISHED (exit $?) ==="
}

run_arm "$OUTOFF" 0      # Arm A: BC OFF
run_arm "$OUTON" 500     # Arm B: BC ON
echo "=== $(date '+%F %H:%M:%S') 18H BC A/B COMPLETE ==="
