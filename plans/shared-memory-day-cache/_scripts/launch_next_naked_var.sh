#!/bin/bash
# Follow-on to the BC A/B (decided 2026-06-03). The A/B showed BC-ON beats
# BC-OFF on locked_per_std composite at EVERY generation (B final +0.132 vs A
# +0.096; top-10 mean 0.096 vs 0.075) — so the base is BC-ON. But BOTH arms
# carry NEGATIVE total P&L = the naked leg's ~zero-EV variance is the binding
# problem. This run adds `--enable-gene naked_variance_penalty_beta` (Phase-5,
# penalises naked variance via reward_overrides; the GA finds the per-agent
# strength) to attack the deployment-critical metric directly
# (feedback_naked_variance_primary_metric). Everything else = the winning
# BC-ON arm, unchanged, for a clean read of the new lever.
set -u
cd "C:/Users/jsmit/source/repos/rl-betfair"
PY="python"
CHAMP="C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json"
RANK="C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json"
DIRM="C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json"
HOLDOUT="2026-05-20 2026-05-21 2026-05-22 2026-05-25 2026-05-27 2026-05-28 2026-05-29"
TS=$(date +%s)
OUT="registry/smdc_nakedvar_${TS}"
echo "out=$OUT" > registry/smdc_nakedvar_dir.txt
echo "=== $(date '+%F %H:%M:%S') launching naked-variance follow-on $OUT (BC-ON + naked_variance_penalty_beta, N=16, 5 gens) ==="
PYTHONIOENCODING=utf-8 $PY -m training_v2.cohort.runner --n-agents 30 --generations 5 \
  --parallel-agents 16 --days 32 --n-eval-days 7 \
  --exclude-days $HOLDOUT --device cuda --seed 42 \
  --output-dir "$OUT" --strategy-mode arb \
  --predictor-bundle-manifests "$CHAMP" "$RANK" "$DIRM" \
  --use-race-outcome-predictor --use-direction-predictor \
  --mature-prob-open-threshold 0.30 --bc-pretrain-steps 500 \
  --enable-gene open_cost --enable-gene stop_loss_pnl_threshold \
  --enable-gene mature_prob_loss_weight --enable-gene arb_spread_target_lock_pct \
  --enable-gene naked_variance_penalty_beta \
  --reward-overrides per_pair_reward_at_resolution=true \
  --reward-overrides locked_pnl_reward_weight=9.0 \
  --composite-score-mode locked_per_std --argmax-eval \
  > "${OUT}.log" 2>&1
echo "=== $(date '+%F %H:%M:%S') naked-variance follow-on FINISHED (exit $?) ==="
