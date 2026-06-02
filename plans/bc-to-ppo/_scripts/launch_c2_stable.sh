#!/bin/bash
# Cohort 2 — the stable-fitness redesign (supersedes c1).
# Key fixes vs c1: 7 eval days (was 2) for a usable selection σ; 12 train days
# (was 25) to buy back the eval cost since train is the expensive lever
# (~867s/agent-day vs ~70s for eval); 20 agents × 4 gens for real breeding on a
# stable fitness. Genes: open_cost, stop_loss, mature_prob_loss_weight,
# arb_spread_target_lock_pct. Pinned: per_pair credit + locked_weight=9.
# Train fc=0 (keep naked signal); winners get an fc=120 holdout re-eval after.
# Holdout May 20-29 EXCLUDED. Select on locked_per_std (now with a stable σ).
set -u
cd "C:/Users/jsmit/source/repos/rl-betfair"
PY=".venv/Scripts/python.exe"
CHAMP="C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json"
RANK="C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json"
DIRM="C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json"
TS=$(date +%s); OUT="registry/scalping_ga_c2_${TS}"
echo "$OUT" > registry/c2_latest_outdir.txt
echo "LAUNCHING $OUT"
$PY -m training_v2.cohort.runner --n-agents 20 --generations 4 --batched \
  --days 19 --n-eval-days 7 \
  --exclude-days 2026-05-20 2026-05-21 2026-05-22 2026-05-25 2026-05-27 2026-05-28 2026-05-29 \
  --device cuda --seed 45 \
  --output-dir "$OUT" --strategy-mode arb \
  --predictor-bundle-manifests "$CHAMP" "$RANK" "$DIRM" \
  --use-race-outcome-predictor --use-direction-predictor \
  --mature-prob-open-threshold 0.30 \
  --enable-gene open_cost --enable-gene stop_loss_pnl_threshold \
  --enable-gene mature_prob_loss_weight --enable-gene arb_spread_target_lock_pct \
  --reward-overrides per_pair_reward_at_resolution=true \
  --reward-overrides locked_pnl_reward_weight=9.0 \
  --bc-pretrain-steps 500 --composite-score-mode locked_per_std --argmax-eval \
  > "${OUT}.log" 2>&1
echo "COHORT 2 EXIT=$? (out=$OUT)"
