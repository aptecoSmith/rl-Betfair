#!/usr/bin/env bash
# Held-out reeval batch — load EXISTING trained weights and eval on
# 7 unseen days under a uniform safe-deployment config (fc=120,
# predictors on, pwin_back=0.20). No retraining. ~45s/agent/day.
#
# Curated set of distinct promising recipes from the campaign,
# chosen by in-sample LOCKED pnl + net structure. Tests which
# TRAINED POLICY generalises out-of-sample. Select on held-out
# LOCKED pnl (structural) not total day_pnl (naked variance).

set -u
REPO=/c/Users/jsmit/source/repos/rl-betfair
cd "$REPO"
WRAPPER_LOG="registry/_holdout_reeval_batch.log"
exec >> "$WRAPPER_LOG" 2>&1
echo "[$(date -Iseconds)] held-out reeval batch started"

HOLDOUT=( 2026-05-07 2026-05-09 2026-05-11 2026-05-13 2026-05-15 2026-05-17 2026-05-19 )
PRED=(
  "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json"
  "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json"
  "C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json"
)

# Curated cohorts: distinct recipes spanning the campaign.
COHORTS=(
  "_round3_E7_pwinback_bc500_1779731177"
  "_round6_G4_e7_bc1000_1779828950"
  "_round7_H2_e7_tight_lock_race_conf_1779869199"
  "_round7_H2_e7_lay_max_race_conf_1779867840"
  "_phase_b_F3_full_aug_1779743267"
  "_env_sweep_C0_baseline_1779693573"
  "_env_sweep_C2_pwin_back_020_1779696342"
  "_round6_G2_e7_tight_lock_1779818533"
  "_round3_E2_bc500_1779722700"
  "_round6_5_K1_e7_fc_off_1779832288"
)

reeval () {
  local cohort="$1"
  local dir="registry/${cohort}"
  if [[ ! -d "$dir/weights" ]]; then
    echo "[$(date -Iseconds)] SKIP ${cohort} — no weights dir"
    return
  fi
  echo "[$(date -Iseconds)] reeval ${cohort}"
  python -m tools.reevaluate_cohort \
    --cohort-dir "$dir" \
    --eval-days "${HOLDOUT[@]}" \
    --device cuda \
    --output reeval_holdout7.jsonl \
    --predictor-bundle-manifests "${PRED[@]}" \
    --use-race-outcome-predictor \
    --use-direction-predictor \
    --predictor-lean-obs \
    --direction-head-manifest models/direction_head/sweep_c11 \
    --strategy-mode arb \
    --predictor-p-win-back-threshold 0.20 \
    --reward-overrides force_close_before_off_seconds=120 \
    --reward-overrides close_feasibility_max_spread_pct=0.05 \
    --reward-overrides matured_arb_expected_random=0.0 \
    >> "registry/_reeval_${cohort}.log" 2>&1
  echo "[$(date -Iseconds)] reeval ${cohort} rc=$?"
}

for c in "${COHORTS[@]}"; do
  reeval "$c"
done

echo "[$(date -Iseconds)] held-out reeval batch complete"
