#!/usr/bin/env bash
# Phase 8 S03 (3-arm probe) + split overnight (12+12 BC vs no-BC).
# Launched 2026-05-05 evening, ~14.5h of GPU expected.
# Each cohort tees its own log into registry/<dir>.log so failures
# are diagnosable independently. `set -e` halts the cascade on first
# non-zero exit so we don't burn 13h of GPU after a broken arm.

set -e
set -o pipefail

cd "$(dirname "$0")"

TS=$(date +%s)

echo "================================"
echo "Phase 8 S03 + overnight launch"
echo "TS=${TS}  start=$(date)"
echo "================================"

# ── Stage 1: S03 3-arm probe (~90 min) ───────────────────────────────
# Per-slot baseline: --per-transition-credit OFF.
# Mature_prob_loss_weight enabled so the GA spreads the gene across
# 8 agents — needed for ρ(weight, mr).
echo "[$(date)] Stage 1.A — per-slot baseline"
python -m training_v2.cohort.runner \
  --n-agents 8 --generations 1 --days 4 --n-eval-days 1 \
  --device cuda --seed 42 --data-dir data/processed \
  --enable-gene mature_prob_loss_weight \
  --output-dir registry/_phase8_s03_A_perslot_${TS} \
  2>&1 | tee registry/_phase8_s03_A_perslot_${TS}.log

echo "[$(date)] Stage 1.B — per-transition credit"
python -m training_v2.cohort.runner \
  --n-agents 8 --generations 1 --days 4 --n-eval-days 1 \
  --device cuda --seed 42 --data-dir data/processed \
  --enable-gene mature_prob_loss_weight \
  --per-transition-credit \
  --output-dir registry/_phase8_s03_B_pertrans_${TS} \
  2>&1 | tee registry/_phase8_s03_B_pertrans_${TS}.log

echo "[$(date)] Stage 1.C — BC + per-transition"
python -m training_v2.cohort.runner \
  --n-agents 8 --generations 1 --days 4 --n-eval-days 1 \
  --device cuda --seed 42 --data-dir data/processed \
  --enable-gene mature_prob_loss_weight \
  --per-transition-credit --bc-pretrain-steps 500 \
  --output-dir registry/_phase8_s03_C_bc_pertrans_${TS} \
  2>&1 | tee registry/_phase8_s03_C_bc_pertrans_${TS}.log

# ── Stage 2: Split overnight (12 + 12, ~13h) ────────────────────────
# Mirrors last night's 24-agent cohort knobs:
#   reward_overrides target_pnl_pair_sizing_enabled, force_close_*,
#                    min_seconds_before_off, open_cost
#   enabled genes:    mark_to_market_weight, mature_prob_loss_weight,
#                     matured_arb_bonus_weight, stop_loss_pnl_threshold
#   maturation_bonus_weight=10
# Difference vs last night:
#   - 12 agents per cohort (vs 24); two cohorts back-to-back
#   - 5 gens (vs 6) — fits 18h budget with buffer
#   - 4 train + 3 eval days (new 50/50 default; was 4 train + 1 eval)
#   - Cohort B adds --bc-pretrain-steps 500 (the only difference vs A)
#   - Both cohorts use --per-transition-credit (Phase 9 GREEN, validated)

echo "[$(date)] Stage 2.A — long overnight, no BC"
python -m training_v2.cohort.runner \
  --n-agents 12 --generations 5 --days 7 \
  --device cuda --seed 42 --data-dir data/processed \
  --per-transition-credit \
  --enable-gene mark_to_market_weight \
  --enable-gene mature_prob_loss_weight \
  --enable-gene matured_arb_bonus_weight \
  --enable-gene stop_loss_pnl_threshold \
  --reward-overrides target_pnl_pair_sizing_enabled=true \
  --reward-overrides force_close_before_off_seconds=60 \
  --reward-overrides min_seconds_before_off=60 \
  --reward-overrides open_cost=1.0 \
  --maturation-bonus-weight 10.0 \
  --output-dir registry/_phase8_overnight_A_nobc_${TS} \
  2>&1 | tee registry/_phase8_overnight_A_nobc_${TS}.log

echo "[$(date)] Stage 2.B — long overnight, BC=500"
python -m training_v2.cohort.runner \
  --n-agents 12 --generations 5 --days 7 \
  --device cuda --seed 42 --data-dir data/processed \
  --per-transition-credit --bc-pretrain-steps 500 \
  --enable-gene mark_to_market_weight \
  --enable-gene mature_prob_loss_weight \
  --enable-gene matured_arb_bonus_weight \
  --enable-gene stop_loss_pnl_threshold \
  --reward-overrides target_pnl_pair_sizing_enabled=true \
  --reward-overrides force_close_before_off_seconds=60 \
  --reward-overrides min_seconds_before_off=60 \
  --reward-overrides open_cost=1.0 \
  --maturation-bonus-weight 10.0 \
  --output-dir registry/_phase8_overnight_B_bc_${TS} \
  2>&1 | tee registry/_phase8_overnight_B_bc_${TS}.log

echo "================================"
echo "ALL COHORTS COMPLETE  end=$(date)"
echo "================================"
