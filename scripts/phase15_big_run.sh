#!/bin/bash
# Phase-15 bigger training run.
#
# Fires AFTER smoke v9 confirms BC + freeze + multi-day calibrates
# direction_prob_head and produces positive eval pnl.
#
# Configuration:
# - 8 agents × 2 generations
# - 5 train + 1 eval day
# - BC pretrain 2000 steps, BC trains direction_prob_head
# - direction_prob_head FROZEN post-BC
# - Gate evolved as a gene (per-agent threshold)
# - Mature bonus + force-close + augmented features (phase-14 S02)
#
# Wall budget: ~4 hours.

set -euo pipefail

TS=$(date +%s)
RUN_NAME="phase15_big_${TS}"
OUT_DIR="registry/_${RUN_NAME}"
LOG="registry/_${RUN_NAME}.log"

echo "Launching ${RUN_NAME}; output: ${OUT_DIR}"

python -m training_v2.cohort.runner \
    --n-agents 8 \
    --generations 2 \
    --days 6 \
    --n-eval-days 1 \
    --device cuda \
    --output-dir "${OUT_DIR}" \
    --bc-pretrain-steps 2000 \
    --enable-gene direction_gate_threshold \
    --reward-overrides bc_direction_target_weight=1.0 \
    --reward-overrides direction_prob_loss_weight=0.1 \
    --reward-overrides direction_gate_enabled=true \
    --reward-overrides matured_arb_bonus_weight=2.0 \
    --reward-overrides force_close_before_off_seconds=60 \
    > "${LOG}" 2>&1

echo "${RUN_NAME} done; tail of log:"
tail -50 "${LOG}"
