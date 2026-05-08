#!/bin/bash
# Phase-15 v10: vanilla BCE BC + freeze + LOW gate threshold (0.55).
# Hypothesis: v9 (no pos_weight, T=0.85) NOOPs because head is
# calibrated to natural distribution (~0.22 mean) and 0.85 blocks
# everything. Lowering threshold to 0.55 lets the head's
# above-baseline predictions (genuine high-confidence runners) pass.
#
# Wall: ~22 min.

set -euo pipefail

TS=$(date +%s)
RUN="phase15_v10_lowgate_${TS}"
OUT="registry/_${RUN}"
LOG="registry/_${RUN}.log"
echo "Launching ${RUN}"

python -m training_v2.cohort.runner \
    --n-agents 2 \
    --generations 1 \
    --days 4 \
    --n-eval-days 1 \
    --device cuda \
    --output-dir "${OUT}" \
    --bc-pretrain-steps 2000 \
    --reward-overrides bc_direction_target_weight=1.0 \
    --reward-overrides direction_bce_use_pos_weight=false \
    --reward-overrides direction_prob_loss_weight=0.1 \
    --reward-overrides direction_gate_enabled=true \
    --reward-overrides direction_gate_threshold=0.55 \
    --reward-overrides matured_arb_bonus_weight=2.0 \
    --reward-overrides force_close_before_off_seconds=60 \
    > "${LOG}" 2>&1

echo "${RUN} done"
python -m tools.phase15_summary "${LOG}" | tail -10
