#!/bin/bash
# Phase-15 v14: replicate v8 EXACTLY but with 8 agents.
# v8 config: 3 train + 1 eval, BC + freeze, T=0.85 (override),
# pos_weight=true. v8 had 2 agents both +pnl. This tests
# whether the 8-agent variance produces a similar positive
# distribution.
#
# Wall: ~3 hours (8 agents × ~22 min each).

set -euo pipefail

TS=$(date +%s)
RUN="phase15_v14_replicate_${TS}"
OUT="registry/_${RUN}"
LOG="registry/_${RUN}.log"
echo "Launching ${RUN} (replicate v8 with 8 agents)"

python -m training_v2.cohort.runner \
    --n-agents 8 \
    --generations 1 \
    --days 4 \
    --n-eval-days 1 \
    --device cuda \
    --output-dir "${OUT}" \
    --bc-pretrain-steps 2000 \
    --reward-overrides bc_direction_target_weight=1.0 \
    --reward-overrides direction_prob_loss_weight=0.1 \
    --reward-overrides direction_gate_enabled=true \
    --reward-overrides direction_gate_threshold=0.85 \
    --reward-overrides matured_arb_bonus_weight=2.0 \
    --reward-overrides force_close_before_off_seconds=60 \
    > "${LOG}" 2>&1

echo "${RUN} done"
python -m tools.phase15_summary "${LOG}"
