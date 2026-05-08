#!/bin/bash
# Phase-15 v13: single-day BC + freeze vs v8's multi-day BC + freeze.
# Tests if multi-day pooling is essential or if single-day suffices
# given the freeze + pos_weight regime.
#
# Same config as v8 but --days 2 (1 train + 1 eval).
# Wall ~12 min.

set -euo pipefail

TS=$(date +%s)
RUN="phase15_v13_singleday_${TS}"
OUT="registry/_${RUN}"
LOG="registry/_${RUN}.log"
echo "Launching ${RUN} (single-day BC + freeze)"

python -m training_v2.cohort.runner \
    --n-agents 2 \
    --generations 1 \
    --days 2 \
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
python -m tools.phase15_summary "${LOG}" | tail -15
