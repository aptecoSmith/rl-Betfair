#!/bin/bash
# Phase-15 gate threshold sweep (v10).
# Tests how mature rate / eval pnl varies with the direction-gate
# threshold, given BC-calibrated head + freeze + multi-day.
#
# Three runs: T=0.70, T=0.85, T=0.95. Each with 2 agents × 1 gen
# × 3 train + 1 eval. Wall ~22 min × 3 = ~70 min.

set -euo pipefail

for THRESHOLD in 0.70 0.85 0.95; do
    TS=$(date +%s)
    RUN="phase15_gate${THRESHOLD/./}_${TS}"
    OUT="registry/_${RUN}"
    LOG="registry/_${RUN}.log"
    echo ""
    echo "=== Launching ${RUN} (T=${THRESHOLD}) ==="
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
        --reward-overrides direction_gate_threshold=${THRESHOLD} \
        --reward-overrides matured_arb_bonus_weight=2.0 \
        --reward-overrides force_close_before_off_seconds=60 \
        > "${LOG}" 2>&1
    echo "${RUN} done; key metrics:"
    grep -E "post_bc_dir_bce|POST-PPO direction|eval \[|matured=" "${LOG}" | tail -10
done

echo ""
echo "=== Gate sweep complete ==="
