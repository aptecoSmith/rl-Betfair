#!/bin/bash
# Phase-15 v12: force_close_before_off_seconds sweep.
# Tests how earlier (30s) vs later (90s) force-close affects pnl
# given v8 config (pos_weight + multi-day BC + freeze + T=0.85).
#
# Default 60s. Three runs at 30/60/90. Wall ~22min × 3 = ~70min.

set -euo pipefail

for FC in 30 60 90; do
    TS=$(date +%s)
    RUN="phase15_fc${FC}_${TS}"
    OUT="registry/_${RUN}"
    LOG="registry/_${RUN}.log"
    echo ""
    echo "=== Launching ${RUN} (force_close=${FC}s) ==="
    python -m training_v2.cohort.runner \
        --n-agents 2 \
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
        --reward-overrides force_close_before_off_seconds=${FC} \
        > "${LOG}" 2>&1
    echo "${RUN} done; key metrics:"
    grep -E "post_bc_dir_bce|POST-PPO direction|eval \[|matured=" "${LOG}" | tail -10
done

echo ""
echo "=== force_close sweep complete ==="
