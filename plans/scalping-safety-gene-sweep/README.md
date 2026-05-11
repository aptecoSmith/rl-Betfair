---
plan: scalping-safety-gene-sweep
status: open
opened: 2026-05-11
predecessor: predictor-integration
---

# Scalping safety-gene sweep

## Why this plan exists

`plans/predictor-integration/` shipped a scalping cohort that
locked +£150–£230 per race-day across all 5 top agents on held-out
evaluation, but the naked tail killed 4 of 5 (mean −£28, median
−£74, only `2f384ae8` profitable at +£347). Hyperparameter audit
of the full 60-agent scoreboard showed:

- 0/60 agents had `stop_loss_pnl_threshold > 0`
- 0/60 agents had `open_cost > 0`
- 0/60 agents had `matured_arb_bonus_weight > 0`
- 0/60 agents had `naked_loss_scale < 1.0`
- 0/60 agents had `mature_prob_loss_weight > 0`
- 0/60 agents had `fill_prob_loss_weight > 0`

The GA had no diversity on the knobs that could suppress naked
losses. This plan re-runs the cohort with those six genes
activated in the GA's mutation pool.

## Hypothesis

Locked PnL transfers — the predictor-driven open mechanic works.
The naked tail is unfinished training, not a structural ceiling.
Activating the six safety/shaping genes will produce agents that
either:

1. Open more selectively (open_cost + mature_prob_loss_weight
   shape the policy to skip pairs likely to naked), OR
2. Cut losing pairs actively (stop_loss_pnl_threshold +
   matured_arb_bonus_weight reward closing-before-naked), OR
3. Both.

Success criterion: ≥3 of top-5 agents profitable on the 3-day
held-out window (vs 1/5 in the predecessor cohort).

## Out of scope

- New code in `env/`, `agents_v2/`, or
  `training_v2/discrete_ppo/`. The gene infrastructure already
  exists (Phase 5 — `plans/rewrite/phase-5-restore-genes/`).
- New predictor variants. Same production champions
  (`1c15250ee90d1b65`, `b23018bf5c8bcc70`,
  `conv1d_k3_s1_9659e9e9c3fb`).
- Re-litigating scalping vs value_win. The Session 07 verdict
  stands; this plan refines scalping only.

## Reference

- Predecessor verdict:
  `plans/predictor-integration/autonomous_run_log.md`
  2026-05-11 entries.
- Gene infrastructure: `training_v2/cohort/genes.py`
  Phase 5 block.
- CLAUDE.md sections on `selective-open shaping`, `naked-loss
  annealing`, `matured-arb bonus`, `mature_prob_head feeds
  actor_head` — the design rationale for each activated gene
  already lives in the project doc.
