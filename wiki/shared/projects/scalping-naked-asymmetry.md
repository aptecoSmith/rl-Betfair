---
id: 01KTJ0K8101P8MSBRVEN5APXND
type: project
cloud: shared
status: stable
created: 2026-06-07
updated: 2026-06-07
tags: [work, research]
sources: [src-04294a]
aliases: [scalping-naked-asymmetry]
---

# Scalping naked-asymmetry fix

Plan that landed the [[naked-pnl-asymmetry-per-pair-fix]] in `env/betfair_env.py` after the activation-A-baseline GA collapsed (best frozen, mean degrading) due to lucky-naked cancellation in the aggregate.

## Goals
- Per-pair aggregation: `sum(min(0, per_pair_naked_pnl))`.
- Restore [[ga|GA]] gradient: best fitness moves across generations.
- Top model uses `close_signal`: `arbs_closed > 0` AND `arbs_closed / arbs_naked > 0.3`.
- Mean fitness stops degrading.
- `raw + shaped == total_reward` invariant stays green ([[reward-invariants]]).

## Status
Targeted single-fix plan. New accessor `BetManager.get_naked_per_pair_pnls(market_id)`. Replaces one line in `_settle_current_race`'s scalping-mode raw branch. Tests added in `tests/test_forced_arbitrage.py` (or new `tests/test_naked_per_pair.py`).

## Inputs
- Symptom: activation-A-baseline run 2026-04-17/18, three frozen generations (best_fitness=0.338), mean degrading.
- Diagnosis from gen-2 episode log — high-volume close-using agents at the bottom of the ranking despite `close_signal` being heavily used.
- Builds on [[scalping-asymmetric-hedging]] (introduced `min(0, naked_pnl)`) and `scalping-close-signal` Session 01 (provided the close mechanic).

## Notes
- Reward-scale change — operators comparing post-fix model PnL to pre-fix scoreboards must know the reward signal changed. Same rule as the activation playbook's Step E.
- Failure mode: if behaviour stays low-volume after the per-pair fix, the next lever is the `naked_penalty_weight` gene range `[0, 1]` — agents rolling near 0 get near-zero shaping. That's a separate plan.

[[shared/index|hub]]
