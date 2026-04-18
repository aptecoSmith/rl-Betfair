# Scalping Naked Asymmetry — Session 01 prompt

Current session: **Session 01 — Per-pair naked P&L in raw reward**.
Detailed brief: [`session_prompts/01_per_pair_naked_pnl.md`](session_prompts/01_per_pair_naked_pnl.md).

Before starting, read:

- [`purpose.md`](purpose.md) — the worked example showing how
  aggregate `min(0, sum(naked_pnls))` lets winning nakeds cancel
  losing nakeds, and the gen-0/1/2 fitness trajectory from the
  overnight 2026-04-17 run that motivated this plan.
- [`hard_constraints.md`](hard_constraints.md) — 15 non-negotiables.
  §1 (one change only), §3 (no-luck-reward invariant), §4 (raw +
  shaped invariant), §9–§10 (reward-scale change protocol),
  §11–§12 (mandatory tests).
- [`master_todo.md`](master_todo.md) — single-session scope and
  exit criteria.
- `CLAUDE.md` — especially "Reward function: raw vs shaped" and
  "Bet accounting: matched orders, not netted positions".
- `plans/scalping-asymmetric-hedging/purpose.md` — the original
  introduction of `min(0, naked_pnl)`. This plan refines its
  aggregation level; the asymmetric design intent is preserved.
- `plans/scalping-close-signal/lessons_learnt.md` — the gen-2
  population analysis from the close_signal run that exposed
  this layer.
