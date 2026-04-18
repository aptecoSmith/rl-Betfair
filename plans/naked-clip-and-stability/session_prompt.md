# Naked-Clip-and-Stability — Session 01 prompt

Current session: **Session 01 — Reward shape: full cash in raw,
95% winner clip + close bonus in shaped**.

Detailed brief:
[`session_prompts/01_reward_shape.md`](session_prompts/01_reward_shape.md).

Before starting, read:

- [`purpose.md`](purpose.md) — gen-2 transformer `0a8cacd3`
  ep-1 blow-up evidence, the three-pathologies diagnosis, and
  the outcome table showing the new per-pair reward contributions.
- [`hard_constraints.md`](hard_constraints.md) — 28
  non-negotiables. §1 (scope), §3–§7 (reward semantics), §20
  (tests green per commit), §23 (worked-example test
  coverage), §24 (reward-scale change protocol), §25
  (CLAUDE.md update) most likely to bite.
- [`master_todo.md`](master_todo.md) — five-session scope and
  per-session exit criteria.
- `CLAUDE.md` — "Reward function: raw vs shaped" and "Bet
  accounting: matched orders, not netted positions".
- `plans/scalping-naked-asymmetry/purpose.md` — the per-pair
  aggregation this plan's clip operates over.
- `plans/scalping-close-signal/purpose.md` — the
  `close_signal` action whose successes now earn a shaped
  bonus.
- `env/betfair_env.py::_settle_current_race` — the function
  being edited. Find the scalping reward branch (grep
  `scalping_locked_pnl`).
- `env/bet_manager.py::get_naked_per_pair_pnls` — accessor
  already exists from `scalping-naked-asymmetry`; Session 01
  uses it directly.
