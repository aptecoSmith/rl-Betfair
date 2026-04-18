# Scalping Equal-Profit Sizing — Session prompt pointer

Current session: **Session 01 — Equal-profit sizing helper +
tests**.
Detailed brief:
[`session_prompts/01_math_helper.md`](session_prompts/01_math_helper.md).

Before starting, read:

- [`purpose.md`](purpose.md) — full derivation of the corrected
  sizing formula, the canonical worked example (Back £16 @ 8.20
  / Lay @ 6.00 / c=5% → locked £4.03), and the reward-scale-
  change protocol that lands in Session 02.
- [`hard_constraints.md`](hard_constraints.md) — 23 non-
  negotiables. §4–§7 (the math), §8 (atomic three-call-site
  switch in Session 02), §11–§13 (reward-scale-change protocol),
  §14–§17 (mandatory tests).
- [`master_todo.md`](master_todo.md) — four-session breakdown
  with deliverables, exit criteria, and acceptance criteria for
  each.
- `CLAUDE.md` — especially "Order matching: single-price, no
  walking" and "Bet accounting: matched orders, not netted
  positions" (Session 03 augments the former).

Session order:

| # | File | What lands | Reward-scale change? |
|---|---|---|---|
| 01 | [`01_math_helper.md`](session_prompts/01_math_helper.md) | Helper functions in `env/scalping_math.py` + tests. No env wiring yet. | No |
| 02 | [`02_wire_placement.md`](session_prompts/02_wire_placement.md) | Wire helper into all three placement paths atomically. **The reward-scale-change commit.** | **Yes** |
| 03 | [`03_docs_and_reset.md`](session_prompts/03_docs_and_reset.md) | CLAUDE.md update + cross-plan `lessons_learnt` note + operator-facing comparability paragraph. | No (docs only) |
| 04 | [`04_ui_display_fix.md`](session_prompts/04_ui_display_fix.md) | Drop `£` from Betfair odds in display strings (the parked UI bug from 2026-04-18). | No (presentation only) |

Sessions 01–03 are the critical path; Session 04 is parked-bug
cleanup that can land any time without affecting the activation
re-run.
