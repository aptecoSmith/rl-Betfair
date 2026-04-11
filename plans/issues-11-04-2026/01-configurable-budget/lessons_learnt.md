# Lessons Learnt — Configurable Budget

Append-only. Date each entry.

---

## 2026-04-11 — Initial analysis

The composite scoring system already normalises P&L by
`starting_budget`, so models trained at different budgets rank
correctly against each other on `composite_score`. The normalisation
happens in two places:

- `pnl_norm = mean_pnl / starting_budget` (clipped to [-1, 1])
- `pnl_per_bet_norm = pnl_per_bet / (starting_budget × 0.1)`

The terminal reward bonus also normalises: `day_pnl / starting_budget`.

This means the work is primarily display + per-plan configurability,
not a fundamental scoring redesign. The biggest risk is that the
scoreboard's normalisation currently reads `self.starting_budget`
from the global config rather than from the per-evaluation record —
Session 03 must fix this to use the recorded budget per eval run.
