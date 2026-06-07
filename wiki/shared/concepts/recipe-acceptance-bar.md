---
id: 01KTGNYHY9R800AYKZ9W0CDE33
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-19293f]
aliases: [acceptance bar, deploy candidate rule, 4 of 5 metrics, seed-robustness rule]
---

# The recipe acceptance bar (deploy-candidate rule)

The multi-metric bar a probe cell must clear to be a deploy candidate — designed so no single metric
(especially naked-driven day_pnl) can carry a cell on its own.

## What it is

Each cell is scored on five targets: **opens/day 100–180, mat% ≥ 5%, fc% ≤ 50%, day_pnl > −£100,
locked/σ_naked > 0.5.** A cell **passing 4 of 5 with day_pnl > −£50 is a deploy candidate.** Separately, a
recipe is judged **robust** if across seed re-runs "day_pnl spreads less than ±£60 and mat% spreads less
than ±2pp" — seed-robustness is a first-class acceptance criterion, not an afterthought. The
`locked/σ_naked` term encodes the project's core selection principle (structural locked edge per unit of
naked-leg variance), so a cell that wins only via a naked tailwind fails it.

## Why it matters

A composite bar resists the single-metric mirage ([[fc0-insample-mirage]], [[gen0-pnl-is-meaningless]]):
requiring 4/5 including `locked/σ_naked` and an opens-band means a deploy candidate has structural edge,
controlled exposure, AND sane volume — not just a lucky P&L. The seed-spread rule operationalises "confirm
it isn't seed-lucky." The canonical metrics behind it are the naked-variance-as-primary-metric and
cohort-metrics-panel disciplines.

## Sources
- `src-19293f` purpose.md (js_desktop:present)
