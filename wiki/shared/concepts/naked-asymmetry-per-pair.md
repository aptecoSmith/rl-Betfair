---
id: 01KTF937MDJYRR2SGJ4YXS7Y4X
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research]
sources: [src-04294a, src-3f548f]
links: [{to: naked-asymmetry-aggregate, type: supersedes}]
aliases: [per-pair naked asymmetry, naked asymmetry]
---

# Per-pair naked-loss asymmetry

The raw reward penalises naked (unhedged) legs **per pair**: the naked term is
`sum(min(0, per_pair_naked_pnl))`, summed over each individual unfilled-paired
aggressive leg.

## What it is

It replaced the aggregate term `min(0, sum(naked_pnls))` (see
[[naked-asymmetry-aggregate]]). Computing `min(0, …)` per pair means each
individual naked **loss** costs reward and a lucky naked **win** can no longer
cancel an unrelated loss in the same race. Losses flow at full cash value; windfalls
are still neutralised (the asymmetric design). This is the raw-side complement of
the shaped naked-windfall neutering described in [[raw-vs-shaped-reward]].

## Why it matters

The aggregate form made "place lots of nakeds, hope for a lucky aggregate" a
positive-EV strategy and starved the close mechanic of purpose. Per-pair
aggregation restores selectivity pressure. A **reward-scale change** (raw means
shift more negative for luck-reliant agents — compare on `raw_pnl_reward`). One
link in the [[reward-shaping-supersessions]] chain.

## Sources
- `src-04294a` purpose.md (js_desktop:present)
- `src-3f548f` CLAUDE.md (js_desktop:present)
