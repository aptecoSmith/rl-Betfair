---
id: 01KTF937MC0M8W5QHQREWDYYDX
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, superseded]
sources: [src-04294a]
aliases: [aggregate naked asymmetry]
---

# Aggregate naked-loss term (superseded)

The **earlier** raw naked penalty `min(0, sum(naked_pnls))` — applied to the
*aggregate* naked P&L of a whole race rather than per pair.

## What it is

Because the `min(0, …)` was taken over the race total, every losing naked was
cancelled out by any unrelated lucky naked in the same race: a +£100 winning naked
plus a −£80 losing naked aggregated to +£20, and `min(0, +£20) = 0` — zero penalty.
The asymmetric intent ("naked losses cost reward, windfalls excluded") only held
when the aggregate happened to be negative.

## Why it matters

Superseded by [[naked-asymmetry-per-pair]]: the fix moved the `min(0, …)` inside
the per-pair sum so luck can no longer launder losses. Kept as the record of the
aggregation bug. Part of the [[reward-shaping-supersessions]] chain.

## Sources
- `src-04294a` Plan: Scalping Naked Asymmetry (purpose.md) (js_desktop:present)
