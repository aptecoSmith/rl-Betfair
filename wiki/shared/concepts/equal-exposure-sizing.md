---
id: 01KTF8ZNV4SGMHJ5MA1JHX3632
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, superseded]
sources: [src-bf778c]
aliases: [equal-exposure sizing, exposure sizing]
---

# Equal-exposure pair sizing (superseded)

The **pre-fix** passive-leg sizing formula `S_lay = S_back × P_back / P_lay`. Its
code comment claimed it was "derived from demanding equal P&L in win and lose
outcomes" — but that derivation only holds when commission is zero.

## What it is

With Betfair's non-zero commission the formula equalises *exposure*
(`stake × (price − 1)`), not profit. The result is an **over-laid** trade whose
two outcomes pay very differently (e.g. +£0.08 if the runner wins vs +£4.78 if it
loses on one observed trade). Consequently `locked_pnl = min(win, lose)` was
systematically understated — the reward path saw the near-zero win-side cliff, not
the trade's true floor.

## Why it matters

Superseded by [[equal-profit-sizing]] on 2026-04-18 (commit `f7a09fc`) after a
live-activity-log observation by the operator. Kept as a **pre-fix reference
only**: garaged models trained before the fix retain their pre-fix `locked_pnl`
scoreboards and are valid *as pre-fix references*, never comparable on shaped
magnitudes to post-fix runs. Part of the [[reward-shaping-supersessions]] chain.

## Sources
- `src-bf778c` Plan: Scalping Equal-Profit Sizing (purpose.md) (js_desktop:present)
