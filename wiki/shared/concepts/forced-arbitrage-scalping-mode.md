---
id: 01KTG4MZS72GP33QPRF1FW0Q3X
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research]
sources: [src-0d7579]
aliases: [forced arbitrage, scalping mode, arb_spread origin]
---

# Forced arbitrage (scalping mode)

The founding scalping design: **every aggressive bet automatically generates a paired passive order on
the opposite side** of the same runner, so the agent learns to scalp — pick spots where the market will
move to fill the second leg and lock a small profit before the off.

## What it is

A new action dimension `arb_spread` (how many ticks from the fill to place the second leg) makes the
agent learn **the tension between spread (profit) and distance (fill probability)**, net of Betfair's
~5% commission. Mechanics: aggressive back fills → system places a passive lay N ticks lower; if the
market drops and the lay fills, profit is locked regardless of race outcome; if it doesn't fill, the
back is naked into the off and settles directionally. A new reward path rewards arb-pair completion,
penalises naked exposure, and ignores directional win/loss metrics (meaningless for market-making). The
**commission breakeven is price-dependent**: at low prices (2-3) a tick is ~£0.02/£10 (need several
ticks), at high prices (10+) ~£0.50/£10 (one may suffice).

## Why it matters

The root of the whole scalping cluster — [[locked-pnl-per-pair-definition]], [[equal-profit-sizing]],
[[naked-asymmetry-per-pair]], [[close-position-action]] all refine this. The pairing is **forced**
because the action space is too large for the agent to stumble into arb behaviour spontaneously, and
directional reward shaping rewards winning bets, not hedges. Scalping is lower-return but far safer — a
candidate for lower-risk live deployment.

## Sources
- `src-0d7579` purpose.md (js_desktop:present)
