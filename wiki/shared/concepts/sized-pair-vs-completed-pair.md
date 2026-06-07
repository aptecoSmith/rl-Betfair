---
id: 01KTGP9R7XG3AVR2TSTBTBY7PM
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-1b344c]
aliases: [properly sized vs completed pair, sizing fix only on completed, 14.5% completion, active management lever]
---

# A "properly sized" pair ≠ a "completed" pair

A scope clarification: the asymmetric-hedging sizing fix only governs pairs that actually **complete**, but
most pair attempts don't — so the fix touches a small minority of trades, and the rest go directional at
the aggressive leg's original stake.

## What it is

The scalping-asymmetric-hedging fix (commit `c218bfb`) ensures completed pairs are properly sized, but
"most pair attempts don't complete — so the sizing fix only applies to 14.5% of attempts. The rest become
directional at the aggressive leg's original stake." So a correct sizing formula is necessary but not
sufficient: it can't help the 85.5% of attempts whose passive never fills. **Active management is the
lever to move that 14.5% up** — raising completion is what makes the sizing fix matter.

## Why it matters

A reminder to separate "is the math right on the trades it applies to?" from "how often does it apply?" —
a fix can be correct and near-irrelevant if its applicability is small. Pairs directly with
[[arbs-naked-is-timing-not-recklessness]] (the 85.5% are timing-outs, not chosen risk) and the
equal-profit sizing line ([[equal-profit-sizing]]); the completion rate is the binding constraint, the
same fill-rate wall as [[fixed-tick-passive-causes-force-close]].

## Sources
- `src-1b344c` lessons_learnt.md (js_desktop:present)
