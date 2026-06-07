---
id: 01KTFQ0JD087TY236VZWTMZTXM
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research]
sources: [src-009382]
aliases: [worst-case-improvement shaping, worst_case improvement]
---

# Worst-case-improvement shaping

A per-closing-leg shaped reward equal to the improvement in the position's guaranteed floor:
`Δ worst_case = worst_case_after − worst_case_before`.

## What it is

Backing a runner leaves a negative worst-case (−stake). A hedge lay lifts that floor; the shaping term
credits the size of the lift, so it specifically rewards *proper sizing* of the hedge (a bigger,
correctly-sized lay lifts the floor more). It is a **dense per-step gradient** that is available before
the race settles, so the signal arrives while the sizing choice is still fresh — unlike the settle-time
locked reward. It complements [[locked-pnl-per-pair-definition]] (the level) by rewarding the
*change* in the floor at the moment of the closing action.

## Why it matters

A shaped term in [[raw-vs-shaped-reward]] that teaches the agent to hedge to a positive floor rather
than place correlated pairs and hope. Part of the scalping-asymmetric-hedging reward fix.

## Sources
- `src-009382` purpose.md (js_desktop:present)
