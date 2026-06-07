---
id: 01KTGP4438J2236MBNK8JTM8KE
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-19b97c]
aliases: [arb_spread hardcoded, DiscreteActionShim 20 ticks, arb_spread_scale global, dead arb_spread action]
---

# arb_spread is a dead action in the v2 stack

A structural finding: in the v2 discrete-action stack the agent **cannot pick arb_spread per-runner — it
cannot pick it at all.** The `DiscreteActionShim` hard-codes `arb_ticks=20` for every OPEN_BACK/OPEN_LAY,
and the env-side decode slot is planted with that constant.

## What it is

During training and eval the arb_spread is **always 20 ticks**, regardless of runner, regardless of obs,
regardless of agent — `shim.step(..., arb_spread=None)` everywhere. The env's `_process_action` still
reads the slot and maps it to ticks, but the shim writes a constant into it. The only per-agent variation
is a global gene `arb_spread_scale ∈ [0.5, 2.0]` applied as a multiplier, so agents get arb_ticks ∈
[10, 25] cohort-wide. Consequence: "no individual agent will EVER pick a different arb_spread for a 1.20
favourite vs a 12.0 long-shot — the architecture forbids it."

## Why it matters

This is the source of the CLAUDE.md note that the per-runner arb_spread action dim is dead code, and it
motivated the later single-gene `arb_spread_target_lock_pct` (a price-adaptive replacement). A dead
*action* is the action-space analogue of a [[sampled-not-used-gene]]: the decode path exists and looks
alive, but nothing ever supplies a varying value. The fix is either pin the global scale
([[fixed-tick-passive-causes-force-close]]) or promote arb_spread to a real per-runner action dim.
Substrate: [[forced-arbitrage-scalping-mode]].

## Sources
- `src-19b97c` findings.md (js_desktop:present)
