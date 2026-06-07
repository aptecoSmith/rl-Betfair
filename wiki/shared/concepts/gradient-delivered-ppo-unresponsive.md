---
id: 01KTGC1SK6MBXT1KRQT961MXHC
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-106b41]
aliases: [gradient delivered but PPO unresponsive, fc-cost probe campaign, representational bottleneck, cohort-scale SNR]
---

# Gradient delivered, PPO unresponsive (the fc-cost probe campaign)

A consolidated meta-finding from ~9 reward-side probes (A/B/C/O/A2/H/D + E1/E2), each a 5-agent × short
cohort attacking force-close cost. **Every one was NO BITE** on the lever-signal metrics (cl_n, fc_n,
bets flat within ±2–3 events). The mechanisms were verified to fire by tests; PPO simply did not respond
at this cohort scale.

## What it is

The probes ruled out, one hypothesis at a time:
- **Magnitude** — `close_signal_bonus` £1 → £10 (A) → £50 (O): £50 produced *fewer* closes (8.4) than
  default £1 (9). "Cleanest possible refutation of the magnitude hypothesis."
- **Training length** — A2 doubled training 3→7 days at bonus £10: *fewer* closes (7.7). Sample-size
  refuted.
- **Timing** — E1 moved the close bonus to per-tick credit (the fix that worked for `open_cost`): cl_n
  9→9.2, flat. Timing refuted.
- **Architecture** — D added an `fc_prob_head` feeding actor_input at BCE weight 3.0: only nudged bets
  178→165, too small to clear naked noise.

The "+£30–40/d pnl mean" lifts across probes were consistently traceable to 1–2 of 5 agents catching
positive naked-variance days, **not** the lever working. Diagnosis: a **representational bottleneck** (no
per-runner "this open will force-close" feature to condition on) plus **cohort-scale signal-to-noise** —
any shaped gradient against fc cost has to propagate through hundreds of ticks against ±£500/day naked
variance, which swamps it at PPO's value-function level.

## Why it matters

This is the empirical wall that motivated the pivot to env-side priors ([[close-feasibility-open-gate]],
[[remove-decisions-beats-teaching]]). It is the cohort-scale companion to the single-agent
[[noop-absorbing-state]] collapse: both say "the shaping signal can't survive the naked-variance noise."
It also explains why per-agent mean-pnl lifts are untrustworthy on short eval windows — always decompose
to find the naked-tailwind agent.

## Sources
- `src-106b41` EXPERIMENTS.md (js_desktop:present)
