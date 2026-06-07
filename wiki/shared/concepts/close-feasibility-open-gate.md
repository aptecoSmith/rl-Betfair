---
id: 01KTGC1SK2G1064X1V075QE60S
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-106b41]
aliases: [E3, close-feasibility gate, close_feasibility_max_spread_pct, if they can't close it]
---

# Close-feasibility open gate (E3 — the breakthrough)

The first lever to clearly move the metrics across ~12 probes, and the first STRONG-band cohort in the
project's history. The operator's principle: **"if they can't close it, they should never have opened
it."** At open time the env peeks the opposite-side ladder, computes the hypothetical cost-to-close at
the current spread, and refuses the open if it exceeds a fraction of stake — a structural prior the
policy doesn't have to learn.

## What it is

`close_feasibility_max_spread_pct` (default None = disabled). At each open candidate in `_process_action`,
peek the junk-filtered opposite-side top price; refuse if `|agg_price − close_price| / close_price >
threshold` or the close side is unpriceable. At threshold 0.05 the probe (5×7d) returned **+£59.4/d vs
baseline −£46** — fc_n 54→34.8 (−36%), fc_£ −£86→−£56, **maturation rate 0.34→0.50**, 4/5 profitable.
Mechanism: refusing opens whose close path is too expensive kills the worst pre-off thin-liquidity opens,
so the remaining pairs mature more reliably.

The full 12×8gen cohort confirmed **STRONG band**: held-out 7-day forward window, top-5 mean **+£55.4/d
fc=120** and +£96.7/d fc=0, **5/5 profitable on BOTH fc settings** — vs the layq null's +£26/d and tnv2's
−£177/d regression. Deploy candidates 11099f65 + 57a42db5 (the ROBUST twin phenotype, locked ~£100,
modest naked, tight span).

## Why it matters

The cleanest validation of the project's recurring lesson — [[remove-decisions-beats-teaching]]: an
env-side prior that REMOVES the structurally-bad opens beats every reward-shaping / selector attempt to
teach the policy to avoid them ([[gradient-delivered-ppo-unresponsive]]). E3 became the production lever
and the anchor of [[scalping-cohort-lineage]]; every close-side / naked-side add-on tested on top of it
net-subtracted ([[probe-to-cohort-regression]]). It directly attacks the [[force-close-population-cost]].

## Sources
- `src-106b41` EXPERIMENTS.md (js_desktop:present)
