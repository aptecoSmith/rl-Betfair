---
id: 01KTGP443B2GVTFYH7A8F532AZ
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-19b97c]
aliases: [EW not the problem, WIN larger fc drag, rate not magnitude, EW vs WIN force-close]
---

# The EW force-close framing collapses (rate, not per-pair cost)

The operator's hypothesis that force-close cost is concentrated in Each Way markets **collapses on the
data**: WIN markets carry the larger absolute drag (−£189 vs −£111 across 7 days, single agent) and a much
higher force-close *rate* (70.5% WIN vs 41.4% EW). What differs is the rate, not the per-pair magnitude.

## What it is

Per-pair force-close cost is essentially identical across market types (≈ −£1.30 each; median −£0.86 EW vs
−£0.85 WIN). "Spread cost per force-closed pair is essentially identical across market types" — the
operator's EW anecdote (−£0.08/pair) is actually *below* the empirical median. So **what dominates the EW
vs WIN cost difference is the rate, not the per-pair magnitude**: the agent force-closes ~1.8× as many WIN
pairs as EW pairs, likely because WIN markets have steeper pre-off price discovery so the +20-tick passive
is less likely to be caught up to.

## Why it matters

A worked example of a plausible cost hypothesis refuted by decomposition: the loss looked like an EW
problem but was a *rate* problem affecting WIN harder. It redirects the fix from "handle EW specially"
([[market-type-filter-gene]]) to "make passives fill" ([[fixed-tick-passive-causes-force-close]]). Read
the rate × magnitude decomposition before attributing a cost to a category — the same discipline as
decomposing a probe's mean-pnl lift to find the naked-tailwind agent ([[gradient-delivered-ppo-unresponsive]]).

## Sources
- `src-19b97c` findings.md (js_desktop:present)
