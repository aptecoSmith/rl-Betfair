---
id: 01KTG4Q6XQMDZSTWBNPTMMK8MX
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-0d94ae]
aliases: [directional value betting fails, scalping only tradeable mode]
---

# Directional value betting fails (scalping is the only tradeable mode)

A pre-registered held-out probe of pure directional value betting (back-only and lay-only) — **both
probes FAIL**; scalping remains the only mode where the project's pwin signal has tradeable edge.

## What it is

5 seeds × 3 held-out days, uniform-random policy with the value-edge gate (threshold 0.05) as the
binding constraint, production predictor bundle. Probe A (back-only, flat £10): per-bet EV −£2.09,
Sharpe −0.10, 0/3 days profitable, −£8,675 cumulative. Probe B (lay-only, £20 liability, price ∈
[2,20]): per-bet EV −£0.69, Sharpe −0.09, 0/3 days, −£1,080 — not a single (seed, day) sub-cohort
recovered. Per the pre-registered decision table this **closes the chapter on directional value
betting** at this predictor's calibration (and updates the reliability-over-upside stance).

## Why it matters

Empirical backing for defaulting to `--strategy-mode arb` (scalping) over directional value betting:
[[forced-arbitrage-scalping-mode]] captures a bounded spread regardless of mid-range calibration, which
directional cannot. The cause is [[predictor-overconfident-middle-deciles]]. Does NOT rule out a
re-calibrated predictor or higher thresholds (the per-bet logs are the calibration raw data).

## Sources
- `src-0d94ae` findings.md (js_desktop:present)
