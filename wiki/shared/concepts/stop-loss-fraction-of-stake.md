---
id: 01KTG8PXV3XC5AA1G40VEDY7X1
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-0eda3f]
aliases: [stop_loss_pnl_threshold, stop-loss units gotcha, cutting losers, ride-to-force-close loss]
---

# Stop-loss: a fraction of stake, and where the toll actually comes from

`stop_loss_pnl_threshold` cuts a losing pair mid-race instead of riding it to the T−120 force-close. Two
findings: a **units gotcha** that made the first sweep a no-op, and the realisation that most of the
force-close toll is a *directional* loss the agent could have cut.

## What it is

The toll decomposition: the −£1.25 force-close toll is **~76% a directional loss from RIDING losers to
the T−120 force-close** — the actual close-cross is only ~£0.30. We never cut losers; the stop-loss
mechanism was wired but defaulted OFF and never used.

The units gotcha (2026-05-02 operator clarification): `stop_loss_pnl_threshold` is a FRACTION of stake,
not £. The first sweep at 0.5–2.0 meant −£5 to −£20 on a £10 stake and never fired; corrected to
0.02–0.10 = −£0.20 to −£1.0. On the BC policy the stop FIRES (sc% 74% at the tight 0.02 end) and the
best config cuts the day loss 36% (−41.85 → −26.81), one day flipped positive — but on the dumb BC
policy it does NOT flip aggregate-positive, and the tradeoff is sharp (the tight stop caps losers but
kills maturation 13%→3%).

## Why it matters

Stop-loss isn't the full unlock, but it caps the downside enough that opening is far less negative-EV,
which is what keeps PPO from collapsing ([[noop-absorbing-state]]). It directly attacks the directional
majority of the [[force-close-population-cost]] / [[toll-to-edge-ratio-wall]], and became one of the
genes handed to the GA cohort ([[economic-wall-was-weak-policy-average]]). Mind the
fraction-of-stake units before reading any sweep.

## Sources
- `src-0eda3f` findings.md (js_desktop:present)
