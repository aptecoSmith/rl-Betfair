---
id: 01KTG846Y2N9RJV4XK35Z90PRG
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research]
sources: [src-0e88d0]
aliases: [K=5 ensemble, min-of-K agreement, consensus uncertainty]
---

# Ensemble consensus as implicit uncertainty

Train **K = 5 independent direction predictors** (different BC seeds) and fire the gate only when **all
K agree** the runner crosses threshold (`min(P_back_k) >= T`) — consensus as an implicit uncertainty
estimate.

## What it is

The intuition: when the input pattern matches what the predictors were trained on, the K converge on
similar outputs and agree; when the input is unfamiliar or noisy, their independent random seeds
**produce different outputs and they disagree** → don't bet. No new loss function, no calibration
tracking — just K parallel heads with an AND-gate. Cost is modest: ~5× BC compute (~5 min/agent), 5×
the (tiny) direction-head parameters. K is fixed at 5 initially; could become a gene later. Success
criterion: fewer false-fire bets than the K=1 baseline while holding mature rate at the fired bets.

## Why it matters

A cheap way to turn the [[prediction-tail-instability]] into a "bet only when sure" gate without a
calibration model — disagreement among seeds is the uncertainty signal. Complements the richer-input
fix [[market-state-and-cross-runner-features]].

## Sources
- `src-0e88d0` purpose.md (js_desktop:present)
