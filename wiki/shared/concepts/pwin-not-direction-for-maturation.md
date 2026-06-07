---
id: 01KTG8PXV2JFHCZ611WFC86KDK
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-0eda3f]
aliases: [pwin mispricing refuted, race-outcome mispricing, pwin-as-direction probe]
---

# pwin mispricing does NOT predict scalp maturation

An operator pivot (2026-05-31) tested whether the race-outcome predictor's mispricing
(`champion_p_win` vs market `implied_prob`) could serve as a price-DIRECTION signal — if the market
corrects toward the predictor, the move is predictable, and a predictable move matures a scalp (pure
scalping, not directional gambling). The probe **refuted** it.

## What it is

`pwin_direction_probe.py`, held-out 7 days, 505k candidates (reuses the maturation dataset). Standalone
maturation AUC: **pwin − implied (mispricing) = 0.502** (chance); champion_p_win raw = 0.500;
dir_fire_shorten/drift = 0.500/0.503. Maturation by mispricing decile is **flat ~12–14%** (most
under-priced 11.7%, most over-priced 13.1%, base 12.3%) — zero monotonic relationship. The market does
not visibly correct toward the race-outcome predictor inside the [T−120, off] window we trade.

The general intuition (a price-direction signal helps) is partly right but already captured: the
direction predictor's median-move quantile **dir_q50_3m reaches 0.599** (best single signal) — but it's
already in the obs and already used by the mature head (that's how it got to 0.745). The race-outcome
pwin adds nothing on top.

## Why it matters

Closes the same door as [[directional-value-betting-fails]] from the other side: the race-outcome pwin
signal, miscalibrated in the middle deciles ([[predictor-overconfident-middle-deciles]]), also has no
maturation-timing power. The maturation signal already lives in the direction-quantile features
([[direction-head-feature-slice]]); the pivot does not move the [[toll-to-edge-ratio-wall]].

## Sources
- `src-0eda3f` findings.md (js_desktop:present)
