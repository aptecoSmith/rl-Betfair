---
id: 01KTG4Q6XSYSGVHWG17P1RZ6W5
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-0d94ae]
aliases: [middle-decile overconfidence, value-edge gate admits miscalibrated]
---

# Predictor over-confident in the middle deciles

The one-line diagnosis for why directional value betting failed: the win-probability predictor is
well-calibrated at the extreme deciles but **systematically over-confident in the middle deciles** —
exactly the region the value-edge gate admits.

## What it is

In the BACK probe's admitted 0.50-0.75 band, predicted 59-72% but realised 26-49%; the LAY probe's
admitted 0.87-0.98 band predicted 87-98% but realised 59-87%. The value-edge gate (threshold 0.05)
admits runners that **LOOK +EV under the predictor's stated probability but are actually closer to fair
odds**, so they lose in bulk; the +0.05 edge floor was supposed to filter this and didn't.
**Scalping sidesteps the issue by capturing a bounded spread per pair regardless of predictor
calibration in the middle** — directional betting is fully exposed to it.

## Why it matters

The mechanism behind [[directional-value-betting-fails]], and a pointer to the fix that *might* unlock
directional: a re-calibrated predictor (Platt/isotonic on held-out) could move the admitted set out of
the miscalibrated middle. A ranking-good predictor can still be a calibration-bad EV gate — the same
calibration-vs-ranking lesson as [[pos-weight-balanced-destroys-calibration]].

## Sources
- `src-0d94ae` findings.md (js_desktop:present)
