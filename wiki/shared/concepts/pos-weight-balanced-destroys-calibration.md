---
id: 01KTFBST2TMH7D82D5A0XDHFG6
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-042412]
links: [{to: direction-head-c11, type: supports}]
aliases: [pos_weight=balanced harm, unweighted BCE wins, BCE calibration finding]
---

# `pos_weight = balanced` destroys calibration here

A counter-intuitive empirical finding from the direction-head sweep: the
"standard practice" `pos_weight = balanced` for an imbalanced binary
classifier (18% positive rate) **slightly hurts ranking AND wrecks
calibration**. Unweighted BCE wins on every metric.

## What it is

Across four architectures, switching from `balanced` to `pos_weight = 1`:

| arch | balanced (mean ρ / Brier) | unweighted (mean ρ / Brier) |
|---|---|---|
| `[64]` | C0: +0.2719 / 0.2282 | C3: +0.2758 / **0.1448** |
| `[256]` | C1: +0.2803 / 0.2185 | C8: +0.2861 / **0.1439** |
| `[256,128]` | C9: +0.2913 / 0.2186 | C11: +0.2921 / **0.1433** |
| `[256,128,64]` | C12: +0.2908 / 0.2268 | C14: +0.2918 / **0.1437** |

Unweighted BCE consistently lifts Pearson +0.001 to +0.006 AND cuts Brier
by 35–37% — and the cut comes from the predictions becoming **calibrated**
(pred mean → empirical rate ~0.18) instead of being pulled to ~0.46 by the
balanced reweighting.

## Why it matters

The class-balancing recipe is reflexive practice in imbalanced
classification. Here it's actively harmful because the head's output is
consumed as a **probability** by `direction_gate_threshold` (and similar
gates) — a model whose "25%" actually fires 25% of the time is the
mechanism the gate needs. Lesson: when the downstream consumer needs
calibrated probabilities (not just rank-order), audit class-balancing —
the standard recipe optimises the wrong objective.

## Sources
- `src-042412` findings.md (js_desktop:present)
