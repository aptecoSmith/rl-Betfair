---
id: 01KTJ0EVN9BK9W1Y5KSKN6B00R
type: concept
cloud: shared
status: stable
created: 2026-06-07
updated: 2026-06-07
tags: [work, research, lessons]
sources: [src-042412]
aliases: [pos-weight-calibration]
---

# pos_weight=balanced harms calibration

For the direction head at ~18% positive rate, `pos_weight=balanced` (class-balanced [[bce]]) pulls the model's output mean to ~0.46 and **destroys calibration** — common practice for imbalanced classification is actively harmful here.

## What it is

Held out comparison at four architectures from the [[direction-head-architecture-sweep]]:

| arch | balanced (ρ / Brier) | unweighted (ρ / Brier) |
|---|---|---|
| [64]         | C0: +0.2719 / 0.2282 | C3: +0.2758 / **0.1448** |
| [256]        | C1: +0.2803 / 0.2185 | C8: +0.2861 / **0.1439** |
| [256,128]    | C9: +0.2913 / 0.2186 | C11: +0.2921 / **0.1433** |
| [256,128,64] | C12: +0.2908 / 0.2268 | C14: +0.2918 / **0.1437** |

Switching to unweighted BCE consistently: slightly improves mean Pearson (+0.001 to +0.006), cuts mean Brier by 35–37%, and produces calibrated outputs (pred mean tracks empirical positive rate). With balanced training, "25% chance" actually meant about 11%; with unweighted training, "25% chance" really means about 25%.

## Why it matters

`direction_gate_threshold` is a gene that operates on these probabilities — it only becomes a *meaningful* lever when the head is calibrated. Without [[c11]] the threshold's nominal value has no semantic relationship to the underlying probability. This finding flipped a "regularisation default" into "explicit footgun" for this domain, and re-applies to any downstream classifier that downstream code threshold-gates.

## Links
- [[direction-head-architecture-sweep]] — the source sweep.
- [[c11]] — the head variant that wins on calibration AND ranking.
- [[shared/index|hub]]
