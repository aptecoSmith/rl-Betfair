---
id: 01KTJ0EVNAC8S5J1JTDRA8FBT1
type: concept
cloud: shared
status: stable
created: 2026-06-07
updated: 2026-06-07
tags: [work, research, lessons]
sources: [src-042412]
aliases: [c11-recipe-ablations, recipe-knobs-negative]
---

# Recipe ablations on C11 — every change hurt

Round 4 of the [[direction-head-architecture-sweep]] tested whether [[c11]]'s training recipe was already optimal. Five common "improvements from the literature" all transferred **negatively**.

## What it is

Each variant changes exactly one knob; everything else is held identical to C11.

| ID | knob changed | mean ρ | delta vs C11 | verdict |
|---|---|---|---|---|
| C11 | (baseline)                          | +0.2921 | —       | (best) |
| C16 | Adam → **AdamW** (wd=1e-3)          | +0.2894 | −0.0027 | small regression |
| C20 | + **label smoothing 0.05**          | +0.2877 | −0.0044 | small regression |
| C19 | ReLU → **GELU**                     | +0.2875 | −0.0046 | small regression |
| C18 | 50 ep / patience 5 → **200/20**     | +0.2775 | −0.0146 | clear overfit |
| C17 | BCE → **focal loss γ=2**            | +0.2584 | −0.0337 | structural regression |

## Why it matters

- **Focal loss is structurally worst** (below the original C0 baseline). Focal down-weights "easy" examples; at the 18% positive rate, the abundant "easy" negatives carry most of the discriminative gradient. Down-weighting them removes signal — the opposite of what focal loss assumes.
- **AdamW / GELU / label-smoothing** each cost ~0.003-0.005 Pearson. Within day-to-day noise individually, but consistent. Conventional wisdom about regularisation, activation functions, and target softening does NOT transfer at this data volume (1M samples) / model size (~40k params).
- **C18 (200 epochs)** mirrors [[c15-pairwise-overfit]]'s story from a different angle: generalisation peaks early; the default `--patience 5` catches it.

## Links
- [[direction-head-architecture-sweep]] — the parent sweep.
- [[c11]] — the recipe these were ablated against.
- [[c15-pairwise-overfit]] — companion overfit evidence.
- [[shared/index|hub]]
