---
id: 01KTFTFE311CM23WJA3J9D2JBG
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-042412]
aliases: [literature tricks hurt, recipe ablations, focal loss worst]
---

# Literature training-tricks transfer negatively (at this scale)

Round-4 recipe ablations on the C11 head: **every common "improvement from the literature" hurt
held-out Pearson** at this data volume (~1M samples) and model size (~40k params).

## What it is

Each variant changed exactly one knob off C11; all regressed: AdamW (wd=1e-3) -0.0027, label-smoothing
0.05 -0.0044, GELU -0.0046, 200-epoch/patience-20 -0.0146 (overfit), and **focal loss γ=2 the worst at
-0.0337** (below the original baseline). Focal is structurally wrong here: it down-weights "easy"
examples, but at the 18% positive rate the abundant easy negatives provide most of the discriminative
gradient, so down-weighting them removes signal. Likewise BN+Dropout (C4) was worse than baseline —
the model isn't overfitting at 1M samples / ~7k params, so regularisation only adds gradient noise.

## Why it matters

Conventional wisdom about regularisation, activation functions, and target softening does not transfer
at this data scale — verify each "improvement" held-out rather than importing it. C11's plain recipe
(Adam, BCE, ReLU, 50 epochs / patience 5, hard targets) is already at the local optimum. Pairs with
[[direction-head-data-ceiling]] (the longer-training overfit) and [[direction-head-c11]].

## Sources
- `src-042412` findings.md (js_desktop:present)
