---
id: 01KTJ0G7GQKCZHN591KZCMNJDN
type: concept
cloud: shared
status: stable
created: 2026-06-07
updated: 2026-06-07
tags: [work, research, lessons]
sources: [src-042412]
aliases: [width-scaling-direction-head]
---

# Width scaling has hard diminishing returns

For the direction head at fixed depth-1, width helps 64 → 256 (+0.0084 Pearson), then each doubling beyond gains less than half the previous step. 1024 (C7) is essentially the same as 512 (C6).

| hidden width | variant | mean ρ | delta from C0 |
|---|---|---|---|
|   64 | C0 | +0.2719 | (baseline) |
|  128 | C4 | +0.2707 | −0.0012 (BN+Dropout hurts) |
|  256 | C1 | +0.2803 | +0.0084 |
|  512 | C6 | +0.2835 | +0.0116 |
| 1024 | C7 | +0.2841 | +0.0122 |

Width itself is sated by 256 once the shape is `[W, W/2]`. Depth-2 ([[depth-plateaus-at-2|see depth result]]) is where the remaining gain lives. Source: [[direction-head-architecture-sweep]].

[[shared/index|hub]]
