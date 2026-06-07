---
id: 01KTFTFE3265MMHNEPCC790028
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-042412]
aliases: [width depth scaling, MLP shape, W W/2]
---

# Direction-head width/depth scaling

The empirical scaling of the per-runner direction-head MLP: width has hard diminishing returns and
depth plateaus at 2 hidden layers — the winning shape is **`[W, W/2]` with W ≥ 256**.

## What it is

**Width:** helps from 64→256 (+0.0084 Pearson), but each doubling beyond gains less than half the
previous step; 1024 ≈ 512. **Depth (at width 256):** [256]→[256,128] gains +0.0110, but [256,128,64]
adds nothing (-0.0005) — depth-2 helps, depth-3 plateaus. **Width+depth combined:** doubling both dims
([256,128]→[512,256]) gains only +0.0007 (noise) — once you have a `[W, W/2]` shape with W ≥ 256, width
itself is sated; the lift is dominated by the depth step. A skip connection (C10) and BN+Dropout (C4)
added nothing. So the dominant shape is "one wide hidden layer + one projection-down layer".

## Why it matters

Concrete sizing guidance for this predictor regime, and corroborates that gains saturate at modest
capacity — consistent with [[direction-head-data-ceiling]] (the limit is signal, not size). Produced
[[direction-head-c11]] ([256,128]).

## Sources
- `src-042412` findings.md (js_desktop:present)
