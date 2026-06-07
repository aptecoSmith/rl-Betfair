---
id: 01KTJ0G7GNBAYHWXXMJZG3WBXK
type: concept
cloud: shared
status: stable
created: 2026-06-07
updated: 2026-06-07
tags: [work, research, lessons]
sources: [src-042412]
aliases: [depth-scaling-direction-head]
---

# Depth helps to 2, plateaus at 3

At fixed width=256, depth-2 lifts Pearson +0.0110 over depth-1; depth-3 then loses −0.0005 vs depth-2. The dominant architecture shape at this data scale is `[W, W/2]`.

| layers | variant | mean ρ | delta |
|---|---|---|---|
| [256]         | C1  | +0.2803 | — |
| [256, 128]    | C9  | +0.2913 | +0.0110 |
| [256, 128, 64]| C12 | +0.2908 | −0.0005 |

Adding more layers below `[W, W/2]` doesn't compound. Combined with [[width-diminishing-returns]] this localises the architecture optimum at `[256, 128]` ([[c11]]). Wider+deeper `[512, 256]` (C13) gains only +0.0007 over `[256, 128]` (within noise). Source: [[direction-head-architecture-sweep]].

[[shared/index|hub]]
