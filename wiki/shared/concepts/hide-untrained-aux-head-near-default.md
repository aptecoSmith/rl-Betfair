---
id: 01KTGP9R7QMR4ZJSTW6WD80Z6E
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-1b344c]
aliases: [hide near init default, untrained head UI gating, non-null but meaningless, self-flipping display]
---

# Hide an untrained aux-head value near its init default, not just on null

A UI/UX lesson: an untrained auxiliary head emits a **non-null but meaningless** value (its init constant),
so a "don't render if null" check isn't enough — the UI must hide values near the init default, or it
trains operators to ignore a signal that later becomes meaningful.

## What it is

Before activation, the fill-prob head "outputs a non-null value on every bet — it's just always ≈ 0.5
because the weights are zero." Without a ±0.02 near-default band, every row would render a middle-of-the-road
"Med" chip during the weeks between the head landing and being trained, "training operators to ignore it."
The fix is data-driven and self-flipping: hide within the near-default band, so once the head is trained
and values spread away from 0.5, chips start appearing automatically — no manual switch.

## Why it matters

A null check guards against *absent* data; it does nothing for *present-but-uninformative* data. Any UI
consuming a model output that has a meaningful init constant (0.5 for a sigmoid, mid-range for a Beta —
cf. [[continuous-head-inits-at-range-center]]) should suppress near that constant and reveal on spread, so
the display tracks the model's actual informativeness. Same "don't show a constant as if it were a signal"
discipline as hiding a benign ~0.5 aux column in the actor.

## Sources
- `src-1b344c` lessons_learnt.md (js_desktop:present)
