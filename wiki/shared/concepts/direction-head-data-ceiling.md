---
id: 01KTFBST2SN7JN6ZKXDEWPY6SS
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-042412]
links: [{to: direction-head-c11, type: supports}]
aliases: [data ceiling not capacity, C15 overfit, C18 overfit]
---

# Direction-head ceiling is data/signal, not capacity

Two independent experiments in the direction-head sweep hit the same
generalisation ceiling from opposite directions, providing unusually clean
evidence that the binding constraint at ~+0.29 Pearson is **the data, not
the architecture**:

- **C15 (more features):** the 23-d per-runner input was expanded with
  outer-product features (552-d) under a `[256, 128]` MLP. Best in-sample
  val_loss of any balanced variant (0.881) — clear capacity gain. Held-out
  Pearson: +0.2614, last place, below the C0 baseline of +0.2719.
- **C18 (more training):** C11's architecture trained for 200 epochs with
  `patience=20` instead of 50/5. Val_bce kept improving (0.426/0.400 →
  0.410/0.386). Held-out Pearson dropped from +0.2921 to +0.2775.

## Why it matters

Both runs make train loss better and held-out worse — the canonical
overfit signature. C11's tight `patience=5` early-stop is not over-
cautious; it's tracking the actual generalisation peak. The implication is
that **adding capacity or compute will not move the ceiling** — the next
gain (if any) must come from new signal: a different label, a different
predictor, or genuinely new features.

## What this means for deferred C5 (full 574-d input)

The "give the head more inputs and it might find better patterns"
hypothesis behind the deferred C5 variant should be approached with much
more caution after C15. If C5 is run it should default to strong
regularisation (dropout sized to keep effective capacity near C11's) and
a tighter held-out-based early-stop, not in-sample.

## Sources
- `src-042412` findings.md (js_desktop:present)
