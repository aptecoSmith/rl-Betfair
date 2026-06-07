---
id: 01KTFTFE30DFZ3NVXJ3S5MFRXP
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-042412]
aliases: [data ceiling, generalisation ceiling, signal ceiling]
---

# Direction-head ceiling is data/signal, not capacity

The ~+0.29 Pearson plateau of the direction head is a **data/signal ceiling, not an architecture or
capacity ceiling** — more capacity or more training makes held-out *worse*.

## What it is

Two independent experiments hit the same ceiling from opposite directions: **C15** (pairwise feature
expansion, 23 → 552 features) got the best in-sample val_loss of any balanced variant but the *worst*
held-out Pearson (+0.2614, below the C0 baseline) — a clean overfit signature. **C18** (C11 trained
200 epochs / patience 20) drove val_bce down but held-out Pearson dropped +0.2921 → +0.2775. So the
binding ceiling is generalisation, not capacity; the optimal weight-space hyperplane is found early,
and `patience=5` early-stop tracks the actual generalisation peak (it is not over-cautious).

## Why it matters

Sets the prior for the deferred **C5 (full 574-d obs)**: "more inputs might find better patterns" is
exactly the C15 hypothesis that overfit, so C5 should be approached with caution — strong default
regularisation + held-out (not in-sample) early-stop. To raise the ceiling, change the *label or the
predictor*, not the head's capacity. See [[direction-head-c11]], [[literature-tricks-transfer-negatively]].

## Sources
- `src-042412` findings.md (js_desktop:present)
