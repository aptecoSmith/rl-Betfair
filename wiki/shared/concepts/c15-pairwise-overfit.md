---
id: 01KTJ0EVN7B1DA039H9XD2ZHV4
type: concept
cloud: shared
status: stable
created: 2026-06-07
updated: 2026-06-07
tags: [work, research, lessons]
sources: [src-042412]
aliases: [input-ceiling-is-generalisation, C15]
---

# C15 pairwise overfit — input ceiling is generalisation, not capacity

C15 expanded the 23 per-runner inputs to `concat([x, outer_product(x, x).flatten()])` = 552 features, then `[256, 128]` [[mlp]]. **Best in-sample val_loss of any balanced variant** (0.881 vs C9's 0.945), **worst held-out Pearson** (+0.2614, below C0 baseline). Pure overfit signature.

## What it is

The hypothesis the sweep was testing: "if architecture saturates at +0.29 Pearson, maybe the input is the ceiling — give it more derived features". The result: more capacity → better train fit, worse generalisation.

C18 reinforced this from the opposite direction — same [[c11]] architecture, but trained for 200 epochs with patience=20: val_bce drove from 0.426/0.400 down to 0.410/0.386, but held-out Pearson dropped +0.2921 → +0.2775. **Two independent experiments hit the same ceiling — one via more features, one via more training.**

## Why it matters

- The 23-d lean obs is NOT a representational ceiling. Adding derived features over the same inputs makes generalisation strictly worse.
- The implication for deferred **C5** (full 574-d input): expected value drops sharply. The 574-d obs contains the same predictor columns plus more — no a priori reason to think it'd avoid the same failure mode. If C5 is run, it should have strong default dropout and held-out-validated early-stop, not in-sample.
- The default `--patience 5` early-stop is tracking the actual generalisation peak; it is NOT over-cautious.

## Links
- [[direction-head-architecture-sweep]] — the source sweep.
- [[c11]] — the calibrated winner this finding protects.
- [[shared/index|hub]]
