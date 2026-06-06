---
id: 01KTFBST2SQ45ES2KYAH4F0WFG
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research]
sources: [src-042412]
links: [{to: pos-weight-balanced-destroys-calibration, type: derived-from}, {to: direction-head-data-ceiling, type: see-also}]
aliases: [C11, direction_head sweep winner, sweep_c11]
---

# Direction-head C11 (sweep winner)

The architecture promoted out of the 2026-05-24 direction-head sweep:
`LayerNorm → Linear(23, 256) → ReLU → Linear(256, 128) → ReLU → Linear(128,
2)`, trained with **pos_weight = 1** (unweighted BCE).

## What it is

C11 combines round-2's wider+deeper architecture winner (C9, `[256, 128]`)
with round-1's calibration finding (unweighted BCE). Pareto-best across all
20 variants on the plan's acceptance criteria:

| metric | C11 | C0 (baseline `[64]`, balanced) | delta |
|---|---|---|---|
| mean Pearson | +0.2921 | +0.2719 | +0.0202 (+7.4%) |
| mean ROC AUC | 0.7098 | 0.6976 | +0.0122 (+1.7%) |
| mean Brier | 0.1433 | 0.2282 | **−0.0849 (−37.2%)** |

The headline is the **calibration win** — pred mean tracks the empirical
positive rate to within 1–2 points across all 10 held-out days, where C0
runs ~2.5× over-confident. C11 leads or ties on per-day Pearson across the
entire eval pool (only beaten on 2026-04-25 by C13, by 0.0023).

## Why it matters

C11 is dominant whenever the actor consumes the head's output as a
probability — every threshold-based decision (the `direction_gate_threshold`
gene becomes meaningful when "25%" actually means 25%, not 11%). Promoted
to the next cohort's `--direction-head-manifest`; mutually exclusive with
`--enable-gene direction_prob_loss_weight` and
`--enable-gene bc_direction_target_weight` per the shared-direction-head
plan's hard constraints.

Round-4 ablations on C11 also showed that every "standard improvement"
(AdamW, GELU, label smoothing, longer training, focal loss) **transferred
negatively** — C11's training recipe is already at the local optimum at
this data scale (1M samples, ~40k params).

## Sources
- `src-042412` findings.md (js_desktop:present)
