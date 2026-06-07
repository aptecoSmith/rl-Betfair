---
id: 01KTG90VS25S8XJWGQB6T9KTP1
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-0fd276]
aliases: [pos_weight happy accident, pos_weight with strict gate, upward bias above threshold]
---

# pos_weight "works" with a strict gate — possibly a happy accident

In the v8 cohort, `direction_bce_use_pos_weight=true` produced positive results **despite** the known
calibration math — a tension with [[pos-weight-balanced-destroys-calibration]] that the writeup itself
flags as possibly a happy accident, pending a vanilla-BCE control (v9).

## What it is

`pos_weight` shifts the loss optimum away from the true conditional probability toward a rebalanced one;
in v8 this shows as post-BC BCE 0.75/0.79 (vs v7 single-day no-pos_weight 0.26/0.35) — i.e. *worse* raw
calibration, consistent with the existing finding. But the head's outputs being biased UP slightly seems
to align with the gate's strict 0.85 threshold: the rebalanced operating point puts more genuine
positives ABOVE 0.85 while the marginal-density true-negatives stay below. So the calibration penalty and
the strict gate happen to cancel favourably. v9 (vanilla BCE) tests whether this is a happy accident or
whether un-weighted BCE calibrates even better.

## Why it matters

A caution against reading "pos_weight helped in the cohort" as contradicting "pos_weight destroys
calibration" — both are true; they operate at different points (raw probability vs post-gate selection).
The clean lesson stays [[pos-weight-balanced-destroys-calibration]] (check Brier, not just ranking); this
note records that a strict downstream gate can mask — not fix — the miscalibration, which is exactly why
the control run matters.

## Sources
- `src-0fd276` lessons_learnt.md (js_desktop:present)
