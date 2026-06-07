---
id: 01KTG90VS00EHPHCTSGBC569PQ
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-0fd276]
aliases: [fix the input pathway, single-knob plans, probe isolates the variable, head capacity vs input]
---

# Fix the input pathway, not head capacity (+ single-knob plans)

The methodological spine of the direction-head arc: phase 13 fixed the **labels** (added direction
labels) → NULL; phase 14 fixed the head's **output** structure (single Linear → per-runner MLP) → probe
lift up; phase 15 fixed the **input** pathway (feed raw per-runner features directly) → the positive
result. When a bottleneck is an input pathway, fix the input pathway directly.

## What it is

The lesson, stated at phase 15's open: "when a bottleneck is suspected to be an input pathway, fix the
input pathway directly. Don't increase head capacity, don't increase training time, don't sweep
hyperparameters" — the supervised probe had already isolated the variable (24–94× OOS top-quintile lift
from the per-runner-slice probe), so the fix was known, not a search.

Paired with a **single-knob discipline**: phase 14 changed three things at once (head architecture,
features, gate), so its cohort outcome couldn't attribute residual lift to any one. Phase 15 changes ONE
thing — which input the direction head reads — so a positive result cleanly credits the input pathway and
a null leaves a known-bounded gap for one more plan.

## Why it matters

A reusable debugging stance for ML pipelines: a probe that isolates the variable converts a
hyperparameter search into a targeted fix, and one-change-per-plan keeps attribution clean. The concrete
fixes the stance produced are [[direction-head-feature-slice]] (the input rewire),
[[layernorm-for-raw-obs-heads]] (making raw input usable), and [[freeze-bc-head-post-pretrain]] (the v8
payoff). Same "probe before cohort" lesson that recurs across the rewrite phases.

## Sources
- `src-0fd276` lessons_learnt.md (js_desktop:present)
