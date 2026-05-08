---
plan: rewrite/phase-15-direction-head-feature-slice
parent_purpose: ./purpose.md
---

# Lessons learnt

Append-only journal. Entries land as work completes.

## Inherited lessons (read before any session)

- `plans/rewrite/phase-14-direction-gate/lessons_learnt.md` —
  the full phase-14 journey. The "Architectural lesson —
  per-runner head, NOT shared output" entry is the direct
  precursor to phase 15: phase 14 fixed the head's *output*
  side; phase 15 fixes the *input* side.

- `plans/rewrite/phase-13-directional-scalping/lessons_learnt.md`
  — the phase-13 NULL writeup. The lesson "probe before cohort"
  is doubly load-bearing here: the supervised probe was the
  diagnostic that motivated phase 15.

## Quantitative ground truth (carried over from phase 14)

The phase-14 ground truth applies unchanged. See
`plans/rewrite/phase-14-direction-gate/lessons_learnt.md`
"Quantitative ground truth" section. Key numbers:

- Per-pair P&L: matured +£3.37, force-closed -£1.80,
  break-even mature rate 34.8%.
- OOS top-quintile lift on per-runner-slice probe with 10
  augmented features: 24-94×.
- 3 of 3 OOS days profitable at empirical cost ratio with
  T ∈ [0.90, 0.95].

The 24-94× number is the load-bearing input-pathway evidence.
If phase 15 reproduces even half of that lift inside the
cohort, the success bar (mature rate ≥35%) is reachable.

## Methodological lesson — input pathway matters more than head capacity

Phase 13 fixed the labels (added direction labels). NULL.
Phase 14 fixed the head's output structure (single Linear →
per-runner MLP). Probe lift went up; cohort BCE may still be
flat (TBD when probeAB lands).

The phase-14 sense_check (item 3) called this exact risk
pre-cohort: "the per-runner MLP fed by the SAME shared
`lstm_last` may have the same issue [as phase 13]." Phase 15
exists because that risk was foreseen and pre-staged.

**Lesson:** when a bottleneck is suspected to be an input
pathway, fix the input pathway directly. Don't increase head
capacity, don't increase training time, don't sweep
hyperparameters. The probe already isolated the variable.

## Methodological lesson — single-knob plans

Phase 14 changed three things at once (head architecture,
features, gate). Each was probe-validated independently, but
the cohort outcome can't cleanly attribute residual lift to
any one of them. Phase 15 changes ONE thing: which input the
direction head reads. If S03 delivers, the input pathway is
the load-bearing fix. If S03 doesn't, the residual is a
known-bounded gap to chase in one more plan.
