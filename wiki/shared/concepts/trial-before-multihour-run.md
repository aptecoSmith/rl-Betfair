---
id: 01KTGJS2NPCJMY8NTH8F6FG5F7
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-184d90]
aliases: [trial run insurance, scaled-down pipeline first, calibrate threshold before full run, session-scoped paths]
---

# Trial-run a scaled pipeline before a multi-hour run

An ops discipline for expensive runs: before a multi-hour GPU run whose output drives a pass/fail
decision, run the **whole pipeline on a scaled-down dataset first** — finding a serialisation bug or a
missing log field on a 23-minute trial is far cheaper than on a 4-hour run.

## What it is

The Session-9 trial (2 train / 1 test day) cost one invariant-threshold calibration and nothing else
before the 42-minute full shakeout. Two attached disciplines: (1) **calibrate thresholds on the trial,
before launching the full run, not after reading its result** — relaxing the dead-gene correlation
threshold from −0.05 to −0.01 *before* the full run is sound; doing it after would be tuning on the test
set; (2) **session-scoped paths** stop a shakeout colliding with the live registry — rewrite
`paths.logs`/`registry_db`/`model_weights` under a session tag, and remember `ModelStore`'s `bet_logs_dir`
is NOT derived from config (forget it and bet logs land in the shared dir, defeating isolation).

## Why it matters

Cheap insurance scales the failure cost down before you pay the full compute. The before-vs-after ordering
of threshold calibration is the line between honest pre-registration and tuning on the test set — the same
held-out discipline as [[fc0-insample-mirage]], applied to invariant thresholds. Pairs with
[[probe-before-cohort-budget]] (cheap falsification first) and [[cost-model-from-per-phase-walls]] (size
the run from measured cost).

## Sources
- `src-184d90` lessons_learnt.md (js_desktop:present)
