---
id: 01KTGF3ZB8SHXGV9N3MQ71GWRE
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-15a08d]
aliases: [0 children bred, survivors >= population_size, breeding pool scope, no offspring]
---

# The "0 children bred" anomaly

A GA-runner gotcha: a generation produces no offspring when **`len(survivors) >= population_size`** —
there are no free slots for children, so breeding is a no-op even though the run looks healthy.

## What it is

It occurs when the survivor set fills or overfills the population: a shakeout script using a non-standard
population size, a selection `top_pct` too generous relative to `population_size`, or a run that started
with more agents than `population_size` (e.g. a 21-agent shakeout under a different pop config). Separately,
the investigation confirmed the breeding pool IS correctly scoped to the current run's agents — the "61
models" the operator saw in a log was the scoreboard's full ranking output, not the breeding pool itself.

## Why it matters

A silent failure mode: the cohort runs to completion but never actually evolves, and the symptom (a large
"models" count in a log) is misleading because it's the scoreboard, not the pool. Check
`survivors` vs `population_size` and the `top_pct` × pop arithmetic before trusting that breeding happened
— a relative of [[degenerate-ga-search]] (a GA that looks like it's searching but isn't). Companion to
[[garaged-models-into-breeding]].

## Sources
- `src-15a08d` lessons_learnt.md (js_desktop:present)
