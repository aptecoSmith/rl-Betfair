---
id: 01KTFTM1P7YFC9F6E21WKRFJ8V
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-0604e6]
aliases: [lock the decision rule, probe discipline, no threshold sliding]
---

# Lock the decision rule (probe discipline)

In an experiment/probe, **locking the decision rule up front is the load-bearing constraint** — the
rule *is* the experiment, not a footnote on it.

## What it is

The strongest temptation in a probe is to slide thresholds after seeing results ("8% is basically 10%,
let's keep going"). A decision rule written into `purpose.md` / `hard_constraints.md` before the run is
the defence against that post-hoc drift, and is worth stating explicitly in plan scaffolding. A related
scaffolding lesson from the same probe: a *side-thread* experiment (real chance of being parked) needs
an explicit `staging.md` — how to commit incrementally without polluting master, where the early-exit
points are, and what survives if it's cancelled mid-flight (plans with operator buy-in don't need this,
side-thread probes do).

## Why it matters

Pre-committing the rule is what keeps a probe an actual test rather than a search for a flattering
threshold — the experimental-hygiene complement to the always-eval-on-held-out discipline.

## Sources
- `src-0604e6` lessons_learnt.md (js_desktop:present)
