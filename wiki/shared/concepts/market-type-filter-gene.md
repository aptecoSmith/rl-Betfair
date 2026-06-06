---
id: 01KTFBMST6VWHW7S31172B7T5R
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research]
sources: [src-039ca3]
links: []
aliases: [market_type filter, WIN-only gene]
---

# Market-type filter gene

A per-agent gene that restricts which races the env plays — WIN-only, EW-
only, or BOTH — without changing the observation/action schema.

## What it is

Implemented as a `str_choice` entry in `config.yaml:hyperparameters.search_
ranges`; the `PopulationManager` handles sampling/crossover/mutation
automatically — no Python changes to the genetic operators. The schema is
deliberately untouched: a WIN-only model has the same obs vector shape as a
BOTH model, with the `market_type_win` / `market_type_each_way` features
always reading `[1, 0]`.

The schema invariance is **load-bearing for inheritance**: a child can
warm-start weights from a BOTH parent and then train as WIN-only without
any cross-arch surgery. Stability of the obs schema across the population
is what makes weight-threading meaningful.

A WIN-only model sees fewer races per day on average (some days are 100%
WIN, some are heavily EW), which affects raw bet counts and P&L magnitude;
the scoreboard normalises by budget so the comparison stays fair.

## Why it matters

This was the first cohort gene to alter the **data the agent trains on**
without altering its architecture — a category distinct from reward shaping
and PPO knobs. It proved the cross-compatible-weights pattern that later
PBT promotion-ladder breeding relies on (warm-start a child from a parent
trained on a different data slice).

## Sources
- `src-039ca3` lessons_learnt.md (js_desktop:present)
