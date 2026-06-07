---
id: 01KTFQ4XDRB8BQQTZ7H35XZVK4
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-039ca3]
aliases: [market type filter, win-only model, market_type gene]
---

# Market-type filter gene

A `str_choice` gene (WIN / each-way / BOTH) deciding which race types an agent trades. Its defining
property: it does **not** change the observation or action schema, so weights stay cross-compatible.

## What it is

Market type is observed as a feature but was not used as a filter (the env played every race). The
filter gene gates which races an agent plays. Crucially a WIN-only model has the **same obs vector
shape** as a BOTH model — the `market_type_win`/`market_type_each_way` features just become a constant
`[1, 0]` — so a child can inherit weights from a BOTH parent and be trained WIN-only; the schema
version stays stable. Adding it was trivial: a YAML entry in `config.yaml:hyperparameters.search_ranges`,
with `PopulationManager` auto-handling sampling/crossover/mutation for `str_choice` types (no Python
for the genetic operators). A WIN-only model sees fewer races/day, shifting bet counts and P&L
magnitude — but the scoreboard's budget-normalisation absorbs that.

## Why it matters

A template for a behaviour-restricting gene that preserves cross-load compatibility — contrast the
[[dead-reward-shaping-genes]] (sampled but unplumbed); this one is properly wired and schema-stable.

## Sources
- `src-039ca3` lessons_learnt.md (js_desktop:present)
