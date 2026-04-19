# Lessons learnt — Arb Curriculum

One entry per surprise (bug, design miss, behavioural
finding) uncovered during this plan's sessions. Dated.
Most recent first.

Format per entry: **Date · Session · Title**, then a short
paragraph on what surprised us, what was wrong, and what
the fix or policy change is.

---

## 2026-04-19 · Pre-plan · Inherited footgun: per-agent BC, never shared

Transferred from
`plans/arb-improvements/lessons_learnt.md`. The prior
(never-shipped) BC-pretrainer design discovered that
sharing BC-pretrained weights across a population was a
severe failure mode — all agents converge to the same
local region, GA diversity collapses immediately, and
the population can't recover. Hard-constrained here as
§16 of `hard_constraints.md`. Every Session 04 test that
touches BC must verify per-agent independence (seed-
divergent parameters after BC).

## 2026-04-19 · Pre-plan · BC ↔ target-entropy controller interaction

Not a bug yet — a predicted interaction logged before
Session 04 lands. After BC, policy entropy is LOW
(confident on oracle targets). `entropy-control-v2`
controller (target 150) will aggressively boost `alpha`
to push entropy up, undoing part of BC. The plan's
handshake: anneal `target_entropy` from post-BC measured
entropy up to 150 over `bc_target_entropy_warmup_eps`
episodes (default 5). If Session 04 lands and validation
shows BC's effect still dissolves over the warmup period,
log the finding here and revisit the anneal schedule.
