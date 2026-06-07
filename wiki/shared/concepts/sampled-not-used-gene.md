---
id: 01KTGJS2NM7QNKJDHYTNTPSKAN
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-184d90]
aliases: [sampled not used, dead gene, every gene needs a use test, correlation is sensitivity not significance]
---

# Sampled ≠ used (and correlation is the detector)

The foundational GA foot-gun: the genetic algorithm can happily mutate a gene that **no downstream code
ever reads**. `reward_precision_bonus` sat in the schema for weeks, got mutated every generation, and
never changed any agent's reward. The rule: every gene must have a test that asserts the env (or trainer)
**actually uses** the sampled value — a grep should find a "read from hp" site AND a "passed to downstream
consumer" site.

## What it is

The runtime detector is a gene↔outcome correlation: "zero correlation is the bug" (a flat `r` means the
gene is dead code). But read `r` carefully — it scales with training volume: `reward_efficiency_penalty`
vs `mean_bet_count` was `r=−0.038` on a 2-day trial and `r=−0.255` on a 4-day run (same sign, ~7× the
magnitude, because each agent saw ~2× more PPO updates). So **the correlation coefficient is a
sensitivity metric, not a significance metric** — don't tighten a dead-gene threshold past what a single
generation can deliver (n=21 has sampling noise ≈ 1/√n ≈ 0.22); any unambiguously-nonzero `r` of the
right sign satisfies "the gene is wired."

## Why it matters

This is the general principle behind [[dead-reward-shaping-genes]] and the same wiring vigilance as the
Path-A launch foot-gun: a knob can look alive (sampled, mutated, logged) while being inert. The two-site
grep + a both-ends use-test is the cheap structural guarantee; the correlation check is the runtime
confirmation. Read sensitivity, not significance, off a single-gen cohort.

## Sources
- `src-184d90` lessons_learnt.md (js_desktop:present)
