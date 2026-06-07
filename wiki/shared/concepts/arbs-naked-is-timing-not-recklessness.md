---
id: 01KTGP9R7C8AJS6S6AK7SYBSNP
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-1b344c]
aliases: [arbs naked terminology trap, naked = passive didn't fill, timing-out scalps, active management not harsher penalty]
---

# "Arbs naked" is timing, not recklessness (terminology trap)

A naming trap that distorts the whole reward-design conversation: in this codebase "arbs naked" does **not**
mean the agent deliberately placed an unhedged bet — it means the agent placed a *pair* whose passive
counter-order never filled before race-off. The aggressive leg then settled directionally by accident.

## What it is

85.5% of one agent's (`ef453cd9`) pair attempts ended this way. "That's not reckless behaviour; it's
timing-out scalps." The implied fix is therefore **active management (re-quote) + fill-probability
awareness, not punishing naked exposure harder** — the agent isn't choosing risk, its passive simply
didn't get caught up to in time. Punishing the naked outcome teaches the agent to stop opening, when the
real lever is making the passive fill.

## Why it matters

Re-frames a metric that *looks* like reckless directional gambling into a *fill-rate* problem — the same
correction that later drove the close-feasibility gate and the fixed-tick diagnosis
([[fixed-tick-passive-causes-force-close]]). It's why the naked channel is handled asymmetrically in the
reward ([[naked-asymmetry-per-pair]]) rather than just penalised, and a sibling of
[[force-close-is-a-crutch]] (a symptom mislabelled as a behaviour to punish). Read the definition before
shaping against the name.

## Sources
- `src-1b344c` lessons_learnt.md (js_desktop:present)
