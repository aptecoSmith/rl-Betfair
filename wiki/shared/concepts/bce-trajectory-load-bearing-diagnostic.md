---
id: 01KTG18MW0GWWVBMA74XX1G24T
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-0c84fe]
aliases: [BCE trajectory diagnostic, flat BCE means deeper bottleneck]
---

# Direction-BCE trajectory is the load-bearing diagnostic

Whether the [[direction-head-feature-slice]] input fix worked is read off the **direction-BCE
trajectory**: a monotone decrease across generations on **both** arms (the head learns from the BCE
auxiliary even with the gate disabled) is the success signal.

## What it is

BCE is the load-bearing diagnostic because it isolates the head's learning from the gate/reward path.
If BCE is **still flat after phase 15**, the bottleneck is even deeper than feature representation —
either the labels carry less signal than the probes claimed, or there is an upstream feature-engineer
bug — and that outcome triggers a separate **diagnostic** plan, not a phase-16 architecture change.
The success bar also requires mature rate ≥ 35% (above the 34.8% break-even) and positive held-out
`eval_day_pnl` on the gate-on arm, but BCE-decreasing is the diagnostic that the *input pathway* fix
landed.

## Why it matters

Pick a metric that isolates the change you made (BCE isolates head-learning from the gate), and
pre-commit what a null result means (flat BCE → deeper bottleneck → diagnostic plan, not more
architecture). Connects the representation fix ([[lstm-compression-bottleneck]]) to the signal-ceiling
question ([[direction-head-data-ceiling]]).

## Sources
- `src-0c84fe` purpose.md (js_desktop:present)
