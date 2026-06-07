---
id: 01KTGC1SK8MRG8K014BBB73C08
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-106b41]
aliases: [probe-shaped budget, probe before cohort, bite criteria, cheap probes first]
---

# Probe before cohort (the probe-shaped budget)

A compute-allocation discipline: before committing ~28h to a full cohort that might fail to bite, spend a
fixed budget on many small **probes** (5 agents × 1 gen × 3 train + 3 eval days, ~1h wall each) and only
escalate a lever to full cohort once it **clearly bites**.

## What it is

After tnv3 was rejected on mechanism, the operator picked a probe-shaped budget — "spend 20h trying
multiple small probes ... and only commit a full cohort once a lever clearly bites." Bite criteria are
lever-specific with a common floor (`locked ≥ +£70/d` and `naked_span ≤ baseline + 50`). The campaign ran
~19 probes this way; the consolidated three-way classification of intervention authority at probe scale:
**env-side action masks — strong; BC supervised — strong; shaped rewards — no authority at probe scale.**

The probes also exposed the method's main failure mode in two directions: a non-binding knob reads as
"no signal" (E6 pair-budget cap of 30 never bound; R4 depth floor £10 was inert), and a probe bite can
fail to compound at cohort scale ([[probe-to-cohort-regression]]). So a probe result needs a "did the
gate actually fire?" check and a cohort confirmation before it's trusted.

## Why it matters

Cheap falsification first: probe A/O/A2 spent ~hours conclusively ruling out magnitude and training-length
before any 28h cohort burned on them, and stopping tnv3/E3-cohort early on mechanism analysis saved ~22h
and ~14h GPU respectively. Pairs with [[gradient-delivered-ppo-unresponsive]] (what the probes found) and
the always-eval-holdout discipline. The durable mechanism for *queuing* these on this Windows/git-bash
box turned out to be plain bash polling chains, not "smart" daemons.

## Sources
- `src-106b41` EXPERIMENTS.md (js_desktop:present)
