---
id: 01KTGNYHY1YP9GR0FFGNTXEP1E
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-19293f]
aliases: [information probe, is the feature load-bearing, mute the obs columns, inert feature ablation, direction signal value]
---

# Ablate a feature to test whether it's load-bearing

An information-probe methodology: to find out whether an obs feature actually earns its place, **ablate it**
— mute its columns or disable its predictor — and check whether the metric moves. If the ablated cell
matches the base, the feature is inert.

## What it is

Round 5 Group 5 tests whether the C11 direction head's obs signal is load-bearing via two cells:
`--direction-signal-gain 0` (mute the 12 direction obs columns) and removing `--use-direction-predictor`
entirely (free its obs columns). "Decision: if the no-direction cell matches the base on day_pnl, the
direction predictor's obs signal isn't earning its place." This is the same dead-code question as
[[sampled-not-used-gene]] but applied to a *feature* rather than a *gene*: presence in the obs vector
proves nothing; only a measured behavioural delta when you remove it does.

## Why it matters

A feature can be plumbed, logged, and confidently believed-in while contributing nothing — the only proof
of value is an ablation that degrades the metric. Cheap to run (two cells), high-leverage (frees obs
columns and compute if inert). The runtime companion to the gene-use test: ablate to confirm load-bearing,
don't assume from presence. Relevant to whether the direction predictor's slice
([[direction-head-feature-slice]]) pays its way once the rest of the recipe is strong.

## Sources
- `src-19293f` purpose.md (js_desktop:present)
