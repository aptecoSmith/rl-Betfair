---
id: 01KTFXZGMF5G5HGR62NKAPEG5B
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-0c3388]
aliases: [batched path silent drops, --batched drops features]
---

# The --batched path silently drops features

The GPU-batched training path (`train_cluster_batched`) silently drops several kwargs the canonical
`train_one_agent` threads: **predictors, feature_cache, input_norm, BC, and per-bet aux-head stamps**.

## What it is

The plan began over a known BC drop; profiling showed the drop is much wider. `train_cluster_batched`
never passes `predictor_bundle` to `_build_env_for_day`, so `use_*_predictor` resolves to the config
default (False): the `--use-race-outcome-predictor`/`--use-direction-predictor` flags were accepted and
**silently ignored**, the predictor obs slots zero-filled, and the pwin/direction/race-confidence gates
became no-ops — meaning **c1 AND c2 were predictor-less runs** (a correctness finding for their
"predictor-gated scalping" science). Also dropped: feature_cache (each agent re-runs `engineer_day`),
input_norm (hardcoded on in solo, absent in batched), and the aux-head bet-log stamps (forensic
fields). The runner only *warns* on BC + per-transition-credit, not the rest.

## Why it matters

When a code path forks (sequential vs batched), **every kwarg the canonical path threads is a
silent-drop candidate in the fork** — diff the two call sites. Re-interpret any batched-cohort science
accordingly (predictors were off). See [[gpu-batch-breaks-discrete-parity]] for why the batched path
also isn't bit-identical.

## Sources
- `src-0c3388` lessons_learnt.md (js_desktop:present)
