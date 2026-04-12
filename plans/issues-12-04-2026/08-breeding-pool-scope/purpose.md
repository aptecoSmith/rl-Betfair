# 08 — Breeding Pool Scope

## What

1. Make the breeding pool configurable per training run: breed from
   just this run's agents, or include garaged models, or the full
   registry.
2. Investigate the "0 children bred" anomaly — if survivors >=
   population_size, breed produces nothing. Understand when this
   can happen and whether it's a bug or a config mismatch.

## Why

- Currently the breeding pool is hardcoded to "this run's agents only"
  (`run_training.py:688-692`). This is safe but may be suboptimal.
- Garaged models are the user's curated best performers. Mixing them
  into the breeding pool as potential parents could accelerate
  convergence — the new generation inherits hyperparameters from
  proven models, not just the current run's survivors.
- The user observed "30 survivors from 61 models, 0 children bred"
  which suggests a run where the breeding pool was larger than
  expected or the population target was already met by survivors.
  This needs to be debugged and understood.

## Options

| Mode | Breeding pool | Use case |
|------|---------------|----------|
| `run_only` (current default) | Only agents from this generation | Clean experiments, isolated runs |
| `include_garaged` | This generation + garaged models | Accelerate convergence by breeding with proven performers |
| `full_registry` | All active models in the registry | Maximum genetic diversity, cross-pollination between runs |
