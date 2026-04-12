# 09 — Adaptive Breeding & Mutation Controls

## What

1. Detect when a generation produces no good candidates (all below
   discard thresholds, or best composite score below a configurable
   floor). Offer configurable responses:
   - **Persist**: keep the cohort, breed normally, hope next gen improves
   - **Persist + boost mutation**: keep the cohort but crank up mutation
     rate to shake up the hyperparameter space
   - **Inject top performers**: breed in garaged/scoreboard top models
     as parents alongside (or replacing) the weakest survivors

2. Make mutation rate configurable and adaptive in the wizard:
   - Base mutation rate (currently hardcoded path from config 0.3)
   - Adaptive mutation: automatically increase mutation rate when the
     generation underperforms. Configurable ramp (e.g. +0.1 per bad
     generation, capped at 0.8)
   - Per-run override in the wizard

## Why

- Currently if a generation is rubbish, the system breeds from rubbish
  and hopes for the best. There's no mechanism to detect "this whole
  cohort is bad" and respond differently.
- Mutation rate 0.3 is fixed in config.yaml. There's no way to tune it
  per run, and no adaptive behaviour. In biology, environmental
  pressure increases mutation — the same principle applies here.
- The wizard has no mutation controls at all. It shows the genetics
  info (step 4) as read-only config.yaml values.
