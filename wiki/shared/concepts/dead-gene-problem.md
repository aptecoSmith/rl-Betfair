---
id: 01KTJ039CJC5XWDCTFZ8PPEKYK
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [work, research]
sources: [src-032073]
aliases: [dead-genes]
---

# Dead-gene problem

A gene that the GA samples and stores on each agent but **no downstream object reads** — so every agent trains as if the gene had its default value, and Gen-0 diversity collapses to whatever variation IS plumbed.

## What it is

Six concrete instances flagged in the [[arch-exploration]] design review:

1. `reward_early_pick_bonus`, `reward_efficiency_penalty`, `reward_precision_bonus` sampled per-agent in `population_manager.py:220`, but the env reads from `config.yaml`. **All agents train with identical reward shaping** regardless of genome.
2. Only 3 [[ppo]] knobs vary (`learning_rate`, `ppo_clip_epsilon`, `entropy_coefficient`). `gamma`, `gae_lambda`, `value_loss_coeff` are hardcoded though the trainer would read them from `hp`.
3. [[lstm]] structural params hardcoded: `num_layers=1`, no dropout, no layer norm. Both `ppo_lstm_v1` and `ppo_time_lstm_v1` share the same encoder/head shapes.
4. `observation_window_ticks` sampled but never read — pure dead gene.
5. `ppo_transformer_v1` named in `PLAN.md` but no scaffolding exists.
6. No planning layer recording what Gen-0 configs have been tried.

## Why it matters

If only 3 PPO knobs actually vary, three generations of evolution have ~no exploration budget — the GA looks busy but is searching a tiny slice. Plumbing every sampled gene to the consuming object is a precondition for ANY claim that the GA found something. This is also a generalisable wiring foot-gun documented in CLAUDE.md (see also: cohort launch flags resolved by `.get(default)` silently dropping CLI flags — same family of bug).

## Links
- [[arch-exploration]] — the plan that catalogued these.
- [[shared/index|hub]]
