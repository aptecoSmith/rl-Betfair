---
id: 01KTG1DSM7EMEA9KN2JM3SQZ3E
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research]
sources: [src-0d0f55]
aliases: [argmax-eval, deterministic eval, argmax action selection]
---

# Argmax-eval (deterministic eval action selection)

An opt-in **deterministic / argmax action-selection path** for eval rollouts that removes
eval-sampling noise at source: `action_dist.sample()` → `argmax(logits)`, and `stake_dist.sample()` →
`Beta.mean`.

## What it is

Gated by a single `deterministic` bool through `RolloutCollector.collect_episode`, with an
`--argmax-eval` CLI flag on the cohort runner / standalone train / `reevaluate_cohort.py`. **Default
mode is stochastic everywhere** (byte-identical to pre-plan), and **training rollouts always stay
stochastic** — PPO is a stochastic-policy algorithm; deterministic-trained PPO is a different algorithm
(out of scope). Design choices: `argmax(logits)` not `argmax(probs)` (equivalent under monotone
softmax; logits are already exposed and masked actions are `-inf` so they lose); **`Beta.mean` not
`Beta.mode`** (mean is always defined for α,β>0; mode needs α,β>1). The `log_prob` invariant
(`transition.log_prob = action_dist.log_prob(action)`) is preserved under both modes, and there are
**no env edits** — the env sees an action int + stake float as before.

## Why it matters

The fix for [[eval-sampling-variance-dominates]]: a reproducible, cheaper eval signal. It is a
*measurement*-signal change, orthogonal to and stacking with the selection-signal work
([[selection-vs-measurement-signal]]); validated by [[argmax-eval-validation-gate]].

## Sources
- `src-0d0f55` purpose.md (js_desktop:present)
