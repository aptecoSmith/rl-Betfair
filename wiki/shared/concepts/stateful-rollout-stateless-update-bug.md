---
id: 01KTFBZF43GE505FWXA0KDEABE
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-094c38]
links: []
aliases: [KL explosion root cause, recurrent PPO hidden state bug, stateful/stateless mismatch]
---

# Stateful-rollout vs stateless-PPO-update bug

The KL explosion observed during arb-signal-cleanup-probe was caused by
a **structural mismatch**: rollout runs the recurrent policy with a carried
`hidden_state`; `_ppo_update` re-runs the same policy with **no
hidden_state**, so the forward pass zero-inits per mini-batch transition.
`old_log_probs` (stateful) and `new_log_probs` (stateless) are drawn from
two different distributions, and `(old − new)` measures the gap between
the rollout policy and a stateless-lobotomised version of the same
network — not anything the agent ever deploys.

## Diagnosis evidence

3,793 PPO updates across cohorts W and A — every single one tripped KL
early-stop after epoch 0:

| stat | KL |
|---|---|
| min | 3.62 |
| median | 12,740 |
| max | 4,620,172 |

Against `kl_early_stop_threshold = 0.03`. The minimum is **120× the
threshold**.

Smoking gun: Spearman ρ(episode_idx, KL) = **+0.435**. Every PPO update
that accepts a tiny gradient drifts the stateless-policy further from the
rollout distribution, so the gap accumulates monotonically — exactly what
the stateful/stateless mismatch predicts. Hypotheses H1–H5 from the
session prompt (advantage-norm off, reward-centering units, alpha drift,
force-close magnitude, entropy explosion) were all refuted by the
correlation table — wrong signs or zero correlation.

## Why it matters

Under this condition the policy effectively takes **one mini-batch
gradient step per rollout** (epochs 1..3 systematically skipped). All
"learning" attributed to PPO during the probe is BC pretrain + a thin
gradient channel + the α-controller (the one PPO-channel hook that
isn't gated by KL). Scoreboard rows from that period measure BC quality
+ genetic selection, not PPO-trained skill.

## Fix path

Store `hidden_state` on `Transition` at rollout time (captured BEFORE the
forward) and thread it through both the mini-batch surrogate forward and
the KL-diagnostics forward. This is the cheapest of the three literature-
standard fixes (sequence-batched BPTT and per-epoch sequential re-rollout
are the alternatives). The action stored must be the **un-clipped**
sampled action so `dist.log_prob(stored)` at update matches rollout. Now
landed in `agents/ppo_trainer.py` per the recurrent-PPO hidden-state
protocol section of CLAUDE.md.

## Sources
- `src-094c38` findings.md (js_desktop:present)
