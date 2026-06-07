---
id: 01KTFXVREPN6ZK1KJKJ07JCFZJ
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-094c38]
aliases: [KL explosion root cause, stateful rollout stateless update, recurrent PPO mismatch]
---

# PPO KL explosion: stateful-rollout / stateless-update mismatch

The root cause of the cohort's KL explosion (approx_kl median ~12,700 vs a 0.03 threshold): a
**structural mismatch between stateful rollout and stateless PPO update** — not advantage-norm,
reward-centering, or force-close.

## What it is

During rollout the policy runs with a carried `hidden_state` (LSTM cell / transformer rolling buffer)
across every tick; during `_ppo_update` the same policy is called **without** `hidden_state`, so the
forward pass is evaluated on a zero-initialised state per mini-batch transition. `old_log_probs`
(stateful) and `new_log_probs` (stateless) come from different distributions, so the reported
`approx_kl` measures "rollout-policy vs stateless-lobotomised-policy" distance — **large by
construction**, not a KL between policies the agent deploys. The smoking gun: KL **grows with episode
index** (Spearman ρ +0.435, the strongest single correlation), because every accepted gradient step
drifts the stateless policy further from the rollout-collection distribution. Fix: **store the hidden
state per transition at rollout time** and pass it through both the mini-batch loss and the KL
diagnostic (cheapest retrofit) — or sequence-batched BPTT.

## Why it matters

This is exactly the failure the CLAUDE.md "Recurrent PPO: hidden-state protocol on update" invariant
now guards against (`Transition.hidden_state_in`). It starved PPO entirely — see
[[ppo-starved-by-kl-early-stop]]. The five session hypotheses were all refuted
([[ppo-kl-hypotheses-refuted]]).

## Sources
- `src-094c38` findings.md (js_desktop:present)
