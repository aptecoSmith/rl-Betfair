---
id: 01KTJ039CVRMJTA9FQZGXNF9XP
type: project
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [work, research]
sources: [src-032073]
aliases: [arch-exploration, Architecture & Hyperparameter Exploration]
---

# Architecture & Hyperparameter Exploration

Plan to make the GA's Gen-0 search actually populate meaningful diversity across reward shaping, [[ppo]] hyperparameters, [[lstm]] hyperparameters, and architecture — so three generations of evolution actually contain signal.

## Goals
- Every gene in the mutation schema is plumbed to the object that uses it ([[dead-gene-problem]]).
- Gen-0 is deliberately planned with even architecture coverage and recorded ranges.
- A third architecture `ppo_transformer_v1` exists and can be mixed into populations.
- Each phase of the rollout delivers a usable intermediate result.
- Every new knob is exposed in the [[ui]].

## Status
Plan written shortly after commit `e76ac98` (the phantom-profit fix), when honest rewards finally made positive-PnL search worth attempting. Predates the v2 cohort rewrite and the PBT breeding mechanism that now drives population search.

## Inputs
- The 6 design-review findings cataloged as [[dead-gene-problem]].
- Hard constraints carried over from CLAUDE.md (see [[reward-invariants]]).

## Notes
- All shaping additions must respect: zero-mean under random policy, `raw + shaped ≈ total_reward`, no walking the [[ladder]], no peeking at unfiltered top-of-book.
- The [[exchange-matcher]] single-price / [[ltp]]-filter / max-price rules are load-bearing.

[[shared/index|hub]]
