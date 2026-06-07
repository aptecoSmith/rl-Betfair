---
id: 01KTHZTN0GQRVBMB5NT5KGMJKC
type: project
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [work, research]
sources: [src-009382]
aliases: [scalping-asymmetric-hedging]
---

# Scalping Asymmetric Hedging

Plan that fixes the scalping reward path so the agent can actually *produce* locked profit by sizing the two legs of a scalp asymmetrically rather than from one stake bucket.

## Goals
- Reward signal that only credits properly-sized hedges (the [[locked-pnl-min-over-outcomes]] redefinition).
- Dense per-step gradient that pushes toward proper sizing ([[worst-case-improvement-shaping]]).
- Honest diagnostics distinguishing locked vs directional vs naked pairs ([[pair-classification-badge]]).
- An action the agent can actually use to close a position at the right stake ([[close-position-action]]).

## Status
Plan written 2026-04 against the post-`98f834b` scalping reward landing (2026-04-15). Sits upstream of [[equal-profit-pair-sizing]] which became the canonical sizing formula in the env. Successor work moved on to selection (`raw_pnl_reward` magnitudes shift but stay comparable on `raw_pnl_reward` only — see `../../../../CLAUDE.md`).

## Inputs
- The pre-fix problem: equal-stake pair makes `min(win_pnl, lose_pnl) = 0` → locked floor of £0 regardless of outcome.
- Joyeuse / Aintree 2026-04-10 12:45 worked example (price 12.5 → 6.0) showing the +£130 was outcome luck not trade skill.
- [[scalping|scalping-mode]] reward path landed in commit `98f834b`.

## Notes
- Out of scope (per the source): observation features around scalping opps, per-tick stake for the opening leg, ladder walking / partial fills (the [[exchange-matcher]] (ExchangeMatcher) no-walk contract is load-bearing), and `ai-betfair` live-inference knock-ons (go via the `ai-betfair/incoming/` postbox).
- The four changes are ordered: (1)+(2) fix the reward signal, (3) makes diagnostics honest, (4) gives the agent the tool to act on the fixed signal.

[[shared/index|hub]]
