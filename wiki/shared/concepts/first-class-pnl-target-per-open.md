---
id: 01KTGJG33WJS9MNJJQJWVGT58D
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-1663dd]
aliases: [target-£-pair sizing, first-class pnl target, target_pair_pnl, £-target per open]
---

# First-class £-target per open (Session 01)

A mechanics change: give the agent a direct **£-target per pair** instead of a tick-space spread. Today
the per-runner `arb_spread` action (0..1) maps to `arb_ticks`, so the agent has no first-class £-target
action — it must learn `(stake, back_price, arb_ticks) → expected_£_profit` from delayed cash signal
alone, a hard mapping under a noisy gradient.

## What it is

Session 01 re-interprets the same `arb_spread` action dim (0..1) as `target_pair_pnl ∈ [£0.20, £5.00]`
linear; the env solves for the lay-price that, given equal-profit sizing and the agent's chosen stake,
produces the target P&L on lock, then quantises to tick. It mirrors how a human scalper thinks — the
operator: "in general, I'm looking to make a single £1 of profit per trade." The action *space* is
unchanged (same dim/range/gene schema), only the env's interpretation — but pre-plan policies can't
cross-load (the dim's semantics differ), so it's an arch-hash break. If the solved lay-price lands inside
the matcher's ±50% junk filter or beyond available-to-lay, the open is **refused** (the refusal is the
signal — no silent fallback to the old behaviour); sizing stays [[equal-profit-sizing]].

## Why it matters

Reframes the action into the trader's natural unit (profit target) so the policy doesn't have to discover
the price↔£ mapping under delayed reward — the same "feed the derived quantity, don't make the policy
rediscover it" philosophy as [[arb-as-observation-feature]]. Half of the minimum scalping toolkit
([[minimum-scalping-toolkit-stacking]]); paired with the stop-close ([[env-stop-close-not-agent-learned]]).
Its behavioural metric is the policy-close fraction.

## Sources
- `src-1663dd` purpose.md (js_desktop:present)
