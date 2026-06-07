---
id: 01KTGC8207GAH767TGJT3J64Q8
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-106b41]
aliases: [GAE discount kills late credit, settle-time credit, 0.95^150, per-tick credit delivery]
---

# GAE discount kills settle-time credit (E1)

Why a reward delivered at race-settle can't teach a decision made hundreds of ticks earlier: the GAE
discount shrinks it to nothing by the time the gradient reaches the action. `close_signal_bonus` landed
at settle, so a close at T−200 saw its bonus ~150+ ticks later.

## What it is

With γ=0.99, λ=0.95 the gradient reaching the close decision is `0.95^150 ≈ 4.6e-4` of the bonus
magnitude — a £50 bonus arrives at the action as ~£0.023 of gradient, invisible against ±£500/d naked
variance. This is the **same trap `open_cost` hit** on 2026-04-25 and fixed by moving to per-tick credit
("GAE smeared the per-race delta back across 5,000 ticks ... per-tick delivery puts the gradient at the
right place"). E1 applied that same fix to `close_signal_bonus` (deliver at the close-tick) — but it was
NO BITE: cl_n 9→9.2, because PPO just doesn't respond to a £10 per-close gradient in 7 days regardless of
when it's delivered.

## Why it matters

The mechanism behind the whole [[per-pair-credit-assignment]] family: settle-time lumping is
mathematically near-zero gradient at the decision, so per-tick / per-resolution delivery is necessary —
but E1 shows it is **not sufficient** when the residual signal is still swamped by naked variance
([[gradient-delivered-ppo-unresponsive]]). A reusable rule: before blaming the policy, compute the
discount factor at the credit lag — `λ^(lag)` may already be ~0.

## Sources
- `src-106b41` EXPERIMENTS.md (js_desktop:present)
