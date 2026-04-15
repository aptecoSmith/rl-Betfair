# Purpose — Scalping Asymmetric Hedging

## Why this work exists

The scalping reward path (landed 2026-04-15, commit `98f834b`)
rewards "locked" profit from back/lay pairs and punishes naked
losses. The missing piece: the agent can't actually *produce*
locked profit, because it sizes both legs of a scalp from the same
stake bucket.

A proper scalp on a shortening price needs **asymmetric stakes**.
To lock profit regardless of outcome:

```
lay_stake  = back_stake × back_price / lay_price
```

Example (Joyeuse, 2026-04-10 Aintree 12:45):

- Back £20 @ 12.5, lay £20 @ 6.0 (current behaviour, equal stakes):
  - Win outcome: +£230 − £100 = +£130
  - Lose outcome: −£20 + £20 = £0
  - Locked floor: **£0**. All the +£130 came from the horse winning.
- Back £20 @ 12.5, lay £41.67 @ 6.0 (properly sized):
  - Win outcome: +£230 − £208.35 = +£21.65
  - Lose outcome: −£20 + £41.67 = +£21.67
  - Locked floor: **~£21.66**. Guaranteed profit from the price move.

The price shortened from 12.5 → 6.0 — a huge favourable move — and
the agent banked £0 of locked profit from it. The +£130 it realised
on the Gold Dancer screenshot came from being lucky on the outcome,
not from trading the price move. We're not scalping; we're placing
correlated pairs and hoping.

## The four changes

Laid out in dependency order. (1) and (2) fix the reward signal.
(3) makes diagnostics honest. (4) gives the agent the tool to act
on the fixed signal.

### 1. Redefine `scalping_locked_pnl`

Current (implicit): realised P&L on the pair contributes to locked
accumulator when it happens to come from a pair.

New: **locked_pnl per pair = max(0, min over outcomes of pair P&L)**.

An equal-stake pair has `min(win_pnl, lose_pnl) = 0` → locks
nothing → contributes £0 to the locked bucket regardless of which
outcome fires. A properly-sized pair has `min > 0` → contributes
its guaranteed floor.

This stops the reward rewarding "accidental pairs" where the agent
got lucky on outcome.

### 2. Worst-case-improvement shaping term

Per closing leg, reward `Δ worst_case = worst_case_after − worst_case_before`.

Example: backing Joyeuse at 12.5 leaves worst-case = −£20.

- Lay £20 @ 6.0 lifts worst-case to £0 → Δ = +£20.
- Lay £41.67 @ 6.0 lifts worst-case to +£21.67 → Δ = +£41.67.

Dense per-step gradient that specifically rewards proper sizing.
Available before the race settles, so the signal arrives while the
choice is still fresh.

### 3. UI classification badge

Bet Explorer currently shows pair P&L with a green/red tint based
on realised P&L. That mis-reads luck as skill. Classify each pair
as **locked / naked / directional** based on its worst-case floor:

- `min(win_pnl, lose_pnl) > 0` → **locked** (green badge)
- `min(win_pnl, lose_pnl) = 0` → **neutral pair** (grey badge)
- otherwise → **directional** (amber badge)
- unpaired matched order → **naked** (red badge)

This is pure diagnostic — no training impact. Lets us watch (1)
and (2) teach the agent before committing to (4).

### 4. "Close position" action

Agent cannot currently pick £41.67 as a stake. Add a dedicated
"close open position on runner X at market" action. The env sizes
the hedge using the formula above, clamped to available ladder
depth. The agent only chooses *whether* and *when* to close.

Mirrors how a live human scalper actually operates.

## What success looks like

- Scalping-locked raw reward reflects *only* properly-sized hedges.
- Shaped worst-case-improvement term appears in
  `logs/training/episodes.jsonl` and the live training chart.
- Bet Explorer distinguishes locked / directional / naked pairs.
- Agent has a close-position action and learns to use it — visible
  as an upward trend in locked_pnl / total_pnl ratio and a
  downward trend in naked_loss count over training.

## What this folder does NOT cover

- Observation-space features around scalping opportunities. The
  agent already has enough price-movement signal.
- Per-tick stake sizing for the *opening* leg (still uses the
  existing discrete stake head).
- Ladder walking or partial-fill logic. `ExchangeMatcher` behaviour
  is unchanged — see CLAUDE.md for why that's load-bearing.
- Live-inference knock-ons. Any `ai-betfair` changes go in
  `ai-betfair/incoming/` per the postbox convention.

## Folder layout

```
plans/scalping-asymmetric-hedging/
  purpose.md              <- this file
  hard_constraints.md     <- non-negotiables
  master_todo.md          <- ordered session list
  progress.md             <- one entry per completed session
  lessons_learnt.md       <- surprising findings, append-only
```
