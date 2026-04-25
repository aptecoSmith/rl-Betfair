# selective-open-shaping-probe — settle-time design (2026-04-25 19:03Z)

Plan: `selective-open-shaping-probe` (`a5f0c7af-…`).
Run aborted partway through gen-0 after 4 agents completed their
18 episodes. The mechanism's gradient signal landed at settle
time and got smeared across 5,000 ticks by GAE — failed to move
the policy.

## Headline numbers

| agent | gene | fc_rate | shaped/race | mean_pnl |
|---|---|---|---|---|
| 8e1185b1 | 0.0564 | 77% | -£31  | -£352 |
| 141877a3 | 0.1972 | 76% | -£101 | -£348 |
| 10978f27 (10/18) | 0.2006 | 75% | -£115 | -£391 |
| 61617a7f | 0.8289 | **77%** | **-£447** | -£362 |

The high-gene agent (0.83) paid 15× more shaped pressure than
the low-gene agent (0.06) and produced the **same** force-close
rate (77% vs 77%). The signal was reaching PPO's gradient
pathway (-£447 = 15× -£31, math checks out) but the agent
wasn't responding by reducing `pairs_opened`.

## Why the settle-time design didn't bite

Per-race shaped contribution arrives on the SETTLE step (last
tick of the race). PPO's GAE then propagates it back across the
5,000-tick race via value-function bootstrapping. With an
untrained value head, the per-tick gradient at the open decision
is essentially noise — even a -£447 race-level shaped delta
becomes ~-£0.09/step after smearing, drowning in ±£3-£5/step of
value-function variance.

The matured-bonus has the same delivery shape but works as
*reinforcement* of an already-existing behavior (open more pairs).
Open-cost was trying to *suppress* a baseline behavior, which is
much harder under delayed credit assignment.

## What replaces it

Same mechanism, per-tick delivery (Session 02 revision in commit
landing alongside this archive):

- Charge `-open_cost` lands on the **open tick** (when
  `bm.place_back/place_lay` returns a non-None bet).
- Refund `+open_cost` lands on the **resolution tick** (when both
  legs of the pair appear in `bm.bets` and the close_leg flag
  doesn't carry `force_close=True`).
- Force-closed and naked outcomes leave the charge in place; the
  cumulative shaped contribution per race matches the original
  formula `open_cost × (refund_count − pairs_opened)` exactly.

Three new integration tests in
`TestSelectiveOpenShaping`:
- `test_per_tick_charge_lands_on_open_step_not_settle`
- `test_per_tick_refund_lands_on_resolution_step`
- `test_per_tick_total_matches_settle_time_formula`

The mathematical equivalence is the same ε-correctness contract:
the per-tick design only changes WHEN the gradient signal
arrives, not its sum.

## Contents

- `models.db` — 76 KB, 4 fully-trained agents + 3 partial.
- `weights/` — 12 `.pt` files. Trained under the settle-time
  design with PPO seeing the open_cost contribution at the
  wrong tick.

## Don't reuse the weights

The per-tick design's gradient regime is different. Cross-loading
these weights would produce a confused starting state. Treat as
historical reference only — the lesson lives in this README.
