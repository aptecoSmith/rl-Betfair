# Hard constraints — Scalping Close-Signal

Non-negotiable rules for every session in this plan. A change that
violates one of these gets rejected in review before it gets a chance
to destabilise production.

## Action / placement

**§1** `close_signal` **never** opens a new naked leg. If no
outstanding aggressive bet with a `pair_id` exists on this runner at
the time of the signal, the action is a silent no-op logged as
`close_reason="no_open_aggressive"` on `action_debug[sid]`. Never
places a bare opposite-side bet.

**§2** Close placements use the existing `ExchangeMatcher`
single-price path. No ladder walking. A close that can't find
sufficient liquidity at one price fills partially (matched_size <
requested_stake) and the pair's `locked_pnl` falls out naturally from
the matched amounts — the agent just takes the resulting realised
P&L. See CLAUDE.md "Order matching: single-price, no walking".

**§3** Close sizing follows the same equal-P&L formula as passive
sizing:
```
S_close = S_aggressive × P_aggressive / P_close
```
The close crosses the spread at the current opposite-side best
(aggressive-lay best for closing a back, aggressive-back best for
closing a lay). Both legs end with matched `S × P` equal, which is
the only sizing that makes `bet_manager.get_paired_positions`
classify the pair as a hedged completion rather than two loose bets.

**§4** Close **bypasses** `env.scalping_math.min_arb_ticks_for_profit`.
The commission-feasibility gate introduced in commit `f37a1d5`
refuses *opening* pairs whose math can't clear commission. Closing is
a deliberate loss-cap; checking feasibility would defeat the
mechanic.

## Reward & accounting

**§5** No new shaped-reward terms. The existing
`scalping_locked_pnl + min(0, naked_pnl)` raw handling stays intact.
Closing a position creates a completed pair whose `locked_pnl` may be
zero (floored) and whose `naked_pnl` is zero (both legs matched) — so
the close contributes 0 to raw reward. The cash cost still flows
through `day_pnl` and the terminal bonus. Any proposal to add a
`close_bonus_weight` shaped term goes in a follow-up plan, not this
one.

**§6** The `locked_pnl = max(0, min(win_pnl, lose_pnl))` floor stays.
Closing-at-a-loss pairs have negative `min(win, lose)` — the floor
makes them register as 0-locked, NOT as negative-locked. This avoids
the double-counting problem (the cash loss is in `day_pnl` already).

**§7** `early_lock_bonus` stays gated on `locked_pnl > 0` (commit
`0bdb3f9`). A close-at-loss pair has `locked_pnl == 0` and thus earns
no bonus — correct behaviour per §5.

**§8** `naked_penalty_weight × naked_exposure` is unchanged. A closed
pair has zero contribution because both legs are matched; a naked
pair still gets the full penalty. The close mechanic therefore gives
the agent a *substitution*: take −0 raw reward for closing or −N raw
reward for going naked.

## Schema

**§9** `ACTION_SCHEMA_VERSION: 3 → 4`. Strict validation on
checkpoint load. Garaged models need a migration helper that pads the
actor head and `action_log_std` for the new dim (zero-init so the
head outputs "no close" by default, preserving pre-plan behaviour
verbatim).

**§10** `OBS_SCHEMA_VERSION: unchanged`. v1 adds no observation
features — the agent already sees `seconds_since_passive_placed` and
`passive_price_vs_current_ltp_ticks` from Session 01, which is the
state it needs to reason about closing.

**§11** `SCALPING_ACTIONS_PER_RUNNER: 6 → 7`. Non-scalping action
layout (`ACTIONS_PER_RUNNER = 4`) is unchanged — the new dim only
exists when `training.scalping_mode=True`.

## Diagnostics

**§12** The env tracks `arbs_closed` per episode, distinct from
`arbs_completed` (passive filled naturally) and `arbs_naked` (passive
never filled). All three sum to the total paired attempts.
`episodes.jsonl` gains an `arbs_closed` column.

**§13** The activity log emits a distinct line format for closes:
```
Pair closed at loss: Back £X / Lay £Y on runner Z → realised −£W
```
Separate from the existing `Arb completed` line. Capped by the same
`_MAX_ARB_EVENTS_PER_EP` as arb_completed to avoid WS flood.

## Testing

**§14** Every close-path change needs at least one test in
`tests/test_forced_arbitrage.py` or a new `tests/test_close_signal.py`
covering the cases:
- Close with open aggressive + unfilled passive (happy path).
- Close when no open aggressive on this runner (no-op).
- Close when the passive already filled (no-op — pair is done).
- Close at unfavorable market (realised loss registered in day_pnl,
  raw reward unchanged).
- Close at favorable market (realised gain, same plumbing as a
  natural completion).
- Legacy checkpoint (action_schema_version=3) refused on load without
  the migration helper.

**§15** Full `pytest tests/ -q` stays green on every commit in this
plan.

## Cross-session

**§16** Do NOT rewrite the `bet_manager.get_paired_positions` locked-
pnl formula to handle closes specially. Closes are just pairs whose
both legs matched. The existing formula produces correct results
(locked floors to 0 for losing pairs, cash loss is in day_pnl).

**§17** Do NOT add commission-feasibility logic anywhere in the close
path. If you're tempted, re-read §4.

**§18** Do NOT extend close to cross-market or multi-runner
coordination. See purpose.md "What this folder does NOT cover".
