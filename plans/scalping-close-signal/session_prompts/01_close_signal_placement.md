# Scalping Close-Signal — Session 01 prompt

## PREREQUISITE — read first

- `plans/scalping-close-signal/purpose.md` — especially the
  "One change, isolated" section. The mechanic is narrowly scoped;
  do not expand it in this session.
- `plans/scalping-close-signal/hard_constraints.md` — the 18
  non-negotiables. §4 (commission gate bypass) and §5–§8 (reward
  semantics untouched) are the ones most likely to trip a careless
  implementation.
- `plans/scalping-active-management/session_prompts/01_re_quote.md`
  — the re-quote mechanic's session prompt. This close action
  mirrors its shape (threshold check, helper method, diagnostic
  tagging, migration helper). Borrow the pattern.
- `env/betfair_env.py` — especially the `_attempt_requote` method
  around line 1494. The close mechanic's placement method is the
  natural mirror.
- `env/scalping_math.py` — the commission-feasibility gate. Understand
  it well enough to confidently skip it on the close path.
- `CLAUDE.md` — "Bet accounting" and "Reward function" sections.

## Before touching anything — locate the code

```
grep -n "SCALPING_ACTIONS_PER_RUNNER\|ACTION_SCHEMA_VERSION\|requote" env/betfair_env.py | head
grep -n "migrate_scalping_action_head" agents/policy_network.py
grep -n "arbs_completed\|arbs_naked" env/betfair_env.py env/bet_manager.py agents/ppo_trainer.py | head -20
```

## What to do

### 1. Schema bump

- `env/betfair_env.py`:
  - `ACTION_SCHEMA_VERSION: 3 → 4`. Docstring comment explains the
    new layout: signal, stake, aggression, cancel, arb_spread,
    requote_signal, **close_signal**.
  - `SCALPING_ACTIONS_PER_RUNNER: 6 → 7`.
- The non-scalping layout (`ACTIONS_PER_RUNNER = 4`) is unchanged —
  the close dim only exists when `training.scalping_mode=True`.

### 2. Action processing

In `_process_action`, after the re-quote pass:

```python
# ── Close-signal pass (scalping-close-signal session 01) ─────────
# Third pass over slots: any runner whose close_signal action is
# raised gets its open pair closed at market — cancel the passive,
# cross the spread with an aggressive opposite-side leg.
if self.scalping_mode and apr > 6:
    for slot_idx in range(self.max_runners):
        close_raw = float(action[6 * self.max_runners + slot_idx])
        if close_raw <= 0.5:
            continue
        sid = self._sid_for_slot(slot_idx)
        if sid is None:
            continue
        runner = runner_by_sid.get(sid)
        if runner is None or runner.status != "ACTIVE":
            continue
        time_to_off = ...
        self._attempt_close(
            sid=sid, runner=runner, race=race,
            time_to_off=time_to_off, action_debug=action_debug,
        )
```

### 3. `_attempt_close` method

Mirror `_attempt_requote`'s structure:

1. Find any outstanding paired passive on this runner (`pair_id`
   matches an aggressive leg in `bm.bets`). If none → silent no-op,
   mark `close_attempted=True, close_reason="no_open_aggressive"`.
2. Locate the aggressive leg from `bm.bets` (the one with
   `pair_id`). If the passive already filled (pair complete) →
   no-op, `close_reason="pair_already_complete"`.
3. Determine the close side: opposite to the aggressive. Back agg →
   close via LAY; lay agg → close via BACK.
4. Use the **matcher's** top opposite-side price as the close price
   (the aggressive-best on the close side). This is
   `runner.available_to_lay[0].price` for closing a back, or
   `runner.available_to_back[0].price` for closing a lay.
5. Size per `S_close = S_agg × P_agg / P_close`. Same formula as
   `_maybe_place_paired` uses for the passive leg's stake.
6. Cancel the outstanding passive (releases budget reservation).
7. Place the close as an aggressive bet via `bm.place_back` or
   `bm.place_lay` with the same `pair_id`. Critically: **skip** any
   commission-feasibility check — this is a deliberate loss-cap.
8. Tag `action_debug[sid]` with `close_attempted=True,
   close_placed=True, close_reason=None`. On any partial/failure
   mark the appropriate reason (e.g. `insufficient_liquidity`).

### 4. Diagnostics

- `env/bet_manager.py::EpisodeStats` gains `arbs_closed: int = 0`.
- `_settle_current_race` increments `arbs_closed` for completed
  pairs whose aggressive placement was the close leg (use a
  tell-tale tag on the `Bet` — e.g. `Bet.close_leg: bool = False`
  defaulting False, set True on the close path). Completed pairs
  with `close_leg=False` on both legs stay in `arbs_completed`.
- `agents/ppo_trainer.py` persists `arbs_closed` to
  `logs/training/episodes.jsonl` alongside `arbs_completed` /
  `arbs_naked`.
- `agents/ppo_trainer.py` adds a distinct activity-log line format
  for each close (capped by `_MAX_ARB_EVENTS_PER_EP` as today):

```python
self.progress_queue.put_nowait({
    "event": "pair_closed",
    "phase": "training",
    "detail": (
        f"Pair closed at loss: Back £{ev['back_price']:.2f}"
        f" / Lay £{ev['lay_price']:.2f}"
        f" on runner {ev['selection_id']}"
        f" → realised £{ev['realised_pnl']:+.2f}"
    ),
    "close_event": ev,
})
```

Where `realised_pnl` is the pair's actual outcome (what
`bet_manager` returns as the pair's settled P&L — typically a
small negative number for a close-at-loss, or a small positive
for a close-at-profit).

### 5. Migration helper

Add `agents.policy_network.migrate_scalping_action_head_v3_to_v4`:
given a pre-close_signal state dict (action_head output dim =
`max_runners × 6`), produce a v4 state dict with dim =
`max_runners × 7`. Zero-init the new weights so the unmigrated
agent outputs `close_signal = 0` identically to pre-plan behaviour.

Mirror the existing `migrate_scalping_action_head` (v1→v2 and
v2→v3 variants) — see its usage pattern in
`plans/scalping-active-management/session_prompts/01_re_quote.md`.

### 6. Tests

New file `tests/test_close_signal.py` covering all six cases from
`hard_constraints.md §14`. Use the same fixture patterns as
`tests/test_forced_arbitrage.py`.

The explicit reward-invariant check: a close-at-loss scenario
must produce `raw_pnl_reward == 0` (neither locked nor naked
contributes) while `info["day_pnl"]` shows the realised cash
loss.

### 7. Exit criteria

- `pytest tests/ -q` → green.
- `cd frontend && npx ng test --watch=false` → green (no frontend
  changes expected in Session 01 but existing tests must still pass).
- A fresh 1-agent 10-episode smoke run with scalping_mode=True and
  `close_signal` stochastically raised on some ticks produces
  non-zero `arbs_closed` in at least one episode.
- The three garaged models (`46187c46`, `ef453cd9`, `ab460eb9`)
  load via the v3→v4 migration helper without producing NaNs in
  the first forward pass on scalping input.

### 8. Commit

One commit for Session 01, referencing this plan + the activation-
A-baseline gen-2 analysis that motivated the mechanic.

---

## Cross-session rules

- Full pytest green on every commit.
- No reward-shaping additions in this session (see
  `hard_constraints.md §5`). Those go in a follow-up plan if at all.
- No observation additions in this session (see §10).
- Do not touch the `ExchangeMatcher` or `PassiveOrderBook` core —
  extend via the same `bm.place_back` / `bm.place_lay` entry points
  the existing aggressive paths use.

## After Session 01

Append an entry to `progress.md` following the convention in
`plans/scalping-active-management/progress.md`: section heading with
session name + date, "Landed." status line, bullet list of what
changed, test count delta, back-compat notes.

Then: reset activation-A-baseline's plan state to `draft` (as we did
on 2026-04-17 after the previous aborted run) and kick it off again.
Compare the gen-0 outcome against the 2026-04-17 gen-2 snapshot —
the comparison document template lives in
`plans/scalping-active-management/activation_playbook.md`.
