# Progress — Scalping Active Management

One entry per completed session. Most recent at the top.

---

## 📋 When you open this plan for the first time

Read in order:

1. `purpose.md` — why this exists, the Gen 1 evidence, the
   four changes.
2. `hard_constraints.md` — 20 non-negotiables (invariant,
   matcher, aux-loss isolation, back-compat).
3. `master_todo.md` — session sequencing.
4. `session_prompt.md` — brief for the immediate next session.

Also re-read `CLAUDE.md`, especially "Order matching:
single-price, no walking" (the matcher is load-bearing) and
"Reward function: raw vs shaped" (the invariant still applies).

---

## Session 01 — Re-quote action + env plumbing (2026-04-16)

**Landed.**

- `SCALPING_ACTIONS_PER_RUNNER` bumped 5 → 6. New dim is
  `requote_signal ∈ [-1, 1]`, read from
  `action[5 * max_runners + slot_idx]` in `_process_action`.
- `SCALPING_POSITION_DIM` bumped 2 → 4. Two new per-runner obs
  features live at offsets +2, +3 of the scalping-extra block:
  - `seconds_since_passive_placed` — elapsed real seconds since
    the paired passive was posted, normalised by that race's
    first/last tick span and clamped to [0, 1].
  - `passive_price_vs_current_ltp_ticks` — signed tick distance
    from current LTP to the resting price, normalised by
    `MAX_ARB_TICKS` and clamped to [-1, 1].
  `PassiveOrder` gained a `placed_time_to_off` field so the
  elapsed computation is drift-free across varying race lengths.
  `_race_durations` is pre-computed once per race in
  `_precompute`, matching the existing static-obs / slot-map
  caching pattern.
- Schemas bumped: `OBS_SCHEMA_VERSION: 5 → 6`,
  `ACTION_SCHEMA_VERSION: 2 → 3`. These invalidate existing
  checkpoints for strict validation, same convention as prior
  schema bumps. A migration helper
  `agents.policy_network.migrate_scalping_action_head` pads the
  actor-head final layer + `action_log_std` for code paths that
  explicitly opt into loading a pre-Session-01 state dict (the
  requote row initialises fresh, existing rows are preserved).
- Re-quote dispatch is a dedicated SECOND PASS over slots after
  the main placement loop so a slot that `continue`'d (no bet
  signal, below-min stake, below min_seconds_before_off) can
  still re-quote if its runner has a paired passive to manage.
  The pass:
    1. Finds the first open paired order for the slot's `sid`.
       If none → no-op with `requote_reason="no_open_passive"`
       (hard_constraints §5: never opens a naked leg).
    2. Computes `arb_ticks` from this tick's `arb_raw` (current
       LTP, not the original fill price).
    3. Computes the new resting price with the same direction
       rule as `_maybe_place_paired` (back → lay below, lay →
       back above).
    4. Applies the junk-filter window explicitly — paired
       `PassiveOrderBook.place` bypasses it for the auto-paired
       path, but an active re-quote sitting outside ±max_dev
       from current LTP IS stale-parked-order risk. On failure
       we cancel the old passive and set
       `requote_reason="junk_band"`, leaving the aggressive
       leg naked for that runner.
    5. Cancels the existing passive via a new
       `PassiveOrderBook.cancel_order(order, reason)` — budget
       reservation is released before the new reservation is
       taken (hard_constraints §6). On Lay-after-Back pairs the
       `place()` freed-budget offset recomputes against the
       aggressive leg still in `bm.bets`, so the net
       `available_budget` change equals the liability delta
       between old and new passive prices.
    6. Re-places via `PassiveOrderBook.place(..., price=...)`
       with the same `pair_id` — ledger continuity is preserved
       and the re-quoted passive, if it fills, shows up as a
       completed pair alongside the aggressive bet.
    7. Records `requote_attempted`, `requote_placed`, and/or
       `requote_failed`/`requote_reason` on
       `action_debug[sid]` for diagnostics.
- Tests: new `TestScalpingRequote` class covering the 11
  scenarios in the session prompt. Existing
  `test_obs_space_grows_when_scalping` updated for the new
  `SCALPING_POSITION_DIM`. Three `test_p1?_*` schema-refusal
  tests had hard-coded `OBS_SCHEMA_VERSION == 5` assertions;
  relaxed to `>= N` so they stay locked on the refusal
  behaviour rather than on a frozen version number.
- Full suite green: `pytest tests/ -q`
  → 1943 passed, 7 skipped, 1 xfailed.

**Back-compat strategy.**

- Scalping OFF: action vector stays 4-per-runner, obs stays
  `SCALPING_POSITION_DIM=0` extras. Byte-identical to pre-
  Session-01. Verified by
  `test_action_space_size_grows` (scalping=False branch).
- Scalping ON with a pre-Session-01 checkpoint: strict
  `validate_action_schema` refuses the version-2 checkpoint.
  Use `migrate_scalping_action_head(state_dict, max_runners,
  old_per_runner=5, new_per_runner=6)` to widen the actor head
  + log-std before `load_state_dict`. Tested by
  `test_legacy_checkpoint_loads`.

**No reward-scale change.** `raw + shaped ≈ total_reward`
invariant verified by the new
`test_raw_plus_shaped_invariant_holds` under active re-quoting.

---

_Plan created 2026-04-16 off the back of the Gen 1 training-run
analysis. The Gen 1 run confirmed the reward-signal fix but
also exposed that even the top scalper only completes 14.5 %
of pair attempts — the rest become accidental directional bets
because the passive never fills. This plan gives the agent the
tools to manage passives actively and to know its own fill
probability and risk._
