# Progress — Scalping Close-Signal

One entry per completed session. Most recent at the top.

---

## Session 01 — Close-signal placement + env plumbing (2026-04-17)

**Landed.**

- `SCALPING_ACTIONS_PER_RUNNER` bumped 6 → 7. New dim is
  `close_signal ∈ [-1, 1]`, read from
  `action[6 * max_runners + slot_idx]` in `_process_action`.
  Non-scalping layout (`ACTIONS_PER_RUNNER = 4`) unchanged.
- `ACTION_SCHEMA_VERSION: 3 → 4`. `OBS_SCHEMA_VERSION` unchanged
  (hard_constraints §10 — no new obs features this session; the
  agent already sees `seconds_since_passive_placed` and
  `passive_price_vs_current_ltp_ticks` from scalping-active-mgmt
  Session 01, which is the state it needs to reason about closing).
- Close dispatch is a dedicated THIRD PASS over slots in
  `_process_action`, after the re-quote pass. Each active slot with
  `close_raw > 0.5` calls `_attempt_close`:
    1. Finds the first open paired passive on the slot's sid. If
       none → silent no-op tagged
       `close_reason="no_open_aggressive"` (hard_constraints §1 —
       never opens a naked leg).
    2. Locates the aggressive partner via `pair_id`; refuses with
       `close_reason="orphan_passive"` if the partner is missing.
    3. Determines the close side (opposite to aggressive) and peeks
       the opposing top-of-book price via `matcher.pick_top_price` —
       the same single-price path `place_back`/`place_lay` use, so
       no ladder walking (hard_constraints §2).
    4. Sizes the close via the equal-P&L formula
       `S_close = S_agg × P_agg / P_close` (hard_constraints §3).
    5. Cancels the outstanding passive first (releases budget
       reservation), then places the aggressive close leg with the
       SAME `pair_id` via `bm.place_back` / `bm.place_lay` — NO
       commission-feasibility check (hard_constraints §4).
    6. Tags the close bet with `close_leg=True` so settlement can
       classify the pair as `arbs_closed` rather than
       `arbs_completed`.
    7. Records `close_attempted`, `close_placed`, and/or
       `close_reason` on `action_debug[sid]` for diagnostics.
- `Bet.close_leg: bool = False` field added in `env/bet_manager.py`.
  Defaults False for every non-close bet (including the original
  aggressive leg of the closed pair).
- `RaceRecord.arbs_closed: int` field added. Populated during
  `_settle_current_race`'s pre-settlement snapshot: pairs whose
  completion includes a `close_leg=True` bet are counted here
  instead of `arbs_completed`. Total paired attempts =
  `arbs_completed + arbs_closed + arbs_naked`.
- Reward accounting (hard_constraints §5). New
  `scalping_closed_pnl` slice is carved out of `naked_pnl` in
  `_settle_current_race`: a closed pair's cash P&L is excluded
  from the asymmetric `min(0, naked_pnl)` term. The pair's
  `locked_pnl` already floors at 0 for losing spreads, so the
  closed pair contributes 0 to `race_reward_pnl` and thus to
  `raw_pnl_reward`. The full cash loss still flows through
  `race_pnl → day_pnl`. This preserves the "close cost me nothing
  reward-signal, naked cost me the full hit" substitution that
  the plan is built around.
- `info["arbs_closed"]` and `info["close_events"]` added as
  top-level env info keys. `EpisodeStats.arbs_closed` and
  `EpisodeStats.close_events` added in `agents/ppo_trainer.py` and
  written to `logs/training/episodes.jsonl` alongside
  `arbs_completed` / `arbs_naked`.
- Activity-log surface: `_publish_progress` emits a distinct
  `pair_closed` event per close (capped by
  `_MAX_ARB_EVENTS_PER_EP`, same as arb_completed) with the line
  format `Pair closed at loss: Back £X / Lay £Y on runner Z →
  realised £W`. The per-episode scalping rollup also gains a
  `closed=N` term.
- `agents.policy_network.migrate_scalping_action_head_v3_to_v4`
  added. Thin wrapper over the existing
  `migrate_scalping_action_head` with `old_per_runner=6`,
  `new_per_runner=7`. Zero-inits the new `close_signal` row on
  the actor head's bias + `action_log_std`, so an unmigrated
  agent's sampled close_signal stays centred at 0.
- Tests: new `tests/test_close_signal.py` with 11 tests covering
  all six cases from hard_constraints §14 — happy-path close,
  no-op without aggressive, no-op when passive already filled,
  close-at-loss with the `raw_pnl_reward == 0` invariant,
  close-at-favourable, and the v3→v4 migration refuse-and-migrate
  pair. `test_action_space_size_grows` in
  `tests/test_forced_arbitrage.py` updated for the 6 → 7 bump.
- Full suite green: `pytest tests/ -q` →
  2140 passed, 7 skipped, 1 xfailed, 133 deselected.

**Back-compat strategy.**

- Scalping OFF: action vector stays 4-per-runner, obs unchanged.
  Byte-identical to pre-plan behaviour. Verified by
  `test_action_space_size_grows` (scalping=False branch) +
  `test_legacy_step_matches_before_scalping`.
- Scalping ON with a v3 checkpoint: strict `load_state_dict`
  refuses it (shape mismatch). Use
  `migrate_scalping_action_head_v3_to_v4(state_dict, max_runners)`
  to widen the actor head + log-std before loading. New rows
  (bias / log-std) zero-init so the migrated agent outputs
  `close_signal = 0` identically to pre-plan behaviour on every
  tick until training diverges.

**Raw + shaped invariant.** Unchanged in behaviour — the new
`scalping_closed_pnl` correction only affects how `naked_pnl` is
partitioned, not the sum `race_reward_pnl + shaped`. The existing
`test_raw_plus_shaped_invariant_holds` (in
`tests/test_forced_arbitrage.py`) continues to pass with
`SCALPING_ACTIONS_PER_RUNNER=7`.

---

_Plan created 2026-04-17 off the back of the activation-A-baseline
gen-2 population analysis which showed high-volume scalpers being
penalised by the asymmetric raw reward, and the observation that
real scalpers resolve this with a "take the red" exit mechanic the
env did not previously support._
