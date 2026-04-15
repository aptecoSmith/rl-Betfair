# Progress — Scalping Asymmetric Hedging

One entry per completed session. Most recent at the top.

---

## Session 01 + 04 merged — Asymmetric hedge sizing + correct locked_pnl (2026-04-15)

**Scope collapsed.** Audit of the existing `_maybe_place_paired`
and `get_paired_positions` (landed in Sprint 5 Sessions 1–2 on
2026-04-14, commit `98f834b`) revealed:

1. `_maybe_place_paired` sized the passive counter-order with
   `stake=aggressive_bet.matched_stake` — **equal** to the
   aggressive leg. That's the bug that made every Gen 0 pair a
   directional bet rather than a lock.
2. `get_paired_positions.locked_pnl` used `stake × spread ×
   (1-comm)` — which is the **MAX-outcome** of an equal-stake
   pair, not the floor. That made the reward path credit the full
   directional windfall as "locked" profit. Gen 0 models learned
   "more pairs = more reward" regardless of sizing.

The existing scalping plumbing (paired orders, pair_id tracking,
arb_spread action, naked-loss reward asymmetry) was therefore
correct in structure but broken in its two central formulas. The
fix is a ~10-line change in each file, not a new action-space
expansion. Our original Session 04 ("close-position action") is
absorbed — the env already places the close automatically; it
just needed to size it properly.

### What changed

- `env/betfair_env.py::_maybe_place_paired` — passive stake now
  computed as `agg_stake × agg_price / passive_price`. Works
  symmetrically for BACK→LAY (passive stake larger) and LAY→BACK
  (passive stake smaller). Guards against a zero/negative
  passive_price.
- `env/bet_manager.py::get_paired_positions` — `locked_pnl`
  redefined as `max(0, min(win_pnl, lose_pnl))`. Win P&L applies
  commission to the back leg's winnings; lose P&L applies it to
  the lay leg's winnings. Equal-stake pairs now correctly report
  £0 locked. Properly-sized pairs report the true guaranteed
  floor.
- `tests/test_forced_arbitrage.py` — previous
  `test_get_paired_positions_complete_yields_locked_spread` was
  asserting the old buggy max-outcome behaviour; renamed and
  flipped to
  `test_equal_stake_pair_locks_nothing` (expects £0). Added
  `test_properly_sized_pair_locks_positive_floor` (asserts > £5)
  and `test_backwards_pair_locks_zero`. Added
  `test_paired_passive_stake_sized_asymmetrically` to pin the
  sizing formula itself. Bumped the arb-spread test action to
  MAX_ARB_TICKS so the commission is beatable.

### Reward-scale break

The fix **changes reward scale** for any model trained with
`scalping_mode: True` on commit `98f834b` or earlier. Those
models' episodes.jsonl entries and scoreboards are no longer
directly comparable to future runs. Called out in
`hard_constraints.md §18`. Pre-scalping models are unaffected —
they never had pair_ids.

### Test results

- `tests/test_forced_arbitrage.py` — 42/42 pass (3 new tests
  added, 2 existing tests updated).
- `tests/test_bet_manager.py`, `tests/test_model_store.py`,
  `tests/test_api_replay.py` — 191/191 pass including pair_id
  round-trip.
- 13 pre-existing test failures on master (sessions 2.8, 4.6,
  4.7) are unchanged — unrelated to this work.

---

## Session 03 — UI classification badge in Bet Explorer (2026-04-15)

Added locked / neutral / directional / naked pair classification
to the Bet Explorer. The badge is derived from the **worst-case
floor** of each pair — exactly the same formula as Session 01's
corrected `locked_pnl` — so the UI cannot mistake luck for skill.

### What changed

End-to-end `pair_id` plumbing:
- `registry/model_store.py::EvaluationBetRecord` — new optional
  `pair_id: str | None`. Parquet write adds the column; read path
  gates on `has_pair_id` so pre-existing parquet files keep
  working.
- `training/evaluator.py` — writes `bet.pair_id` to the evaluation
  record.
- `api/schemas.py::ExplorerBet` — new `pair_id` field.
- `api/routers/replay.py` — passes pair_id through to the API
  response.
- `frontend/src/app/models/bet-explorer.model.ts` — added
  `pair_id?: string | null`.

UI additions in `frontend/src/app/bet-explorer/`:
- `bet-explorer.ts` — pure functions `pairFloorPnl` and
  `classifyBet` compute the worst-case floor client-side; new
  computed `legsByPair` groups bets by pair_id; new computed
  `classCounts` tallies the four categories. `pairClass(bet)` and
  `pairClassLabel(c)` exposed to the template.
- `bet-explorer.html` — new `.pair-class-bar` showing counts for
  locked / neutral / directional / naked; per-bet
  `.pair-class-badge` next to the existing settlement badge.
- `bet-explorer.scss` — styling for the badge (green / grey /
  amber / red) and counter bar, matching the existing
  summary-bar visual language.

### Verified in browser

- Frontend compiled clean (no TS / template errors).
- Page loads at `/bets`, selecting model `a7e9ef4f` renders:
  - Counter bar: **NAKED 990** (all other categories 0).
  - Per-bet badges render red-tinted "NAKED" spans.
- `preview_inspect` confirms colour `rgb(248, 113, 113)` and
  expected background on the NAKED badge.
- Pre-scalping Gen 0 models correctly show all bets as NAKED
  (they have no pair_ids). Post-fix scalping models will show
  LOCKED badges when pairs are properly sized, and
  NEUTRAL/DIRECTIONAL when the agent places unhedged pairs —
  exactly the signal the user wanted the UI to reveal.

### Test results

- `tests/test_model_store.py::test_parquet_schema_correct` —
  expected-columns set updated to include `pair_id`.
- Full affected test suite (191 tests across 4 files) passes.

---

## Session 02 — Worst-case-improvement shaping term (DEFERRED)

Not needed at this stage. With Sessions 01 and 03 landed, the
reward signal is honest and the UI makes pair quality legible.
Observing agent behaviour over the next training run will
reveal whether a dense per-step shaping term is required to
speed learning; adding it now would be premature and could
fight with the honest raw signal.

Keep this session open in `master_todo.md` for a future
training-driven decision.

---

## Session 05 — Training run + analysis (PENDING)

Awaiting a fresh training run with `scalping_mode: True` on the
corrected code. Success criteria:
- Ratio of locked_pnl to total_pnl trends up.
- Naked-loss count trends down.
- Bet Explorer shows a growing share of LOCKED badges.
