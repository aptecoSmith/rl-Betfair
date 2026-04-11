# Master TODO — EW Settlement

Ordered session list. Tick boxes as sessions land.

When a session completes:
1. Tick its box here.
2. Add an entry to `progress.md`.
3. Append any learnings to `lessons_learnt.md`.
4. Note cross-repo follow-ups for `ai-betfair/plans/ew-settlement/`.

---

## Phase 1 — Settlement fix

- [ ] **Session 01 — BetManager EW settlement logic**

  Core settlement fix in `bet_manager.py`:

  - Add an `each_way_divisor: float | None = None` parameter to
    `settle_race()`.
  - When divisor is provided AND market_type is EACH_WAY:
    - Split each bet's `matched_stake` into two virtual legs
      (win = S/2, place = S/2).
    - **Winner** (selection_id is the race winner): win leg pays
      at full odds, place leg pays at `(price - 1) / divisor + 1`.
    - **Placed** (selection_id in winning_selection_ids but not
      the winner): lose win leg (-S/2), place leg pays at place
      fraction.
    - **Unplaced**: lose both legs (-S total, same as today).
  - Back and lay bets both need the split treatment. For lay bets
    the logic inverts: layer loses on legs where the runner
    qualifies.
  - Commission applies per-leg to the winning side of each leg.
  - When divisor is None or market_type is WIN: existing path,
    no change.

  **Requires:** `settle_race()` must also know which selection_id
  is the *winner* vs merely *placed*. Currently
  `winning_selection_ids` is a flat set. Options:
    - (a) Add a `winner_selection_id: int | None` parameter
      (the single race winner).
    - (b) Pass a dict `{selection_id: "WINNER"|"PLACED"}`.
  Prefer (a) — simpler, and a race has exactly one winner.

  **Tests:**
  - Back bet, horse wins EW race: assert P&L = both legs paid.
  - Back bet, horse places EW race: assert P&L = lose win leg,
    win place leg.
  - Back bet, horse unplaced EW race: assert P&L = -stake.
  - Lay bet equivalents of the above three.
  - Win market settlement unchanged (regression).
  - Commission applied correctly per-leg.
  - Edge case: EW divisor is None → falls back to Win logic.
  - Edge case: EW race with no placed runners (small field).

- [ ] **Session 02 — Environment integration**

  Wire the corrected settlement into `betfair_env.py`:

  - `_settle_current_race()` must pass `each_way_divisor` and
    `winner_selection_id` (the actual race winner, distinct from
    placed runners) to `BetManager.settle_race()`.
  - `Race` dataclass already carries `each_way_divisor` and
    `winner_selection_id`. Verify these are populated correctly
    by `episode_builder.py`.
  - Confirm `winning_selection_ids` (the set) includes both
    winner and placed runners, and `winner_selection_id` (the
    int) is the single race winner.
  - Raw reward component reflects corrected P&L.
  - Shaped components (early_pick, efficiency, precision,
    drawdown, spread_cost) use the original requested stake, not
    the half-legs. Verify this is the case.

  **Tests:**
  - Full env step-settle cycle on an EW race fixture.
  - Reward invariant: raw + shaped ~ total.
  - Day P&L accumulates correctly across mixed Win and EW races.

- [ ] **Session 03 — Fix incorrect comments + episode builder audit**

  - Correct the comment in `episode_builder.py` that says
    "Betfair EACH_WAY markets already quote the place-adjusted
    price."
  - Correct the comment in `feature_engineer.py` that says the
    place_odds_fraction is "informational" and "not used in
    settlement."
  - Audit `episode_builder.py` to confirm:
    - `Race.winner_selection_id` is populated from the data.
    - `Race.winning_selection_ids` includes placed runners.
    - `Race.each_way_divisor` is populated from the data.
  - If `winner_selection_id` does not exist on `Race`, add it.
    The upstream data (MySQL `marketResults` table) has
    `Status = "WINNER"` vs `"PLACED"` — use this to distinguish.

  **Tests:**
  - Episode builder constructs Race with correct winner vs placed
    distinction.
  - Parquet/MySQL data round-trip preserves EW terms.

## Phase 2 — Validation

- [ ] **Session 04 — Historical P&L comparison**

  - Take a sample of historical EW race days from the training
    data.
  - Run the corrected settlement against these races.
  - Compare old vs new P&L per race to quantify the error.
  - Document the magnitude of the training reward distortion in
    `lessons_learnt.md`.
  - This is analysis only — no code changes.

---

## Summary

| Session | What | Phase |
|---------|------|-------|
| 01 | BetManager EW settlement logic | 1 |
| 02 | Environment integration | 1 |
| 03 | Comments + episode builder audit | 1 |
| 04 | Historical P&L comparison | 2 |

Total: 4 sessions. All training done after Session 02 will use
correct EW rewards. Session 04 quantifies how wrong the old
rewards were.
