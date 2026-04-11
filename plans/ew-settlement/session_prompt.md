# EW Settlement — All Sessions (01–04)

Work through all four sessions sequentially. Complete each session
fully (code + tests + progress.md entry) before moving to the next.
Commit after each session.

## Before you start — read these

- `plans/ew-settlement/purpose.md` — why this work exists and the
  correct EW settlement math.
- `plans/ew-settlement/hard_constraints.md` — non-negotiables.
- `plans/ew-settlement/lessons_learnt.md` — discovery context.
- `CLAUDE.md` — especially "Reward function: raw vs shaped" section
  and the `raw + shaped ≈ total_reward` invariant.

---

## Session 01 — BetManager EW settlement logic

### Context

`env/bet_manager.py:704-775` — current `settle_race()`. It takes
`winning_selection_ids: int | set[int]` and treats all IDs in the
set identically: back bets on any winning ID pay at full
`average_price`, lay bets lose full liability. There is no concept
of "winner vs placed" or "place fraction".

The `Race` dataclass (`data/episode_builder.py:174-199`) already
has `winner_selection_id: int | None` (the single race winner) and
`winning_selection_ids: set[int]` (winner + placed). It also has
`each_way_divisor: float | None`.

### What to do

Add EW settlement to `BetManager.settle_race()`:

1. Add two new parameters:
   - `each_way_divisor: float | None = None`
   - `winner_selection_id: int | None = None` (the single winner)

2. When `each_way_divisor` is not None, apply EW logic for each bet:

   **Place odds** = `(average_price - 1.0) / each_way_divisor + 1.0`

   **BACK bets:**
   - **Winner** (`bet.selection_id == winner_selection_id`):
     Both legs pay. Win leg profit = `(S/2) × (price - 1)`.
     Place leg profit = `(S/2) × ((price - 1) / divisor)`.
     Commission applies to each leg's gross profit separately.
     Budget gets back both half-stakes plus net profits.
     `pnl = win_net + place_net`
   - **Placed** (`bet.selection_id in winners` but not the winner):
     Win leg loses: -S/2. Place leg pays: `(S/2) × ((price - 1) / divisor)`.
     Commission on place leg profit only.
     Budget gets back place half-stake + place net profit (win
     half-stake was already deducted at placement).
     `pnl = -(S/2) + place_net`
   - **Unplaced**: lose full stake. `pnl = -matched_stake` (same
     as today).

   **LAY bets** (logic inverts — layer pays when runner qualifies):
   - **Winner**: layer loses on BOTH legs.
     Win liability = `(S/2) × (price - 1)`.
     Place liability = `(S/2) × ((price - 1) / divisor)`.
     `pnl = -(win_liability + place_liability)`
   - **Placed**: layer loses place leg, wins win leg.
     Place liability = `(S/2) × ((price - 1) / divisor)`.
     Win profit = `S/2` (backer's win half-stake, minus commission).
     `pnl = win_net - place_liability`
   - **Unplaced**: layer wins both legs.
     Gross profit = matched_stake. Commission on profit.
     `pnl = matched_stake × (1 - commission)` (same as today).

   Budget bookkeeping must stay correct: back bets deducted full
   stake at placement; lay bets reserved total liability. The EW
   path must account for this when returning funds.

3. When `each_way_divisor` is None: existing path unchanged.

4. Set `bet.outcome` to WON if net pnl > 0, LOST if < 0. For the
   "placed back bet" case this depends on the odds and divisor —
   at short odds the place fraction may not cover the lost win leg.

### Bet.pnl semantics

`bet.pnl` should reflect the total net P&L of the EW bet (both
legs combined). Do NOT split into two Bet objects — the agent
placed one action, settlement produces one P&L figure.

### Tests

Add to `tests/test_bet_manager.py` in a new class
`TestEachWaySettlementCorrected`. Use concrete numbers that are
easy to verify by hand. Use commission=0.05 in most tests.

Example fixture: stake=10, price=10.0, divisor=4.0, commission=0.05.
Place odds = (10-1)/4 + 1 = 3.25.

1. **Back bet, horse wins EW:**
   Win leg: profit = 5 × 9 × 0.95 = 42.75
   Place leg: profit = 5 × 2.25 × 0.95 = 10.6875
   Total pnl = 53.4375. Budget = 100 - 10 + 10 + 53.4375 = 153.4375.

2. **Back bet, horse places EW:**
   Win leg: lose 5.
   Place leg: profit = 5 × 2.25 × 0.95 = 10.6875
   Total pnl = -5 + 10.6875 = 5.6875.

3. **Back bet, horse unplaced EW:** pnl = -10.

4. **Lay bet, horse wins EW:**
   Win liability = 5 × 9 = 45. Place liability = 5 × 2.25 = 11.25.
   Total pnl = -(45 + 11.25) = -56.25.

5. **Lay bet, horse places EW:**
   Win leg: layer wins, gross = 5, net = 5 × 0.95 = 4.75.
   Place leg: layer loses liability = 5 × 2.25 = 11.25.
   Total pnl = 4.75 - 11.25 = -6.50.

6. **Lay bet, horse unplaced EW:**
   Gross = 10, net = 10 × 0.95 = 9.50. Same as non-EW.

7. **Win market regression:** existing `TestEachWaySettlement` tests
   must still pass. Run the full test suite.

8. **EW with divisor=None falls back to Win logic.**

9. **EW with short-odds placed runner:** stake=10, price=2.0,
   divisor=5.0. Place odds = (2-1)/5 + 1 = 1.2. Place leg profit =
   5 × 0.2 × 0.95 = 0.95. Net = -5 + 0.95 = -4.05. The back bet
   still loses money even though the horse placed. Confirm outcome
   is LOST.

10. **Commission=0 variant** of tests 1 and 2 to verify the
    commission path is truly per-leg.

### Exit criteria

- All new tests pass. All existing tests pass.
- `progress.md` updated. `lessons_learnt.md` if anything surprising.
- Commit.

---

## Session 02 — Environment integration

### Context

`env/betfair_env.py:973-1102` — `_settle_current_race()`. Currently
calls `bm.settle_race(winning_ids, market_id=..., commission=...)`.
Does not pass `each_way_divisor` or `winner_selection_id`.

`Race` dataclass already has `winner_selection_id`, `each_way_divisor`,
and `winning_selection_ids`.

### What to do

1. Update the `settle_race` call at line ~1013 to pass:
   ```python
   race_pnl = bm.settle_race(
       winning_ids,
       market_id=race.market_id,
       commission=self._commission,
       each_way_divisor=race.each_way_divisor,
       winner_selection_id=race.winner_selection_id,
   )
   ```

2. Verify that `race.winner_selection_id` is correctly populated.
   Check `data/episode_builder.py:635` — it reads from
   `first.winner_selection_id` which comes from the tick data. This
   should be the single race winner. Confirm it's not None for races
   that have results.

3. Verify shaped components use `bet.matched_stake` (the full
   original stake), not the half-legs. Since Session 01 doesn't
   split `Bet` objects, `bet.matched_stake` stays at the original
   value — confirm this is the case.

4. Check the `raw + shaped ≈ total_reward` invariant still holds.
   The raw component is `race_pnl` which now reflects correct EW
   settlement. Shaped components are independent. The invariant
   should hold naturally.

### Tests

Add to `tests/test_betfair_env.py` or a new file
`tests/test_ew_env_integration.py`:

1. **EW race fixture step-settle:** Create a minimal Day with one
   EW race. Place a back bet on a runner. Settle with the runner
   as placed-only. Assert the reward reflects correct EW P&L
   (not full-odds P&L).

2. **Mixed Win + EW day:** Two races, one WIN one EACH_WAY.
   Assert day_pnl accumulates correctly.

3. **raw + shaped invariant on EW race.**

### Exit criteria

- All tests pass (new and existing).
- `progress.md` updated.
- Commit.

---

## Session 03 — Fix incorrect comments + episode builder audit

### What to do

1. **Fix the comment at `data/episode_builder.py:622-623`:**
   Current: "Betfair EACH_WAY markets are place markets — the quoted
   odds already reflect the place fraction, so PLACED pays at full
   price."
   Replace with: "For EACH_WAY markets, WINNER + PLACED runners are
   both included. Settlement applies the place fraction via
   each_way_divisor — see BetManager.settle_race()."

2. **Fix the comment at `data/feature_engineer.py:581-584`:**
   Current: "Betfair EACH_WAY markets already quote the
   place-adjusted price, so this is informational for the agent
   rather than used in settlement"
   Replace with: "Place odds fraction — the agent can use this to
   reason about EW value. Settlement applies this divisor via
   BetManager.settle_race()."

3. **Fix the docstring at `env/bet_manager.py:713-717`:**
   Current: "Betfair EACH_WAY odds already include the place
   fraction so PLACED runners pay at the quoted price."
   Replace with accurate description of the new EW settlement
   behaviour.

4. **Fix the comment at `data/episode_builder.py:196-198`:**
   Current: "all of them pay out on a back bet."
   Replace with: "Settlement distinguishes winner from placed
   using each_way_divisor — see BetManager.settle_race()."

5. **Audit `Race` field population:**
   - `winner_selection_id` — populated from `first.winner_selection_id`
     (line 635). Confirm this is always the single race winner
     (not placed).
   - `winning_selection_ids` — built from runner statuses "WINNER"
     and "PLACED" (lines 624-628). Confirm both are included.
   - `each_way_divisor` — from `first_row.get("each_way_divisor")`
     (line 642). Confirm it's populated for EACH_WAY markets.

6. **If any field is not correctly populated**, fix the population
   code. The data source (Parquet) should have the raw values from
   the BetfairPoller.

### Tests

- Verify existing episode builder tests still pass.
- If not already tested: assert `Race.winner_selection_id` !=
  `Race.winning_selection_ids` for an EW race fixture (winner is
  one ID, winning_selection_ids is a superset).

### Exit criteria

- No incorrect EW comments remain in the codebase. Grep for
  "already quote", "already include the place", "already reflect
  the place" — should return zero results outside of plan docs
  and lessons_learnt.
- All tests pass.
- `progress.md` updated.
- Commit.

---

## Session 04 — Historical P&L comparison

### What to do

Analysis only — no production code changes.

Write a standalone script `scripts/ew_pnl_comparison.py` that:

1. Loads a sample of historical days from the training data (use
   `data/extractor.py` or the Parquet files directly).

2. Filters to EACH_WAY races only.

3. For each EW race that had bets placed by any trained model
   (or simulate a simple back-the-favourite strategy if no bets
   exist):
   - Compute P&L under the **old** method (full odds for all
     winning_selection_ids, single stake).
   - Compute P&L under the **new** method (place fraction,
     split stake, both legs for winner).
   - Record the delta.

4. Output a summary table:
   - Number of EW races analysed.
   - Mean / median / max absolute P&L delta per race.
   - Total P&L delta across all races.
   - Percentage of races where the sign of P&L changed.

5. Save the per-race details to a CSV for manual inspection.

### Deliverable

- `scripts/ew_pnl_comparison.py` — the script.
- A summary in `lessons_learnt.md` documenting the magnitude of
  the training reward distortion.
- `progress.md` updated.
- Commit.

---

## Cross-session rules

- Run `pytest tests/ -x` after each session. All tests must pass.
- Do not touch observation features, action space, or the training
  loop. This plan is settlement-only.
- Do not "improve" unrelated code you happen to read. Scope is
  tight.
- Commit after each session with a clear message referencing the
  session number.
