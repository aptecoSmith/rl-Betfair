# Session 18 — R-2 self-depletion fix

## Before you start — read these

- `../purpose.md`
- `../hard_constraints.md` — **constraint 4 (single-price rule) and
  constraint 7 (matcher stays simulation-only / vendorable)** are
  the two that govern this session. Read them.
- `../bugs.md` — R-2 entry. The full sketch is there.
- `../master_todo.md` Phase 0
- `../design_decisions.md` — entry "R-2 self-depletion is a sim-side
  bug" explains the scope split.
- `../downstream_knockon.md` §0a — the live-side equivalent. Not in
  scope for this session, but read it so you don't accidentally
  bake in assumptions that conflict with the live-side fix.
- `../initial_testing.md`
- repo-root `CLAUDE.md` — sections "Bet accounting" and "Order
  matching: single-price, no walking".
- `env/exchange_matcher.py`
- `env/bet_manager.py`

## Goal

Fix the matcher's failure to deduct the agent's own previously-
matched volume from visible top-of-book size when the agent stacks
multiple bets at the same price level on the same selection within
one race. After this session, two back bets at the same price on
the same runner cannot collectively match more stake than the
historical top-of-book size at that level showed at first
placement.

## Inputs — constraints to obey

1. **Matcher stays stateless and vendorable.** The accumulator
   lives on `BetManager`, never on `ExchangeMatcher`. The matcher
   gains *one* new optional parameter
   (`already_matched_at_top: float = 0.0`) and stays a pure
   function of its inputs. A "put the dict on the matcher"
   implementation is rejected at code review — see `bugs.md` R-2
   notes and `hard_constraints.md` #7.
2. **Single-price rule unchanged.** Stake exceeding the *adjusted*
   top-of-book size is unmatched, not spilled into the next level.
3. **No new behaviour for the first bet at a price.** If the
   accumulator for `(selection, side, price)` is zero, the matcher
   must produce byte-identical results to the current
   implementation. The existing matcher tests must all still pass
   without modification.

## Steps

1. **Add `_matched_at_level` to `BetManager.__init__`.**
   `dict[tuple[int, BetSide, float], float]`. Empty on
   construction. Resets implicitly when the env recreates the
   manager between races (already handled by the env).

2. **Extend `ExchangeMatcher._match` signature.** Add a keyword-
   only parameter `already_matched_at_top: float = 0.0`. Inside
   `_match`, after picking `top` but before computing `matched`,
   replace `min(stake, top.size)` with
   `min(stake, max(0.0, top.size - already_matched_at_top))`.
   Returning a `MatchResult` with `skipped_reason = "self-
   depletion exhausted level"` when the adjusted size is zero is
   nice-to-have, not required.

3. **Plumb the parameter through `match_back` and `match_lay`.**
   Both accept `already_matched_at_top` and forward it to
   `_match`. Default 0.0 so external callers (live wrapper, tests)
   keep working unchanged.

4. **Update `BetManager.place_back`/`place_lay`.** Before calling
   the matcher, look up
   `key = (runner.selection_id, BetSide.BACK_or_LAY, top_price)`
   on `_matched_at_level`. The tricky bit: we don't *know* the
   final fill price until the matcher picks `top`, so we need to
   peek at the same junk-filtered top the matcher will pick. Two
   options:
    - **(A)** Compute the filtered top in `BetManager` first, then
      pass `(top.price, already_matched_at_top)` into the matcher.
      Cleanest separation but duplicates filter logic.
    - **(B)** Add a small helper on `ExchangeMatcher` —
      `pick_top_price(levels, reference_price, lower_is_better)` —
      that returns the post-filter top price (or None) without
      doing a fill. `BetManager` calls it, looks up the
      accumulator, then calls `match_back`/`match_lay` with the
      result. Less duplication, slightly more matcher API.
   **Recommended: (B).** It keeps the filter logic in one place
   and the helper is still a pure function.

5. **After a successful match, increment the accumulator.** In
   `place_back` / `place_lay`, after `self.bets.append(bet)`,
   add `result.matched_stake` to
   `self._matched_at_level[key]` (creating the entry if absent).

6. **Run the existing matcher and bet-manager test suites.** Both
   must still pass with no modification. If anything breaks, the
   default-zero behaviour was not preserved — fix before adding
   new tests.

## Tests to add

Create `tests/research_driven/test_r2_self_depletion.py`:

1. **Two back bets at the same price, same runner.** Top-of-book
   £21 at price P. Place £12.10, then place £17 *with the same
   ladder snapshot*. First fill = £12.10 at P. Second fill =
   £8.90 at P (not £17). Total matched stake at P = £21.
2. **Two back bets at *different* prices, same runner.** First
   fill at P1, second fill at P2 ≠ P1. Both bets fill at the
   stated prices regardless of accumulator state at P1. (Sanity
   check that the key is per-price.)
3. **Two back bets at the same price, *different* runners.** Both
   fill at the stated prices regardless of accumulator state for
   the other runner. (Sanity check that the key is per-selection.)
4. **Back bet at P, then lay bet at P, same runner.** Both fill
   independently — back accumulator does not affect lay
   accumulator. (Sanity check that the key is per-side.)
5. **Same-price back bets across two races.** Race A fills £21 at
   P; race B starts with a fresh `BetManager` and again fills £21
   at P at first placement. (Sanity check that the accumulator
   resets per race — should be implicit from env behaviour but
   worth pinning.)
6. **Skipped-reason on self-exhaustion.** A second bet at a fully-
   self-exhausted price returns `matched_stake == 0.0` with a
   non-None `skipped_reason`. The exact string is not asserted —
   just that it's truthy and mentions self-depletion.

All CPU, all fast.

## Manual tests

None. This is a unit-testable bug fix and the manual replay UI
will reflect it automatically once the unit tests pass.

## Session exit criteria

- All 6 new tests pass.
- All existing matcher and bet-manager tests pass with no
  modification.
- `raw + shaped ≈ total_reward` invariant test still passes.
- `progress.md` Session 18 entry: what shipped, what files
  changed, the `(A)` vs `(B)` choice for the helper, any surprise.
- `bugs.md` R-2 entry updated with "fixed in session 18, commit
  <sha>" — keep the entry, add the close-out line at the bottom.
- `lessons_learnt.md` updated only if there *was* a surprise.
- `master_todo.md` Phase 0 R-2 box ticked.
- Commit.

## Do not

- Do not put the accumulator on `ExchangeMatcher`. The matcher is
  vendored into `ai-betfair`; adding sim-only state breaks that
  contract. See `hard_constraints.md` #7.
- Do not change the single-price rule or the junk filter. R-2 is
  a self-depletion bug, not a ladder-walking bug. Walking the
  ladder to "find more liquidity" after self-depletion is exactly
  the phantom-profit incident reborn.
- Do not skip the same-side / different-runner / different-side /
  cross-race sanity tests. Each one pins a different axis of the
  accumulator key.
- Do not start work on P3/P4 in the same session. R-2 lands
  before P3/P4 specifically because P3/P4 make the bug worse;
  bundling them would make the fix harder to verify.
