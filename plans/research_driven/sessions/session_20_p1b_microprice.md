# Session 20 — P1b: weighted microprice feature

## Before you start — read these

- `../purpose.md`
- `../hard_constraints.md` — constraints 12, 13 (same as session 19).
- `../analysis.md` §3
- `../proposals.md` P1
- `../progress.md` — confirm session 19 (P1a OBI + schema bump
  infrastructure) has landed.
- `../initial_testing.md`
- `env/features.py` — created in session 19.
- `env/betfair_env.py` — observation builder.

## Goal

Add a second per-runner feature `weighted_microprice` to the
observation vector. Reuses the schema-bump infrastructure from
session 19. Smaller scope than session 19 because the
infrastructure is already in place.

`weighted_microprice` = size-weighted midpoint of the top-N
levels per side:

```
mp = (Σ back_size_i × back_price_i + Σ lay_size_i × lay_price_i)
   / (Σ back_size_i + Σ lay_size_i)
```

with fallback to LTP when both sides sum to zero.

## Inputs — constraints to obey

1. **Same vendoring rule as session 19.** Implementation is a pure
   function in `env/features.py`. No `numpy`, no env imports.
2. **Bounded by best back and best lay.** For any non-degenerate
   book, the result must lie within the best-back / best-lay
   range. Tested explicitly.
3. **Schema bump.** Adding a new feature column requires a fresh
   schema bump (one bump per feature is fine — the loader is
   already built; just increment the version constant). Pre-P1b
   checkpoints (which include P1a) get refused with a clear error.
4. **`N` is configurable separately from OBI's `N`.** New config
   key `features.microprice_top_n: 3`. Different features may
   want different windows.

## Steps

1. **Add `compute_microprice(back_levels, lay_levels, n,
   ltp_fallback)` to `env/features.py`.** Returns float. If both
   sides sum to zero, returns `ltp_fallback`. If `ltp_fallback`
   itself is None or non-positive, raises `ValueError` (a runner
   with no LTP and no liquidity is unpriceable; do not silently
   return zero).

2. **Wire into the observation builder** after the OBI column.
   Order matters: existing P1a code must not move.

3. **Add `features.microprice_top_n: 3` to `config.yaml`.** Plumb
   through `BetfairEnv.__init__`.

4. **Bump obs schema version by 1.** Update the loader's expected
   version.

5. **Expose in `info["debug_features"]`** alongside `obi_topN`.

## Tests to add

Create `tests/research_driven/test_p1b_microprice.py`:

1. **Pure function: symmetric book.** Equal sizes, equal price
   gaps → microprice equals the simple midpoint.
2. **Pure function: asymmetric sizes.** More size on the back
   side → microprice pulls *toward* the back-best price.
3. **Pure function: empty book with LTP fallback.** Returns the
   LTP value.
4. **Pure function: empty book without LTP fallback.** Raises.
5. **Pure function: bounded by best-back and best-lay.** A
   randomised property test (5 random books) — the result is
   within `[best_back_price, best_lay_price]` for every one.
6. **Env smoke.** A 1-race fixture run; assert
   `weighted_microprice` appears in `info["debug_features"]` and
   lies between best back and best lay.
7. **Schema-bump loader refuses pre-P1b checkpoint.** Including
   a P1a checkpoint — both must be refused.

All CPU, all fast.

## Manual tests

- **Open one race in the replay UI**, find a tick where the
  visible book is asymmetric (clearly more on one side). Confirm
  microprice pulls toward the heavier side.

## Session exit criteria

- All 7 new tests pass.
- All existing tests (including P1a) still pass.
- `env/features.py` now contains exactly two functions: OBI and
  microprice. No accidental drift into windowed features.
- `progress.md` Session 20 entry.
- `ui_additions.md` row for microprice in per-runner panel.
- `master_todo.md` P1 sub-bullet for P1b ticked.
- Commit.

## Do not

- Do not change the OBI function. It's done. Touching it here
  invites scope creep.
- Do not add `traded_delta_T` or `mid_drift_T` in this session.
  They're session 21.
- Do not let microprice depend on the OBI value. They're
  independent features even though they sometimes correlate.
- Do not silently substitute LTP when LTP is also missing. Raise.
  A runner with no LTP and no liquidity is genuinely unpriceable
  and the env's existing "skip this runner" path should already
  handle it upstream.
