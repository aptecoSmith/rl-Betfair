# Session 5 — Wire arb features into env + schema bump

## Before you start — read these

- `plans/arb-improvements/purpose.md`
- `plans/arb-improvements/master_todo.md` — Phase 2, Session 5.
- `plans/arb-improvements/testing.md`
- `plans/arb-improvements/hard_constraints.md` — obs schema bump is
  loud; features are unconditional (not gated on scalping_mode);
  `data/feature_engineer.py` stays in lockstep.
- `plans/arb-improvements/progress.md` — read Session 4.
- `env/betfair_env.py:97` — "These MUST match the keys produced by
  `data/feature_engineer.py` exactly". This rule is non-negotiable.

## Goal

Expose the four Session-4 feature functions in the observation
vector. Bump `OBS_SCHEMA_VERSION`. Mirror the new keys in
`data/feature_engineer.py` so cached features on disk stay aligned.

## Scope

**In scope:**

- Extend `RUNNER_KEYS` in `env/betfair_env.py` with three new keys
  at the end of the runner feature block:
  - `arb_lock_profit_pct`
  - `arb_spread_ticks_norm` (the raw ticks value divided by
    `MAX_ARB_TICKS` so the feature is in `[-1, 1]`)
  - `arb_fill_time_norm`
- Extend `MARKET_KEYS` with one new key:
  - `arb_opportunity_density_60s`
- Increment `OBS_SCHEMA_VERSION` by 1.
- `_build_observation()` populates the new keys using the pure
  functions from `env/features.py`. The "any arb available" flag
  for density is computed in one pass per tick: iterate runners,
  compute `arb_lock_profit_pct`, set the flag if any runner has a
  positive value.
- A new tick-history entry for density: per-tick
  `(timestamp_s, any_arb_available)` buffer on `BetfairEnv` that
  trims to the last 60 s window.
- Mirror the new keys in `data/feature_engineer.py` — add the same
  four keys with the same computation. Whatever code path produces
  cached features must also produce these new columns.
- Update the `MARKET_DIM` / `RUNNER_DIM` derived constants
  (`env/betfair_env.py:201–203`) so the comment and computed values
  are correct.

**Out of scope:**

- Oracle / BC work (Phase 3).
- Gating on `scalping_mode`. Features are always on — this has been
  decided and documented in `purpose.md`.
- Changing any existing feature's value. Adding, not editing.

## Exact code path

1. `env/betfair_env.py:132–189` — `RUNNER_KEYS` extension. Append
   the three arb keys after `book_churn`. Update the `cross-runner`
   / `velocity` block counts in the comments.
2. `env/betfair_env.py:99–120` — `MARKET_KEYS` extension with
   `arb_opportunity_density_60s`. Add a comment marker for the new
   dim.
3. `env/betfair_env.py` — bump `OBS_SCHEMA_VERSION` (search for the
   constant, it's near the top of the file).
4. `env/betfair_env.py::_build_observation` — for each runner, call
   the Session-4 functions and insert the values at the right
   positions. For the density feature, maintain a rolling buffer on
   `self._arb_density_history` (trimmed in `_build_observation`).
5. `data/feature_engineer.py` — add the same keys. Use the same
   pure functions so the values are identical by construction.
6. `env/betfair_env.py:201–203` — update `RUNNER_DIM` / `MARKET_DIM`
   expected values in the comment.

## Tests to add (all CPU-only, fast)

Create `tests/arb_improvements/test_arb_features_wiring.py`:

1. **Obs shape has the new dims.** Instantiate `BetfairEnv` on a
   fixture day; assert `observation_space.shape[0]` equals the
   pre-session shape + (3 × max_runners) + 1.

2. **Schema bump refuses pre-bump checkpoints.** Construct a dummy
   checkpoint dict with the previous `obs_schema_version`; call
   `validate_obs_schema` and assert it raises `ValueError` with a
   message mentioning the new version.

3. **Env features match pure-function output.** On a synthetic
   first tick where the book is crossed on one runner,
   construct observation and confirm the value at the
   `arb_lock_profit_pct` slot equals the direct call to
   `compute_arb_lock_profit_pct` on the same inputs.

4. **Density buffer trims to 60 s.** Advance the env by
   synthetic ticks covering 70 s; assert only the last 60 s of
   the density history is retained.

5. **Density value in obs matches direct call.** Build a synthetic
   history, set `self._arb_density_history` to it, call
   `_build_observation`, assert the density slot equals the
   direct call to `compute_arb_opportunity_density`.

6. **`data/feature_engineer.py` produces the same keys.**
   Introspect the engineer's output column list; assert the three
   runner keys and one market key are present in the same order
   as `RUNNER_KEYS` / `MARKET_KEYS`.

7. **`raw + shaped ≈ total_reward` invariant still holds** on a
   synthetic scalping race with the new features enabled (reward
   path is unchanged — sanity check).

8. **Unpriceable runner → zero feature values** (no NaN leakage
   into the obs vector).

9. **Obs vector has no NaNs after 100 synthetic ticks** on a
   fixture day including the usual liquidity gaps. Use
   `np.isfinite(obs).all()`.

## Session exit criteria

- All 9 tests pass.
- All 1844+ existing scalping / forced-arb tests pass.
- `progress.md` Session 5 entry written. End-of-Phase-2 note: "the
  policy can now see arb opportunities as first-class features;
  training signal and representation stability are both in place
  for Phase 3."
- `lessons_learnt.md` updated if anything surprising came up —
  especially around the density buffer or feature-engineer
  alignment.
- Commit: `feat(env): arb-opportunity features + obs schema v{N}`.
- `git push all`.

## Do not

- Do not silently zero-pad old checkpoints. Refuse to load — the
  invariant is "loud schema bumps".
- Do not gate on `scalping_mode`. Features are unconditional.
- Do not forget `data/feature_engineer.py`. The cached-feature
  invariant is load-bearing — a mismatch is an obscure bug that
  surfaces far from where it was caused.
- Do not add GPU tests.
