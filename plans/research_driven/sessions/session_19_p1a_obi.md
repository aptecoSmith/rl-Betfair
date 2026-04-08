# Session 19 — P1a: OBI feature + obs schema bump

## Before you start — read these

- `../purpose.md`
- `../hard_constraints.md` — **constraint 12 (features byte-
  identical sim ↔ live) and constraint 13 (schema bumps refuse
  old checkpoints)** apply directly.
- `../analysis.md` §3
- `../proposals.md` P1
- `../master_todo.md` Phase 1
- `../progress.md` — confirm session 18 (R-2 fix) has landed.
- `../initial_testing.md`
- `../downstream_knockon.md` §1 — knock-on for `ai-betfair`. Not
  in scope for this session, but informs the design.
- `env/betfair_env.py` — wherever the observation vector is
  assembled per runner per tick.
- `data/episode_builder.py` — how ladder rows are produced today.

## Goal

Add a single per-runner feature `obi_topN` to the observation
vector and bump the obs schema version. This is the **smallest**
P1 feature; it sets the precedent for the schema-bump
infrastructure that sessions 20 and 21 reuse.

`obi_topN = (sum(back_size_top_N) − sum(lay_size_top_N))
            / (sum(back_size_top_N) + sum(lay_size_top_N))`

with `obi_topN = 0.0` when both sides sum to zero.

## Inputs — constraints to obey

1. **Feature is computed by code that can be vendored verbatim
   into `ai-betfair`.** Implementation lives in a new module
   `env/features.py` (or similar) that has no dependencies beyond
   the standard library and the existing `PriceLevel`-shaped
   inputs. No `numpy`, no env-internal imports.
2. **Schema bump is loud.** Loading a pre-P1 checkpoint with the
   new env raises a clear error mentioning the schema version
   mismatch. Loading a P1 checkpoint with a pre-P1 env (e.g. an
   older copy of the code) also raises. Silent zero-pad or
   zero-truncate is forbidden.
3. **`N` is configurable.** Default `N=3`. Lives in `config.yaml`
   as `features.obi_top_n` so future tuning is one knob.
4. **Determinism.** For a given ladder snapshot, the feature value
   is byte-identical across runs.

## Steps

1. **Create `env/features.py`.** Pure functions only. Add
   `compute_obi(back_levels, lay_levels, n)` returning `float`.
   Empty/zero handling: returns `0.0` if either or both sides
   have zero summed size.

2. **Wire `compute_obi` into the observation builder.** Find the
   per-runner row assembly in `env/betfair_env.py` (or wherever
   `_build_observation` lives) and append `obi_topN` after the
   existing ladder columns. The row layout is part of the schema
   bump in step 4.

3. **Add `features.obi_top_n: 3` to `config.yaml`** under a new
   `features:` section. Plumb through `BetfairEnv.__init__` as
   `self._obi_top_n`.

4. **Bump the obs schema version.** Locate the existing schema
   version constant (search for `OBS_SCHEMA_VERSION` or similar
   — it will exist because the LSTM/transformer arch sessions
   already established the pattern). Increment by 1. Update the
   loader to refuse mismatched versions with a clear error.

5. **Expose the feature in `info["debug_features"]`.** Per-runner
   dict, keyed by selection_id, value is `{"obi_topN": <float>}`.
   This becomes the inspection point for the manual test in
   `manual_testing_plan.md`.

## Tests to add

Create `tests/research_driven/test_p1a_obi.py`:

1. **Pure function: balanced book.** Equal back and lay sums →
   `obi == 0.0`.
2. **Pure function: all back, no lay.** `obi == 1.0`.
3. **Pure function: all lay, no back.** `obi == -1.0`.
4. **Pure function: empty book.** `obi == 0.0` (no exception).
5. **Pure function: respects `n`.** A book with `N+1` levels
   where the (N+1)-th level is huge and asymmetric — the result
   is unaffected, because only top-N are summed.
6. **Env smoke.** A 1-race fixture run; assert `obi_topN` appears
   in `info["debug_features"]` for at least one runner on at
   least one tick, and is in `[-1, 1]`.
7. **Env determinism.** Same tick, same ladder → same value
   across two `BetfairEnv` runs.
8. **Schema-bump loader refuses pre-P1 checkpoint.** Build a
   minimal fake checkpoint with the *previous* schema version,
   try to load it with the new env, assert a clear error is
   raised. (If a real existing checkpoint is on disk you can use
   it instead, but don't *commit* one.)

All CPU, all fast.

## Manual tests

- **Open one race in the replay UI**, find a tick where the
  operator can visually confirm the OBI value matches the
  visible book balance. Add the snippet you spot-checked to
  `progress.md`.

## Session exit criteria

- All 8 new tests pass.
- All existing tests pass.
- `env/features.py` exists, is dependency-free, and contains
  only the OBI function (microprice and windowed features come
  in sessions 20 and 21).
- `info["debug_features"]` populated for at least one race.
- `progress.md` Session 19 entry.
- `ui_additions.md` row for "show OBI in per-runner panel"
  filed against the replay UI section.
- `master_todo.md` Phase 1 P1 box gets a sub-tick for P1a (if
  it doesn't already have sub-bullets, add them).
- Commit.

## Do not

- Do not add `weighted_microprice`, `traded_delta_T`, or
  `mid_drift_T` in this session. They live in sessions 20 and 21
  specifically so each one can be reviewed in isolation.
- Do not add a `numpy` import to `env/features.py`. The whole
  point of the file is that it can be vendored into `ai-betfair`
  without dragging dependencies along.
- Do not silently zero-pad an old checkpoint to load. Refuse it
  loudly. Loaders that silently fix mismatches hide bugs.
- Do not couple OBI's `n` to the existing `top_n` used elsewhere
  in the env (if any). Different features may want different
  windows; the knob is feature-specific.
