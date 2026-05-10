# Predictor-integration data-bridging — follow-on plan

**Origin:** spawned out of `plans/predictor-integration/` Session 02
(2026-05-10). Session 02 closed cleanly with the env wiring +
flag plumbing + byte-identical regression guard, but the
`data/feature_engineer.py::_inject_predictor_outputs` injection
(integration_contract.md §2) was identified as a non-trivial
data-shaping problem that warranted its own plan rather than
bundling.

## What's outstanding

The `PredictorBundle` is loaded; the env's RUNNER_KEYS already
carry slots for the 18 predictor outputs; the
`use_race_outcome_predictor` / `use_direction_predictor` flags
are plumbed. When BOTH flags are off, the env produces the
flag-off byte-identical baseline (`hard_constraints.md §1`,
verified by
`tests/test_predictor_integration.py::test_flag_off_is_byte_identical_to_pre_plan`).

The remaining piece: when a flag is True, the env must call
`bundle.predict_race(race_card_df)` /
`bundle.predict_tick(ladder_window)` and write the outputs into
the per-runner feature dict. Today that injection raises by
construction (`PredictorBundle` requires inputs the env
doesn't yet construct).

## The data-shape gap

### Champion + ranker (race-level)

`bundle.predict_race(race_card)` expects a `pandas.DataFrame` with:

- The 21-column F2 contract for the champion (course / race_class /
  race_type / surface / sex / headgear categoricals + 9 race-card
  numerics + 6 prior-form aggregates).
- The 43-column F5 contract for the ranker (F2 + jockey aggregates
  + trainer aggregates + jockey-trainer combo aggregates +
  course/jockey/trainer target encodings).
- `selection_id` and `market_id` for routing outputs.

What rl-betfair's `Race` / `RunnerMetadata` already carries:
course, race_class, race_type, surface (in `Race`); sex,
headgear, age, weight (in `RunnerMetadata`). What it does NOT
carry: the F2 prior-form aggregates (per-runner cumulative wins
/ places / runs over historical races) and the F5 jockey/trainer
rolling-window aggregates.

The predictor repo's
`scripts/outcome_predictor/features/aggregates.py::add_aggregates_for_variant`
is the canonical computation, but it operates on the predictor
repo's parquet pipeline (one row per runner-per-race, ordered
by `race_date`, joined against historical races via `selection_id`,
`jockey_id`, `trainer_id`).

### Direction predictor (per-tick)

`bundle.predict_tick(ladder_window)` expects a `(32, 26)` float32
matrix. The 26 columns are:

- 16 V1 columns: `ltp, back_p1..3, back_s1..3, lay_p1..3,
  lay_s1..3, traded_volume_runner, num_active_runners,
  time_to_off_sec`.
- 10 V2 lag/window stats: `ltp_lag_{1,5,10,30}, ltp_w32_{mean,
  std,min,max,first,n}`.

rl-betfair's `Tick` / `Runner` snapshots carry every V1 column
verbatim and the env's `TickHistory` already buffers per-runner
state for velocity features — so the V2 lag/window stats are
mechanically computable on the rl-betfair side, just not yet
done.

## Two design options

### Option A — embed the predictor's aggregates module in rl-betfair

Pros:
- Pure-Python wiring; no cross-repo coupling at runtime.
- The encoder lazy-fit at `PredictorBundle` startup already
  pulls the predictor's training corpus, so adding a one-shot
  aggregates fit at startup is symmetric.

Cons:
- The aggregates module is 100+ lines and depends on
  pandas-heavy operations rl-betfair would inherit.
- Duplicates a code path that the predictor repo owns.

### Option B — let the trainer pre-compute aggregates per day, hand them to the env

Pros:
- The trainer already knows which day it's training on; can
  pre-compute the day's aggregates once, attach them to the
  `Race` object, and hand off.
- Keeps the env's per-tick path cheap.

Cons:
- New aggregation step in the day-loading pipeline.
- Coupling between training startup and predictor-side feature
  contracts is hard to test in isolation.

## Recommendation

Option B with a fallback: use the predictor repo's parquet
shards directly when present (the encoder workaround already
relies on this), join against rl-betfair's `selection_id` /
`jockey_id` / `trainer_id`. If the predictor parquets are
absent, refuse loudly per hard_constraints §10. This avoids
re-implementing the aggregation logic.

## Smoke test

`tests/test_predictor_integration.py::test_flag_on_populates_predictor_keys`
is currently `@pytest.mark.skip`-ed pending this work. Once
the bridging lands, the test wires up against a real
`PredictorBundle` + a one-day rl-betfair env construction
with `use_race_outcome_predictor=True`, and asserts the runner
obs slice has non-zero values at the predictor-key indices for
at least one runner.

## Adjacent items also gated on this follow-on (added 2026-05-10)

After completing predictor-integration Session 03 (strategy-mode +
genes), three additional items were identified as gated on the
PredictorBundle being instantiated by the trainer/worker:

1. **Trainer registry record extension** — `model_store.create_model`
   should capture `strategy_mode` + `predictor_champion_experiment_id`
   + `predictor_ranker_experiment_id` +
   `predictor_direction_experiment_id` per
   integration_contract.md §5 + hard_constraints.md §7. The
   experiment_ids only exist when a bundle is constructed; no bundle
   is constructed today (predictor flags default off; flag-on path is
   data-bridging-blocked). Easiest landing once the bridging plan
   instantiates bundles.

2. **`tools/reevaluate_cohort.py` predictor experiment_id read** —
   refuse re-eval when the bundle on disk doesn't match the cohort
   row's recorded experiment_ids. Trivial once (1) lands.

3. **`registry/model_store.py::purge_incompatible` extension** —
   refuse a checkpoint whose recorded predictor experiment_ids
   don't match the live bundle. Same dependency.

The schema-migration / JSON-blob approach for (1) is a design
question for the follow-on plan; suggested default: append the
`strategy_mode` + 3 experiment_ids into the existing
`hyperparameters` JSON column rather than adding 4 new SQL columns.
That sidesteps a schema migration and stays Pythonic.

## Out of scope for this follow-on

- Direction-predictor per-tick injection cost profiling
  (Session 03's job).
- Strategy-mode switch (Session 03; doesn't depend on real
  predictor outputs landing).
- Each-way action surface (Session 04).

## Hand-off

This file lives in `rl-betfair/incoming/` per the cross-repo
postbox convention. When the operator picks it up, the natural
home is a new `plans/predictor-integration-data-bridging/` plan
folder with its own `purpose.md` / `master_todo.md` /
`hard_constraints.md` skeleton, modelled after the existing
plan structure.
