# Predictor-integration data-bridging — follow-on plan

**Origin:** spawned out of `plans/predictor-integration/` Session 02
(2026-05-10). Session 02 closed cleanly with the env wiring +
flag plumbing + byte-identical regression guard, but the
`data/feature_engineer.py::_inject_predictor_outputs` injection
(integration_contract.md §2) was identified as a non-trivial
data-shaping problem that warranted its own plan rather than
bundling.

**Revised scope (2026-05-10, after operator pointer to ai-betfair):**
the data-bridging is significantly smaller than the original
write-up suggested. Both rl-betfair (training) and ai-betfair
(live) already carry the raw race-card data in their respective
runtime parquets / MySQL tables. Both repos already share
`data.episode_builder.RunnerMeta` + `PastRace`. The remaining
work is a pure-function aggregator on top of those shared shapes.

## What's outstanding

The `PredictorBundle` is loaded; the env's RUNNER_KEYS already
carry slots for the 18 predictor outputs; the
`use_race_outcome_predictor` / `use_direction_predictor` flags
are plumbed. When BOTH flags are off, the env produces the
flag-off byte-identical baseline (`hard_constraints.md §1`,
verified by
`tests/test_predictor_integration.py::test_flag_off_is_byte_identical_to_pre_plan`).

The remaining piece: when a flag is True, the env must build a
DataFrame in the shape the GBMs expect and call
`bundle.predict_race(race_card_df)` (or `predict_tick(window)`).
The DataFrame's contents come from rl-betfair's existing race-card
data; this plan writes the aggregation functions that fill the
prior-form columns the GBMs expect, then wires them into
`data/feature_engineer.py::_inject_predictor_outputs`.

## What's actually in rl-betfair / ai-betfair already

### Shared `RunnerMeta` (in `data.episode_builder`)

Both repos already construct `RunnerMeta` objects per runner.
The runner parquet (rl-betfair) and the MySQL `RunnerMetaData`
+ `RaceCardRunners` tables (ai-betfair) carry every F1 raw field
the predictor needs:

- Categoricals: `course` (in `Race`), `race_class`, `race_type`,
  `surface` (in `Race`), `sex_type`, `wearing` / `headgear`.
- Numerics: `age`, `weight_value`, `stall_draw`,
  `forecastprice_*`, `official_rating`, `adjusted_rating`,
  `days_since_last_run`, distance (in `Race`).

Plus a `past_races: tuple[PastRace, ...]` field with full
prior-race history per runner (date, course, distance, going,
bsp, position-as-`"4/8"`, jockey, race_type). On ai-betfair
this is parsed from the same JSON. On rl-betfair it's parsed
from the parquet's `past_races_json` column.

### What's missing

The GBMs' input contract requires per-runner aggregates derived
FROM `past_races`:

- **F2 (champion, 21 cols):** F1 + 6 prior-form aggregates —
  `prior_runs`, `prior_wins`, `prior_places`, `prior_win_rate`,
  `prior_place_rate`, `days_since_prior_run`. All derivable from
  `past_races` by counting + parsing the `position` field
  (`"4/8"` → position 4 of 8 runners → won iff position==1,
  placed iff position ≤ N where N depends on field_size).
- **F5 (ranker, 43 cols):** F2 + jockey rolling-window aggregates,
  trainer rolling-window aggregates, jockey-trainer combo
  aggregates, course/jockey/trainer target encodings. Same
  source data (`past_races` carries jockey per past race;
  rl-betfair has `jockey_name` / `trainer_name` per current
  runner; aggregates are computed across the day's runner set).
- **V2 (direction, 32×26 window):** every column is in the tick
  parquet that the env already iterates over. The 10 lag/window
  stats are computable from `TickHistory` which the env already
  buffers for velocity features.

## Design — single shared aggregator

Land ONE module in `data.episode_builder` (or a new
`data/predictor_features.py`) that exposes:

```python
def compute_f2_aggregates(
    runner_meta: RunnerMeta,
    *,
    as_of_date: date,
) -> dict[str, float]:
    """Compute the 6 F2 prior-form aggregates from runner_meta.past_races.

    Strict <as_of_date filter (no leakage). Returns
    {'prior_runs', 'prior_wins', 'prior_places', 'prior_win_rate',
    'prior_place_rate', 'days_since_prior_run'}. NaN-safe — a
    rookie runner with no prior_races returns
    {'prior_runs': 0.0, 'prior_wins': 0.0, ...}.
    """


def compute_f5_aggregates(
    race_card: list[RunnerMeta],
    *,
    as_of_date: date,
    history: PastFormCache,  # day-scoped cache; lives across runners
) -> dict[int, dict[str, float]]:
    """F2 + per-jockey / per-trainer / per-combo rolling aggregates.
    Returns {selection_id: {col_name: value}}.
    """


def build_predict_race_dataframe(
    race: Race,
    runner_metas: dict[int, RunnerMeta],
    feature_variant: str,  # 'F2' for champion, 'F5' for ranker
) -> pd.DataFrame:
    """Stitch race-level + per-runner data + computed aggregates into
    the exact column shape PredictorBundle.predict_race expects."""
```

This module is **shared by both consumers**:

- **rl-betfair** imports it into
  `data/feature_engineer.py::_inject_predictor_outputs` (the
  Session 02 path that's currently `@pytest.mark.skip`-ed).
- **ai-betfair** imports it into its inference pipeline once
  it's ready to consume predictors live (no plan there yet, but
  ai-betfair already imports from `rl-betfair.data.episode_builder`,
  so this lands automatically).

Aggregation functions are pure (no side effects), live in a
shared dependency-free module, and the predictor repo's
`add_aggregates_for_variant` becomes a reference oracle for
unit-testing rather than a runtime dependency.

## Why this is now smaller than originally scoped

The earlier scoping flagged "vendor the predictor's aggregates
module" as a substantial cost. It was overestimated:

- The predictor's aggregates module operates on its training
  parquet pipeline (one row per runner-per-race, joined across
  ALL training races globally). At inference time, rl-betfair /
  ai-betfair only need the per-runner aggregates LOCAL to the
  runner's own past_races — which is a 5-line group-by inside
  `compute_f2_aggregates`, not a global join.
- For F5 jockey/trainer aggregates, the day's race card
  (current day's `RunnerMeta` set) provides the universe.
  Rolling-window stats over the day are local computations, not
  cross-day joins.
- Both repos share `data.episode_builder` already, so the new
  module gets free reuse with zero plumbing.

The encoder workaround (re-fit `EncoderState` from the predictor
repo's training shards at PredictorBundle startup) STAYS — that
provides the integer mapping the GBM was trained against, which
must match exactly. But that's a 2-second one-time cost, not a
runtime concern.

## Smoke test

`tests/test_predictor_integration.py::test_flag_on_populates_predictor_keys`
is currently `@pytest.mark.skip`-ed pending this work. Once
the bridging lands, the test wires up against a real
`PredictorBundle` + a one-day rl-betfair env construction
with `use_race_outcome_predictor=True`, and asserts the runner
obs slice has non-zero values at the predictor-key indices for
at least one runner.

A second, stricter test cross-checks the aggregation: take a
known-runner from the predictor repo's val split, compute
aggregates via the new shared module, compare against the
predictor repo's own
`add_aggregates_for_variant(variant='F2')` output. Numpy
allclose to floating-point tolerance.

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
- Each-way action surface (Session 04 — separate follow-on
  for the parts 3+ that didn't land in the autonomous run).
- ai-betfair-side live inference wiring. The shared module
  this plan builds is the dependency; the consumer-side
  integration in ai-betfair is a separate small plan there.

## Hand-off

This file lives in `rl-betfair/incoming/` per the cross-repo
postbox convention. When the operator picks it up, the natural
home is a new `plans/predictor-integration-data-bridging/` plan
folder with its own `purpose.md` / `master_todo.md` /
`hard_constraints.md` skeleton, modelled after the existing
plan structure.

The shared-module decision means a sibling drop-note in
`ai-betfair/incoming/` is also worth filing to flag the
upcoming `data.episode_builder` extension to the live-inference
team.
