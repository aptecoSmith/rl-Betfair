# Predictor-integration autonomous run log

This file tracks per-iteration progress through the
predictor-integration plan. Newest entries at the bottom.
The autonomous-run prompt
(`session_prompts/00_autonomous_full_run.md`) reads this file
to decide what to do next.

## 2026-05-10 13:30 — Session 01, iteration 1

**State entering iteration:** No log existed; bootstrap iteration.
Sibling repo `betfair-predictors/` confirmed present at
`C:/Users/jsmit/source/repos/betfair-predictors`; rl-betfair has
no `predictors/` package yet.

**Work done:**
- Created `plans/predictor-integration/autonomous_run_log.md`
  (this file).
- Read `predictor_contracts.md`, `integration_contract.md`,
  `hard_constraints.md`, `master_todo.md`, and Session 01 prompt.
- Inspected production manifests:
  - `betfair-predictors/production/race-outcome/manifest.json`
    (`experiment_id = 1c15250ee90d1b65`).
  - `betfair-predictors/production/race-outcome-ranker/manifest.json`
    (`experiment_id = b23018bf5c8bcc70`).
  - `betfair-predictors/production/direction-predictor/manifest.json`
    (`experiment_id = conv1d_k3_s1_9659e9e9c3fb`).
- Created `predictors/__init__.py` exporting the public API.
- Created `predictors/segment_router.py` with
  `SegmentRouter.from_path` reading the real
  `segment_performance.json` sidecars. Discovered the real JSON
  carries an undocumented `consumer_hint = "neutral"` value;
  added `ConsumerHint.NEUTRAL` and logged the lesson in
  `lessons_learnt.md`.
- Created `predictors/loader.py` with:
  - `PredictorLoaderError` (loud-fail per hard_constraints §10).
  - `RaceLevelOutputs`, `TickLevelOutputs` dataclasses
    matching `integration_contract.md` §1.
  - `_Manifest` dataclass + `_read_manifest` validating
    required keys, weights-on-disk presence, architecture block.
  - `PredictorBundle.from_manifests` reading all three manifests
    + sidecar segment routers; sibling-repo `sys.path.insert`
    pathway honoured (operator-recommended Path A from
    master_todo.md "After Session 01").
  - `predict_race` / `predict_tick` raise `NotImplementedError`
    with TODO markers — model-load + inference path lands in
    iteration 2 of Session 01.

**Tests run:**
- `python -c "from predictors import ..."` → imports OK.
- End-to-end smoke against the three real production manifests:
  experiment_ids surface correctly; both segment routers index
  all 7 axes (`field_size`, `sp_band`, `distance`, `race_type`,
  `surface`, `agree_disagree_sp`, `confidence_threshold`); known
  strong/weak/unseen lookups return the contractually-correct
  `ConsumerHint` value.
- No formal pytest tests yet — those land in iteration 2 alongside
  the model-load wiring.

**Outstanding for this session (Session 01):**
- Wire the model-load path: champion/ranker `joblib.load(weights)`
  via `betfair-predictors/scripts/outcome_predictor/models.py`;
  direction predictor via `scripts/predictor/models.build_model` +
  `torch.load(weights.pt)`. Champion/ranker are GBMs with custom
  `_GBMTwoHead` / lambdarank wrappers — need to re-pickle through
  the sibling repo on `sys.path` (the loader's
  `_ensure_betfair_predictors_on_path()` already arranges this).
- Implement `predict_race`: cache by `race.market_id`; call champion
  to get `p_win`/`p_placed`; call ranker, compute softmax + rank +
  top1 + top1_high_confidence_flag; lookup champion segment_router
  per market and emit `segment_strong_flag`.
- Implement `predict_tick`: build the 32x26 V2 window, forward
  through the Conv1D, derive `fire_*` from quantile thresholds.
- Write `tests/test_predictor_loader.py` and
  `tests/test_segment_router.py` with the test names enumerated in
  the Session 01 prompt's "Success bar" block.
- Add `predictors.*` paths to `config.yaml` (Session 01 prompt
  §Deliverables) — these are pure path additions, not flag
  plumbing (which is Session 02).

**Next iteration's focus:** Implement the model-load path (start
with the champion's `joblib.load` since it's simplest and
unblocks `predict_race`'s p_win/p_placed branch). Stand up
`tests/test_predictor_loader.py::test_loads_three_manifests` and
`tests/test_segment_router.py::test_loads_segment_performance`
as the first concrete test bar.

**Operator decisions pending:** None — sibling-import via
`sys.path.insert` was the recommended default per master_todo.md
"After Session 01"; iteration 1 honoured it without escalation.

## 2026-05-10 13:42 — Session 01, iteration 2

**State entering iteration:** Skeleton in place; iteration 1
deferred model-load + tests. No regression test exists yet.

**Work done:**
- Inspected pickled production weights:
  - Champion `weights.joblib` is a dict
    `{'win': LGBMClassifier, 'placed': LGBMClassifier,
    'feature_names': list[21], 'artifacts': FitArtifacts,
    'params': dict}`.
  - Ranker `weights.joblib` is a dict
    `{'win_ranker': LGBMRanker, 'placed_model': LGBMClassifier,
    'feature_names': list[43], 'artifacts': ..., 'n_trees': 300,
    'max_depth': 5}`.
  - Direction `weights.pt` is a torch state_dict (8 conv weights
    + 4 head weights, OrderedDict).
- Added typed payload dataclasses in `predictors/loader.py`:
  `_ChampionPayload(win_model, placed_model, feature_names)`,
  `_RankerPayload(win_ranker, placed_model, feature_names)`,
  `_DirectionPayload(model, n_features, n_horizons, n_quantiles,
  quantiles, horizons, time_window)`.
- Promoted `champion`/`ranker`/`direction` from optional to
  required fields on `PredictorBundle`. Eager load at
  `from_manifests` time per hard_constraints §10.
- Implemented private loaders:
  - `_load_champion(manifest)` — `joblib.load`, validate dict
    keys (`win`, `placed`, `feature_names`), wrap in payload.
  - `_load_ranker(manifest)` — same pattern with
    `win_ranker`/`placed_model` keys.
  - `_load_direction(manifest)` — resolves
    `feature_columns(variant)` from
    `betfair-predictors/scripts/predictor/datasets.py`,
    cross-checks `manifest.input_shape.n_features` vs the
    feature-list length, calls
    `scripts.predictor.models.build_model(family='conv1d',
    n_features=..., n_horizons=3, n_quantiles=3,
    arch_kwargs=...)` then `model.load_state_dict(strict=True)`
    + `model.eval()`. Strict load is the architecture-hash
    guard (CLAUDE.md §"fill_prob feeds actor_head" pattern).
- Added `tests/test_segment_router.py` (6 tests):
  loads / strong / weak / insufficient_data / weak-axis-
  dominates-strong / unknown-hint-raises.
- Added `tests/test_predictor_loader.py` (5 tests):
  `test_loads_three_manifests` (full bundle smoke against real
  weights — verifies experiment_ids
  `1c15250ee90d1b65` / `b23018bf5c8bcc70` /
  `conv1d_k3_s1_*`, payload shapes, segment-router axes),
  `test_missing_manifest_raises`,
  `test_schema_mismatch_raises`,
  `test_weights_missing_raises`,
  `test_experiment_ids_captured_for_registry`. Each test
  skips cleanly when the sibling repo is absent — CI on a
  fresh checkout without `betfair-predictors/` won't fail.

**Tests run:**
`pytest tests/test_segment_router.py tests/test_predictor_loader.py -v`
→ **11 passed in 19.27s.** No skips (sibling repo present).

**Outstanding for this session (Session 01):**
- Implement `predict_race(race_card)`: cache by
  `race.market_id`; transform race_card features into the
  21-col F2 / 43-col F5 numeric matrices the GBMs expect
  (categorical encoders + numeric column ordering); call
  champion `predict_proba` → `p_win`, `p_placed`; call ranker
  `predict` → raw scores + softmax + rank + top1 + top1_high_conf.
- Implement `predict_tick(runner, ladder_window)`: build
  32×26 V2 window via
  `scripts.predictor.datasets.feature_columns('V2')`; forward
  through Conv1D; derive `fire_drift / fire_shorten /
  fire_no_signal` from the 7m horizon's `q10/q50/q90` per the
  manifest's `signal_description`.
- Add the remaining Session 01 tests:
  `test_predict_race_returns_per_runner_dict`,
  `test_predict_race_caches_by_market_id`,
  `test_predict_tick_fire_logic`. These need `predict_*`
  implemented.
- Add `predictors.*` paths to `config.yaml` (Session 01 prompt
  §Deliverables).

**Next iteration's focus:** Implement `predict_race`. The
hardest piece is the categorical-encoder handoff — the
GBMs read `course_idx, race_class_idx, race_type_idx,
surface_idx, sex_idx, headgear_idx` (21-col F2 column list).
The encoders themselves are NOT in the joblib payload; they
must be discoverable from the predictor repo's training
artefacts. Investigate
`betfair-predictors/scripts/outcome_predictor/datasets.py`
for the encoder fit/load API, and look at how the predictor
repo's S04 training script bakes the encoder state for
inference.

**Operator decisions pending:** None.

## 2026-05-10 13:49 — Session 01, iteration 3

**State entering iteration:** Loader + first 11 tests green.
Iteration 2's hand-off flagged the categorical-encoder handoff
as the long pole for `predict_race`.

**Work done:**
- Confirmed the GBMs' `EncoderState` is NOT in `weights.joblib`
  or in the registry artifact — has to be re-fit from the
  predictor repo's training corpus at consumer-side startup.
  Logged the finding in `lessons_learnt.md` (entry #2).
- Filed a cross-repo request at
  `betfair-predictors/incoming/persist_encoder_state_alongside_weights.md`
  asking for the encoder to ship alongside weights so future
  consumers don't need the parquet tree to fit them. Per the
  cross-repo-postbox feedback memory.
- Threaded `feature_variant` through `_ChampionPayload` and
  `_RankerPayload` (champion = "F2", ranker = "F5"); both are
  read from each manifest's `training.feature_variant` block;
  loader raises if missing.
- Added `_fit_categorical_encoder(train_corpus, feature_variant)`
  which routes through
  `scripts.outcome_predictor.datasets.load_split` +
  `fit_encoders`. Refuses loudly if
  `betfair-predictors/data/outcome_dataset/` is missing.
- `PredictorBundle.from_manifests` now fits both encoders eagerly
  and stores them on the bundle as `champion_encoder` /
  `ranker_encoder`. The state is held read-only thereafter
  (hard_constraints §4 — predictors frozen).

**Tests run:**
- End-to-end smoke: `PredictorBundle.from_manifests(...)` → 15.93s
  total (manifests + weights + encoder fit). Cardinalities sane
  (course=57, race_class=1, race_type=6, surface=3, sex=16,
  headgear=26).
- `pytest tests/test_segment_router.py tests/test_predictor_loader.py -v`
  → **11 passed in 29.59s.** The full-bundle test now pays the
  encoder-fit cost; isolated segment-router tests are still fast.

**Outstanding for this session (Session 01):**
- Implement `predict_race(race_card)`. Inputs land in Session 02
  via the env's data layer; for Session 01 the implementation can
  accept a `pandas.DataFrame` of runners and the test fabricates
  one. Steps:
    1. `apply_encoders(df, bundle.champion_encoder)` to add
       `_idx` columns.
    2. `numeric_feature_matrix(df, variant='F2')` → champion's X.
    3. Champion: `win_proba = m.predict_proba(X)[:, 1]`,
       `placed_proba = m.predict_proba(X)[:, 1]`.
    4. Ranker: same encoder pattern but variant='F5'; ranker's X
       needs F3/F4/F5 aggregates which the predictor repo's
       `add_aggregates_for_variant` computes — defer adding this
       at Session 01 to "use the simple path: load via
       `load_split` for tests; env feeds it ready-made later".
    5. softmax(scores) → `ranker_softmax_share`; argmax → top1;
       gate top1_high_conf at `softmax_share >= 0.30`.
    6. Cache by `race_card.market_id`.
- Implement `predict_tick`: 32x26 V2 window forward through
  Conv1D; derive `fire_*` from the 7m horizon's `q10/q50/q90`.
- Add the remaining Session 01 tests:
  `test_predict_race_returns_per_runner_dict`,
  `test_predict_race_caches_by_market_id`,
  `test_predict_tick_fire_logic`.
- Add `predictors.*` paths to `config.yaml`.

**Next iteration's focus:** Implement `predict_race` accepting a
`pd.DataFrame` of runners with the F2 raw column set. Build the
test fixture from the predictor repo's val split (one market) so
the test cross-checks against the model's own validation
behaviour. This is one focused chunk: encoder application +
GBM forward + softmax + caching + the test.

**Operator decisions pending:** None.

## 2026-05-10 13:57 — Session 01, iteration 4

**State entering iteration:** Bundle + encoders ready; predict_race
was the long pole.

**Work done:**
- Implemented `PredictorBundle.predict_race(race_card, *,
  high_confidence_threshold=0.30)`:
  - Accepts a `pandas.DataFrame` with the F2/F5 raw column union +
    `selection_id` + `market_id`. The env data layer will assemble
    this in Session 02; tests fabricate from the predictor repo's
    val split.
  - One market per call (`ValueError` if `market_id.unique()` > 1).
  - `selection_id` column required (raises early — moved BEFORE the
    cache lookup so a malformed call still raises even on a cache hit).
  - Champion path: `apply_encoders(df, champion_encoder)` →
    `numeric_feature_matrix(df, 'F2')` → check column ordering matches
    `feature_names` → `predict_proba(X)[:, 1]` → `p_win` + `p_placed`.
  - Ranker path: same encoder pattern with `ranker_encoder` + 'F5' →
    `predict(X)` raw scores → softmax-within-market → rank from
    argsort → `top1` from argmax → `top1_high_confidence` gated at
    `softmax_share >= 0.30`.
  - Champion segment_router: `lookup({'field_size', 'race_type',
    'surface'})` → `segment_strong = (hint == STRONG)`. Distance
    bucket axis deliberately abstains (no canonical bucketer in the
    rl-betfair runtime context yet).
  - All output dict values cast to plain Python `bool` / `float` /
    `int` (sidesteps `np.True_ is True` test confusion downstream).
  - Cache by `market_id`; same `RaceLevelOutputs` object returned on
    repeat calls.
- Added `_market_features_for_segment_lookup(race_card)` helper —
  small, isolated; documents which axes are derived vs abstained.
- Added 6 tests to `tests/test_predictor_loader.py`:
  `test_predict_race_returns_per_runner_dict` (shapes + invariants:
  per-runner dicts keyed by selection_id, p_win in [0,1], softmax
  sums to 1.0, exactly one top1, ranks are 1..n with no dupes),
  `test_predict_race_caches_by_market_id` (`is`-identity check),
  `test_predict_race_top1_high_confidence_threshold` (gate at 0.30),
  `test_predict_race_rejects_multi_market_dataframe`,
  `test_predict_race_requires_selection_id`,
  `test_predict_race_rejects_non_dataframe`.
- Module-scoped `_bundle_and_val` pytest fixture so the 16-second
  bundle + val-split load is paid once for all 6 tests.

**Tests run:**
- Smoke: predict on real val market `1.257386603` (9 runners). Top
  pick has `p_win=0.493, ranker_softmax_share=0.899, top1=True,
  top1_high_confidence=True`. `sum(p_win) ≈ 0.99` (sane — calibrated
  champion's win-prob rows are independent, not a softmax). `sum
  (softmax) = 1.0` exactly.
- Two test failures on first run; both fixed:
  - `np.True_ is True` → cast booleans to Python `bool`.
  - `requires_selection_id` failed because the cache short-circuited
    before the column-presence check; moved the check earlier.
- Re-run: `pytest tests/test_segment_router.py
  tests/test_predictor_loader.py -v` → **17 passed in 44.16s.**

**Outstanding for this session (Session 01):**
- Implement `predict_tick(runner, ladder_window)`: forward a
  32x26 V2 window through the Conv1D, derive `fire_*` from the
  7m horizon's `q10/q50/q90`.
- Add `test_predict_tick_fire_logic` (the third Session-01-prompt
  success-bar item).
- Add `predictors.*` paths to `config.yaml`.
- Commit Session 01 once those land + byte-identical regression
  test gate (the regression test itself is Session 02's
  deliverable, but Session 01 must merge cleanly).

**Next iteration's focus:** Implement `predict_tick`. Conv1D
forward is straightforward; the layered ladder-window construction
needs care because the predictor's V2 column list is per-tick
(`ltp`, `back_p1..3`, `back_s1..3`, `lay_p1..3`, `lay_s1..3`,
`traded_volume_runner`, `num_active_runners`, `time_to_off_sec`,
+ V2 lag/window stats). For Session 01 the test fixture can pull
a real (32, 26) window from the predictor repo's val parquets.

**Operator decisions pending:** None.

## 2026-05-10 14:05 — Session 01, iteration 5

**State entering iteration:** predict_race + 6 tests green;
predict_tick was the last predict-method outstanding.

**Work done:**
- Implemented `PredictorBundle.predict_tick(ladder_window)`:
  - Validates `(time_window, n_features)` shape against
    `direction.time_window` / `direction.n_features` per the
    manifest. ndim != 2 OR shape mismatch → `ValueError`.
  - Forward through Conv1D under `torch.no_grad()` with
    `(B=1, T, F)` batch shape; output `(1, n_horizons,
    n_quantiles)` reduced to a 9-element grid.
  - `q10/q50/q90 × 1m/3m/7m` mapped via the manifest's
    `horizons` / `quantiles` tuples (no positional assumption —
    the manifest is the source of truth).
  - Fire logic from the manifest's `signal_description` block:
    `fire_drift = (q50_7m >= +5) AND (q10_7m >= 0)`,
    `fire_shorten = (q50_7m <= -5) AND (q90_7m <= 0)`,
    `fire_no_signal = NOT (drift OR shorten)`. Mutually exclusive
    AND exhaustive by construction.
  - All output values cast to plain Python `float` / `bool`.
  - No caching — per-tick is cheap (~ms) and the window changes
    every tick.
- Renamed param from `(runner, ladder_window)` to just
  `(ladder_window)` to match what the env's data layer can supply
  by Session 02. The integration_contract mention of `runner` was
  shorthand; the real handoff is the engineered window.
- Added 4 tests to `tests/test_predictor_loader.py`:
  - `test_predict_tick_returns_dataclass` — output type contract.
  - `test_predict_tick_fire_logic` — runs 5 random windows, asserts
    mutual-exclusion `sum(flags) == 1` AND that each flag matches
    the manifest's threshold derivation against the actual
    quantile output.
  - `test_predict_tick_rejects_wrong_shape` — covers wrong rank
    (3-D), wrong feature dim, wrong time dim.
  - `test_predict_tick_is_deterministic` — cross-checks
    `intended_consumer.md` §Determinism guarantee.

**Tests run:**
- `pytest tests/test_predictor_loader.py tests/test_segment_router.py -v`
  → **21 passed in 44.84s.**

**Session 01 success bar (all items now green):**
- ✅ test_loads_three_manifests
- ✅ test_predict_race_returns_per_runner_dict
- ✅ test_predict_race_caches_by_market_id
- ✅ test_predict_tick_fire_logic
- ✅ test_missing_manifest_raises
- ✅ test_schema_mismatch_raises
- ✅ test_loads_segment_performance
- ✅ test_lookup_strong_segment
- ✅ test_lookup_weak_segment
- ✅ test_lookup_insufficient_data

**Outstanding for this session (Session 01):**
- Add `predictors.*` paths to `config.yaml` (Session 01 prompt
  §Deliverables).
- Commit Session 01 work to master with the project's commit
  conventions: `feat(predictor-integration): Session 01 — predictor
  loader + segment router + 21 tests`. Reference the session
  number per the autonomous-run prompt's commit template.

**Next iteration's focus:** Add `predictors.*` paths to
`config.yaml`; commit Session 01; flip into Session 02
(observation wiring — RUNNER_KEYS extension, OBS_SCHEMA_VERSION
7→8, byte-identical regression test).

**Operator decisions pending:**
- "After Session 01" decision per master_todo.md is the
  sibling-import vs installable-package question. Iteration 1
  honoured the recommendation (sys.path.insert) without
  escalation; the cross-repo postbox drop in
  `betfair-predictors/incoming/persist_encoder_state_alongside_weights.md`
  notes the encoder issue is the more material follow-on but is
  separately filed and doesn't block Session 02.

