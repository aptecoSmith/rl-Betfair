# Market Type Filter — All Sessions (01–03)

Work through sessions sequentially. Complete each session fully
(code + tests + progress.md entry) before moving to the next.
Commit after each session.

## Before you start — read these

- `plans/issues-11-04-2026/04-market-type-filter/purpose.md` — design
  and rationale.
- `plans/issues-11-04-2026/04-market-type-filter/hard_constraints.md`
  — non-negotiables (especially: no schema version bumps).
- `plans/issues-11-04-2026/04-market-type-filter/master_todo.md` —
  session breakdown.
- `config.yaml` — existing gene definitions under
  `hyperparameters.search_ranges`.
- `agents/population_manager.py` — how genes are sampled, crossed
  over, and mutated. The new gene is a `str_choice` — same type as
  `architecture_name`.
- `env/betfair_env.py` — `reset()` method where races are loaded.
- `data/episode_builder.py` — `Race.market_type` field.

---

## Session 01 — Add gene + env filtering

### Context

The environment iterates through `self.day.races` in `reset()` and
processes every race. `Race.market_type` is a string: `"WIN"`,
`"EACH_WAY"`, or `""`.

Genes are defined in `config.yaml:hyperparameters.search_ranges` and
automatically handled by `PopulationManager` for sampling, crossover,
and mutation. A `str_choice` gene requires only the YAML definition.

### What to do

1. **config.yaml** — add under `hyperparameters.search_ranges`:
   ```yaml
   market_type_filter:
     type: str_choice
     choices:
       - WIN
       - EACH_WAY
       - BOTH
       - FREE_CHOICE
   ```

2. **env/betfair_env.py** — in `reset()`, after the day's races are
   loaded (after `self.day = ...`), filter based on `market_type_filter`:
   ```python
   mtf = self._config.get("market_type_filter", "BOTH")
   if mtf not in ("BOTH", "FREE_CHOICE"):
       self.day.races = [
           r for r in self.day.races
           if (r.market_type or "").upper() == mtf
       ]
   ```
   BOTH and FREE_CHOICE both present all races (no filtering).
   Read the gene from whatever config dict is passed to the env.
   Find where hyperparameters are merged into the env config in the
   orchestrator (`training/run_training.py`) and ensure
   `market_type_filter` is included.

3. **Handle zero races:** If `self.day.races` is empty after
   filtering, the episode should complete immediately. Check how
   `step()` handles reaching the end of the race list — it likely
   already returns `terminated=True`. Verify this works when the
   race list starts empty.

4. **training/run_training.py** — find where per-agent env config is
   built. The hyperparameters dict already flows from the model's
   record. Confirm that `market_type_filter` will be present in the
   env config without special-casing (it should be, since it's just
   another hyperparameter key).

### Tests

Add to an appropriate test file (or new `tests/test_market_type_filter.py`):

1. **Filtering works:** Create a Day with 2 WIN and 2 EW races.
   Reset env with filter=WIN → `len(env.day.races) == 2` and all WIN.
2. **BOTH is default:** Reset without filter → all 4 races.
3. **EW filter:** Reset with filter=EACH_WAY → only EW races.
4. **Zero races:** Day with only WIN races, filter=EACH_WAY → episode
   completes with reward=0, no crash.
5. **Gene in population:** Verify `market_type_filter` appears in
   sampled hyperparameters and is one of the three choices.

### Exit criteria

- All tests pass (new and existing). `progress.md` updated. Commit.

---

## Session 02 — Evaluator filtering

### Context

`training/evaluator.py:_evaluate_day()` creates a fresh env per day,
running the model's policy. The env config comes from the model's
hyperparameters. The filter gene should already be in the
hyperparameters dict from Session 01.

### What to do

1. Verify that the evaluator passes the model's hyperparameters
   (including `market_type_filter`) into the env config. If not,
   wire it through.

2. Handle zero-race eval days: the evaluator must create a valid
   `EvaluationDayRecord` with `bet_count=0`, `day_pnl=0.0`,
   `profitable=False`, etc. It must NOT skip the day.

3. Verify the bet logs are correctly filtered: a WIN-only model's
   bet log parquet should contain no bets from EW races.

### Tests

1. WIN-only model evaluated on mixed data → bet log contains only
   WIN market bets.
2. Zero-race eval day (all races filtered out) → day record exists
   with bet_count=0.
3. BOTH model → unchanged behaviour (regression).

### Exit criteria

- All tests pass. `progress.md` updated. Commit.

---

## Session 03 — Scoreboard + model detail display

### Context

The API returns `ScoreboardEntry` objects. The frontend displays them
in the scoreboard and garage tables, and in the model detail page.

### What to do

1. **api/schemas.py** — add `market_type_filter: str | None = None`
   to `ScoreboardEntry`.

2. **api/routers/models.py** — in `_score_to_entry()`, read
   `model.hyperparameters.get("market_type_filter", "BOTH")` and
   include in the response.

3. **Frontend scoreboard** — add a badge/tag column or inline badge
   next to the model ID. Colour: WIN=blue, EW=amber, BOTH=grey.

4. **Frontend garage** — same badge.

5. **Frontend model-detail** — show in the metadata/header area.

### Tests

- API response includes market_type_filter for models that have it.
- Old model (no gene) → API returns "BOTH" (or null, handled by
  frontend as BOTH).

### Exit criteria

- All tests pass. `progress.md` updated. Commit.

---

## Cross-session rules

- Run `pytest tests/ -x` after each session. All tests must pass.
- Do not change OBS_SCHEMA_VERSION or ACTION_SCHEMA_VERSION.
- Do not "improve" unrelated code. Scope is tight.
- Commit after each session with a clear message referencing the
  session number.
