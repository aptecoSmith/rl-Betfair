# Progress — Market Type Filter

One entry per completed session.

---

## Session 01 — Add gene + env filtering (2026-04-12)

### Changes
- `config.yaml` — added `market_type_filter` gene under
  `hyperparameters.search_ranges` with choices WIN/EACH_WAY/BOTH/FREE_CHOICE.
- `env/betfair_env.py` — new `market_type_filter` param in `__init__`.
  Filters `self.day.races` before `_total_races` is computed. BOTH and
  FREE_CHOICE skip filtering. Filtered subsets bypass feature cache to
  avoid cross-filter cache collisions.
- `agents/ppo_trainer.py` — reads `market_type_filter` from hyperparams
  and passes to `BetfairEnv`.
- `training/evaluator.py` — `evaluate()` and `_evaluate_day()` accept
  `market_type_filter` param, passed to `BetfairEnv`.
- `training/run_training.py` — orchestrator reads
  `agent.hyperparameters.market_type_filter` and passes to evaluator
  in both `_eval_agent` and garaged re-evaluation.

### Tests (10 new, all passing)
- WIN filter keeps only WIN races (2/4)
- EACH_WAY filter keeps only EW races (2/4)
- BOTH keeps all, FREE_CHOICE keeps all
- Default (no filter) keeps all
- Zero races after filtering → episode completes gracefully
- Case insensitive filtering
- Gene exists in config, correct choices
- PopulationManager samples the gene

---

## Session 02 — Evaluator filtering (2026-04-12)

Included in Session 01 — the evaluator already passes
`market_type_filter` through to `BetfairEnv` via the new param.
Zero-race days produce valid `EvaluationDayRecord` with 0 bets/pnl.

---

## Session 03 — Scoreboard + model detail display (2026-04-12)

### Changes
- `api/schemas.py` — added `market_type_filter: str | None` to
  `ScoreboardEntry`.
- `api/routers/models.py` — reads `market_type_filter` from model's
  hyperparameters in `_score_to_entry()`.
- Frontend `scoreboard.model.ts` — added `market_type_filter` field.
- Frontend `scoreboard.html` — "Filter" column with colour-coded badge
  (WIN=blue, EW=amber, BOTH=grey).
- Frontend `scoreboard.scss` — `.filter-badge` styles.
- Frontend `garage.html` / `garage.scss` — same filter badge column.
- Frontend `model-detail.html` / `model-detail.scss` — filter badge
  in header next to architecture.

### Tests
- No new backend tests (Session 03 is display-only).
  All 174 tests pass.
