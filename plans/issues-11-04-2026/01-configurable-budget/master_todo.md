# Master TODO — Configurable Budget & Percentage-Based P&L

Ordered session list. Tick boxes as sessions land.

When a session completes:
1. Tick its box here.
2. Add an entry to `progress.md`.
3. Append any learnings to `lessons_learnt.md`.
4. Note cross-repo follow-ups in `knockon_ai_betfair.md`.

---

## Phase 1 — Per-plan budget & recording

- [x] **Session 01 — Per-plan starting_budget override**

  Allow training plans to carry an optional `starting_budget` field.
  The orchestrator uses the plan's value if set, else falls back to
  `config.yaml:training.starting_budget`.

  - `registry/model_store.py` — add `starting_budget` to the
    training plan schema (optional float, nullable).
  - `training/run_training.py` — read `plan.starting_budget` and
    pass it to the environment config.  Fall back to global config.
  - `api/routers/training.py` — accept optional `starting_budget`
    in plan creation endpoint.
  - Frontend training plan form — optional budget field with
    placeholder showing the global default.

  **Tests:**
  - Plan with budget=10 → env receives starting_budget=10.
  - Plan without budget → env receives global config value.
  - API rejects budget ≤ 0.

- [x] **Session 02 — Record starting_budget per evaluation run**

  Without this, we can't retroactively compute percentage return for
  models trained at different budgets.

  - `registry/model_store.py` — add `starting_budget: float` to
    `EvaluationDayRecord`.  Schema migration: add column to
    `evaluation_days` table with default=100.0 for existing rows.
  - `training/evaluator.py` — write `starting_budget` from the
    environment config into each day record.
  - Bet log parquets — include `starting_budget` column.

  **Tests:**
  - Evaluation at budget=10 → day record has starting_budget=10.
  - Old records without the column → default to 100.0.

## Phase 2 — Percentage display

- [x] **Session 03 — Percentage return in scoreboard**

  Add `mean_daily_return_pct` to the scoring pipeline and API.

  - `registry/scoreboard.py` — compute
    `mean_daily_return_pct = (mean_daily_pnl / starting_budget) * 100`
    in `ModelScore`.  Use the budget from evaluation day records
    (they should all be the same for a given model).
  - `api/schemas.py` — add `mean_daily_return_pct: float | None`
    to `ScoreboardEntry`.
  - `api/routers/models.py` — pass through.
  - Frontend scoreboard — add "Return %" column alongside raw P&L.
    Format: `+4.2%` / `-1.8%`.  Colour green/red.
  - Frontend garage — same column.

  **Tests:**
  - Model at budget=10, mean_pnl=1.0 → return_pct = 10.0%.
  - Model at budget=100, mean_pnl=10.0 → return_pct = 10.0%.
  - Both rank identically on composite score (already true).

- [x] **Session 04 — Model detail + bet explorer budget context**

  - Model detail page — show "Budget: £10/race" in metrics grid.
  - Bet explorer stats bar — show budget context.
  - Bet explorer P&L column — optionally show as % of budget.

  **Tests:**
  - Model detail displays correct budget.
  - Bet explorer shows budget in stats bar.

## Phase 3 — Discard policy update

- [x] **Session 05 — Percentage-based discard threshold**

  - `config.yaml` — add `min_mean_return_pct: 0.0` alongside
    existing `min_mean_pnl`.
  - `registry/scoreboard.py` — check percentage threshold in
    `check_discard_candidates()`.  Use `min_mean_return_pct` if set,
    else fall back to `min_mean_pnl` for backward compat.
  - Document the migration path: once all models have
    `starting_budget` recorded, `min_mean_pnl` can be removed.

  **Tests:**
  - Model at budget=10, pnl=0.5 → return=5% → survives threshold=0%.
  - Model at budget=100, pnl=-2.0 → return=-2% → discarded if
    threshold=0% and other conditions met.
  - Backward compat: config with only `min_mean_pnl` still works.

---

## Summary

| Session | What | Phase |
|---------|------|-------|
| 01 | Per-plan starting_budget override | 1 |
| 02 | Record starting_budget per eval run | 1 |
| 03 | Percentage return in scoreboard | 2 |
| 04 | Model detail + bet explorer context | 2 |
| 05 | Percentage-based discard threshold | 3 |

Total: 5 sessions. Sessions 01-02 are the foundation. Session 03
delivers the headline feature (% return display). Sessions 04-05
are polish.
