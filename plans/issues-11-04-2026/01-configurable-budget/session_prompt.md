# Configurable Budget — All Sessions (01–05)

Work through sessions sequentially. Complete each session fully
(code + tests + progress.md entry) before moving to the next.
Commit after each session.

## Before you start — read these

- `plans/issues-11-04-2026/01-configurable-budget/purpose.md` — why
  this work exists and the analysis of what's already normalised.
- `plans/issues-11-04-2026/01-configurable-budget/hard_constraints.md`
  — non-negotiables.
- `plans/issues-11-04-2026/01-configurable-budget/master_todo.md` —
  session breakdown with tests.
- `CLAUDE.md` — especially "Reward function: raw vs shaped".

---

## Session 01 — Per-plan starting_budget override

### Context

`config.yaml:training.starting_budget` is currently the only place
the budget is set (default £100).  Training plans
(`registry/training_plans/*.json`) don't carry a budget field.
`training/run_training.py` reads from global config and passes it
to the environment.

### What to do

1. Add optional `starting_budget: float | None` to the training plan
   schema in `registry/model_store.py`.

2. In `training/run_training.py`, when building the environment config:
   - Check if the active plan has `starting_budget` set.
   - If yes, use it.  If no, use `config["training"]["starting_budget"]`.

3. In `api/routers/training.py`, accept optional `starting_budget`
   in the plan creation request body.  Validate > 0 if provided.

4. Frontend training plan form — add an optional "Budget per race"
   field with the global default shown as placeholder text.

### Tests

- Plan with budget=10 → env receives starting_budget=10.
- Plan without budget → env receives global config value (100).
- API rejects budget ≤ 0.

### Exit criteria

- All tests pass. `progress.md` updated. Commit.

---

## Session 02 — Record starting_budget per evaluation run

### Context

`EvaluationDayRecord` (in `registry/model_store.py`) has `day_pnl`,
`bet_count`, `winning_bets`, etc. but no `starting_budget`.  Without
it we can't compute percentage return retroactively.

### What to do

1. Add `starting_budget: float = 100.0` to `EvaluationDayRecord`.

2. Schema migration: add `starting_budget REAL DEFAULT 100.0` column
   to the `evaluation_days` table.  Existing rows get 100.0.

3. In `training/evaluator.py`, when creating `EvaluationDayRecord`,
   populate `starting_budget` from the environment config.

4. Bet log parquets: include `starting_budget` as a column.

### Tests

- Evaluation at budget=10 → day record has starting_budget=10.
- Load old records (no column) → default to 100.0.
- Parquet round-trip preserves starting_budget.

### Exit criteria

- All tests pass. `progress.md` updated. Commit.

---

## Session 03 — Percentage return in scoreboard

### Context

`registry/scoreboard.py:compute_score()` already computes
`pnl_norm = mean_pnl / self.starting_budget`, but this uses the
*global* config budget, not the per-evaluation budget.  With
Session 02 landed, each `EvaluationDayRecord` now carries the budget
it was evaluated with.

### What to do

1. In `compute_score()`, use the recorded `starting_budget` from
   the day records (they should all be the same for a given model's
   evaluation run).

2. Add `mean_daily_return_pct` to `ModelScore`:
   ```python
   mean_daily_return_pct = (mean_daily_pnl / recorded_budget) * 100
   ```

3. Add `mean_daily_return_pct: float | None` to `ScoreboardEntry`
   in `api/schemas.py`.

4. Frontend scoreboard — add "Return %" column.  Format: `+4.2%`.
   Green if positive, red if negative.

5. Frontend garage — same column.

### Tests

- Model at budget=10, mean_pnl=1.0 → 10.0%.
- Model at budget=100, mean_pnl=10.0 → 10.0%.
- Both display correctly in the API response.

### Exit criteria

- All tests pass. `progress.md` updated. Commit.

---

## Session 04 — Model detail + bet explorer budget context

### What to do

1. Model detail page — show "Budget: £10/race" in the metrics grid.
   Read from the model's evaluation records.

2. Bet explorer stats bar — show budget context (e.g. "Budget:
   £10/race").

3. Bet explorer P&L values — no change to display (raw £), but the
   budget context makes them interpretable.

### Exit criteria

- Budget displayed on model detail and bet explorer. Commit.

---

## Session 05 — Percentage-based discard threshold

### Context

`config.yaml:discard_policy.min_mean_pnl` is an absolute £ threshold
(default 0.0).  A model making £0.50/day on a £10 budget is a 5%
return — healthy — but would be near the discard floor if we ever
tightened the threshold.

### What to do

1. Add `min_mean_return_pct: float` to `config.yaml:discard_policy`
   (default: 0.0).

2. In `scoreboard.py:check_discard_candidates()`, compute return %
   from the recorded budget and compare against the new threshold.

3. Keep `min_mean_pnl` as a backward-compat fallback: if
   `min_mean_return_pct` is not set, use `min_mean_pnl` as before.

### Tests

- Budget=10, pnl=0.5 → return=5% → survives threshold=0%.
- Budget=100, pnl=-2.0 → return=-2% → discarded if other conditions met.
- Config with only `min_mean_pnl` → still works.

### Exit criteria

- All tests pass. `progress.md` updated. Commit.

---

## Cross-session rules

- Run `pytest tests/ -x` after each session. All tests must pass.
- Do not touch the reward function, observation space, or action space.
- Do not "improve" unrelated code. Scope is tight.
- Commit after each session with a clear message referencing the
  session number.
