# 01 — Configurable Budget & Percentage-Based P&L Display

## Problem

`starting_budget` is fixed at £100 in `config.yaml` and applies to all
models.  The operator wants to experiment with different budgets (e.g. £10
per race) but if two models train on different budgets, the scoreboard's
`mean_daily_pnl` column (raw £) makes the £100 model look 10x better even
if both achieved the same *percentage return*.

## Good news: ranking already works

The composite score normalises P&L by `starting_budget`:

```python
pnl_norm = np.clip(mean_pnl / self.starting_budget, -1.0, 1.0)         # scoreboard.py:110
pnl_per_bet_norm = np.clip(pnl_per_bet / (self.starting_budget * 0.1))  # scoreboard.py:113
```

And the terminal reward bonus already normalises:

```python
day_pnl / starting_budget  # betfair_env.py:806
```

So **ranking, genetic selection, and reward are all already
budget-independent**.  The problem is purely display + per-plan
configurability.

## What needs to change

### 1. Per-plan `starting_budget` override

Currently `starting_budget` lives in global `config.yaml:training`.
Training plans (`registry/training_plans/*.json`) don't carry it.

**Change:** Allow training plans to include `starting_budget`.  The
training orchestrator should use the plan's value if present, else fall
back to the global config.

- `training/run_training.py` — read `plan.starting_budget` if set
- `api/routers/training.py` — accept optional `starting_budget` in plan
  creation endpoint
- Frontend training plan form — optional budget field

### 2. Record `starting_budget` per evaluation run

`EvaluationDayRecord` doesn't store the budget used.  Without it we can't
retroactively compute percentage return.

**Change:** Add `starting_budget: float` to `EvaluationDayRecord` and
persist it in `evaluation_days` table + bet log parquets.

- `registry/model_store.py` — add field + schema migration
- `training/evaluator.py` — write starting_budget into day record

### 3. Percentage return in scoreboard + model detail

Add `mean_daily_return_pct` to `ModelScore` and the API response:

```python
mean_daily_return_pct = (mean_daily_pnl / starting_budget) * 100
```

Display in scoreboard: `+4.2%` instead of (or alongside) `£4.20`.

- `registry/scoreboard.py` — compute and include in `ModelScore`
- `api/schemas.py` — add `mean_daily_return_pct` to `ScoreboardEntry`
- Frontend scoreboard + garage — show % return column
- Frontend model detail — show % return in metrics grid

### 4. Discard policy: percentage threshold

`min_mean_pnl: 0.0` is absolute.  With £10 budgets, even a profitable
model making £0.50/day would be above threshold, so this isn't urgent
— but for correctness it should be percentage-based.

**Change:** Rename to `min_mean_return_pct` or add it alongside.
Default: `0.0` (break-even).

- `config.yaml` — add `min_mean_return_pct`
- `registry/scoreboard.py` — use percentage in discard check
- Back-compat: keep `min_mean_pnl` working if set

### 5. Bet explorer + model detail: show budget context

When viewing a model's bets or detail page, show what budget it trained
with so the raw P&L numbers have context.

- Model detail page — show "Budget: £10/race" in the header
- Bet explorer — show budget in the stats bar

## Files touched

| Layer | File | Change |
|---|---|---|
| Config | `config.yaml` | Document `starting_budget` as overridable |
| Training plan | `registry/training_plans/*.json` | Optional `starting_budget` field |
| Orchestrator | `training/run_training.py` | Read plan-level budget |
| Environment | `env/betfair_env.py` | Already parameterised (no change) |
| Evaluator | `training/evaluator.py` | Write budget to day record |
| Model store | `registry/model_store.py` | New field on `EvaluationDayRecord` |
| Scoreboard | `registry/scoreboard.py` | Compute % return, % discard |
| API schemas | `api/schemas.py` | `mean_daily_return_pct` field |
| API models | `api/routers/models.py` | Pass through new field |
| Frontend | scoreboard, garage, model-detail | Show % return |
| Frontend | bet-explorer | Show budget context |

## ai-betfair knock-on

`ai-betfair` loads trained checkpoints and runs inference.  The budget
used at inference time is configured in `ai-betfair`'s own config (it
controls the real stake sizing).  However:

- **Checkpoint metadata** should carry the training budget so the
  live system can display "trained at £10/race, running at £5/race"
  for transparency.
- **Signal strength** calibration may differ if the policy was trained
  at a very different budget from the live stake.  Worth logging
  `training_budget / live_budget` ratio as a sanity-check metric.
- No structural code change needed — the policy's forward pass is
  budget-agnostic (actions are stake fractions, not absolute amounts).

## Test plan

1. Train two models at £10 and £100 budgets on the same data
2. Verify composite scores are comparable (not 10x different)
3. Verify scoreboard shows % return and raw P&L
4. Verify discard policy works with percentage threshold
5. Verify evaluation day records include starting_budget
