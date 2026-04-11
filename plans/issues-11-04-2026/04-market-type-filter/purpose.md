# 04 — Market Type Filter (WIN / EW / BOTH)

## Problem

Currently the environment plays every race in a day regardless of
market type. A model trained on both WIN and EACH_WAY markets sees
a mixed distribution, and there's no way to specialise a model on
one type or the other.

The operator wants to:
- Train models that only play WIN markets.
- Train models that only play EW markets.
- Train models that play both (current default).
- Have this choice travel as a gene so the genetic algorithm can
  evolve it.
- At live inference time, enforce the same filter the model was
  trained with (or let the operator override via the UI).

## Design

### New gene: `market_type_filter`

```yaml
market_type_filter:
  type: str_choice
  choices:
    - WIN
    - EACH_WAY
    - BOTH
    - FREE_CHOICE
```

Four options:
- **WIN** — only play WIN markets. Agent never sees EW races.
- **EACH_WAY** — only play EW markets. Agent never sees WIN races.
- **BOTH** — play all markets. Agent sees both but cannot opt out
  of a race type mid-episode.
- **FREE_CHOICE** — play all markets (same data as BOTH), but the
  agent is free to learn when to engage vs abstain. The difference
  from BOTH is semantic: BOTH means "you must play everything",
  FREE_CHOICE means "you see everything but you choose". In
  practice, both present all races — the agent already has the
  ability to place zero bets on a race it doesn't like. The
  distinction is useful as metadata at inference time: a FREE_CHOICE
  model was explicitly trained to self-select, whereas a BOTH model
  was not given that framing.

Default for new random models: `BOTH` (backward-compatible).
Existing models without the gene: treated as `BOTH`.

### Where the filter applies

1. **Environment (`betfair_env.py`)** — at episode start (`reset()`),
   filter `self.day.races` based on `market_type_filter`. Races that
   don't match are skipped entirely (not just masked). This means:
   - A WIN-only model never sees EW races in training.
   - An EW-only model never sees WIN races.
   - A BOTH model sees everything (current behaviour).
   - A FREE_CHOICE model sees everything (same as BOTH — the filter
     is `BOTH` or `FREE_CHOICE` → no filtering).

2. **Evaluator (`training/evaluator.py`)** — apply the same filter
   when evaluating. A model trained on WIN-only must be evaluated on
   WIN-only races for scores to be meaningful.

3. **Registry** — the gene is stored in `hyperparameters` JSON like
   all other genes. No schema change needed.

4. **Scoreboard** — models with different filters are still ranked
   together on composite score (which is normalised). But the filter
   should be displayed as a badge/tag so the operator can see it.

5. **ai-betfair** — at inference time, the loaded model's
   `market_type_filter` determines which live markets it bids on.
   The operator can override via radio buttons on the go-live page.

### Gene behaviour in evolution

- **Sampling:** Random from `{WIN, EACH_WAY, BOTH}`.
- **Crossover:** Inherited from parent A or B (like `architecture_name`).
- **Mutation:** Jump to adjacent choice (like other `str_choice` genes).
  No cooldown needed (unlike architecture, this doesn't invalidate
  weights — the obs/action shapes are identical regardless of filter).

### Important: no observation schema change

The `market_type_win` and `market_type_each_way` observation features
remain. A WIN-only model still has these features in its obs vector
(they'll always be `[1.0, 0.0]`). This keeps the obs schema version
stable and allows weight compatibility across filter values if the
operator wants to re-deploy a WIN-trained model on BOTH markets.

### Important: no action schema change

The filter only controls which races are presented to the agent. The
action space per race is unchanged.

## Edge cases

- **Day with no matching races:** If a WIN-only model gets a day with
  only EW races, the episode has zero races. The env should handle
  this gracefully (zero reward, no bets). The evaluator should
  record it as a 0-bet, 0-pnl day (not skip it).

- **Mixed populations:** A population can contain WIN-only, EW-only,
  and BOTH models. Each model filters its own training/eval races.
  The scoreboard normalisation handles different bet counts and P&L
  scales.

- **Crossover between WIN and EW parents:** Produces a child with one
  parent's filter. The weights may come from either parent. If the
  child inherits WIN-parent's weights but EW filter, the weights
  are still compatible (same obs/action shape) — just the training
  distribution shifts. This is fine; the next training session will
  adapt the weights.

## Files touched

| Layer | File | Change |
|---|---|---|
| Config | `config.yaml` | Add `market_type_filter` to search_ranges |
| Environment | `env/betfair_env.py` | Filter races in `reset()` based on gene |
| Evaluator | `training/evaluator.py` | Apply same filter during eval |
| Orchestrator | `training/run_training.py` | Pass gene to env config |
| Scoreboard | Frontend scoreboard/garage | Display filter badge |
| Model detail | Frontend model-detail | Show market type filter |
| API | `api/schemas.py` | Include filter in ScoreboardEntry |
