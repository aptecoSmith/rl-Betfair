# Master TODO — Market Type Filter

Ordered session list. Tick boxes as sessions land.

When a session completes:
1. Tick its box here.
2. Add an entry to `progress.md`.
3. Append any learnings to `lessons_learnt.md`.
4. Note cross-repo follow-ups in `knockon_ai_betfair.md`.

---

## Phase 1 — Gene + environment filter

- [ ] **Session 01 — Add gene + env filtering**

  1. **config.yaml** — add to `hyperparameters.search_ranges`:
     ```yaml
     market_type_filter:
       type: str_choice
       choices:
         - WIN
         - EACH_WAY
         - BOTH
         - FREE_CHOICE
     ```

  2. **env/betfair_env.py** — in `reset()`, after loading the day's
     races, filter based on `market_type_filter` from the config:
     ```python
     mtf = self.config.get("market_type_filter", "BOTH")
     if mtf != "BOTH":
         self.day.races = [
             r for r in self.day.races
             if (r.market_type or "").upper() == mtf
         ]
     ```
     Handle the empty-races case: if no races remain, the episode
     should complete immediately with zero reward.

  3. **training/run_training.py** — when building the per-agent env
     config, include `market_type_filter` from the model's
     hyperparameters.

  **Tests:**
  - Gene sampled → one of WIN, EACH_WAY, BOTH, FREE_CHOICE.
  - Env with filter=WIN on a mixed day → only WIN races played.
  - Env with filter=EACH_WAY on a mixed day → only EW races played.
  - Env with filter=BOTH → all races (regression).
  - Env with filter=FREE_CHOICE → all races (same as BOTH).
  - Env with filter=WIN on an all-EW day → zero races, zero reward,
    episode completes cleanly.
  - Crossover between WIN-parent and EW-parent → child gets one
    parent's filter.
  - Mutation of market_type_filter → jumps to adjacent choice.

- [ ] **Session 02 — Evaluator filtering**

  `training/evaluator.py` — apply the same `market_type_filter`
  when evaluating a model. The filter value comes from the model's
  `hyperparameters` dict in the registry.

  - Pass `market_type_filter` into the env config for each eval run.
  - Handle zero-race days in eval: record as 0-bet, 0-pnl day
    (not skip).

  **Tests:**
  - WIN-only model evaluated on mixed data → only WIN race bets in
    the bet log.
  - EW-only model evaluated → only EW race bets.
  - Zero-race eval day → day record with bet_count=0, pnl=0.

## Phase 2 — Display

- [ ] **Session 03 — Scoreboard + model detail display**

  1. **api/schemas.py** — add `market_type_filter: str | None` to
     `ScoreboardEntry`.  Default `None` (backward compat for old
     models without the gene → treated as "BOTH").

  2. **api/routers/models.py** — read from model's hyperparameters
     and pass to the response.

  3. **Frontend scoreboard** — display as a small badge/tag next to
     the model ID: `WIN`, `EW`, or `BOTH`.  Colour-code:
     - WIN → blue badge
     - EW → amber badge
     - BOTH → grey/neutral badge
     - FREE → green badge

  4. **Frontend garage** — same badge.

  5. **Frontend model-detail** — show in the metadata section.

  **Tests:**
  - Scoreboard entry includes market_type_filter.
  - Old model (no gene) → displays as "BOTH".

---

## Summary

| Session | What | Phase |
|---------|------|-------|
| 01 | Gene definition + env filtering | 1 |
| 02 | Evaluator filtering | 1 |
| 03 | Scoreboard + model detail display | 2 |

Total: 3 sessions. Session 01 is the core work. Session 02 ensures
eval consistency. Session 03 is display.
