# Master TODO — Anti-Passivity

Single session — small reward shaping addition.

---

- [ ] **Session 01 — Inactivity penalty gene + reward integration**

  1. **config.yaml** — add to `hyperparameters.search_ranges`:
     ```yaml
     inactivity_penalty:
       type: float
       min: 0.0
       max: 2.0
     ```

  2. **env/betfair_env.py** — add `"inactivity_penalty"` to
     `_REWARD_OVERRIDE_KEYS` frozenset.

  3. **env/betfair_env.py** — in `__init__()`, read the gene:
     ```python
     self._inactivity_penalty = reward_cfg.get("inactivity_penalty", 0.0)
     ```

  4. **env/betfair_env.py** — in `_settle_current_race()`, after
     computing all other shaped terms and before summing:
     ```python
     if race_bet_count == 0:
         inactivity_term = -self._inactivity_penalty
     else:
         inactivity_term = 0.0
     ```
     Add `inactivity_term` to the `shaped` sum.

  5. **Logging** — include `inactivity_term` in
     `self._cum_shaped_reward` accumulator (already handled by
     adding it to `shaped`). Optionally add a separate accumulator
     `self._cum_inactivity_penalty` for diagnostic logging.

  **Tests:**
  - Zero-bet race with penalty=0.5 → reward = -0.5 (not 0.0).
  - One-bet race with penalty=0.5 → inactivity_term = 0.0 (no penalty).
  - Penalty=0.0 → no change (backward compat, current behaviour).
  - raw + shaped ≈ total_reward invariant still holds (inactivity
    is part of shaped).
  - Gene appears in sampled hyperparameters.

  **Exit criteria:**
  - All tests pass (new and existing). `progress.md` updated. Commit.
