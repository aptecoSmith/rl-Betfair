# Anti-Passivity — Session 01

## Before you start — read these

- `plans/issues-11-04-2026/07-anti-passivity/purpose.md` — the
  design rationale and why force-bet was rejected.
- `plans/issues-11-04-2026/07-anti-passivity/hard_constraints.md`
- `CLAUDE.md` — "Reward function: raw vs shaped" section and the
  `raw + shaped ≈ total_reward` invariant.
- `env/betfair_env.py` — `_settle_current_race()` (lines ~985-1107),
  `_REWARD_OVERRIDE_KEYS` (line ~330), reward config reading in
  `__init__()` (lines ~390-407).

## What to do

1. **config.yaml** — add gene:
   ```yaml
   inactivity_penalty:
     type: float
     min: 0.0
     max: 2.0
   ```

2. **env/betfair_env.py** — add `"inactivity_penalty"` to
   `_REWARD_OVERRIDE_KEYS`.

3. **env/betfair_env.py** — in `__init__()`, alongside the other
   reward config reads:
   ```python
   self._inactivity_penalty = reward_cfg.get("inactivity_penalty", 0.0)
   ```

4. **env/betfair_env.py** — in `_settle_current_race()`, after
   `spread_cost_term` (line ~1080) and before the `shaped` sum
   (line ~1082):
   ```python
   inactivity_term = -self._inactivity_penalty if race_bet_count == 0 else 0.0
   ```
   Then add it to the shaped sum:
   ```python
   shaped = (
       early_pick_bonus
       + precision_reward
       - efficiency_cost
       + drawdown_term
       + spread_cost_term
       + inactivity_term
   )
   ```

5. Optionally add a diagnostic accumulator:
   ```python
   self._cum_inactivity_penalty += inactivity_term
   ```
   And expose it in `info` dict if there's a pattern for that.

## Tests

Add to `tests/test_betfair_env.py` or a new test file:

1. **Zero-bet race, penalty=0.5:** Create a day with one race.
   Step through without placing any bets. Assert total reward
   includes -0.5 inactivity term.

2. **One-bet race, penalty=0.5:** Place one bet. Assert
   inactivity_term = 0.0 (no penalty for betting races).

3. **Penalty=0.0 (default):** Zero-bet race → reward = 0.0
   (backward compat, same as current behaviour).

4. **raw + shaped invariant:** Episode with mix of betting and
   non-betting races. Assert `raw + shaped ≈ total_reward`.

5. **Gene in config:** Verify `inactivity_penalty` appears in
   sampled hyperparameters from the population manager.

## Exit criteria

- All tests pass (new and existing).
- `progress.md` updated.
- Commit.
