# Progress — Anti-Passivity

One entry per completed session.

---

## Session 01 — 2026-04-11

Implemented the inactivity penalty gene:

- Added `inactivity_penalty: 0.0` to `config.yaml` reward section (default off)
- Added gene to `hyperparameters.search_ranges` (float, 0.0–2.0)
- Added `"inactivity_penalty"` to `_REWARD_OVERRIDE_KEYS` in `betfair_env.py`
- Read gene in `__init__()`: `self._inactivity_penalty = reward_cfg.get("inactivity_penalty", 0.0)`
- Computed `inactivity_term = -self._inactivity_penalty if race_bet_count == 0 else 0.0`
  in `_settle_current_race()`, included in shaped sum
- Added to `_REWARD_GENE_MAP` in `ppo_trainer.py` for HP → override plumbing
- Added 5 tests in `test_reward_plumbing.py`:
  - Zero-bet race with penalty → reward includes -0.5
  - Bet-placed race → no penalty applied
  - Default penalty=0.0 → backward compatible (zero shaped)
  - raw + shaped invariant holds with penalty active
  - Gene appears in sampled hyperparameters
