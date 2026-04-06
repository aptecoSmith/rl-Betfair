# Progress — Architecture & Hyperparameter Exploration

Update this file at the end of each session. One short entry per
session: what shipped, what files changed, what tests were added, what
didn't ship and why.

Don't use this file for running thoughts — that's what
`lessons_learnt.md` is for. Keep entries factual.

---

## Session 0 — Design review & planning (2026-04-06)

**Shipped:**
- Created `plans/arch-exploration/` folder with planning docs.
- Design review: traced the reward-gene-plumbing bug, audited every
  hyperparameter in the current schema, confirmed `ppo_transformer_v1`
  is the named third architecture in `PLAN.md`.

**Files added:**
- `plans/arch-exploration/purpose.md`
- `plans/arch-exploration/master_todo.md`
- `plans/arch-exploration/progress.md` (this file)
- `plans/arch-exploration/lessons_learnt.md`
- `plans/arch-exploration/testing.md`
- `plans/arch-exploration/ui_additions.md`
- `plans/arch-exploration/session_1_reward_plumbing.md`
- `plans/arch-exploration/session_2_ppo_schema.md`
- `plans/arch-exploration/session_3_reward_schema.md`
- `plans/arch-exploration/session_4_training_plan.md`
- `plans/arch-exploration/session_5_lstm_structural.md`
- `plans/arch-exploration/session_6_transformer_arch.md`
- `plans/arch-exploration/session_7_drawdown_shaping.md`
- `plans/arch-exploration/session_8_ui_additions.md`
- `plans/arch-exploration/session_9_gpu_shakeout.md`

**Not shipped:**
- No code changes. This session was planning only.

**Next:** Session 1 — plumb reward genes through to env.

---

## Session 1 — Plumb reward genes through to env (2026-04-06)

**Shipped:**
- `BetfairEnv.__init__` accepts a new `reward_overrides: dict | None`
  kwarg. It overlays the overrides on top of a per-instance copy of
  `config["reward"]` (shared config is never mutated). Unknown keys are
  silently dropped after a one-shot `logger.debug` so a typoed gene
  name can't crash a multi-day run.
- `PPOTrainer.__init__` now caches `self.hyperparams` and builds a
  `self.reward_overrides` dict via the new `_reward_overrides_from_hp`
  helper. `_collect_rollout` passes that dict into every `BetfairEnv`
  it constructs — closing the dead path from `population_manager.py:220`
  to `env/betfair_env.py:226`.
- Gene-name → reward-config-key mapping lives in `_REWARD_GENE_MAP`
  inside `agents/ppo_trainer.py`. `reward_early_pick_bonus` pins both
  `early_pick_bonus_min` and `early_pick_bonus_max` to the same value
  (min == max → constant multiplier, no reward-formula change).
  Session 3 will split it into proper min/max genes.
- Retired `observation_window_ticks`: removed from `config.yaml`
  `search_ranges` and from the assertion list in `test_config.py`. No
  downstream code reads it, so no migration shim was needed.
- Added `gpu` and `slow` pytest markers in `pyproject.toml` and
  deselected them in the default `addopts`, matching the rules in
  `plans/arch-exploration/testing.md`.
- New test file `tests/arch_exploration/test_reward_plumbing.py` with
  8 CPU-only tests (covers all 6 session-plan items plus a
  shared-config-mutation guard and a direct unit test of
  `_reward_overrides_from_hp`). Runs in ~2.5 s.

**Files changed:**
- `env/betfair_env.py` — added `reward_overrides` kwarg,
  `_REWARD_OVERRIDE_KEYS` class attribute, merge logic.
- `agents/ppo_trainer.py` — added `_REWARD_GENE_MAP`,
  `_reward_overrides_from_hp`, `self.hyperparams`,
  `self.reward_overrides`, and updated the `BetfairEnv(...)`
  construction site in `_collect_rollout`.
- `config.yaml` — removed `observation_window_ticks` from
  `hyperparameters.search_ranges`.
- `pyproject.toml` — registered `gpu`/`slow` markers, updated
  `addopts` to deselect them by default.
- `tests/test_config.py` — drop `observation_window_ticks` from the
  expected-params list.
- `tests/arch_exploration/__init__.py` — new empty package marker.
- `tests/arch_exploration/test_reward_plumbing.py` — new 8-test file.
- `plans/arch-exploration/progress.md`, `lessons_learnt.md`,
  `ui_additions.md` — session notes.

**Tests:**
- `pytest tests/arch_exploration/` → 8 passed (~2.5 s).
- `pytest tests/test_betfair_env.py tests/test_config.py
  tests/test_population_manager.py tests/test_genetic_operators.py
  tests/test_orchestrator.py tests/test_genetic_selection.py
  tests/arch_exploration/` → 227 passed, 1 pre-existing failure
  deselected (`test_obs_dim_matches_env` hardcodes 1630 but actual is
  1636; confirmed stale on plain `master` via `git stash` → not
  caused by this session).

**Not shipped:**
- Did not touch the UI; Session 8 will consume `ui_additions.md`.
- Did not add `gamma`/`gae_lambda`/`value_loss_coeff` genes — that's
  Session 2.

**Next:** Session 2 — expand PPO hyperparameter schema.

---

## Session 2 — Expand PPO hyperparameter schema (2026-04-06)

**Shipped:**
- Added `gamma`, `gae_lambda`, and `value_loss_coeff` as float genes in
  `config.yaml` `hyperparameters.search_ranges`, matching the shape of
  the existing `ppo_clip_epsilon` / `entropy_coefficient` entries. Ranges
  are the ones specified in the session plan:
  - `gamma`: float [0.95, 0.999]
  - `gae_lambda`: float [0.9, 0.98]
  - `value_loss_coeff`: float [0.25, 1.0]
- No code changes in `sample_hyperparams` or `PPOTrainer.__init__`
  were needed — the sampler already handles `type: float`, and the
  trainer already reads all three via `hp.get(..., default)`. The
  `hp.get(..., default)` fallbacks were kept to protect old checkpoints
  loaded without these genes (explicit regression test added).
- New test file `tests/arch_exploration/test_ppo_schema.py` with 4
  CPU-only tests:
  1. `test_sampler_produces_ppo_genes_in_range` — sampler emits all
     three genes inside their declared ranges (50-iteration loop).
  2. `test_trainer_reads_ppo_genes_from_hyperparams` — extreme-value
     round-trip: construct `PPOTrainer` with values pinned near range
     edges and assert `trainer.gamma` / `.gae_lambda` / `.value_loss_coeff`
     match. This is the "gene is actually read" test mandated by
     `lessons_learnt.md`.
  3. `test_trainer_defaults_survive_when_genes_missing` — hardening
     around the `hp.get` fallbacks so a future refactor can't silently
     break old-checkpoint loading.
  4. `test_mutation_keeps_ppo_genes_in_range` — seeded RNG, 200
     mutation rounds at `mutation_rate=1.0`, assert every mutated
     value remains clamped inside `[spec.min, spec.max]`.
- `tests/test_config.py::test_hyperparameter_search_ranges_present`
  extended to require the three new keys.
- `ui_additions.md` Session 2 checklist already contained the three
  items; left as pending-UI (Session 8 consumes them — the server side
  is now live so they're eligible to be ticked there).

**Files changed:**
- `config.yaml` — added three `float` entries under
  `hyperparameters.search_ranges`.
- `tests/test_config.py` — added `gamma`, `gae_lambda`, `value_loss_coeff`
  to the required-params assertion list.
- `tests/arch_exploration/test_ppo_schema.py` — new 4-test file.
- `plans/arch-exploration/progress.md` — this entry.

**Tests:**
- `pytest tests/arch_exploration/test_ppo_schema.py tests/test_config.py`
  → 17 passed (~4.5 s).
- `pytest tests/arch_exploration/ tests/test_population_manager.py
  tests/test_genetic_operators.py` → 91 passed, 1 pre-existing failure
  (`test_obs_dim_matches_env` stale `1630` vs actual `1636` — documented
  in Session 1 progress/lessons as unrelated drive-by; out of scope).

**Not shipped:**
- `mini_batch_size`, `ppo_epochs`, `max_grad_norm` — explicitly out of
  scope per the session plan (GPU-memory / rollout-length interactions).
- Nothing touched in the UI; Session 8 consumes `ui_additions.md`.

**Next:** Session 3 — expand reward hyperparameter schema.
