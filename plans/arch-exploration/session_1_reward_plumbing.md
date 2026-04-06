# Session 1 — Plumb reward genes through to env

## Before you start — read these

- `plans/arch-exploration/purpose.md` — why this work exists.
- `plans/arch-exploration/master_todo.md` — where this session sits in
  the bigger plan.
- `plans/arch-exploration/testing.md` — **testing rules. No GPU. Fast
  feedback. After each feature, add tests, run them, commit.**
- `plans/arch-exploration/progress.md` — read the last entry to see
  what state the repo was left in.
- `plans/arch-exploration/lessons_learnt.md` — read it; the "sampled ≠
  used" lesson is literally this session's problem.
- `plans/arch-exploration/ui_additions.md` — this session appends
  verification tasks for Session 8.
- Repo root `CLAUDE.md` — the hard invariants in "Order matching" and
  "Reward function: raw vs shaped" are non-negotiable.

## Goal

Fix the dead path from per-agent hyperparams to env reward
calculation. After this session, mutating `reward_early_pick_bonus`,
`reward_efficiency_penalty`, or `reward_precision_bonus` in the
genetic algorithm must visibly change that agent's training signal.

## Scope

**In scope:**
- Accept a reward-overrides dict in `BetfairEnv.__init__`. Merge it
  over `config["reward"]` before reading values.
- `PPOTrainer` extracts reward keys from `self.hyperparams` and passes
  them into every `BetfairEnv` it constructs.
- Retire `observation_window_ticks`: it's currently sampled but not
  read anywhere. Remove from `config.yaml` search_ranges and from
  `sample_hyperparams`. Add a migration shim **only** if needed for
  loading existing checkpoints — prefer deleting cleanly.
- Add CPU-only tests (see below).

**Out of scope:**
- Do not add new reward knobs yet (that's Session 3).
- Do not promote more PPO knobs (that's Session 2).
- Do not touch architecture code.
- Do not touch the UI — update `ui_additions.md` instead.

## Exact code path to fix

The design review traced the dead path as:

1. `agents/population_manager.py:220` samples all genes including
   reward keys.
2. `training/run_training.py:448` passes `hyperparams=agent.hyperparameters`
   into `PPOTrainer`.
3. `agents/ppo_trainer.py:150-159` reads only PPO learning keys from
   `hp`. Reward keys are silently dropped.
4. `agents/ppo_trainer.py:248` constructs `BetfairEnv(day, self.config,
   feature_cache=self.feature_cache)` — no per-agent reward config.
5. `env/betfair_env.py:226-232` reads reward values from `config["reward"]`
   only. All agents get identical values.

Fix: change step 4 to also pass reward overrides, and change step 5 to
accept and honour them.

Do not route reward genes through `config["reward"]` as a mutation —
config is shared. Pass a separate `reward_overrides: dict | None = None`
kwarg into `BetfairEnv.__init__`.

## Tests to add (all CPU-only, fast)

Create `tests/arch_exploration/test_reward_plumbing.py`:

1. **Sampling still works.** `sample_hyperparams` called with the schema
   produces a dict containing `reward_early_pick_bonus`,
   `reward_efficiency_penalty`, `reward_precision_bonus`, and the values
   are within the configured ranges.

2. **Env honours overrides.** Construct `BetfairEnv` with
   `reward_overrides={"efficiency_penalty": 0.99, "precision_bonus": 0.0}`
   and assert `env._efficiency_penalty == 0.99` and
   `env._precision_bonus == 0.0`. Use the smallest available fixture
   day — do NOT load a full day of real market data.

3. **Env ignores unknown keys.** Overrides dict containing an unknown
   key does not crash; unknown keys are silently ignored (or logged
   once — your choice, just document it).

4. **Trainer passes overrides into env.** Unit test that mocks
   `BetfairEnv` and asserts `PPOTrainer` constructs it with a
   `reward_overrides` kwarg derived from `self.hyperparams`. This is
   the test that proves the plumbing is live end-to-end.

5. **Raw + shaped invariant still holds.** Run the existing reward
   invariant test (or write one if it doesn't exist) with non-default
   overrides. Assert `raw + shaped ≈ total_reward` on a synthetic
   settled race.

6. **`observation_window_ticks` is gone.** Assert the gene is not in
   the default schema returned by the sampler.

If `conftest.py` does not already define `gpu` / `slow` pytest markers
that skip by default, add the scaffolding as part of this session's
first commit.

## Session exit criteria

- All 6 tests above pass locally with `pytest tests/arch_exploration/ -x`.
- Existing tests still pass: `pytest tests/ -m "not gpu and not slow"`.
- `progress.md` updated with a Session 1 entry listing files changed.
- `lessons_learnt.md` updated with anything surprising you found along
  the way. (If nothing was surprising, note that — absence of news is
  still data.)
- `ui_additions.md` Session 1 checklist items are still accurate
  (nothing new to add unless you discovered more dead genes).
- Commit the work (do not push — user reviews before push).

## Do not

- Do not regress the CLAUDE.md invariants on reward symmetry or
  raw/shaped bucketing. The fix is purely plumbing — the reward
  *formulas* stay the same.
- Do not add any GPU-touching tests.
- Do not widen the scope of this session. "While I'm here, let me
  also..." is how we end up with three half-finished sessions.
