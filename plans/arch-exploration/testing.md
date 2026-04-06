# Testing Rules — Architecture & Hyperparameter Exploration

These rules exist so that each session delivers fast feedback. The goal
is that a developer editing code can run the full relevant test set in
under 30 seconds, iterate, and re-run.

## Golden rules

1. **No GPU during development sessions.** Sessions 1–8 must use
   CPU-only tests. GPU training is deferred to Session 9. If a test
   would require CUDA, mark it `@pytest.mark.gpu` and skip it by
   default (configure in `conftest.py`).

2. **No full training runs during development.** A test that spins up
   a PPO training loop, even on CPU, is too slow for fast feedback.
   Unit-test the pieces (`sample_hyperparams`, `BetfairEnv.__init__`,
   policy `forward()`) in isolation.

3. **After each feature, add tests, run them, then commit.** Do not
   batch up features and test at the end. Each session in
   `master_todo.md` lists the tests it must add — the session is not
   done until those tests pass.

4. **Tests run on CPU-only PyTorch if possible.** Where a policy must
   be instantiated, force `device="cpu"` in the test fixture. This
   avoids accidental GPU warm-up delays.

5. **Mock heavy dependencies.** If a test needs a `BetfairEnv` but
   doesn't need real market data, use the smallest fixture day or the
   existing test fixtures in `tests/fixtures/` (check what's there
   before inventing new fixtures).

## What fast tests look like in this codebase

- **Gene plumbing tests:** construct a fake hyperparams dict with
  extreme values, pass through `PPOTrainer` / `BetfairEnv`, read the
  attribute that should have changed, assert it matches. Should run
  in <100 ms.
- **Sampler tests:** call `sample_hyperparams(spec, rng)` with a
  deterministic RNG, assert each returned value is within the spec's
  range. <10 ms.
- **Policy instantiation tests:** construct a policy with every
  supported combination of structural knobs, call `forward()` once on
  a zero tensor of the right shape, assert output shape. <1 s.
- **Reward invariant tests:** given a synthetic race, compute reward,
  assert `raw + shaped ≈ total_reward` to within floating-point
  tolerance. <50 ms.

## What fast tests do NOT look like

- Running `training/run_training.py` end-to-end.
- Loading a full day's worth of market data.
- Any test whose name starts with `test_full_` or `test_integration_`
  unless it's explicitly marked `@pytest.mark.slow` and skipped by
  default.
- Any test that creates a CUDA tensor.

## Running tests

```bash
# Fast feedback loop (what every session uses):
pytest tests/ -m "not gpu and not slow" -x

# Session 9 only (the one GPU session):
pytest tests/ -m gpu
```

`conftest.py` must define `gpu` and `slow` markers and skip them by
default unless `--run-gpu` / `--run-slow` is passed. If it doesn't
already do this, Session 1 should add the scaffolding as part of its
first commit.

## Test locations

- New CPU-only tests for arch-exploration work live in
  `tests/arch_exploration/` (create the folder if it doesn't exist).
- Do not add tests into the existing `tests/integration/` tree during
  Phase 1–3 — that's for slower end-to-end tests we explicitly skip.
