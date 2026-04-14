# Testing Rules — Arb Improvements

These rules exist so each session delivers fast feedback. Goal: a
developer editing code can run the full relevant test set in under
30 seconds, iterate, re-run.

## Golden rules

1. **No GPU during development sessions.** Sessions 1–9 are CPU-only.
   GPU training is deferred to Session 10 (the verification run).
   Mark any test that needs CUDA with `@pytest.mark.gpu` and skip by
   default in `conftest.py`.

2. **No full training runs during development.** Do not spin up a
   PPO training loop even on CPU — it's too slow for iteration.
   Unit-test the pieces (`sample_hyperparams`, `BetfairEnv.__init__`,
   policy `forward()`, `bc_pretrainer.pretrain()`,
   `arb_oracle.scan_day()`) in isolation.

3. **After each feature, add tests, run them, then commit.** No
   batching up features and testing at the end. Each session's
   `session_N_*.md` lists the tests it must add — the session is
   not done until those tests pass.

4. **Force CPU where a policy is instantiated.** Use `device="cpu"`
   explicitly in test fixtures. Avoids accidental GPU warm-up.

5. **Reuse existing fixtures.** Check `tests/fixtures/` before
   inventing new ones. The forced-arbitrage sessions already added
   synthetic day fixtures with pair-tracking — lean on those.

## What fast tests look like for this plan

- **Clipping tests (Phase 1):** synthesise a large step reward,
  run the training-signal clipping function, assert clipped value
  is bounded and telemetry value is untouched. <50 ms.
- **Entropy-floor tests (Phase 1):** feed a sequence of entropies
  to the floor controller, assert the coefficient scales up when
  below floor and restores when recovered. <20 ms.
- **Feature function tests (Phase 2):** construct synthetic
  `PriceLevel` lists (duck-typed objects with `.price` and `.size`),
  call the function, assert returned float equals hand-computed
  value. <10 ms per test.
- **Obs schema tests (Phase 2):** instantiate `BetfairEnv` on the
  smallest fixture, read `observation_space.shape`, compare to
  `MARKET_DIM + VELOCITY_DIM + RUNNER_DIM × max_runners + ...`
  with the new dims counted in. <500 ms.
- **Oracle scan tests (Phase 3):** synthetic episode with one
  injected crossed-book tick, assert exactly one oracle sample
  returned and filtered correctly when an env rule would refuse it.
  <1 s.
- **BC pretrain tests (Phase 3):** build 100 synthetic
  `(obs, oracle_action)` pairs, call `pretrain()` for a few steps,
  assert loss drops and policy action mean converges toward
  oracle. <2 s.

## What fast tests do NOT look like

- Running `training/run_training.py` end-to-end.
- Loading a full day of real market data.
- Anything that creates a CUDA tensor.
- Scanning a real day for oracle samples (that's integration, mark
  `@pytest.mark.slow`).

## Running tests

```bash
# Fast feedback loop — every session uses this:
pytest tests/ -m "not gpu and not slow" -x

# Session 10 only (the one GPU session):
pytest tests/ -m gpu
```

If `conftest.py` already defines the `gpu` / `slow` markers from
the `arch-exploration` plan, reuse them. If it doesn't, Session 1
adds the scaffolding as part of its first commit.

## Test locations

- New CPU-only tests for this plan live in
  `tests/arb_improvements/` (create if it doesn't exist).
- Do NOT add tests into `tests/integration/` during Phases 1–4 —
  that's for the explicitly-slow end-to-end tests skipped by
  default.
- Extending existing `tests/test_forced_arbitrage.py` is fine when
  a test is a direct extension of forced-arb behaviour (e.g. "arb
  features return 0 when scalping-mode bets are placed on a
  one-sided book"). Otherwise use the new folder.

## Deterministic oracle testing

The oracle scan is deterministic by contract. Tests must:

- Use a fixed seed where randomness is involved (none should be).
- Compare byte-for-byte against a checked-in expected output when
  the input is a committed fixture.
- Never depend on wall-clock or environment state.

Non-determinism in the oracle invalidates BC pretraining diagnostics
across runs.
