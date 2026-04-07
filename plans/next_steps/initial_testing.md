# Initial Testing — Next Steps

**Scope:** the fast CPU-only tests written during a session for
immediate feedback. "Initial" meaning: the first line of defence,
run after every feature, before commit.

**Not in scope:** GPU tests, full training runs, human-in-the-loop
verification. Those live in `integration_testing.md` and
`manual_testing_plan.md`.

## Golden rules

1. **CPU only.** If a test would need CUDA, it's not an initial
   test. Mark it `@pytest.mark.gpu` and it moves to integration.

2. **Fast.** The full initial-test suite for a session must run in
   under 30 seconds on a laptop. If you're pushing past that,
   something belongs in integration.

3. **After each feature, add tests, run them, commit.** Do not
   batch. Every bug the arch-exploration phase caught early was
   caught by a per-feature test; every bug it caught late was
   caught by a batched test.

4. **No full training runs.** Unit-test the pieces —
   `sample_hyperparams`, `BetfairEnv.__init__`, policy `forward()`,
   `TrainingPlan` round-trips, coverage math — in isolation.

5. **`device="cpu"` in fixtures.** Force it explicitly; don't rely
   on default device selection.

6. **Mock heavy dependencies.** Smallest fixture day. Patch
   `BetManager.settle_race` when a synthetic race outcome is
   needed (technique documented in
   `arch-exploration/lessons_learnt.md` Session 3 entry).

## What a good initial test looks like

- **Gene plumbing test** — construct extreme values, pass them
  through the relevant module, assert the downstream attribute
  changed. <100 ms per assertion.
- **Sampler test** — deterministic RNG in, value range out.
  <10 ms.
- **Coverage math test** — pure-function inputs/outputs, no env.
  <10 ms.
- **Policy instantiation test** — CPU forward pass on a zero
  tensor, assert output shapes. <1 s per variant.
- **Reward invariant test** — synthetic race, assert
  `raw + shaped ≈ total_reward`. <50 ms.

## What an initial test does NOT look like

- Running `training/run_training.py` end-to-end.
- Loading a full day of market data when a fixture day would do.
- Any test that creates a CUDA tensor.
- Any test whose name starts with `test_full_` or
  `test_integration_`.
- Sleeps, retries, or polling loops.

## Running

```bash
# Fast feedback loop — what every session uses:
pytest tests/ -m "not gpu and not slow" -x
```

The `gpu` and `slow` markers and their skip-by-default behaviour
live in `conftest.py`, added during Session 1 of arch-exploration.
Do not disable them.

## Test locations

- New CPU-only tests for next-steps work live in
  `tests/next_steps/`. Create the folder when the first session
  lands code there.
- `tests/arch_exploration/` is frozen — do not add new tests there
  unless fixing a regression in arch-exploration code.

## When a test starts to feel slow

Don't silently ignore it. Either:
- Move it to `integration_testing.md` (mark `@pytest.mark.slow` or
  `@pytest.mark.gpu`, update the integration doc), **or**
- Reduce its scope until it fits the fast budget again.

Never let a test get away with being slow-but-tolerated; that
erodes the 30-second budget and the entire initial-test contract
breaks within a few sessions.
