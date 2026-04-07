# Integration Testing — Next Steps

**Scope:** slower, GPU-touching, or end-to-end tests that are NOT
run on every feature commit. These exist to catch bugs that only
appear when real components are wired together.

**Not in scope:** the fast CPU feedback loop (that's
`initial_testing.md`) or human-in-the-loop verification (that's
`manual_testing_plan.md`).

## When integration tests run

- **Explicitly opted-in** via pytest markers:
  ```bash
  pytest tests/ -m gpu             # GPU sanity checks
  pytest tests/ -m slow            # slow CPU integration tests
  pytest tests/ -m "gpu or slow"   # everything integration-y
  ```
- Never in the default test run.
- Never as part of a "fix and re-run" inner loop during feature
  development — they are too slow for that.
- At the end of a dedicated integration session (see below).

## What belongs here

- **GPU forward/backward sanity checks.** Instantiate each
  architecture on CUDA, do one forward + one backward pass,
  assert no NaNs and no exploding gradients. Fast on a GPU but
  excluded from the CPU fast loop because it needs CUDA.
- **Short training-loop smoke tests.** A few dozen environment
  steps on a tiny fixture day, asserting the optimiser step
  doesn't crash and reward bookkeeping stays consistent.
- **End-to-end TrainingPlan round-trips.** Load a plan, spin up
  a population, run one generation on a fixture day, persist
  outcomes, reload, assert state matches.
- **Multi-race reward invariant checks.** `raw + shaped ≈ total`
  across many synthetic days and many random policies. Catches
  statistical drift that single-race tests miss.
- **Long-running coverage math tests** that stress the planner
  with thousands of synthetic historical agents.

## What does NOT belong here

- Running a full multi-generation search. That is **Session 11**,
  and it is a *manual* action with a dedicated plan (see
  `manual_testing_plan.md`). Integration tests are automated; the
  real run is not.
- Anything that takes more than ~5 minutes wall time for a single
  test. At that point it is either a shakeout session or a
  manual-test item, not an integration test.
- Tests that require network access, external APIs, or anything
  the sandbox can't provide reliably.

## When to write integration tests

- When a CPU unit test can't reach the failure mode. Example: a
  transformer-specific NaN that only surfaces under real
  half-precision training.
- When crossing a major boundary (env ↔ trainer ↔ population
  manager ↔ plan registry) for the first time after a structural
  change.
- After a manual run in Session 11 surfaces a failure mode —
  codify the repro as an integration test so it doesn't regress.

## When NOT to write integration tests

- When the same confidence can be gained from a fast CPU unit
  test. Always prefer the fast test first.
- Speculatively, before the code under test exists.
- As a substitute for thinking about edge cases in the fast
  tests.

## Location

`tests/next_steps/integration/` — create the folder when the
first integration test lands. Do not mix with fast tests.

Every file in that folder must have its tests marked
`@pytest.mark.gpu` or `@pytest.mark.slow` (or both). A file with
unmarked tests is a bug — the file-level marker alone is not
enough; pytest reads per-test markers.

## Running during a session

Development sessions are CPU-only. The only time integration
tests run during a session is:

1. At the start of **Session 10** (housekeeping sweep) — run the
   existing integration suite on the current code to confirm
   nothing is already broken before we make changes.
2. At the end of **Session 11** (real run) — the run itself is
   manual, but supporting integration tests (GPU sanity checks,
   short training smokes) run before the manual launch to catch
   obvious breakage.
3. Any future session that is explicitly labelled as an
   integration session in its prompt.
