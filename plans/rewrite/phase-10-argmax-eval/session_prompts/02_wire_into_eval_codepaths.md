---
session: phase-10-argmax-eval / S02
phase: rewrite/phase-10-argmax-eval
parent_purpose: ../purpose.md
depends_on: S01
---

# S02 — wire `argmax_eval` flag through eval code paths

## Context

S01 added the `deterministic` kwarg to
`RolloutCollector.collect_episode`. This session plumbs an
operator-facing `--argmax-eval` flag from the cohort runner CLI
all the way down to the eval-rollout collector calls. Training
rollouts stay stochastic by hard-constraint.

The plumbing path:

```
CLI --argmax-eval
  → run_cohort(argmax_eval=...)
    → train_one_agent(argmax_eval=...)        # sequential path
    → train_cluster_batched(argmax_eval=...)  # batched path
        → eval_collector.collect_episode(deterministic=...)
```

Plus a parallel path through the standalone train CLI and
`tools/reevaluate_cohort.py`.

## Pre-reqs

- Read [`training_v2/cohort/runner.py`](../../../../training_v2/cohort/runner.py)
  — focus on `def run_cohort(...)` signature (line ~102) and
  the eval-loop call sites at lines ~280-310 (sequential) and
  ~252-270 (batched).
- Read [`training_v2/cohort/worker.py`](../../../../training_v2/cohort/worker.py)
  — focus on `def train_one_agent(...)` signature and the eval
  rollout loop (introduced in Phase 8 multi-day work — should be
  around line 700+ where the per-day eval loop iterates over
  `eval_days`).
- Read [`training_v2/cohort/batched_worker.py`](../../../../training_v2/cohort/batched_worker.py)
  — same focus on the eval rollout loop (around line 405+).
- Read [`training_v2/discrete_ppo/train.py`](../../../../training_v2/discrete_ppo/train.py)
  — focus on the eval rollout call (around line 700+, after the
  multi-day training loop).
- Read [`tools/reevaluate_cohort.py`](../../../../tools/reevaluate_cohort.py)
  — entire file. Note the `RolloutCollector` instantiation and
  `collect_episode` call inside the per-day loop.

## What to do

### 1. Add `argmax_eval` to `train_one_agent`

In `training_v2/cohort/worker.py`:

- Add `argmax_eval: bool = False` to the kwarg list of
  `def train_one_agent(...)`.
- In the eval-day loop, change the `collect_episode()` call to
  `collect_episode(deterministic=argmax_eval)`.
- That's it — the per-day loop infrastructure already exists
  from Phase 8.

### 2. Add `argmax_eval` to `train_cluster_batched`

In `training_v2/cohort/batched_worker.py`:

- Same: add `argmax_eval: bool = False` to the signature.
- Change the eval `collect_episode()` call inside the
  per-agent / per-day loop.

### 3. Add `argmax_eval` to `run_cohort`

In `training_v2/cohort/runner.py`:

- Add `argmax_eval: bool = False` to the `run_cohort(...)`
  signature.
- Pass it through to `train_one_agent_fn(..., argmax_eval=argmax_eval)`
  in the sequential branch.
- Pass it through to `train_cluster_batched(..., argmax_eval=argmax_eval)`
  in the batched branch.

### 4. Add CLI flag to the runner

In `training_v2/cohort/runner.py::_parse_args`:

```python
p.add_argument(
    "--argmax-eval", action="store_true",
    help=(
        "Use deterministic argmax action selection at eval time "
        "(no per-tick Categorical sampling, no per-tick Beta sampling). "
        "Default: stochastic eval, byte-identical to pre-2026-05-05. "
        "Per-day eval results become bit-identical at fixed seed; "
        "use this for reproducible architecture / gene comparisons "
        "where action-sampling RNG would otherwise dominate the cash "
        "signal. Training rollouts stay stochastic regardless."
    ),
)
```

In `main()`, pass `argmax_eval=bool(args.argmax_eval)` into
`run_cohort(...)`.

Add a startup log line when the flag is on:
```python
if args.argmax_eval:
    logger.info("Eval mode: argmax (deterministic action + Beta.mean stake)")
```

### 5. Add `eval_mode` to scoreboard rows

In `training_v2/cohort/runner.py::_agent_result_to_scoreboard_row`:

- Add `argmax_eval: bool = False` to the kwarg list.
- Add a new field to the returned dict:

```python
"eval_mode": "argmax" if argmax_eval else "stochastic",
```

Update both call sites in `run_cohort` to pass
`argmax_eval=argmax_eval`.

### 6. Add `--argmax-eval` to the standalone train CLI

In `training_v2/discrete_ppo/train.py`:

- Add the same CLI flag.
- Plumb `deterministic=args.argmax_eval` into the eval rollout's
  `collect_episode` call (just one site).

### 7. Add `--argmax-eval` to `tools/reevaluate_cohort.py`

- Add the flag to `_parse_args`.
- Plumb `deterministic=args.argmax_eval` into the per-day
  `eval_collector.collect_episode()` call inside the agent loop.
- Add an output field `reeval_mode` (`"argmax"` or `"stochastic"`)
  to the scoreboard row written.

## New tests in `tests/test_v2_argmax_eval.py` (extending the file from S01)

### 6. `test_train_one_agent_argmax_eval_flag_reaches_collector`

Spy-style integration test. Monkey-patch
`training_v2.cohort.worker.RolloutCollector` (or the
`collect_episode` method on instances) to record the
`deterministic` kwarg it was called with. Run a tiny synthetic
`train_one_agent` call with `argmax_eval=True` and assert the
recorded `deterministic` was `True`.

You may need to stub other heavy dependencies (BetfairEnv, the
real shim) similarly to the existing `_stub_train_one_agent` in
`tests/test_v2_cohort_runner.py` — but more invasively because
we need the collector spy to fire. Easier path: call into the
real worker but stub at the env-build / policy-build boundary.

If full plumbing-through-the-real-worker is too much wall-time
for a unit test, a lower-cost alternative: just call
`run_cohort(..., argmax_eval=True, train_one_agent_fn=stub)`
where the stub records its own `argmax_eval` kwarg. That tests
the runner-level plumbing without needing the worker internals
to actually run.

### 7. `test_run_cohort_argmax_eval_flag_plumbs_through`

End-to-end flag-routing test using the existing
`_stub_train_one_agent` infrastructure in
`tests/test_v2_cohort_runner.py`. Update the stub to accept and
record `argmax_eval` (default False). Run two cohorts:

- Cohort A: `run_cohort(..., argmax_eval=False)` — confirm stub
  saw `False`, scoreboard `eval_mode == "stochastic"`.
- Cohort B: `run_cohort(..., argmax_eval=True)` — confirm stub
  saw `True`, scoreboard `eval_mode == "argmax"`.

### 8. `test_reevaluate_cohort_argmax_eval_reproducible`

Integration test that uses a real saved model (smallest in the
existing s06 cohort). Run `tools.reevaluate_cohort.main(...)`
twice with `--argmax-eval` on the same agent + same eval day.
Assert the two output JSONL rows have bit-identical
`reeval_day_pnl`, `reeval_locked_pnl`, `reeval_naked_pnl`.

This test needs CUDA to match the production path. Mark with
`@pytest.mark.cuda` (or whatever convention the repo uses) so it
can be skipped on CPU-only runners. If no convention, just put
it in a separately-importable test file
(`tests/test_v2_argmax_eval_cuda.py`) so CI can opt in/out.

If running this CUDA test inside the test suite is infeasible,
substitute a manual repro instruction in the session's
`findings.md` (Session 03 will redo this anyway as part of
validation).

## Done when

- All 8 tests in `tests/test_v2_argmax_eval.py` pass.
- The existing ~68-test cohort suite still passes.
- A `--argmax-eval` invocation of the cohort runner produces
  scoreboard rows with `"eval_mode": "argmax"`.
- A `--argmax-eval` invocation of `tools/reevaluate_cohort.py`
  produces output rows with `"reeval_mode": "argmax"`.
- Commit: `feat(cohort): --argmax-eval flag plumbs through eval
  rollouts (training stays stochastic)`.

## Stop conditions

- If the spy test (test 6) requires more than 30 lines of
  scaffolding, switch to the runner-level approach noted in the
  test description. The point is to verify the kwarg flows; not
  to test the full worker.
- If `--argmax-eval` invocation produces non-reproducible output
  (test 8), the env or shim may have its own RNG path that's
  not seeded by the collector. Audit the env-build path for
  unseeded `random.Random()` or `np.random.RandomState()` calls.
  Document the finding in `findings.md` even if it blocks the
  test — that's exactly the kind of issue this phase exists to
  surface.
- If the existing 68 tests fail after the wiring: a kwarg name
  or default got wrong somewhere. Default should always be
  `False`, kwarg name always `argmax_eval` at every level, and
  the collector-level kwarg always `deterministic`.

## Out of scope

- Validation cohort / cross-eval analysis — that's Session 03.
- Frontend display of the new `eval_mode` field — out of plan
  scope per `purpose.md`.
- Per-mini-batch or per-update logging of any new fields. The
  collector-level kwarg is internal; only the operator-facing
  scoreboard `eval_mode` field is external.
- Adding `argmax_eval` as a per-agent gene. It's a measurement
  choice, not a per-agent variation knob — operator-level only.
