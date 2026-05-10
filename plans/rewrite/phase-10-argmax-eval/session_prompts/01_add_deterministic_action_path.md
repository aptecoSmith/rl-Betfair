---
session: phase-10-argmax-eval / S01
phase: rewrite/phase-10-argmax-eval
parent_purpose: ../purpose.md
---

# S01 — deterministic / argmax action path in `RolloutCollector`

## Context

Read `purpose.md` first. The motivating evidence:
- Smoke test on s06 cohort 2026-05-05 19:36 showed identical-weights /
  identical-day eval producing pnl swings of £100-£300 from action-
  sampling RNG alone (`658a7f72` +£178 → +£55; `81c80d76` -£0.19 → -£336).
- The two stochastic sources are `out.action_dist.sample()` at
  [`training_v2/discrete_ppo/rollout.py:464`](../../../../training_v2/discrete_ppo/rollout.py)
  and `stake_dist.sample()` at `:482`.

This session adds a `deterministic: bool = False` kwarg to
`RolloutCollector.collect_episode` that, when `True`, replaces both
samples with their deterministic counterparts (`logits.argmax` and
`Beta.mean`). Default `False` preserves byte-identical pre-plan
behaviour. **No trainer / worker / runner changes in this session** —
just the collector primitive and its tests.

## Pre-reqs

Read in this order:
- [`agents_v2/discrete_policy.py`](../../../../agents_v2/discrete_policy.py)
  — focus on `PolicyOutput` (around line 120) and the forward-pass
  return at line 532. Confirm `out.action_dist` is a
  `torch.distributions.Categorical` exposing `.logits` and
  `.log_prob(action)`. Confirm `out.stake_alpha` and `out.stake_beta`
  are float tensors of shape `(1,)`.
- [`training_v2/discrete_ppo/rollout.py`](../../../../training_v2/discrete_ppo/rollout.py)
  full file. Focus on `_collect_rollout` (or `collect_episode` —
  inspect to find the actual loop body around lines 460-500).
  Identify:
  - The `Categorical.sample()` call (line ~464).
  - The `Beta.sample()` call (line ~482).
  - The `log_prob_action_t` and `log_prob_stake_t` calculations
    (lines ~470, ~499).
  - Where transitions are appended.
- [`training_v2/discrete_ppo/transition.py`](../../../../training_v2/discrete_ppo/transition.py)
  — confirm `Transition` carries `action`, `stake`, `log_prob_action`,
  `log_prob_stake` (or equivalent).

## What to do

### 1. Add `deterministic` kwarg to `collect_episode`

Find the `def collect_episode(...)` signature. Add a kwarg
`deterministic: bool = False` after the existing args. Plumb it
into `_collect_rollout` if that's a separate inner function.

### 2. Gate the action sample

Replace:

```python
action = out.action_dist.sample()              # (1,) long
```

with:

```python
if deterministic:
    action = out.action_dist.logits.argmax(dim=-1)
else:
    action = out.action_dist.sample()
```

The `.argmax(dim=-1)` returns a long tensor of the same shape as
`.sample()` — drop-in replacement. Action-mask handling is
preserved automatically because masked-out logits are `-inf` and
will never win argmax.

### 3. Gate the stake sample

Replace:

```python
stake_unit_t = stake_dist.sample()              # (1,)
```

with:

```python
if deterministic:
    # Beta(α, β).mean = α / (α + β). Always defined for α, β > 0.
    stake_unit_t = out.stake_alpha / (out.stake_alpha + out.stake_beta)
else:
    stake_unit_t = stake_dist.sample()
```

`stake_dist` should be constructed regardless of mode (the
`log_prob_stake_t` calculation downstream needs it) — only the
sampling step is replaced. Confirm by re-reading the code around
line 479-485.

### 4. log_prob remains computed against the chosen action

The existing line:

```python
log_prob_action_t = (
    out.action_dist.log_prob(action).detach().squeeze()
)
```

works unchanged — it computes `log_prob` of whatever `action` ended
up being, deterministic or sampled. Same for the stake `log_prob`
at line ~499. **No edit needed there.** This is the load-bearing
invariant that keeps the rollout buffer schema stable and any future
"train under argmax" experiment compatible.

### 5. Run the existing test suite to confirm no regression

```bash
python -m pytest tests/test_v2_cohort_runner.py tests/test_v2_cohort_worker.py \
  tests/test_v2_multi_day_train.py tests/test_v2_aux_heads.py -x --tb=short
```

All ~68 tests should pass with no edits — the `deterministic=False`
default keeps the existing call paths byte-identical.

## New tests in `tests/test_v2_argmax_eval.py` (new file)

Five tests, all on a stub policy / fixture rollout (no real env, no
GPU required):

### 1. `test_collector_deterministic_action_is_argmax_of_logits`

Build a stub policy whose forward returns rigged
`Categorical(logits=...)` with known argmax (e.g. `logits =
[1.0, 5.0, 2.0, 0.5]` → argmax index 1). Run `collect_episode(...,
deterministic=True)` for a few steps. Assert every transition's
`action` equals 1.

Hint: a minimal stub policy can subclass `torch.nn.Module` and
return a hand-built `PolicyOutput` from forward. Look at how
existing tests in `tests/test_v2_aux_heads.py` build stub policies
for inspiration.

### 2. `test_collector_deterministic_stake_is_beta_mean`

Same stub-policy approach. Set `stake_alpha=2.0, stake_beta=1.0`
on the returned `PolicyOutput`. The Beta mean is `2 / (2 + 1) = 0.6667`.
Assert every transition's `stake_unit` equals 0.6667 within fp
epsilon.

Note: only assert on transitions where the action's mask says
"uses stake" (NOOP doesn't). Reuse the existing `action_uses_stake`
helper from `agents_v2/env_shim.py`.

### 3. `test_collector_default_is_stochastic_byte_identical`

Two-pass byte-identity check. Set torch + numpy + python random
seeds before each pass. Run `collect_episode()` (no kwarg) twice
and assert the resulting transitions are bit-identical (action,
stake, log_prob, rewards). This is the regression guard for the
default-False kwarg — proves we haven't accidentally changed the
stochastic path.

### 4. `test_collector_log_prob_invariant_holds_under_deterministic`

Run `collect_episode(deterministic=True)`. For each transition,
recompute `expected_log_prob = action_dist.log_prob(transition.action)`
on the same-step policy output and assert it equals
`transition.log_prob_action` within fp epsilon. The PPO buffer's
load-bearing invariant must hold under the new mode.

This test is more involved than 1-3 because it requires capturing
the policy outputs alongside the transitions. Easiest path: have
the stub policy record its own outputs in a list the test can
inspect.

### 5. `test_collector_action_mask_respected_under_deterministic`

Stub policy returns `Categorical(logits=[1.0, -inf, 5.0, -inf])`
(actions 1 and 3 masked out). Under `deterministic=True`, every
transition's action must be 2 (the highest non-masked logit). The
mask logic already produces -inf for illegal actions; this test
just confirms argmax respects it.

## Done when

- All 5 new tests pass under `pytest tests/test_v2_argmax_eval.py -xvs`.
- The existing 68-test cohort/worker/aux-head suite still passes.
- A short commit message: `feat(rollout): deterministic action path
  for eval (argmax + Beta.mean)`.

## Stop conditions

- If the `Beta.mean` formula introduces a NaN or inf on any test
  fixture, stop and check whether `stake_alpha` / `stake_beta` are
  guaranteed positive at the policy boundary. Look at
  `agents_v2/discrete_policy.py` to see the activation that
  produces them (probably softplus or +1).
- If test 4 (`log_prob_invariant`) fails: the chosen-action
  `log_prob` is computed wrong somewhere. Don't paper over by
  changing the test — find the bug.
- If a randomly-rigged stub fails to produce the expected argmax,
  re-check whether `action_dist.sample()` is being called somewhere
  ELSE in the same loop (e.g. for diagnostic logging). Grep the
  collector for all `.sample()` calls.

## Out of scope

- Wiring the flag through the trainer / worker / runner — that's
  Session 02.
- Validation cohort / re-eval — that's Session 03.
- Changing the policy's forward pass — already exposes everything
  needed.
- Removing the stake distribution construction (still needed for
  log_prob).
