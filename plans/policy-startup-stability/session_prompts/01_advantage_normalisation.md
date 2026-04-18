# Policy Startup Stability — Session 01 prompt

Implements per-mini-batch advantage normalisation in the PPO
update loop. Single targeted change to `agents/ppo_trainer.py`,
plus a synthetic test that demonstrates spike-prevention on a
fake high-magnitude rollout.

This session is designed to be run unattended end-to-end: the
brief specifies what to change, what to test, what the smoke
check looks like, and an explicit decision rule for whether to
ship the optional defence-in-depth (LR warmup) at the end.

## PREREQUISITE — read first

- [`../purpose.md`](../purpose.md) — root cause, evidence
  (agent `3e37822e-c9fa` trajectory), and the literature
  reference.
- [`../hard_constraints.md`](../hard_constraints.md) — 21
  non-negotiables. §1 (single conceptual change), §5–§7 (the
  math), §8 (one function in `ppo_trainer.py`), §15–§17
  (mandatory tests).
- `agents/ppo_trainer.py` — read end-to-end before changing.
  Specifically locate:
  - The advantage computation (likely uses GAE — search for
    `gae` or `advantage`).
  - The mini-batch loop in the PPO update.
  - The surrogate-loss calculation (search for `ratio` or
    `surrogate`).
- `tests/test_ppo_trainer.py` — existing tests' structure;
  match the fixture pattern.
- `CLAUDE.md` — "Reward function: raw vs shaped" section.

## Locate the code

```
grep -n "advantage\|surrogate\|ratio.*adv\|policy_loss" agents/ppo_trainer.py | head -30
grep -n "def.*update\|def.*train" agents/ppo_trainer.py | head
ls tests/test_ppo_trainer.py
```

Confirm before editing: `agents/ppo_trainer.py` has a clear
PPO update function (probably called `update`, `_update`, or
`learn`) that iterates over mini-batches. Inside that loop,
the advantage tensor is computed (or referenced from the
buffer) and used in the surrogate loss. The change goes
directly above the surrogate-loss computation, inside the
mini-batch loop.

## What to do

### 1. Add advantage normalisation

Find the surrogate-loss block. The pattern looks something
like:

```python
ratio = torch.exp(new_logprobs - old_logprobs)
surrogate_1 = ratio * advantages
surrogate_2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
```

Immediately above the `ratio` line, add:

```python
# Per-batch advantage normalisation (plans/policy-startup-stability,
# Session 01, 2026-04-18). Stabilises the PPO update against
# large-magnitude rewards that would otherwise produce a first-
# rollout policy_loss spike, saturating action heads (most
# notably close_signal — see purpose.md). Literature precedent:
# Engstrom et al. 2020, "Implementation Matters in Deep Policy
# Gradients". Standard recipe; ships in stable-baselines3,
# CleanRL, RLlib.
adv_mean = advantages.mean()
adv_std = advantages.std() + 1e-8
advantages = (advantages - adv_mean) / adv_std
```

Critical notes:
- The normalisation is **per-mini-batch**, not per-rollout-buffer.
  If the variable holding advantages is shared across mini-
  batches, slice into the per-batch view first; don't normalise
  the whole buffer's tensor.
- Apply ONLY to the policy loss. The value loss continues to
  use the un-normalised return targets (per
  `hard_constraints.md §6`).
- `eps = 1e-8` for numerical stability — guards against the
  degenerate case where all advantages in the batch are
  identical (std = 0).

### 2. Add the synthetic test

In `tests/test_ppo_trainer.py` (or a new file
`tests/test_ppo_advantage_normalisation.py` if the existing
file is large), add:

```python
class TestAdvantageNormalisationStability:
    """Per-batch advantage normalisation prevents the first-update
    policy-loss explosion that would otherwise saturate action
    heads.

    Reproduces the failure mode observed in production agent
    3e37822e-c9fa (2026-04-18 morning training run): on a fresh
    policy with large-magnitude rewards, the first PPO update
    produced policy_loss = 3.35e+14 and saturated close_signal.
    """

    def _fresh_policy_and_rollout(self, with_normalisation: bool):
        """Build a freshly-initialised policy + a synthetic
        rollout buffer with deliberately-large reward magnitudes,
        run one PPO update, return (policy_loss, action_head_mean_shift).
        """
        # ... fixture setup mirroring existing PPO tests'
        # initialisation pattern.
        # ... synthesise advantages in the ±£500 to ±£2000 range
        #     (matches real scalping rollout magnitudes).
        # ... run one update with or without normalisation
        #     (use a feature flag or a monkey-patch to toggle).
        # ... capture policy_loss + the action_head's output mean
        #     shift.
        ...

    def test_unnormalised_first_update_spikes(self):
        """Without normalisation, large-reward rollouts produce a
        catastrophic policy_loss on the first update."""
        loss, _ = self._fresh_policy_and_rollout(with_normalisation=False)
        assert loss > 100, (
            f"expected spike >100 on un-normalised update; got {loss}"
        )

    def test_normalised_first_update_stays_bounded(self):
        """With normalisation, the same rollout produces a bounded
        policy_loss on the first update."""
        loss, _ = self._fresh_policy_and_rollout(with_normalisation=True)
        assert loss < 5, (
            f"expected bounded loss <5 with normalisation; got {loss}"
        )

    def test_normalisation_dampens_action_head_shift(self):
        """The action_head's output mean must shift materially less
        in the normalised case — this is the principled check that
        the fix prevents head saturation, not just dampens the
        loss."""
        _, shift_un = self._fresh_policy_and_rollout(with_normalisation=False)
        _, shift_norm = self._fresh_policy_and_rollout(with_normalisation=True)
        assert abs(shift_norm) < abs(shift_un) / 5, (
            f"expected normalised shift << un-normalised; "
            f"got norm={shift_norm}, un-norm={shift_un}"
        )
```

The fixture details (`_fresh_policy_and_rollout`) depend on
how the trainer is structured — when implementing, mirror the
shape of any existing test that exercises the update path.
The `with_normalisation` toggle can be implemented by:
- A module-level flag the test patches (simplest), or
- An optional kwarg on the update method (cleaner if the
  existing API allows it), or
- Direct manipulation of the line you added (e.g. monkey-
  patching the normalisation step in/out).

Pick whichever is least invasive given the existing trainer's
structure.

### 3. Run tests

```
pytest tests/test_ppo_advantage_normalisation.py -v   # or wherever you put them
pytest tests/test_ppo_*.py -q
pytest tests/ -q
```

All MUST be green. The pre-existing
`test_invariant_raw_plus_shaped_equals_total_reward` is the
load-bearing reward-accumulator check — if it fails, the
normalisation has somehow leaked into the raw/shaped accumulator
plumbing (it shouldn't — the normalisation is purely on the
gradient pathway, not on `info["raw_pnl_reward"]`). Trace and
fix; do NOT relax the invariant test.

### 4. Smoke test

Manually launch a tiny training run (1 agent, 5 episodes) with
the fix in place. Easiest path: spin up a `TrainingPlan` with
`population_size=1`, `n_generations=1`, `n_epochs=1` via the
existing API or by direct invocation of the orchestrator (see
`scripts/run_scalping_gen1.py` for the pattern; adapt to a
1-agent run).

Once it's run, inspect `logs/training/episodes.jsonl`:

```
python -c "
import json
with open('logs/training/episodes.jsonl') as f:
    eps = [json.loads(l) for l in f]
# Filter to the smoke run (most recent agent)
mid = eps[-1]['model_id']
for e in eps:
    if e['model_id'] == mid:
        pl = e.get('policy_loss', 0)
        print(f\"ep {e['episode']:>2}  loss={pl:.4f}\")
"
```

### 5. Decision: ship LR warmup or not?

**If episode 1's `policy_loss < 100`:** the normalisation
alone is sufficient. Do NOT add LR warmup. Proceed to commit.

**If episode 1's `policy_loss >= 100`:** the normalisation
isn't enough. Add the optional first-update LR warmup as
defence-in-depth:

```python
# In the trainer's __init__ or update method, before the
# first PPO update:
self._update_count = 0  # initialise in __init__

# In the update method, before stepping the optimiser:
warmup_factor = min(1.0, (self._update_count + 1) / 5.0)
for param_group in self.optimizer.param_groups:
    param_group['lr'] = self.base_learning_rate * warmup_factor
self._update_count += 1
```

(The `base_learning_rate` reference depends on how the trainer
stores the learning rate — adapt to the existing structure.)

Add a corresponding test that asserts the LR is reduced on
update 0, ramped over updates 1–4, and at full strength from
update 5 onward.

Re-run the smoke test. If `policy_loss < 100` on episode 1,
proceed to commit. If still spiking, escalate by appending a
note to the commit body and `progress.md` — Session 02 (or a
follow-up plan) will need to address action-head initialisation
directly.

### 6. Commit

One commit, type `fix(agents)`. First line:

```
fix(agents): per-batch advantage normalisation in PPO update
```

Body:

```
Stabilises the first-rollout PPO update against large-magnitude
rewards that would otherwise produce policy_loss spikes (10^4 to
10^14 observed in production scalping runs), saturating action-
head outputs and rendering close_signal / requote_signal
permanently un-trainable.

Concrete evidence — agent 3e37822e-c9fa (2026-04-18 morning
activation-A-baseline run): episode 1 produced policy_loss =
3.35e+14, saturating the close_signal head; from episode 2
onward closed=0 across the agent's remaining 14 episodes.
Same pattern observed in three independent runs to date.

Fix: per-mini-batch advantage normalisation in the PPO
surrogate-loss branch:

    adv_mean = advantages.mean()
    adv_std  = advantages.std() + 1e-8
    advantages = (advantages - adv_mean) / adv_std

Literature standard (Engstrom et al. 2020, "Implementation
Matters in Deep Policy Gradients"). Ships in stable-baselines3,
CleanRL, RLlib. Single-line change in agents/ppo_trainer.py;
no schema bumps; no env changes; no reward function changes.

Pre-existing checkpoints (including 3 garaged models) load
unchanged. Reward magnitudes in episodes.jsonl are unchanged
(the fix is purely on the gradient pathway, not on
info["raw_pnl_reward"]). Scoreboard rows are directly
comparable pre- and post-fix.

[If LR warmup also shipped, add: "Also adds first-5-update
linear LR warmup as defence-in-depth — synthetic test showed
normalisation alone reduced policy_loss from 1e+12 to ~80,
still above the smoke-test threshold of 100; warmup brings
update-1 loss to <2."]

Tests: TestAdvantageNormalisationStability (3 tests) +
TestLRWarmup (X tests, if applicable). pytest -q: <delta>
passed.

See plans/policy-startup-stability/.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## Exit criteria

- `tests/test_ppo_*` green.
- Pre-existing
  `test_invariant_raw_plus_shaped_equals_total_reward` green.
- Full `pytest tests/ -q` green.
- Smoke run produces ep-1 `policy_loss < 100` (and ideally
  `< 5`).
- Commit on master with clear first-line summary.

## Acceptance

The synthetic test
`test_normalisation_dampens_action_head_shift` ASSERTS the
action_head's output mean shifts materially less in the
normalised case than the un-normalised case. This is the
principled check that the fix prevents head saturation, not
just dampens loss magnitudes.

## After Session 01

Append a Session-01 entry to
[`../progress.md`](../progress.md). Include:
- The single-line code change citation (file + line + recipe).
- Test count delta.
- Smoke-test ep-1 `policy_loss` value (the actual number).
- Whether LR warmup also shipped (yes/no + reason).

Then proceed to [`02_docs_and_reset.md`](02_docs_and_reset.md).
