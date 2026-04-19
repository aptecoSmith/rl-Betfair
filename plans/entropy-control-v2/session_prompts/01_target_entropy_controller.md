# Session 01 prompt — Target-entropy controller (learned log_alpha)

## PREREQUISITE — read first

- [`../purpose.md`](../purpose.md) — Baseline-A entropy drift
  evidence, the diagnosis, and the controller design sketch.
- [`../hard_constraints.md`](../hard_constraints.md). §4–§10
  (controller semantics), §11–§12 (inherited agents and
  checkpoint format), §18–§19 (tests green + specific test
  names), §21 (NOT a reward-scale change — commit message
  doesn't need worked-example numerics), §24 (later sessions
  block if this one fails).
- [`../master_todo.md`](../master_todo.md) — Session 01
  deliverables.
- `agents/ppo_trainer.py` — the file being edited. Key
  existing references to locate before editing:
  - `self.entropy_coeff` — the effective coefficient used
    in the surrogate-loss computation. Grep
    `entropy_coeff`; confirm there's exactly one write site
    (construction in `__init__`) and one or more read sites
    (inside the loss formula).
  - `self._entropy_coeff_base` — the arb-improvements
    Session 2 scaffolding. Grep it; confirm it stores a
    constant that is never scaled. Remove it per §10.
  - `self.entropy_coefficient = hp.get("entropy_coefficient",
    0.005)` around line ~482 — this is where the Session 03
    default lives. The controller reads from the same hp
    key for its initial `log_alpha`.
  - `_ppo_update` — the PPO-update loop. The controller
    step runs once per call to this method.
  - `_log_episode` — where per-episode JSONL rows are
    written. New `alpha` and `log_alpha` fields land here.
  - `save_checkpoint` / `load_checkpoint` — checkpoint format
    extensions per §11. Grep these names; find the dict that
    carries state across save/load.
- `plans/policy-startup-stability/progress.md` — the
  per-mini-batch advantage normalisation (`adv_mean`,
  `adv_std`) this session must not disturb.

## Locate the code

```
grep -n "entropy_coeff\|entropy_coefficient\|_entropy_coeff_base" agents/ppo_trainer.py
grep -n "save_checkpoint\|load_checkpoint" agents/ppo_trainer.py
grep -n "_log_episode\|episodes.jsonl" agents/ppo_trainer.py
```

Confirm before editing:

1. `self.entropy_coeff` has one write at init and one or
   more reads in the loss formula.
2. `_entropy_coeff_base` is unused (stores a constant
   never scaled). If found to be used by any downstream
   test or caller, update that caller to read
   `self.entropy_coeff` directly.
3. The advantage normalisation (from commit `8b8ca67`) is
   untouched — the controller step runs on the entropy
   value from the forward pass, which is orthogonal to the
   advantage pipeline.

## What to do

### 1. Controller state in `__init__`

Replace the existing fixed-coefficient setup:

```python
# Before (Session 03 of naked-clip-and-stability):
self.entropy_coeff = hp.get("entropy_coefficient", 0.005)
self._entropy_coeff_base = float(self.entropy_coeff)  # REMOVE
```

with:

```python
# After (this plan):
import math  # at top of file if not already imported

initial_entropy_coeff = float(hp.get("entropy_coefficient", 0.005))
self._target_entropy = float(hp.get("target_entropy", 112.0))
self._log_alpha_min = math.log(1e-5)
self._log_alpha_max = math.log(0.1)
self._log_alpha = torch.tensor(
    math.log(initial_entropy_coeff),
    dtype=torch.float32,
    device=self.device,
    requires_grad=True,
)
self._alpha_optimizer = torch.optim.Adam(
    [self._log_alpha],
    lr=float(hp.get("alpha_lr", 1e-4)),
)
# Effective coefficient the loss formula reads. Kept in
# sync with log_alpha after every controller step.
self.entropy_coeff = self._log_alpha.exp().item()
```

Note: the `.exp().item()` conversion is a one-time snapshot
that gets refreshed after every controller update. The loss
formula uses `self.entropy_coeff` (a Python float), not
`self._log_alpha` (a Tensor) — this keeps the policy
optimiser's graph clean of the controller.

Add a comment cross-linking this plan:

```python
# Target-entropy controller (SAC-style). Entropy coefficient
# is a learned variable rather than a fixed hyperparameter.
# See plans/entropy-control-v2/purpose.md for the motivation
# (Baseline-A 2026-04-19 entropy drift 139.6 → 201.3).
# The separate _alpha_optimizer holds its own momentum state
# and does NOT share anything with the policy optimiser.
```

### 2. Controller-update method

Add a new private method:

```python
def _update_entropy_coefficient(self, current_entropy: float) -> None:
    """Target-entropy controller step. Drives the entropy
    coefficient to hold `current_entropy` at
    `self._target_entropy`. Uses a separate Adam optimiser
    over log_alpha; does not backprop through the policy.

    Call ONCE per _ppo_update, after the entropy value is
    computed on the current rollout. Must be called with a
    detached Python float — no tensor leakage.
    """
    # Loss formula: gradient descent on log_alpha moves it
    # DOWN when entropy > target (shrinking the bonus,
    # letting entropy fall) and UP when entropy < target.
    alpha_loss = -self._log_alpha * (self._target_entropy - current_entropy)
    self._alpha_optimizer.zero_grad()
    alpha_loss.backward()
    self._alpha_optimizer.step()
    # Clamp to prevent runaway during calibration.
    self._log_alpha.data.clamp_(self._log_alpha_min, self._log_alpha_max)
    # Refresh the effective coefficient the loss formula reads.
    self.entropy_coeff = self._log_alpha.exp().item()
```

### 3. Wire the controller into `_ppo_update`

Inside the PPO update loop, after the entropy value is
available from the forward pass (usually as
`entropy.mean()` on the current minibatch), call the
controller BEFORE the policy optimiser's `.step()`:

```python
# Existing: forward pass produces policy_loss, value_loss,
# entropy_value from the current minibatch.

# NEW: controller step. Detach and convert to Python
# float — the controller must not share autograd graph
# with the policy.
current_entropy = float(entropy.mean().detach().item())
self._update_entropy_coefficient(current_entropy)

# Existing: policy optimiser step.
self.optimizer.zero_grad()
total_loss.backward()
self.optimizer.step()
```

Important: the controller reads the entropy value BEFORE
the policy takes its gradient step, so the next update uses
the updated coefficient. Calling after the policy step is
also defensible but introduces a one-step lag between what
the coefficient responded to and what the policy sees. The
"before" ordering is what §6 specifies.

### 4. Per-episode logging

Extend `_log_episode` (or whatever writes rows into
`logs/training/episodes.jsonl`) to include the new fields:

```python
row = {
    # ... existing fields ...
    "alpha": float(self._log_alpha.exp().item()),
    "log_alpha": float(self._log_alpha.item()),
    # ... existing fields ...
}
```

These let the learning-curves panel plot the controller's
trajectory alongside entropy. Not a breaking change — new
optional keys. Downstream tooling that reads the JSONL must
tolerate their absence (pre-existing rows don't have them).

### 5. Checkpoint format extension

Extend `save_checkpoint`:

```python
checkpoint = {
    # ... existing keys ...
    "log_alpha": float(self._log_alpha.item()),
    "alpha_optim_state": self._alpha_optimizer.state_dict(),
    # ... existing keys ...
}
```

Extend `load_checkpoint`:

```python
if "log_alpha" in checkpoint:
    self._log_alpha.data = torch.tensor(
        float(checkpoint["log_alpha"]),
        dtype=torch.float32,
        device=self.device,
    )
else:
    # Backward-compat: checkpoint predates the controller.
    # Fresh-init from the default entropy_coefficient.
    logger.warning(
        "Checkpoint missing log_alpha; fresh-initing from default. "
        "Expected for checkpoints saved before entropy-control-v2."
    )

if "alpha_optim_state" in checkpoint:
    self._alpha_optimizer.load_state_dict(checkpoint["alpha_optim_state"])
# else: optimiser stays at its fresh-init state (Adam's
# momentum starts at 0 — no warning needed, same as the
# above behaviour).

# Refresh effective coefficient after load.
self.entropy_coeff = self._log_alpha.exp().item()
```

### 6. Remove `_entropy_coeff_base`

Per §10. Grep for all references:

```
grep -n "_entropy_coeff_base" agents/ ppo_trainer.py tests/
```

Remove the attribute assignment in `__init__` and any
downstream scaling logic that reads it (shouldn't be any
— the scaffolding was unused). If any test reads
`_entropy_coeff_base`, update it to read `self.entropy_coeff`.

### 7. Tests

New class `TestTargetEntropyController` in
`tests/test_ppo_trainer.py`:

```python
class TestTargetEntropyController:
    def test_log_alpha_initialises_from_entropy_coefficient(self):
        """hp={'entropy_coefficient': 0.01} →
        trainer._log_alpha.exp().item() ≈ 0.01 (within 1e-6).
        Default (no hp) → 0.005."""

    def test_controller_shrinks_alpha_when_entropy_above_target(self):
        """target=100, feed current_entropy=200 → one
        _update_entropy_coefficient call → log_alpha strictly
        smaller than before."""

    def test_controller_grows_alpha_when_entropy_below_target(self):
        """target=100, feed current_entropy=50 → one
        _update_entropy_coefficient call → log_alpha strictly
        larger than before."""

    def test_log_alpha_clamped_within_bounds(self):
        """Stress with a pathological current_entropy=1e6
        pulling log_alpha hard downward; after one step
        log_alpha equals the lower clamp bound, not below
        it. Symmetric for the upper clamp."""

    def test_controller_optimizer_separate_from_policy(self):
        """After a _update_entropy_coefficient call, the
        policy optimiser's state_dict is byte-identical to
        its pre-call state. The controller's optimiser
        state, however, has changed."""

    def test_effective_entropy_coeff_matches_log_alpha_exp(self):
        """After _update_entropy_coefficient returns,
        self.entropy_coeff ==
        self._log_alpha.exp().item() (within float eps)."""

    def test_real_ppo_update_updates_log_alpha(self):
        """Run a real _ppo_update on a synthetic rollout;
        assert that log_alpha changed from its initial
        value. Exercises the wired-in code path, not the
        controller method in isolation (per the 2026-04-18
        units-mismatch lesson)."""
```

New tests in `tests/test_ppo_checkpoint.py` (or extend the
existing checkpoint-test file — locate with
`find tests -name "*checkpoint*"`):

```python
def test_checkpoint_roundtrip_preserves_log_alpha():
    """Save, load, assert log_alpha matches to within float
    eps; alpha_optim_state also round-trips."""

def test_checkpoint_backward_compat_missing_log_alpha():
    """Load a checkpoint dict without the log_alpha or
    alpha_optim_state keys; trainer fresh-inits from the
    default; no crash; logger emits a warning."""
```

The `test_real_ppo_update_updates_log_alpha` test is the
load-bearing regression guard per the 2026-04-18
units-mismatch lesson — it exercises the full wired path.
Unit tests on the controller method alone are insufficient.

### 8. Synthetic-rollout qualitative probe

Documented in `progress.md`, not a pytest. Construct:

- `PPOTrainer` with `target_entropy=112`, default `alpha_lr`.
- 15-episode synthetic rollout: each "episode" is a
  rollout of N=512 transitions with random advantages and
  a forward-pass entropy drawn from a slowly-rising
  sequence (start at 140, rise 5/episode without the
  controller, to simulate the A-baseline dynamic).
- After each synthetic `_ppo_update` call, record
  `self.entropy_coeff`, `self._log_alpha.item()`, and the
  post-step forward-pass entropy.

Assert in the progress-entry text:

- By ep 10, `forward-pass entropy` is within ±15% of 112
  (the target).
- `self._log_alpha` does NOT saturate at either clamp
  bound on a well-behaved rollout.
- `alpha` trajectory is monotonically decreasing until it
  finds the setpoint, then oscillates with small amplitude.

This is the smoke check that matches the A-baseline
pathology. Don't gate the commit on it — gate on the
pytest suite. But document the result in the progress
entry.

### 9. CLAUDE.md

Add a new paragraph under "PPO update stability" (or create
a new "Entropy control" subsection immediately after it):

```
## Entropy control — target-entropy controller (2026-04-19)

Entropy coefficient is a *learned variable*, not a fixed
hyperparameter. A small separate Adam optimiser (lr=1e-4)
optimises `log_alpha = log(entropy_coefficient)` to hold
the policy's current entropy at `target_entropy=112`
(≈ 80% of observed ep-1 pop-avg on a fresh-init population).

When the forward-pass entropy exceeds the target, log_alpha
shrinks (less entropy bonus, entropy falls); when below,
log_alpha grows (more bonus, entropy rises). log_alpha is
clamped to [log(1e-5), log(0.1)] to prevent runaway.

The controller's optimiser is SEPARATE from the policy
optimiser — it does not share momentum state or LR schedule.
The effective `entropy_coefficient` the surrogate loss uses
is `log_alpha.exp().item()`, refreshed after each
controller step.

Load-bearing for any training run on the current scalping
reward shape. Without the controller, entropy drifts
monotone 139 → 200+ across a 15-episode run (observed on
64 agents during `activation-A-baseline` 2026-04-19), the
policy diffuses toward the uniform distribution, and
close-signal / requote-signal actions lose their
probability mass. See `plans/entropy-control-v2/
purpose.md` for the drift evidence and controller design.

Reward magnitudes in `episodes.jsonl` are UNCHANGED by
the controller — the fix is purely on the gradient
pathway. Scoreboard rows pre-controller are directly
comparable to post-controller rows.
```

Historical entries from `policy-startup-stability` and
`naked-clip-and-stability` stay preserved.

### 10. Full suite

```
pytest tests/ -q
```

Must be green. Regression guards:

- `tests/test_ppo_advantage_normalisation.py` — 8 tests.
- `tests/test_ppo_stability.py` — 16 tests (Session 02 of
  predecessor plan).
- `tests/test_ppo_trainer.py::TestEntropyAndCentering` —
  6 tests. The entropy default test needs updating:
  previously asserted `entropy_coeff == 0.005`; now assert
  that `_log_alpha.exp().item() ≈ 0.005` at init (same
  numeric value, different attribute).
- `tests/test_forced_arbitrage.py::TestScalpingReward::
  test_invariant_raw_plus_shaped_equals_total_reward` —
  still green (controller doesn't touch reward).
- `tests/test_smoke_test.py` — stays green. Session 02 of
  this plan updates the smoke-gate assertion; Session 01
  does NOT touch it. If Session 01's changes break any
  smoke-test test, that's a bug in the controller wiring.

### 11. Commit

```
feat(agents): target-entropy controller (learned log_alpha)

Replace the fixed entropy_coefficient with a SAC-style
target-entropy controller. A small separate Adam optimiser
(lr=1e-4) optimises log_alpha to hold the policy's entropy
at target_entropy=112 (≈ 80% of the observed ep-1 pop-avg
on a fresh-init population, 2026-04-19 activation-A-baseline).

Why: Baseline-A validation (2026-04-19, commit 1d5acc9)
found entropy drifting monotone 139.6 → 201.3 across 64
agents × 15 episodes — no agent showed entropy trending
down, close_signal couldn't stick, mean reward stayed
deeply negative. The fixed-coefficient approach from
naked-clip-and-stability Session 03 (halved default +
reward centering) flattened entropy enough to pass the
3-episode smoke gate but couldn't hold the policy stable
over the 15-episode horizon. Sparse scalping rewards
produce ~zero policy gradient on quiet steps; a fixed
entropy bonus dominates and the policy diffuses. A
controller that auto-tunes the coefficient closes the
loop.

Changes:
- New _log_alpha tensor + _alpha_optimizer in PPOTrainer.
- _update_entropy_coefficient runs once per _ppo_update,
  BEFORE the policy step, with the forward-pass entropy
  as input (detached, Python float).
- log_alpha clamped to [log(1e-5), log(0.1)].
- self.entropy_coeff refreshed from log_alpha.exp() after
  each controller step — the loss formula is unchanged.
- Checkpoint format extended with log_alpha and
  alpha_optim_state; backward-compat for pre-controller
  checkpoints (fresh-init + warning).
- Per-episode JSONL row gains `alpha` and `log_alpha`
  fields for learning-curves visualisation.
- _entropy_coeff_base scaffolding (arb-improvements
  Session 2) removed — replaced by the controller.

GA gene `entropy_coefficient` still exists and still
mutates; it now defines the initial log_alpha for fresh
agents, with the controller taking over from there. Gene
range unchanged.

Tests: N new in tests/test_ppo_trainer.py
(TestTargetEntropyController) and
tests/test_ppo_checkpoint.py. The real-_ppo_update test is
the load-bearing regression guard per the 2026-04-18
units-mismatch lesson.

Not changed: reward shape, matcher, action/obs schemas,
PPO stability defences (ratio clamp, KL early-stop,
per-arch LR, warmup), advantage normalisation, reward
centering, gene ranges. Per
plans/entropy-control-v2/hard_constraints.md §1–§3.

pytest tests/ -q: <delta>.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## Cross-session rules

- No reward-path changes. Reward shape stays byte-identical
  to `naked-clip-and-stability`.
- No smoke-gate changes (that's Session 02). If a smoke-gate
  test fails on a legitimate Session 01 change, that's a bug
  in the controller wiring — fix, don't weaken the
  assertion.
- No GA gene-range changes.
- No PPO-stability changes (those landed in
  `naked-clip-and-stability` Session 02).

## After Session 01

1. Append a `progress.md` entry: commit hash, the
   controller wiring, test counts, synthetic-rollout probe
   result.
2. Hand back for Session 02 (smoke-gate slope assertion).
