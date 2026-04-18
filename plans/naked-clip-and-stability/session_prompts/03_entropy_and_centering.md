# Session 03 prompt — Entropy control: halved coefficient + reward centering

## PREREQUISITE — read first

- [`../purpose.md`](../purpose.md) — §3 entropy-diffusion
  description and the 139→189 monotone-rising entropy in
  transformer `0a8cacd3` ep 1–7.
- [`../hard_constraints.md`](../hard_constraints.md). §13
  (entropy_coefficient default halves), §14 (reward
  centering preserves advantage ordering), §20 (tests green),
  §27 (later sessions block if earlier fail).
- [`../master_todo.md`](../master_todo.md) — Session 03
  deliverables.
- `agents/ppo_trainer.py` — the file being edited. In
  particular:
  - `self.entropy_coeff = hp.get("entropy_coefficient",
    0.01)` around line 468 — default changes here.
  - The advantage-computation path (search
    `advantages = `, `returns`, `gamma`) — reward centering
    slots in BEFORE advantage normalisation.
- `plans/policy-startup-stability/progress.md` — the
  per-mini-batch advantage normalisation this session must
  not disturb.

## Locate the code

```
grep -n "entropy_coeff\|entropy_coefficient" agents/ppo_trainer.py
grep -n "returns\|advantages\|gamma\|gae\|_compute_advantage\|_compute_return" agents/ppo_trainer.py
```

Confirm before editing:
1. The entropy coefficient default is currently `0.01`
   (line ~468).
2. Advantages are computed in one place (most PPO
   implementations do). Reward centering hooks in BEFORE
   this computation.
3. The existing per-mini-batch advantage normalisation
   (line ~1271) remains downstream of the centering — they
   compose, they don't conflict.

## What to do

### 1. Halve entropy coefficient default

```python
# Before
self.entropy_coeff = hp.get("entropy_coefficient", 0.01)

# After
self.entropy_coeff = hp.get("entropy_coefficient", 0.005)
```

Add a comment cross-linking this plan:

```python
# Default halved 2026-04-18 per
# plans/naked-clip-and-stability/purpose.md §3 — with
# per-mini-batch advantage normalisation in place, 0.01
# dominates the surrogate term and flattens the policy
# under negative-reward pressure (observed rising entropy
# 139→189 across transformer 0a8cacd3 ep 1–7).
```

GA-expressed values stay unchanged. The gene range for
`entropy_coefficient` is NOT modified — only the default
that fresh agents initialise with (§13).

### 2. Reward centering

Add a running-mean baseline. Class-level state:

```python
# In __init__:
self._reward_ema: float = 0.0
self._reward_ema_alpha: float = 0.01
self._reward_ema_initialised: bool = False
```

In the rollout-processing path (wherever rewards are
converted to returns), update the EMA and subtract BEFORE
the return/advantage computation:

```python
def _update_reward_baseline(self, episode_reward: float) -> None:
    """EMA-updated reward baseline. Subtracted from returns
    before advantage computation to prevent the policy from
    flattening under uniformly-negative rewards (see
    plans/naked-clip-and-stability/purpose.md §3). Does
    not change advantage ordering within a rollout — the
    subtraction is a pure translation of returns."""
    if not self._reward_ema_initialised:
        self._reward_ema = float(episode_reward)
        self._reward_ema_initialised = True
        return
    self._reward_ema = (
        (1.0 - self._reward_ema_alpha) * self._reward_ema
        + self._reward_ema_alpha * float(episode_reward)
    )
```

Call once per episode (or once per rollout, consistent with
whatever unit the trainer counts). Then in the
return-computation path:

```python
# Subtract baseline from per-step rewards before GAE/returns.
centered_rewards = rewards - self._reward_ema
# (existing return computation continues with centered_rewards)
```

Subtraction is a pure translation. Advantages computed on
`centered_rewards` differ from advantages computed on
`rewards` by a constant term that cancels within the
per-mini-batch normalisation. Unit test below asserts this.

Do NOT initialise the EMA to 0 and let it drift — that
produces biased advantages for the first rollout. Either
initialise on the first observed reward (as above) or
bootstrap from a short warmup (5 rollouts, then switch on).
The first-observed-reward approach is simpler and behaves
well empirically.

### 3. Tests

New tests in `tests/test_ppo_trainer.py`:

```python
class TestEntropyAndCentering:
    def test_entropy_default_is_halved(self):
        """Fresh PPOTrainer with no entropy_coefficient
        hyperparameter — self.entropy_coeff == 0.005."""

    def test_entropy_explicit_hp_overrides_default(self):
        """PPOTrainer with hp={'entropy_coefficient':
        0.02} — self.entropy_coeff == 0.02."""

    def test_reward_baseline_initialises_on_first_episode(self):
        """First call to _update_reward_baseline(42.0)
        sets self._reward_ema == 42.0 exactly. Second call
        with a different value applies the EMA."""

    def test_reward_baseline_ema_update_is_monotonic(self):
        """Feeding a monotonically-increasing reward
        sequence produces a monotonically-increasing EMA."""

    def test_centering_preserves_advantage_ordering(self):
        """Compute advantages on a synthetic rollout twice:
        once with reward centering, once without. After
        per-mini-batch normalisation, the two advantage
        tensors agree to within 1e-5 tolerance — centering
        is a pure translation that normalisation erases."""

    def test_centering_fixes_uniformly_negative_rewards(self):
        """Synthetic rollout with all rewards in [-900,
        -200] (replicating transformer 0a8cacd3 scale). With
        centering + advantage normalisation, the resulting
        advantage mean is ≈ 0; without centering, the
        advantage tensor pre-normalisation is strongly
        negative-biased. Sanity check that centering does
        what we think it does."""
```

The ordering-preserved test is the load-bearing principled
check per `hard_constraints.md §14`. If it fails, something
in the normalisation path isn't absorbing the translation
as expected — fix the plumbing, don't weaken the test.

### 4. Qualitative probe

Not a pytest; a scratch-notebook run documented in
`progress.md`. Construct a synthetic rollout mimicking
transformer `0a8cacd3` ep 1–3 magnitudes (rewards ∈ [-900,
+300], 15 episodes). Run `PPOTrainer._ppo_update` with the
combined Session 02 + Session 03 changes. Assert in the
progress-entry text:

- Ep 1 `policy_loss` no longer reaches 10¹⁷ (expect < 100).
- Entropy at ep 3 ≤ entropy at ep 1.

This is the signal that matches the purpose-table
pathology. Don't gate the commit on it — gate on the
pytest suite. But document the result in the progress
entry.

### 5. Full suite

```
pytest tests/ -q
```

Must be green. Regression: existing
`test_invariant_raw_plus_shaped_equals_total_reward` still
passes (Session 01 regression guard remains). Existing
advantage-normalisation test from
`plans/policy-startup-stability/` still passes.

### 6. Commit

```
fix(agents): halve entropy coefficient default + reward centering

Two changes targeting the rising-entropy pathology observed
in transformer `0a8cacd3` gen-2 training (ep 1 entropy 139
→ ep 7 entropy 189):

1. Default entropy_coefficient halves from 0.01 to 0.005.
   With per-mini-batch advantage normalisation in place the
   surrogate-loss term is O(1), and 0.01 was dominating the
   gradient — flattening the policy under uniformly-negative
   reward pressure.
2. Reward centering: subtract an EMA baseline (α=0.01) from
   per-step rewards before return/advantage computation.
   Pure translation; advantage ordering within a rollout is
   preserved. Fixes the "everything is negative → explore
   wider" dynamic without changing the optimal policy.

GA-expressed entropy_coefficient values are unchanged —
only the default for fresh agents is halved. Gene range
unchanged.

Motivation: plans/naked-clip-and-stability/purpose.md §3.

Tests: N new in tests/test_ppo_trainer.py
(TestEntropyAndCentering). The centering-preserves-ordering
test is the principled regression guard (hard_constraints
§14). pytest tests/ -q: <delta>.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## Cross-session rules

- No reward-path changes (that's Session 01).
- No PPO-stability changes (that's Session 02).
- No new shaped terms.
- No GA gene-range changes.

## After Session 03

1. Append a `progress.md` entry: commit hash, the two
   changes, test counts, qualitative-probe result.
2. Hand back for Session 04 (smoke-test gate).
