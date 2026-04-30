# Session prompt — Phase 2, Session 01: rollout collector + per-runner GAE

Use this prompt to open a new session in a fresh context. The prompt
is self-contained — it briefs you on the task, the design decisions
already locked, and the constraints. Do not require any context from
the session that scaffolded this prompt.

---

## The task

Build the rollout-collection layer for the v2 PPO trainer. **No PPO
update, no surrogate loss, no optimiser** — that's Session 02. This
session ships:

1. A `Transition` dataclass carrying everything the PPO update will
   eventually need.
2. A `RolloutCollector` that drives Phase 1's `DiscreteActionShim` +
   `DiscreteLSTMPolicy` through one episode and returns a list of
   `Transition`s.
3. A `compute_per_runner_gae` pure function with hand-checked
   correctness on a synthetic toy.

End-of-session bar: roll out one episode, GAE produces a per-runner
advantage tensor shaped `(n_steps, max_runners)` whose values match
a hand-computed reference on a 3-step / 2-runner toy.

## What you need to read first

1. `plans/rewrite/README.md` — rewrite plan overview, hard
   constraints.
2. `plans/rewrite/phase-2-trainer/purpose.md` — locked algorithm
   shape, per-runner GAE math, hyperparameter table. Section
   "Per-runner credit assignment" is the load-bearing piece.
3. `plans/rewrite/phase-1-policy-and-env-wiring/findings.md` —
   Phase 1 hand-offs. The five constraints listed there apply to
   this session verbatim.
4. `agents_v2/__init__.py`, `agents_v2/discrete_policy.py`,
   `agents_v2/env_shim.py` — Phase 1's deliverables. **Import
   these; do not re-implement.**
5. `CLAUDE.md` §"Recurrent PPO: hidden-state protocol on update" —
   the rollout-time hidden-state capture rule. v1's `ppo-kl-fix`
   plan got this wrong on the first try and paid for it; we don't
   need to re-derive it.
6. `agents/ppo_trainer.py::_collect_rollout` — v1 reference for the
   rollout shape. **Read, don't import.** v2 has a much simpler
   `Transition` (no fill/mature labels, no risk targets, no
   forward-looking returns) but the rollout loop's outer shape is
   the same.

## What to do

### 1. `training_v2/discrete_ppo/transition.py` (~30 min)

```python
@dataclass(frozen=True)
class Transition:
    obs: np.ndarray                            # (obs_dim,) float32
    hidden_state_in: tuple[np.ndarray, ...]    # before-forward state
    mask: np.ndarray                           # (action_space.n,) bool
    action_idx: int
    stake_unit: float                          # ∈ (0, 1), Beta sample
    log_prob_action: float                     # categorical log-prob
    log_prob_stake: float                      # Beta log-prob (or 0 if action doesn't use stake)
    value_per_runner: np.ndarray               # (max_runners,) float32
    per_runner_reward: np.ndarray              # (max_runners,) float32
    done: bool
```

Notes:

- `hidden_state_in` is stored as a tuple of CPU numpy arrays so
  serialising / re-loading rollouts later is trivial. The collector
  detaches + `.cpu().numpy()`s before stashing.
- `log_prob_stake` is **zero** when the chosen action doesn't use
  stake (NOOP, CLOSE_*). The PPO update will multiply by an
  "action-uses-stake" mask before contributing to the surrogate
  loss; storing zero here is a placeholder, not a real log-prob.
- Use the `DiscreteActionSpace` to derive "uses stake" (`OPEN_BACK`
  / `OPEN_LAY` only) — don't hard-code the index ranges.

### 2. `training_v2/discrete_ppo/rollout.py` (~90 min)

```python
class RolloutCollector:
    def __init__(
        self,
        shim: DiscreteActionShim,
        policy: BaseDiscretePolicy,
        device: str = "cpu",
    ): ...

    def collect_episode(self) -> list[Transition]:
        """Run one full episode end-to-end.

        Returns the list of transitions in order. The final
        transition's ``done`` is True; intermediate transitions'
        ``done`` is False. Uses the policy's masked categorical
        sampling + Beta stake sampling exactly as Phase 1's smoke
        driver does.
        """
```

Implementation notes:

- **Capture hidden state BEFORE the forward pass.** This is the
  ppo-kl-fix gotcha. The state at step `t` that goes into the
  transition is the state that was passed INTO `policy.forward`,
  not the state returned by it.
- Per-runner reward attribution: at each step, read
  `info["per_pair_pnl_delta"]` (env exposes pair-level P&L deltas;
  if it doesn't expose per-step deltas, derive them from
  `bm.bets` + the previous step's snapshot). Map each pair's
  `selection_id` to a runner slot via
  `env._slot_maps[env._race_idx]`. Sum per-runner P&L deltas;
  distribute non-attributable reward components (terminal bonus,
  episode-level shaping if any) equally across `max_runners` slots.
- Sanity check: `per_runner_reward.sum(axis=1)` should match
  `info["raw_pnl_reward"] + info["shaped_bonus"]` to within
  floating-point noise. If not, the attribution is wrong. Add this
  as an assertion in the collector itself (cheap, catches drift
  early).
- The Beta stake sample: use the unit-interval `s ∈ (0, 1)` directly
  — the shim's `step(stake=…)` expects £, but for THIS rollout the
  shim is configured to accept the raw unit-interval value; pass
  `stake=s * shim.env.bet_manager.budget` (or revisit if the shim's
  contract is stricter). Confirm by reading
  `agents_v2/env_shim.py::step` first.

### 3. `training_v2/discrete_ppo/gae.py` (~45 min)

```python
def compute_per_runner_gae(
    rewards: np.ndarray,        # (n_steps, max_runners)
    values: np.ndarray,         # (n_steps, max_runners)
    bootstrap_value: np.ndarray, # (max_runners,) — V(s_T) for the final state
    dones: np.ndarray,          # (n_steps,) bool — episode boundary
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Standard GAE applied per-runner.

    Returns (advantages, returns), each shape (n_steps, max_runners).
    """
```

Pure NumPy — no torch, no env. The function is reusable in tests and
in the trainer.

Math:

```
δ_t^{(i)}  = r_t^{(i)} + γ * (1 - done_t) * V_{t+1}^{(i)} - V_t^{(i)}
A_t^{(i)}  = δ_t^{(i)} + γλ * (1 - done_t) * A_{t+1}^{(i)}
return_t^{(i)} = A_t^{(i)} + V_t^{(i)}
```

`done_t` zeroes the bootstrap when the episode ends. For the final
step, `V_{t+1}` is `bootstrap_value` (a separate forward pass on the
terminal observation, or zero if the episode terminated naturally).

### 4. Tests (~75 min)

`tests/test_discrete_ppo_transition.py`:

- `test_transition_round_trip` — construct, read fields back.
- `test_uses_stake_only_for_open_actions` — the helper that decides
  whether `log_prob_stake` contributes returns True for
  OPEN_BACK_*, OPEN_LAY_*, False for NOOP, CLOSE_*.

`tests/test_discrete_ppo_rollout.py` (slow-marked, skip-if-scorer-
absent — same pattern as Phase 1's smoke test):

- `test_collect_episode_emits_one_transition_per_step` on a tiny
  synthetic day.
- `test_hidden_state_in_captured_before_forward_pass` — t=0's
  `hidden_state_in` is zero; a later step's is non-zero. Catches
  the post-forward-capture bug Phase 2's purpose.md flags.
- `test_per_runner_reward_sums_to_total_reward` on a 50-step
  synthetic episode. Within 1e-5 absolute.
- `test_mask_is_carried_with_transition` — random sample of 10
  transitions, the stored mask matches what `shim.get_action_mask()`
  returned at that step (re-run the collector with the same seed
  and check).

`tests/test_discrete_ppo_gae.py`:

- `test_gae_per_runner_shape` — input `(5, 3)` → output `(5, 3)`.
- `test_gae_matches_hand_reference` — 3-step / 2-runner toy,
  hand-compute advantages with γ=0.9, λ=0.5, assert
  `np.allclose`. Use simple round-numbered rewards and values so
  the hand-compute is auditable in the test docstring.
- `test_gae_returns_equal_advantages_plus_values` — definitional
  guard: `returns = advantages + values` exactly.
- `test_gae_zero_rewards_zero_advantages_iff_values_constant` — if
  `rewards == 0` and `values` is constant across time, advantages
  are zero. Sanity guard for the discount machinery.

## Stop conditions

- All 4 success-bar conditions for THIS session pass → write a
  short note in your session output, message operator "Phase 2
  Session 01 GREEN, ready for Session 02", **stop**.
- The collector's per-runner reward attribution doesn't match
  `info["raw_pnl_reward"] + info["shaped_bonus"]` to floating-
  point noise → **stop and investigate**. This is the load-bearing
  invariant for the whole phase. Likely cause: a reward component
  in `_settle_current_race` we forgot to attribute. Document the
  component, file the attribution rule, then resume.
- The GAE hand-reference test fails on its first run → **stop**.
  Either the math is wrong or the test reference is wrong; both
  block Session 02.

## Hard constraints

- **No PPO update.** No optimiser, no loss, no `.backward()`. The
  rollout collector and the GAE function are pure data
  transformations.
- **No env edits.** Same as Phase 1.
- **No re-import of v1 classes.** Parallel tree.
- **No new shaped rewards** (rewrite hard constraint §5; reiterating
  because the per-runner attribution code is exactly where the
  temptation to "add a small per-runner shaping term" might appear
  — don't).
- **Per-runner reward attribution must SUM to the env's scalar
  reward.** Within float tolerance. The unit test enforces this;
  the collector's runtime assertion enforces it on every step.
- **Hidden state captured BEFORE forward.** Non-negotiable. The
  load-bearing test asserts t=0 state is zero.

## Out of scope

- PPO update / surrogate loss (Session 02).
- Optimiser construction (Session 02).
- Real training run on a real day (Session 03).
- Cohort / GA / multi-day (Phase 3).
- Frontend wiring (Phase 3).
- Performance benchmarking — collector wall time is fine to log,
  but no optimisation work.

## Useful pointers

- `agents_v2/env_shim.py::DiscreteActionShim` — `step(action_idx,
  stake=…, arb_spread=…)` returns `(obs, reward, term, trunc, info)`.
- `agents_v2/discrete_policy.py::DiscreteLSTMPolicy.forward` —
  returns `DiscretePolicyOutput`. The `action_dist.sample()` and
  `Beta(stake_alpha, stake_beta).sample()` give you the discrete
  index and the stake unit.
- `env/bet_manager.py::BetManager.bets` — list of `Bet` objects;
  each has `selection_id`, `pnl`, `outcome`. The collector reads
  these to compute per-step per-runner P&L deltas.
- `env/betfair_env.py::_settle_current_race` — the env's reward
  computation. Read this to understand which components are
  per-runner attributable and which aren't (terminal bonus is
  episode-level; per-pair P&L is per-runner).
- `tests/test_betfair_env.py::_make_day` — synthetic day fixture
  for the slow-marked rollout test.

## Estimate

3–5 hours.

- 30 min: `Transition` + helper.
- 90 min: `RolloutCollector`.
- 45 min: `compute_per_runner_gae`.
- 75 min: tests.
- 30 min: session writeup (a short note, not a full findings.md —
  Session 03 writes the phase-level findings).

If past 6 hours, stop and check scope. The most likely overrun is
the per-runner reward attribution — `_settle_current_race` has 8+
reward components and getting the per-runner split right for ALL
of them takes more reading than the prompt budgets. If the
synthetic-day test passes but the real-day attribution drifts, ship
Session 01 with the synthetic-day bar and file a Session 01b
follow-on.
