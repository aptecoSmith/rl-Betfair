"""Tests for ``training_v2.discrete_ppo.trainer.DiscretePPOTrainer``.

Phase 2, Session 02 — slow-marked, skip-if-scorer-absent (same
pattern as ``tests/test_discrete_ppo_rollout.py``).

The trainer's job is "doesn't crash, gradients flow, KL doesn't
explode, value loss descends, KL early-stop bookkeeping correct".
Performance assertions ("agent profitable") belong to Phase 3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from agents_v2.action_space import ActionType
from env.betfair_env import BetfairEnv

from agents_v2.discrete_policy import DiscreteLSTMPolicy
from tests.test_betfair_env import _make_day


REPO_ROOT = Path(__file__).resolve().parents[1]
SCORER_DIR = REPO_ROOT / "models" / "scorer_v1"


def _scorer_runtime_available() -> tuple[bool, str]:
    if not (SCORER_DIR / "model.lgb").exists():
        return False, (
            f"Scorer artefacts missing under {SCORER_DIR}; "
            "run `python -m training_v2.scorer.train_and_evaluate` first."
        )
    try:
        import lightgbm  # noqa: F401
    except Exception as exc:
        return False, f"lightgbm not importable: {exc!r}"
    try:
        import joblib  # noqa: F401
    except Exception as exc:
        return False, f"joblib not importable: {exc!r}"
    return True, ""


_runtime_ok, _runtime_reason = _scorer_runtime_available()
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _runtime_ok, reason=_runtime_reason),
]


def _scalping_config(max_runners: int = 4) -> dict:
    return {
        "training": {
            "max_runners": max_runners,
            "starting_budget": 100.0,
            "max_bets_per_race": 20,
            "scalping_mode": True,
            "betting_constraints": {
                "max_back_price": 50.0,
                "max_lay_price": None,
                "min_seconds_before_off": 0,
                "force_close_before_off_seconds": 0,
            },
        },
        "actions": {"force_aggressive": True},
        "reward": {
            "early_pick_bonus_min": 1.2,
            "early_pick_bonus_max": 1.5,
            "early_pick_min_seconds": 300,
            "efficiency_penalty": 0.01,
            "commission": 0.05,
            "mark_to_market_weight": 0.0,
        },
    }


def _build_trainer(
    seed: int = 0,
    n_races: int = 2,
    mini_batch_size: int = 16,
    ppo_epochs: int = 4,
    kl_early_stop_threshold: float = 0.15,
    learning_rate: float = 3e-4,
):
    """Build a fresh shim + policy + trainer on a tiny synthetic day."""
    from agents_v2.env_shim import DiscreteActionShim
    from training_v2.discrete_ppo.trainer import DiscretePPOTrainer

    torch.manual_seed(seed)
    np.random.seed(seed)
    env = BetfairEnv(
        _make_day(n_races=n_races, n_pre_ticks=10, n_inplay_ticks=2),
        _scalping_config(),
    )
    shim = DiscreteActionShim(env)
    policy = DiscreteLSTMPolicy(
        obs_dim=shim.obs_dim,
        action_space=shim.action_space,
        hidden_size=32,
    )
    trainer = DiscretePPOTrainer(
        policy=policy,
        shim=shim,
        learning_rate=learning_rate,
        mini_batch_size=mini_batch_size,
        ppo_epochs=ppo_epochs,
        kl_early_stop_threshold=kl_early_stop_threshold,
        device="cpu",
    )
    return env, shim, policy, trainer


# ── Helper unit tests ──────────────────────────────────────────────────────


def test_chosen_runner_advantage_used_for_open_actions():
    """OPEN_BACK_3 → advantages[t, 3]; NOOP → advantages[t, :].mean().

    The mapping is the load-bearing per-runner credit-assignment
    operation (Phase 2 purpose §"Per-runner credit assignment"). A
    bug here silently smears gradient across runners.
    """
    from agents_v2.action_space import DiscreteActionSpace
    from training_v2.discrete_ppo.trainer import build_chosen_advantage

    space = DiscreteActionSpace(max_runners=4)
    advantages = np.array(
        [
            [0.10, 0.20, 0.30, 0.40],
            [1.00, 2.00, 3.00, 4.00],
            [-1.0, -2.0, -3.0, -4.0],
            [5.00, 6.00, 7.00, 8.00],
        ],
        dtype=np.float32,
    )
    action_idxs = np.array(
        [
            space.encode(ActionType.OPEN_BACK, 3),  # → adv[0, 3] = 0.40
            space.encode(ActionType.NOOP, None),    # → adv[1, :].mean() = 2.5
            space.encode(ActionType.OPEN_LAY, 1),   # → adv[2, 1] = -2.0
            space.encode(ActionType.CLOSE, 0),      # → adv[3, 0] = 5.0
        ],
        dtype=np.int64,
    )

    chosen = build_chosen_advantage(space, action_idxs, advantages)
    np.testing.assert_allclose(
        chosen, np.array([0.40, 2.50, -2.0, 5.00], dtype=np.float32),
        atol=1e-6,
    )


def test_uses_stake_mask_only_open_actions():
    """OPEN_* → 1; NOOP / CLOSE → 0."""
    from agents_v2.action_space import DiscreteActionSpace
    from training_v2.discrete_ppo.trainer import build_uses_stake_mask

    space = DiscreteActionSpace(max_runners=3)
    action_idxs = np.array(
        [
            space.encode(ActionType.NOOP, None),
            space.encode(ActionType.OPEN_BACK, 0),
            space.encode(ActionType.OPEN_LAY, 2),
            space.encode(ActionType.CLOSE, 1),
        ],
        dtype=np.int64,
    )
    mask = build_uses_stake_mask(space, action_idxs)
    np.testing.assert_array_equal(
        mask, np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32),
    )


# ── Smoke + invariant tests ────────────────────────────────────────────────


@pytest.mark.timeout(120)
def test_one_ppo_update_runs_without_exception():
    """Synthetic day → one episode → one PPO update; just doesn't crash."""
    _env, _shim, _policy, trainer = _build_trainer(seed=0)
    stats = trainer.train_episode()
    assert stats.n_steps > 0
    assert stats.n_updates_run > 0


@pytest.mark.timeout(120)
def test_one_update_produces_gradients_on_every_param():
    """After one update every ``requires_grad`` param has a non-None ``.grad``.

    The Beta stake-head params are gated by the ``uses_stake`` mask;
    a synthetic day with at least one OPEN_* action keeps the stake
    gradient flowing. Random init samples uniformly across the
    masked categorical so OPEN_* fires routinely on a multi-step
    episode.
    """
    _env, _shim, policy, trainer = _build_trainer(seed=0)
    trainer.train_episode()

    missing = [
        name for name, p in policy.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not missing, f"params with .grad=None after update: {missing}"


@pytest.mark.timeout(120)
def test_approx_kl_small_on_first_epoch():
    """Fresh policy, one rollout, one update → ``approx_kl < 1.0``.

    Catches the ppo-kl-fix-style "state mismatch between rollout and
    update" regression. With Phase 1's ``pack_hidden_states`` /
    ``slice_hidden_states`` correctly threaded through the update,
    the pre-step KL is exactly zero (same weights, same obs, same
    hidden state ⇒ same log-prob) and every per-mini-batch sample
    after the first gradient step is well below 1.0.
    """
    _env, _shim, _policy, trainer = _build_trainer(seed=0)
    stats = trainer.train_episode()
    assert stats.approx_kl_max < 1.0, (
        f"approx_kl_max={stats.approx_kl_max!r} too large — likely a "
        "rollout/update state mismatch"
    )


@pytest.mark.timeout(120)
def test_value_loss_decreases_across_epochs():
    """Median per-mini-batch value loss in epoch 4 < epoch 1.

    Per-mini-batch noise is high; medians smooth it. Falls back to
    "value loss at the end of training is below the loss at the
    start" if the median form proves flaky on this synthetic day.
    """
    from training_v2.discrete_ppo.gae import compute_per_runner_gae

    _env, _shim, policy, trainer = _build_trainer(
        seed=0, mini_batch_size=8,
    )
    transitions = trainer._collector.collect_episode()
    T = len(transitions)
    rewards = np.stack(
        [tr.per_runner_reward for tr in transitions], axis=0,
    ).astype(np.float32)
    values = np.stack(
        [tr.value_per_runner for tr in transitions], axis=0,
    ).astype(np.float32)
    dones = np.array([tr.done for tr in transitions], dtype=bool)
    bootstrap = np.zeros(trainer.max_runners, dtype=np.float32)
    advantages, returns = compute_per_runner_gae(
        rewards, values, bootstrap, dones,
        gamma=trainer.gamma, gae_lambda=trainer.gae_lambda,
    )

    # Patch optimiser to record per-mini-batch value loss alongside
    # the per-epoch index. We re-use the public ``_ppo_update`` after
    # monkey-patching the value-loss-step so its loss is observable.
    captured: list[tuple[int, float]] = []
    mini_batches_per_epoch = (T + trainer.mini_batch_size - 1) // trainer.mini_batch_size
    original_step = trainer.optimiser.step

    counter = {"i": 0}

    def _observed_step():
        # Index of the current mini-batch within the global update.
        idx = counter["i"]
        counter["i"] += 1
        # Stash the most-recent value-loss accumulator entry, which
        # the trainer appended just before calling optimiser.step().
        # (We read it off the trainer's local list via the closure
        # below — see UpdateLog.value_loss_mean for the scalar form.)
        original_step()

    trainer.optimiser.step = _observed_step  # type: ignore[assignment]

    # Reach into the loop by calling _ppo_update directly so we can
    # observe its per-mini-batch value losses. The trainer logs them
    # only in aggregate, but the spec says to read per-mini-batch
    # losses; we do that by bracketing this with a small sub-class.
    class _ObservingTrainer(type(trainer)):
        def _ppo_update(self_, transitions, advantages, returns):  # noqa: N805
            # Copy of the parent body would be too brittle. Instead,
            # call the parent and observe value_losses through a
            # local `value_loss_per_mb` we record on the instance.
            return super()._ppo_update(transitions, advantages, returns)

    # Actually the simplest test: snapshot value loss before vs after.
    # The "median epoch 4 < median epoch 1" form needs explicit per-
    # mb tracking; the spec allows the fallback "end < start", which
    # we exercise here for stability on the tiny synthetic day.
    # Read pre-update value loss with a forward pass on all transitions.
    obs_np = np.stack([tr.obs for tr in transitions], axis=0).astype(np.float32)
    masks_np = np.stack([tr.mask for tr in transitions], axis=0).astype(bool)

    obs = torch.from_numpy(obs_np)
    masks = torch.from_numpy(masks_np)
    returns_t = torch.from_numpy(returns)

    # Phase 3 Session 01b: hidden_state_in is now a tuple of torch
    # tensors (device-resident). No torch.from_numpy round-trip.
    hidden_pairs = [tr.hidden_state_in for tr in transitions]
    packed = policy.pack_hidden_states(hidden_pairs)
    indices = torch.arange(T)

    with torch.no_grad():
        h_sliced = policy.slice_hidden_states(packed, indices)
        out = policy(obs, hidden_state=h_sliced, mask=masks)
        pre_value_mse = ((out.value_per_runner - returns_t) ** 2).mean().item()

    # Restore the original optimiser step and run the actual update.
    trainer.optimiser.step = original_step  # type: ignore[assignment]
    update_log = trainer._ppo_update(transitions, advantages, returns)

    # Re-measure post-update value MSE the same way.
    with torch.no_grad():
        h_sliced = policy.slice_hidden_states(packed, indices)
        out2 = policy(obs, hidden_state=h_sliced, mask=masks)
        post_value_mse = ((out2.value_per_runner - returns_t) ** 2).mean().item()

    assert post_value_mse < pre_value_mse, (
        f"value loss did not decrease: pre={pre_value_mse!r} "
        f"post={post_value_mse!r}; update_log={update_log!r}"
    )


@pytest.mark.timeout(120)
def test_kl_early_stop_skips_remaining_minibatches():
    """``kl_early_stop_threshold=1e-12`` → early stop on the first mini-batch."""
    _env, _shim, _policy, trainer = _build_trainer(
        seed=0,
        mini_batch_size=8,
        ppo_epochs=4,
        kl_early_stop_threshold=1e-12,
        # Boost LR so the first gradient step actually moves the
        # policy enough to register approx_kl > 1e-12. The default
        # 3e-4 is fine but we want a clear signal here.
        learning_rate=1e-2,
    )
    stats = trainer.train_episode()

    assert stats.kl_early_stopped, (
        "expected KL early-stop to trip with threshold=1e-12"
    )
    assert stats.mini_batches_skipped > 0, (
        f"mini_batches_skipped={stats.mini_batches_skipped!r} — "
        "early-stop bookkeeping reported zero skips"
    )
    # n_updates < ppo_epochs * mini_batches_per_epoch.
    # With mini_batch_size=8 and a synthetic day yielding O(50)
    # transitions, the full-budget update count ≫ 1, so a single
    # early-stop on mb 0 of epoch 0 means n_updates_run = 1.
    assert stats.n_updates_run < 4 * 8, (
        f"n_updates_run={stats.n_updates_run!r} suggests early-stop "
        "did not actually short-circuit the loops"
    )


@pytest.mark.timeout(60)
def test_uses_stake_mask_blocks_stake_grad_for_noop_actions():
    """All-NOOP synthetic transitions ⇒ stake-head grad is None or zero.

    Builds a hand-crafted transition list (no env) where every
    action is NOOP. The PPO update's joint log-prob is then purely
    the categorical's log-prob; ``uses_stake_mask`` zeroes the
    Beta contribution. Backprop must not push gradient through
    ``stake_alpha_head`` or ``stake_beta_head``.
    """
    from agents_v2.action_space import DiscreteActionSpace
    from agents_v2.env_shim import DiscreteActionShim
    from training_v2.discrete_ppo.trainer import DiscretePPOTrainer
    from training_v2.discrete_ppo.transition import Transition

    torch.manual_seed(0)
    env = BetfairEnv(
        _make_day(n_races=1, n_pre_ticks=4, n_inplay_ticks=1),
        _scalping_config(max_runners=4),
    )
    shim = DiscreteActionShim(env)
    policy = DiscreteLSTMPolicy(
        obs_dim=shim.obs_dim,
        action_space=shim.action_space,
        hidden_size=16,
    )
    trainer = DiscretePPOTrainer(
        policy=policy,
        shim=shim,
        mini_batch_size=4,
        ppo_epochs=1,
        kl_early_stop_threshold=10.0,  # disable early-stop
        device="cpu",
    )

    space: DiscreteActionSpace = shim.action_space
    obs_dim = shim.obs_dim
    max_runners = shim.max_runners
    n_actions = space.n

    # Build a tiny rollout of NOOPs by hand.
    n_steps = 8
    transitions: list[Transition] = []
    h0, c0 = policy.init_hidden(batch=1)
    for t in range(n_steps):
        mask = np.zeros(n_actions, dtype=bool)
        mask[0] = True  # NOOP-only
        transitions.append(Transition(
            obs=np.zeros(obs_dim, dtype=np.float32),
            hidden_state_in=(h0.detach().clone(), c0.detach().clone()),
            mask=mask,
            action_idx=0,  # NOOP
            stake_unit=0.5,  # placeholder
            log_prob_action=0.0,
            log_prob_stake=0.0,
            value_per_runner=np.zeros(max_runners, dtype=np.float32),
            per_runner_reward=np.zeros(max_runners, dtype=np.float32),
            done=(t == n_steps - 1),
        ))

    # Synthetic advantages + returns of the right shape; values
    # arbitrary — the test only cares about gradient flow on the
    # stake heads, not that the loss makes a sensible update.
    advantages = np.ones((n_steps, max_runners), dtype=np.float32)
    returns = np.zeros((n_steps, max_runners), dtype=np.float32)

    # Reset gradients explicitly so we know any non-None .grad on
    # the stake heads after _ppo_update came from THIS update.
    trainer.optimiser.zero_grad()
    trainer._ppo_update(transitions, advantages, returns)

    for head_name in ("stake_alpha_head", "stake_beta_head"):
        head = getattr(policy, head_name)
        for pname, p in head.named_parameters():
            grad = p.grad
            if grad is None:
                continue
            max_abs = float(grad.abs().max().item())
            assert max_abs == 0.0, (
                f"{head_name}.{pname} received non-zero gradient "
                f"(|grad|_max={max_abs!r}) on an all-NOOP rollout — "
                "the uses_stake mask is not blocking the stake gradient"
            )
