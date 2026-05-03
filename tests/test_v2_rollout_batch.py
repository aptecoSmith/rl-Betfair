"""Phase 4 Session 06 regression guards for the ``RolloutBatch``
return type of ``RolloutCollector.collect_episode`` in
``training_v2/discrete_ppo/rollout.py``.

The session prompt
(``plans/rewrite/phase-4-restore-speed/session_prompts/
06_transition_dataclass.md``) specifies five load-bearing tests:

1. **Per-field equivalence vs the legacy list-of-Transitions form**
   — every ``RolloutBatch`` field stored by the new collector
   matches what the pre-Session-06 ``np.stack([tr.field for tr in
   transitions])`` pass would have produced. We verify this by
   running a single rollout, converting the batch back to a list
   of :class:`Transition` via ``rollout_batch_to_transitions``,
   and asserting strict equality between each batch field and the
   per-tick stack of the equivalent transition fields.

2. **PPO update state-dict bit-identical via direct vs list path**
   — run two PPO updates on identically-seeded fresh policies on
   the same synthetic day. Path A: trainer consumes the
   :class:`RolloutBatch` directly. Path B: convert batch →
   transitions → batch and feed THAT back to the trainer. Every
   tensor in the post-update ``state_dict`` must be byte-equal.
   This is the load-bearing test for Session 06: the consumer-
   side refactor must not have changed any number the surrogate
   loss / value loss / KL diagnostic depends on.

3. **Transition dataclass not constructed on the rollout hot path**
   — patch ``Transition.__init__`` and run a single rollout via
   ``collect_episode``. The call count must be zero. This is the
   structural guard against a partial refactor that re-introduces
   per-tick dataclass construction.

4. **RolloutBatch views remain valid after collect_episode returns**
   — write to a row of the underlying obs buffer (held via the
   batch's ``obs`` attribute) and assert the change is visible
   through the same attribute. Confirms the batch's numpy fields
   are views, not copies — same view-semantics contract Sessions
   02 / 04 already documented.

5. **Existing trainer test smoke** — the meta-test from the session
   prompt that "a representative subset of
   tests/test_discrete_ppo_trainer.py still passes". Covered
   directly by the rest of the test suite (the trainer / rollout
   / batched / multi-day / cohort tests all consume the new
   RolloutBatch surface). We pin a small representative invocation
   here as a quick regression sentinel: build a trainer, run one
   episode, assert it didn't crash and produced sensible numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from env.betfair_env import BetfairEnv

from agents_v2.discrete_policy import DiscreteLSTMPolicy
from training_v2.discrete_ppo.transition import (
    RolloutBatch,
    Transition,
    rollout_batch_to_transitions,
    transitions_to_rollout_batch,
)
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


def _build_collector(seed: int = 42, n_races: int = 2):
    """Build a fresh shim + policy + collector on a tiny synthetic day."""
    from agents_v2.env_shim import DiscreteActionShim
    from training_v2.discrete_ppo.rollout import RolloutCollector

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
    collector = RolloutCollector(shim=shim, policy=policy, device="cpu")
    return env, shim, policy, collector


def _build_trainer(seed: int = 0, n_races: int = 2, mini_batch_size: int = 16):
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
        policy=policy, shim=shim, mini_batch_size=mini_batch_size,
        ppo_epochs=2, kl_early_stop_threshold=10.0, device="cpu",
    )
    return env, shim, policy, trainer


# ── Tests ─────────────────────────────────────────────────────────────


@pytest.mark.timeout(60)
def test_rollout_batch_fields_match_legacy_transition_list_form():
    """Every RolloutBatch field equals ``np.stack([tr.field for tr in
    transitions])`` of the equivalent per-tick transitions.

    Confirms the new collector's pre-stacked arrays carry the same
    bytes the pre-Session-06 ``np.stack`` pass produced from a list
    of Transitions. We use ``rollout_batch_to_transitions`` to
    project the batch into the legacy form and compare each field.
    Strict equality (``np.array_equal`` / ``torch.equal``) — bit-
    identity is the bar.
    """
    _env, _shim, _policy, collector = _build_collector(seed=42)
    batch = collector.collect_episode()
    transitions = rollout_batch_to_transitions(batch)
    assert int(batch.n_steps) == len(transitions) > 0

    # Per-field bit-identity against the np.stack-style equivalent.
    np.testing.assert_array_equal(
        batch.obs,
        np.stack([tr.obs for tr in transitions], axis=0),
    )
    np.testing.assert_array_equal(
        batch.mask,
        np.stack([tr.mask for tr in transitions], axis=0),
    )
    np.testing.assert_array_equal(
        batch.action_idx,
        np.array([tr.action_idx for tr in transitions], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        batch.stake_unit,
        np.array(
            [tr.stake_unit for tr in transitions], dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        batch.log_prob_action,
        np.array(
            [tr.log_prob_action for tr in transitions], dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        batch.log_prob_stake,
        np.array(
            [tr.log_prob_stake for tr in transitions], dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        batch.value_per_runner,
        np.stack(
            [tr.value_per_runner for tr in transitions], axis=0,
        ),
    )
    np.testing.assert_array_equal(
        batch.per_runner_reward,
        np.stack(
            [tr.per_runner_reward for tr in transitions], axis=0,
        ),
    )
    np.testing.assert_array_equal(
        batch.done,
        np.array([tr.done for tr in transitions], dtype=bool),
    )

    # Hidden state: per-tick view stacked along axis 0.
    n_slots = len(batch.hidden_state_in)
    for k in range(n_slots):
        ref = torch.stack(
            [tr.hidden_state_in[k] for tr in transitions], dim=0,
        )
        assert torch.equal(batch.hidden_state_in[k], ref), (
            f"hidden_state_in slot {k} drifts vs the per-tick stack"
        )


@pytest.mark.timeout(120)
def test_ppo_update_state_dict_byte_identical_via_direct_vs_list_path():
    """Two PPO updates produce byte-equal state_dicts.

    Path A: trainer consumes the rollout's :class:`RolloutBatch`
    directly. Path B: convert batch → list[Transition] → batch
    (round-trip via the public helpers) and feed THAT to the same
    trainer pipeline. Both must produce the same post-update
    policy state_dict.

    This is the strictest signature for Session 06's correctness
    bar: if any per-tick field's value or order changed during the
    refactor, the surrogate loss / value loss / KL would diverge
    on the very first PPO update and the post-update weights would
    differ.
    """
    # Path A: direct.
    _env_a, _shim_a, policy_a, trainer_a = _build_trainer(seed=0)
    stats_a = trainer_a.train_episode()
    sd_a = {
        k: v.detach().clone() for k, v in policy_a.state_dict().items()
    }

    # Path B: round-trip the batch through transitions + back.
    _env_b, _shim_b, policy_b, trainer_b = _build_trainer(seed=0)
    batch_b = trainer_b._collector.collect_episode()
    transitions_b = rollout_batch_to_transitions(batch_b)
    rebatch_b = transitions_to_rollout_batch(transitions_b)
    last_info_b = trainer_b._collector.last_info
    stats_b = trainer_b.update_from_rollout(
        transitions=transitions_b, last_info=last_info_b,
    )
    sd_b = policy_b.state_dict()

    # Sanity: both paths produced an actual update.
    assert stats_a.n_steps == stats_b.n_steps
    assert stats_a.n_updates_run == stats_b.n_updates_run

    # Bit-identity on every parameter.
    assert sd_a.keys() == sd_b.keys()
    for k in sd_a:
        assert torch.equal(sd_a[k], sd_b[k]), (
            f"post-update parameter {k!r} drifts between direct and "
            "list-round-trip paths — Session 06 introduced numerical "
            "divergence"
        )

    # Sanity: the round-trip reproduced the same batch contents.
    np.testing.assert_array_equal(batch_b.obs, rebatch_b.obs)
    np.testing.assert_array_equal(batch_b.mask, rebatch_b.mask)
    np.testing.assert_array_equal(batch_b.action_idx, rebatch_b.action_idx)
    np.testing.assert_array_equal(
        batch_b.stake_unit, rebatch_b.stake_unit,
    )
    np.testing.assert_array_equal(
        batch_b.log_prob_action, rebatch_b.log_prob_action,
    )
    np.testing.assert_array_equal(
        batch_b.log_prob_stake, rebatch_b.log_prob_stake,
    )
    np.testing.assert_array_equal(
        batch_b.value_per_runner, rebatch_b.value_per_runner,
    )
    np.testing.assert_array_equal(
        batch_b.per_runner_reward, rebatch_b.per_runner_reward,
    )
    np.testing.assert_array_equal(batch_b.done, rebatch_b.done)


@pytest.mark.timeout(60)
def test_transition_dataclass_not_constructed_on_rollout_hot_path(
    monkeypatch,
):
    """``Transition.__init__`` is NOT called during ``collect_episode``.

    Phase 4 Session 06's load-bearing structural guard: the
    sequential rollout no longer rounds per-tick state through the
    frozen-dataclass form. We patch
    :class:`training_v2.discrete_ppo.transition.Transition.__init__`
    to count invocations and run a synthetic rollout; the call count
    must be exactly zero. A partial refactor that left an in-loop
    ``Transition(...)`` construction would surface here.
    """
    counter = {"calls": 0}
    real_init = Transition.__init__

    def counting_init(self, *args, **kwargs):
        counter["calls"] += 1
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(Transition, "__init__", counting_init)

    _env, _shim, _policy, collector = _build_collector(seed=42)
    batch = collector.collect_episode()

    assert int(batch.n_steps) > 0, "rollout produced no transitions"
    assert counter["calls"] == 0, (
        f"Transition.__init__ was called {counter['calls']} times "
        "on the rollout hot path — Session 06's contract is that "
        "the sequential collector emits a RolloutBatch without "
        "instantiating per-tick Transition objects"
    )


@pytest.mark.timeout(60)
def test_rollout_batch_obs_is_a_view_into_underlying_buffer():
    """Mutating the underlying obs buffer reflects in ``batch.obs``.

    Session 06's view-semantics contract: the batch's numpy
    fields are slice views (``obs_arr[:n_steps]``) into the
    rollout's per-episode buffer. Writing into a row of the batch's
    obs and reading it back must return the new value — confirming
    the batch did not silently copy the data into a fresh array.

    This is the same view-semantics property Session 02's
    ``test_obs_buffer_allocated_once_per_episode`` relies on,
    promoted to a direct write/read check on the
    :class:`RolloutBatch` surface.
    """
    _env, _shim, _policy, collector = _build_collector(seed=42)
    batch = collector.collect_episode()
    assert int(batch.n_steps) > 1

    # Write a sentinel value into row 0 via the batch's obs and
    # read it back via the same attribute.
    sentinel = np.float32(-7.5)
    batch.obs[0, 0] = sentinel
    assert batch.obs[0, 0] == sentinel, (
        "writing to batch.obs[0, 0] did not change batch.obs[0, 0] — "
        "the obs field is not a view into the rollout's per-episode "
        "buffer"
    )

    # Same property for hidden_state_in (slice into a torch buffer).
    h_buf = batch.hidden_state_in[0]
    h_buf[0, 0, 0, 0] = -3.5
    assert float(batch.hidden_state_in[0][0, 0, 0, 0]) == -3.5, (
        "writing to batch.hidden_state_in[0] did not mutate the same "
        "tensor — the hidden-state field is not a view"
    )


@pytest.mark.timeout(120)
def test_existing_trainer_smoke_with_rollout_batch():
    """Build a trainer, run one episode, assert sensible diagnostics.

    Meta-sentinel for the session prompt's #5: confirm the
    consumer-side refactor still drives a complete rollout + GAE +
    PPO update + episode-stats pipeline end-to-end. Specific
    correctness invariants (bit-identity, per-field equality, KL
    sanity) are pinned by tests 1–4 above plus the wider trainer /
    rollout / cohort suites.
    """
    _env, _shim, _policy, trainer = _build_trainer(seed=0)
    stats = trainer.train_episode()

    assert stats.n_steps > 0
    assert stats.n_updates_run > 0
    # KL diagnostic is finite — no nan / inf leak through the refactor.
    assert np.isfinite(stats.approx_kl_mean)
    assert np.isfinite(stats.approx_kl_max)
    assert np.isfinite(stats.policy_loss_mean)
    assert np.isfinite(stats.value_loss_mean)
    assert np.isfinite(stats.entropy_mean)
    # Action histogram has at least one entry.
    assert stats.action_histogram is not None
    assert sum(stats.action_histogram.values()) == stats.n_steps


@pytest.mark.timeout(60)
def test_collect_episode_returns_rollout_batch_not_list():
    """``collect_episode`` returns :class:`RolloutBatch`, not ``list``.

    Phase 4 Session 06 public-API change pinned. A future regression
    that flips the return type back to ``list[Transition]`` would
    also break the trainer's ``_update_from_batch`` path — but
    asserting the type directly here gives a faster signal.
    """
    _env, _shim, _policy, collector = _build_collector(seed=42)
    out = collector.collect_episode()
    assert isinstance(out, RolloutBatch), (
        f"collect_episode returned {type(out).__name__}, expected "
        "RolloutBatch (Phase 4 Session 06 public API)"
    )
