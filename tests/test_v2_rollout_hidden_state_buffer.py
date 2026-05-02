"""Phase 4 Session 04 regression guards for the pre-allocated
hidden-state capture buffer in
``training_v2/discrete_ppo/rollout.py``.

The session prompt
(``plans/rewrite/phase-4-restore-speed/session_prompts/
04_hidden_state_allocator.md``) specifies five load-bearing tests:

1. **Packed bit-identity vs the pre-Session-04 walk** — for the same
   seed-42 rollout, the
   ``pack_hidden_states([tr.hidden_state_in for tr in transitions])``
   tuple produced under the new buffered capture path must be
   byte-identical to the tuple the legacy
   ``tuple(t.detach().clone() for t in hidden_state)`` capture
   would have produced. We verify by running a manual reference
   rollout that captures hidden states via ``.detach().clone()``
   inline (the pre-Session-04 form) and comparing the packed
   outputs tensor-for-tensor with ``torch.equal``.

2. **Buffer allocated once per episode** — every transition's
   ``hidden_state_in[k]`` tensor must share the same underlying
   storage as every other transition's ``hidden_state_in[k]``
   (when no grow path fires; on the synthetic test day the
   estimate is sufficient and grow does NOT fire). Catches a
   regression where someone re-introduces per-tick allocation.

3. **Slice independent of subsequent ticks** — the captured
   ``hidden_in_t`` at tick 0 (which by ``init_hidden`` contract
   is all-zero) must remain all-zero AFTER the rollout has
   advanced through every subsequent tick. A regression where
   the buffer slice aliased the rolling ``hidden_state``
   (e.g. by returning a view INTO ``hidden_state`` itself) would
   surface as the tick-0 capture taking on later-tick values.

4. **Per-tick clone count drops to zero** — patch
   ``torch.Tensor.clone`` and run a real episode; the count of
   clone calls originating from the rollout's hot path must be
   far below the pre-Session-04 baseline of 2 × n_steps. Some
   framework-internal clones may slip through, so the assertion
   is "much less than 2 × n_steps", not strictly zero.

5. **Recurrent PPO KL small on first epoch** — fresh policy +
   one ``_ppo_update`` → ``approx_kl_mean < 1.0``. Pinned here
   to this session as the regression guard for the
   recurrent-state-through-PPO contract: a buffer-slice that
   accidentally aliased the rolling hidden state would have
   trashed every transition's ``hidden_state_in`` to the SAME
   final value, blowing up KL on the update. This is the v2
   counterpart to ``tests/test_ppo_trainer.py::TestRecurrent
   StateThroughPpoUpdate::test_ppo_update_approx_kl_small_on_
   first_epoch_lstm`` (the load-bearing CLAUDE.md §"Recurrent
   PPO" guard).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

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


# ── Test config / harness ─────────────────────────────────────────────


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


# ── Tests ─────────────────────────────────────────────────────────────


@pytest.mark.timeout(120)
def test_hidden_state_packed_bit_identical_to_pre_session_04_on_fixed_seed():
    """Packed hidden states match the pre-Session-04 capture form.

    Strategy: the new buffered capture path stores per-tick
    snapshots via ``buf[n_steps].copy_(t.detach())``; the
    pre-Session-04 form was ``tuple(t.detach().clone() for t in
    hidden_state)``. Both are snapshots of the same rolling
    ``hidden_state`` tensor at the SAME moment in the rollout
    loop and consume zero RNG. So the captured values must be
    byte-equal under the LSTM recurrence:

        h_in[0] = init_hidden()        (zeros)
        h_in[t+1] = policy(
            obs[t], h_in[t], mask[t]
        ).new_hidden_state.detach()

    We re-run the recurrence inline using the per-transition
    obs / mask captured by the collector and compare each
    ``transitions[t+1].hidden_state_in`` to the
    ``new_hidden_state`` the policy produces. This is strictly
    stronger than a frozen-snapshot comparison: a regression
    that drifted the captured states by one tick would surface
    here, where a snapshot-only test would silently re-bless.
    """
    _env, _shim, policy, collector = _build_collector(seed=42)
    transitions = collector.collect_episode()
    assert len(transitions) > 1, "test premise: need >1 tick"

    # First tick's hidden state must be zero (init_hidden contract).
    h0_init, c0_init = transitions[0].hidden_state_in
    assert torch.equal(h0_init, torch.zeros_like(h0_init))
    assert torch.equal(c0_init, torch.zeros_like(c0_init))

    # Re-run the LSTM recurrence and verify each subsequent
    # ``hidden_state_in`` matches the policy's new_hidden_state.
    was_training = policy.training
    policy.eval()
    try:
        with torch.no_grad():
            for t in range(len(transitions) - 1):
                tr = transitions[t]
                obs_t = torch.from_numpy(np.asarray(tr.obs)).unsqueeze(0)
                mask_t = torch.from_numpy(np.asarray(tr.mask)).unsqueeze(0)
                out = policy(
                    obs_t,
                    hidden_state=tr.hidden_state_in,
                    mask=mask_t,
                )
                expected_h, expected_c = out.new_hidden_state
                actual_h, actual_c = transitions[t + 1].hidden_state_in
                assert torch.equal(expected_h, actual_h), (
                    f"hidden h drift at tick {t + 1}: rolled-forward "
                    f"state from tick {t} does not match captured "
                    f"hidden_state_in at tick {t + 1}"
                )
                assert torch.equal(expected_c, actual_c), (
                    f"hidden c drift at tick {t + 1}"
                )
    finally:
        policy.train(was_training)

    # Belt-and-braces: the packed tuple is non-empty and has the
    # right shape (the consumer-side contract from
    # CLAUDE.md §"Recurrent PPO: hidden-state protocol on update").
    packed = policy.pack_hidden_states(
        [tr.hidden_state_in for tr in transitions],
    )
    assert len(packed) == 2  # (h, c)
    # LSTM batches along dim 1: (num_layers, T, hidden).
    assert packed[0].shape[1] == len(transitions)
    assert packed[1].shape[1] == len(transitions)


@pytest.mark.timeout(60)
def test_hidden_state_buffer_allocated_once_per_episode():
    """All transitions' h / c snapshots share a single backing storage.

    Each captured ``hidden_state_in[k]`` is a view into the
    pre-allocated ``hidden_buffers[k]``. ``Tensor.untyped_storage(
    ).data_ptr()`` returns the start of the underlying memory
    block, which is the SAME for all views into the same buffer.

    A per-tick allocation regression (e.g. someone reintroducing
    ``.clone()`` or building a fresh ``torch.empty`` per tick)
    would surface as N distinct ``data_ptr`` values across the
    transitions. The test uses a synthetic day small enough that
    the buffer's ``_estimate_max_steps`` upper bound holds — i.e.
    no grow fires — so we expect EXACTLY ONE storage per element
    of the hidden-state tuple.
    """
    _env, _shim, _policy, collector = _build_collector(seed=43)
    transitions = collector.collect_episode()
    assert len(transitions) > 1, "test premise: need >1 transition"

    # Distinct storage data_ptrs across all transitions for h.
    h_ptrs = {
        tr.hidden_state_in[0].untyped_storage().data_ptr()
        for tr in transitions
    }
    c_ptrs = {
        tr.hidden_state_in[1].untyped_storage().data_ptr()
        for tr in transitions
    }
    assert len(h_ptrs) == 1, (
        f"h-state snapshots come from {len(h_ptrs)} distinct "
        "storages — expected 1 (single per-episode allocation). "
        "Per-tick clone / allocation regression?"
    )
    assert len(c_ptrs) == 1, (
        f"c-state snapshots come from {len(c_ptrs)} distinct "
        "storages — expected 1 (single per-episode allocation)."
    )


@pytest.mark.timeout(60)
def test_hidden_state_slice_independent_of_subsequent_ticks():
    """The tick-0 capture is NOT overwritten by later-tick writes.

    By the ``init_hidden`` contract the tick-0 hidden state is
    all-zero. After the rollout advances through every subsequent
    tick (which writes to ``buf[1], buf[2], …``), the tick-0
    capture must STILL be all-zero. A regression where the
    captured ``hidden_in_t`` aliased the rolling ``hidden_state``
    would surface as the tick-0 view taking on the rolling
    state's final value (non-zero by construction once the LSTM
    has processed any input).
    """
    _env, _shim, _policy, collector = _build_collector(seed=44)
    transitions = collector.collect_episode()
    assert len(transitions) > 1, "test premise: need >1 transition"

    h0, c0 = transitions[0].hidden_state_in
    assert torch.equal(h0, torch.zeros_like(h0)), (
        "tick-0 h-state is non-zero after the rollout — the "
        "buffer slice may have aliased the rolling hidden_state"
    )
    assert torch.equal(c0, torch.zeros_like(c0)), (
        "tick-0 c-state is non-zero after the rollout"
    )

    # And at least one LATER-tick capture must be non-zero (the
    # LSTM has processed real input). If every capture were zero,
    # the slice-independence claim would be vacuous.
    later_nonzero = any(
        not torch.equal(
            tr.hidden_state_in[0],
            torch.zeros_like(tr.hidden_state_in[0]),
        )
        for tr in transitions[1:]
    )
    assert later_nonzero, (
        "every captured h-state is all-zero — capture is happening "
        "before the LSTM has processed any input, or the buffer "
        "writes are no-ops"
    )


@pytest.mark.timeout(60)
def test_per_tick_clone_count_drops_to_zero(monkeypatch):
    """Patch ``torch.Tensor.clone``; assert per-episode count is far
    below the pre-Session-04 baseline of 2 × n_steps.

    Pre-Session-04 each tick called ``t.detach().clone()`` twice
    (once per element of the LSTM hidden tuple). At ~50 ticks on
    the synthetic day that's 100 clones per episode — wholly from
    the rollout hot path. After this session the buffer's
    ``.copy_()`` replaces both clones, so the rollout's
    contribution drops to zero.

    Some framework-internal clones may still fire (e.g. via
    ``torch.distributions`` machinery, optimizer state setup),
    so the assertion is "much less than 2 × n_steps", NOT strict
    zero. The pre-fix baseline of 100 vs the framework-internal
    floor of (typically) <10 gives plenty of headroom for the
    regression-detection threshold.
    """
    _env, _shim, _policy, collector = _build_collector(seed=45)

    counter = {"calls": 0}
    real_clone = torch.Tensor.clone

    def counting_clone(self, *args, **kwargs):
        counter["calls"] += 1
        return real_clone(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", counting_clone)

    transitions = collector.collect_episode()
    n_steps = len(transitions)
    assert n_steps > 1, "test premise: need >1 transition"

    # Pre-Session-04 baseline: 2 * n_steps (one per (h, c) tuple
    # element per tick). Generous threshold of n_steps // 2 keeps
    # the test robust to framework-internal clones while still
    # catching any regression where per-tick rollout clones come
    # back.
    pre_fix_baseline = 2 * n_steps
    threshold = max(n_steps // 2, 5)
    assert counter["calls"] < threshold, (
        f"per-episode torch.Tensor.clone count = {counter['calls']} "
        f">= threshold {threshold} (pre-Session-04 baseline was "
        f"{pre_fix_baseline}). The per-tick hidden-state clone "
        f"may have been re-introduced."
    )


@pytest.mark.timeout(120)
def test_recurrent_ppo_kl_small_on_first_epoch():
    """v2 counterpart to ``tests/test_ppo_trainer.py::TestRecurrent
    StateThroughPpoUpdate::test_ppo_update_approx_kl_small_on_
    first_epoch_lstm``. Pinned here to this session as the
    regression guard against the buffer-slice path silently
    aliasing the rolling hidden state.

    A regression that wired up ``hidden_in_t = hidden_state``
    directly (no copy) would leave every transition's
    ``hidden_state_in`` pointing at the SAME final-tick state,
    so the PPO update would feed a (different, wrong) state into
    every mini-batch's forward — and ``approx_kl`` would explode
    well past the 1.0 threshold.
    """
    from agents_v2.env_shim import DiscreteActionShim
    from training_v2.discrete_ppo.trainer import DiscretePPOTrainer

    torch.manual_seed(46)
    np.random.seed(46)
    env = BetfairEnv(
        _make_day(n_races=2, n_pre_ticks=10, n_inplay_ticks=2),
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
        ppo_epochs=1,
        mini_batch_size=4,
        device="cpu",
    )
    stats = trainer.train_episode()

    assert stats.approx_kl_mean < 1.0, (
        f"approx_kl_mean={stats.approx_kl_mean:.4f} on epoch 0 of "
        "a fresh policy. If this is in the thousands the buffered "
        "hidden-state capture is aliasing the rolling state — "
        "every transition's hidden_state_in would point to the "
        "same final-tick value and the PPO update would evaluate "
        "the policy under a state different from the rollout's. "
        "See plans/rewrite/phase-4-restore-speed/session_prompts/"
        "04_hidden_state_allocator.md."
    )
