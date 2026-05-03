"""Phase 4 Session 02 regression guards for the single-allocation
obs / mask rollout buffers in
``training_v2/discrete_ppo/rollout.py``.

The session prompt
(``plans/rewrite/phase-4-restore-speed/session_prompts/
02_obs_mask_double_copy.md``) specifies five load-bearing tests:

1. **Bit-identity vs the pre-Session-02 walk** — every transition's
   ``obs`` / ``mask`` byte-equals the array the legacy code would
   have produced (``np.asarray(obs, dtype=np.float32)`` /
   ``np.asarray(mask_np, dtype=bool)``). We verify by recording
   the obs / mask each tick via spy hooks on
   ``shim.step`` / ``shim.reset`` / ``shim.get_action_mask`` and
   comparing to ``transitions[i].obs`` / ``transitions[i].mask``
   element-wise.

2. **Obs buffer allocated once per episode** — patch
   ``training_v2.discrete_ppo.rollout.np.empty`` and assert it's
   called exactly twice per episode (once for obs, once for mask).
   Catches a future regression where someone re-introduces per-tick
   allocation.

3. **Mask buffer allocated once per episode** — same constraint
   covered by the np.empty count above; pinned as a separate test
   per the session prompt.

4. **Buffer grow path warns and continues** — patch
   ``_estimate_max_steps`` to return 1, run an episode, verify
   the warning fires and produces the same final transitions.

5. **Transition obs not aliased after buffer grow** — after a
   grow, the transitions stored from before the grow must still
   hold valid data (the grow re-allocates; views into the OLD
   buffer would risk pointing at freed memory if the old buffer
   were dropped before the transitions were built).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from env.betfair_env import BetfairEnv

from agents_v2.discrete_policy import DiscreteLSTMPolicy
from training_v2.discrete_ppo.transition import (
    rollout_batch_to_transitions,
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
def test_obs_mask_buffers_bit_identical_to_pre_session_02_on_fixed_seed():
    """Every transition's obs / mask byte-equals the legacy
    ``np.asarray(obs, dtype=np.float32)`` / ``np.asarray(mask_np,
    dtype=bool)`` materialisation.

    We intercept the obs returned by ``shim.reset`` / ``shim.step``
    and the mask returned by ``shim.get_action_mask`` per tick. Each
    transition must store byte-equal data to those captures —
    confirming the contiguous-buffer write produces the same bytes
    the per-tick ``np.asarray`` cast would have.
    """
    _env, shim, _policy, collector = _build_collector(seed=42)

    captured_obs: list[np.ndarray] = []
    captured_masks: list[np.ndarray] = []

    real_reset = shim.reset
    real_step = shim.step
    real_mask = shim.get_action_mask

    # Tick-by-tick capture: in the rollout loop
    # ``mask_np = shim.get_action_mask()`` runs FIRST in the tick,
    # then the env steps. So obs[t] is the pre-step obs and mask[t]
    # is the pre-step mask. Initial obs comes from reset(); after
    # each step the new obs becomes the next tick's obs.
    pending_obs: list[np.ndarray] = []

    def reset_spy(*args, **kwargs):
        obs, info = real_reset(*args, **kwargs)
        # Copy so a later in-place mutation by the env can't
        # retroactively change what we observed.
        pending_obs.append(np.array(obs, dtype=np.float32, copy=True))
        return obs, info

    def step_spy(*args, **kwargs):
        next_obs, reward, terminated, truncated, info = real_step(
            *args, **kwargs,
        )
        if not (terminated or truncated):
            pending_obs.append(
                np.array(next_obs, dtype=np.float32, copy=True),
            )
        return next_obs, reward, terminated, truncated, info

    def mask_spy():
        m = real_mask()
        captured_masks.append(np.array(m, dtype=bool, copy=True))
        return m

    shim.reset = reset_spy  # type: ignore[method-assign]
    shim.step = step_spy  # type: ignore[method-assign]
    shim.get_action_mask = mask_spy  # type: ignore[method-assign]
    try:
        transitions = rollout_batch_to_transitions(
            collector.collect_episode()
        )
    finally:
        shim.reset = real_reset  # type: ignore[method-assign]
        shim.step = real_step  # type: ignore[method-assign]
        shim.get_action_mask = real_mask  # type: ignore[method-assign]

    captured_obs = pending_obs
    assert len(transitions) > 0
    assert len(captured_obs) == len(transitions), (
        f"obs capture count {len(captured_obs)} != "
        f"transitions {len(transitions)}"
    )
    assert len(captured_masks) == len(transitions), (
        f"mask capture count {len(captured_masks)} != "
        f"transitions {len(transitions)}"
    )

    for i, tr in enumerate(transitions):
        np.testing.assert_array_equal(
            tr.obs, captured_obs[i],
            err_msg=f"obs drift at tick {i}",
        )
        np.testing.assert_array_equal(
            tr.mask, captured_masks[i],
            err_msg=f"mask drift at tick {i}",
        )
        assert tr.obs.dtype == np.float32
        assert tr.mask.dtype == np.bool_


@pytest.mark.timeout(60)
def test_obs_buffer_allocated_once_per_episode():
    """All transitions' obs share a single backing buffer.

    Checks ``tr.obs.base`` — a numpy view's ``.base`` is the
    underlying ndarray it slices into. If the obs buffer is
    allocated ONCE per episode, every transition's obs slice
    has the SAME ``.base`` object. A per-tick ``np.empty`` /
    ``np.asarray(..., dtype=…)`` regression would produce a
    distinct ``.base`` (or ``None``) per transition.

    Counting ``np.empty`` calls directly is too coarse —
    numpy is a singleton, so monkeypatching ``np.empty``
    catches calls from the env, the scorer, the policy, etc.
    The shared-base check is precise and scoped exactly to
    the rollout's own buffer.
    """
    _env, _shim, _policy, collector = _build_collector(seed=43)
    transitions = rollout_batch_to_transitions(
        collector.collect_episode()
    )

    assert len(transitions) > 1, "test premise: need >1 transition"

    bases = {id(tr.obs.base) for tr in transitions}
    assert len(bases) == 1, (
        f"obs slices come from {len(bases)} distinct buffers — "
        "expected 1 (single per-episode allocation). Per-tick "
        "allocation regression?"
    )

    # The shared base must be a non-None ndarray — if the obs
    # were a fresh allocation per tick (ndarray.base is None for
    # arrays that own their data), the set would have collapsed
    # to {id(None)} and been size 1, masking the bug. Pin the
    # positive check separately.
    base = transitions[0].obs.base
    assert isinstance(base, np.ndarray), (
        f"tr.obs.base is {type(base).__name__}, not ndarray — "
        "the obs is not a view into a shared buffer"
    )
    assert base.shape[1:] == transitions[0].obs.shape, (
        "shared buffer has wrong row shape"
    )


@pytest.mark.timeout(60)
def test_mask_buffer_allocated_once_per_episode():
    """All transitions' masks share a single backing buffer.

    Symmetric to the obs-buffer check via ``tr.mask.base``. The
    obs and mask buffers are allocated independently, so one
    check covers BOTH the "single allocation" property AND the
    fact that obs and mask share neither memory (they have
    different dtypes — float32 vs bool — so ``id(obs.base) ==
    id(mask.base)`` would be impossible by construction, but
    the per-buffer single-allocation property is what we care
    about).
    """
    _env, _shim, _policy, collector = _build_collector(seed=44)
    transitions = rollout_batch_to_transitions(
        collector.collect_episode()
    )

    assert len(transitions) > 1, "test premise: need >1 transition"

    bases = {id(tr.mask.base) for tr in transitions}
    assert len(bases) == 1, (
        f"mask slices come from {len(bases)} distinct buffers — "
        "expected 1 (single per-episode allocation)"
    )

    base = transitions[0].mask.base
    assert isinstance(base, np.ndarray), (
        f"tr.mask.base is {type(base).__name__}, not ndarray"
    )
    assert base.dtype == np.bool_


@pytest.mark.timeout(60)
def test_buffer_grow_path_warns_and_continues(monkeypatch, caplog):
    """Inject an undersized estimate; verify grow fires and the
    final transitions are still consistent.

    Patches ``RolloutCollector._estimate_max_steps`` to return 1
    so the very first tick after the initial fill triggers a
    grow. Multiple grows fire across the episode (1 → 2 → 4 →
    8 → … until the buffer accommodates all ticks). The collector
    must not raise; the resulting transitions must have the same
    count as a non-grow run (determinism on fixed seed) and the
    obs / mask shapes must still be sane.
    """
    import logging

    from training_v2.discrete_ppo.rollout import RolloutCollector

    # Reference run with the normal estimate.
    _env_ref, _shim_ref, _policy_ref, collector_ref = _build_collector(
        seed=45,
    )
    transitions_ref = rollout_batch_to_transitions(
        collector_ref.collect_episode()
    )

    # Forced-grow run.
    _env, _shim, _policy, collector = _build_collector(seed=45)
    monkeypatch.setattr(
        RolloutCollector, "_estimate_max_steps", lambda self, env: 1,
    )

    with caplog.at_level(logging.WARNING, logger="training_v2.discrete_ppo.rollout"):
        transitions = rollout_batch_to_transitions(
            collector.collect_episode()
        )

    assert len(transitions) == len(transitions_ref), (
        "grow path produced a different number of transitions "
        f"({len(transitions)} vs {len(transitions_ref)})"
    )
    assert len(transitions) > 1, (
        "test premise broken: the synthetic day must produce >1 "
        "tick for the grow path to fire"
    )

    grow_warnings = [
        rec for rec in caplog.records
        if "obs/mask buffer grow fired" in rec.getMessage()
    ]
    assert len(grow_warnings) >= 1, (
        "expected at least one grow-path warning; got "
        f"{[r.getMessage() for r in caplog.records]}"
    )

    # Per-transition sanity — obs and mask are still the right
    # shape and dtype after surviving the grows.
    obs_dim = collector.shim.obs_dim
    n_actions = collector.action_space.n
    for tr in transitions:
        assert tr.obs.shape == (obs_dim,)
        assert tr.obs.dtype == np.float32
        assert tr.mask.shape == (n_actions,)
        assert tr.mask.dtype == np.bool_
        assert tr.mask[0], "NOOP must always be legal"


@pytest.mark.timeout(60)
def test_transition_obs_not_aliased_after_buffer_grow(monkeypatch):
    """Transitions stored across a grow event hold distinct, valid data.

    Forces multiple grows by patching ``_estimate_max_steps`` to 1.
    After the rollout completes, distinct ticks' obs arrays must
    NOT all read the same row — a regression where the grow
    accidentally reused the same backing storage (or where the
    Transition stored a stale view into the OLD buffer that got
    overwritten) would surface as identical bytes across many
    transitions.

    The strict signature: at least two transitions whose obs
    differ. On a real rollout the obs evolves every tick (the
    runner LTPs change, the time-to-off ticks down, etc.), so a
    healthy run has many distinct obs rows.
    """
    from training_v2.discrete_ppo.rollout import RolloutCollector

    _env, _shim, _policy, collector = _build_collector(seed=46)
    monkeypatch.setattr(
        RolloutCollector, "_estimate_max_steps", lambda self, env: 1,
    )
    transitions = rollout_batch_to_transitions(
        collector.collect_episode()
    )

    assert len(transitions) > 2, (
        "test premise: need >2 ticks for a meaningful no-alias check"
    )

    # Every Transition's obs must be the right shape (no stale
    # view into a freed / re-shaped buffer would silently fail
    # this — np would error on mismatched stride access).
    obs_dim = collector.shim.obs_dim
    for i, tr in enumerate(transitions):
        assert tr.obs.shape == (obs_dim,), (
            f"obs at tick {i} has wrong shape {tr.obs.shape!r} — "
            "stale view into a pre-grow buffer?"
        )

    # Distinct ticks have distinct obs (the env evolves). A
    # regression where every transition aliased the SAME buffer
    # row would produce identical bytes across all transitions.
    distinct_pairs = 0
    for i in range(len(transitions) - 1):
        if not np.array_equal(transitions[i].obs, transitions[i + 1].obs):
            distinct_pairs += 1
    assert distinct_pairs >= 1, (
        "all transitions' obs are bytewise identical — "
        "grow path may have collapsed all rows to one slot"
    )

    # Stronger guard: an early-tick obs and a late-tick obs are
    # distinct. The early one would have been stored when the
    # buffer was tiny (size 1 or 2); the late one after several
    # grows. If the early view aliased the post-grow buffer it
    # would now read the late row's data → bytewise equal.
    assert not np.array_equal(transitions[0].obs, transitions[-1].obs), (
        "tick 0's obs equals the last tick's obs — early-tick "
        "view appears to have drifted onto a post-grow buffer row"
    )
