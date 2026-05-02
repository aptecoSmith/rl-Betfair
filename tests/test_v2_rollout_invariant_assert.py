"""Phase 4 Session 05 regression guards for the sampled / strict
per-tick attribution-invariant assert in
``training_v2/discrete_ppo/rollout.py``.

The session prompt
(``plans/rewrite/phase-4-restore-speed/session_prompts/
05_invariant_assert.md``) specifies six load-bearing tests:

1. **Strict mode fires per tick** — with ``PHASE4_STRICT_ATTRIBUTION=1``
   (the test-suite default; cf. root ``conftest.py``) the per-tick
   ``np.isclose`` call count equals the rollout's transition count.
   Catches a regression where strict mode silently drops to sampled.

2. **Sampled mode fires at most once per N + on every settle tick** —
   with strict OFF the call count is ``floor(n_steps / N) +
   n_settle_ticks`` (give or take one for the off-by-one at episode
   end). Catches a regression where settle-step always-check stops
   firing.

3. **Strict mode raises on injected drift** — the production path
   inside ``_attribute_step_reward`` is preserved: when the
   ``np.isclose`` check sees drift, the assert raises and the rollout
   terminates. Tested by patching ``np.isclose`` to return False on a
   chosen tick.

4. **Sampled mode raises on settle-step drift** — the always-check
   carve-out for settle-step ticks is non-negotiable per the session
   prompt's hard_constraints §2. Tested by patching ``np.isclose``
   to return False on every call: in sampled mode the only firing
   ticks on a short test episode are settle ticks, so the first raise
   IS the settle-step check.

5. **Sampled mode misses drift on non-sample non-settle tick** — the
   explicit trade-off the session prompt documents. With strict OFF
   and N=100 (default), an injected drift on tick 1 is not caught;
   the rollout completes without raising. Catches a regression where
   sampled mode silently checks every tick.

6. **Attribution outputs unchanged across modes** — the same fixed-
   seed rollout in strict and sampled mode produces byte-equal
   per-tick ``per_runner_reward`` arrays. Catches a future regression
   where the assert path mutates state (it shouldn't — the assert is
   a side-effect-free read of ``per_runner.sum()``).

Note on the test-suite default: the root ``conftest.py``
unconditionally sets ``os.environ.setdefault("PHASE4_STRICT_ATTRIBUTION",
"1")`` BEFORE pytest imports any test module. So the
``_STRICT_ATTRIBUTION`` module attribute on ``rollout`` is True by
default in this test process. Tests that need to exercise sampled-
mode behaviour monkeypatch the attribute back to False for the
duration of the test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable
from unittest import mock

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


def _build_collector(
    seed: int = 42, n_races: int = 2, n_pre_ticks: int = 10,
):
    """Build a fresh shim + policy + collector on a tiny synthetic day."""
    from agents_v2.env_shim import DiscreteActionShim
    from training_v2.discrete_ppo.rollout import RolloutCollector

    torch.manual_seed(seed)
    np.random.seed(seed)
    env = BetfairEnv(
        _make_day(
            n_races=n_races,
            n_pre_ticks=n_pre_ticks,
            n_inplay_ticks=2,
        ),
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


def _make_isclose_spy() -> tuple[Callable, list]:
    """Wrap ``np.isclose`` in a spy that records every call.

    Returns ``(spy_fn, calls)`` — a callable suitable for
    ``mock.patch(..., side_effect=spy_fn)`` and the list it appends
    to. Each entry is the ``(args, kwargs)`` of a single call.
    Returns the real ``np.isclose`` result so the assert path
    behaves identically to production.
    """
    real = np.isclose
    calls: list[tuple[tuple, dict]] = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    return spy, calls


# ── Tests ─────────────────────────────────────────────────────────────


@pytest.mark.timeout(120)
def test_strict_mode_fires_per_tick(monkeypatch):
    """``np.isclose`` is called exactly once per env step in strict mode.

    The conftest default already sets strict mode for the test suite;
    we re-pin it here to be explicit so the test still passes if the
    conftest is restructured.
    """
    from training_v2.discrete_ppo import rollout as rollout_mod

    monkeypatch.setattr(rollout_mod, "_STRICT_ATTRIBUTION", True)

    _env, _shim, _policy, collector = _build_collector(seed=42)

    spy, calls = _make_isclose_spy()
    with mock.patch.object(rollout_mod.np, "isclose", side_effect=spy):
        transitions = collector.collect_episode()

    assert len(transitions) > 0
    assert len(calls) == len(transitions), (
        f"strict-mode np.isclose call count ({len(calls)}) should equal "
        f"transition count ({len(transitions)}); a smaller count means "
        "strict mode silently dropped to sampled"
    )


@pytest.mark.timeout(120)
def test_sampled_mode_fires_at_most_once_per_n_plus_settle_ticks(
    monkeypatch,
):
    """Sampled call count ≈ n_steps / N + n_settle_ticks.

    On the synthetic 2-race day each race has ~12 ticks plus a
    settle step, so n_steps ≈ 26 and n_settle ≈ 2. With the default
    ``N=100`` sample window no sample-rate firings happen on this
    short episode — every fire must be a settle tick. We use a smaller
    ``N=4`` to also exercise the sample-rate path so the test
    covers BOTH gating sources.
    """
    from training_v2.discrete_ppo import rollout as rollout_mod

    monkeypatch.setattr(rollout_mod, "_STRICT_ATTRIBUTION", False)
    monkeypatch.setattr(rollout_mod, "_SAMPLED_ATTRIBUTION_EVERY_N", 4)

    _env, _shim, _policy, collector = _build_collector(
        seed=42, n_races=2, n_pre_ticks=20,
    )

    spy, calls = _make_isclose_spy()
    with mock.patch.object(rollout_mod.np, "isclose", side_effect=spy):
        transitions = collector.collect_episode()

    n_steps = len(transitions)
    state = collector.last_attribution_state
    n_settle = len([0 for _ in range(state.settled_count)]) > 0
    # ``settled_count`` watermark equals the number of bets seen via
    # ``_settled_bets``; it doesn't directly tell us how many settle
    # *ticks* there were. Two races → exactly two settle ticks, by
    # construction of the synthetic day.
    n_settle_ticks = 2

    expected_max = (n_steps // 4) + n_settle_ticks + 2  # +2 slack
    expected_min = n_settle_ticks
    assert expected_min <= len(calls) <= expected_max, (
        f"sampled-mode np.isclose call count ({len(calls)}) outside "
        f"expected range [{expected_min}, {expected_max}] "
        f"(n_steps={n_steps}, N=4, n_settle_ticks={n_settle_ticks})"
    )
    # Sanity: must be strictly less than n_steps (otherwise sampled
    # mode is degenerating to strict).
    assert len(calls) < n_steps, (
        f"sampled-mode call count ({len(calls)}) must be strictly less "
        f"than n_steps ({n_steps}); got {len(calls)} which means "
        "sampled mode is silently strict"
    )


@pytest.mark.timeout(60)
def test_strict_mode_raises_on_injected_drift(monkeypatch):
    """A patched ``np.isclose=False`` triggers the assert in strict mode.

    Because strict fires the check on every tick, the very first call
    raises. Equivalent to "if drift exists, strict catches it on the
    tick it occurs."
    """
    from training_v2.discrete_ppo import rollout as rollout_mod

    monkeypatch.setattr(rollout_mod, "_STRICT_ATTRIBUTION", True)

    _env, _shim, _policy, collector = _build_collector(seed=42)

    with mock.patch.object(rollout_mod.np, "isclose", return_value=False):
        with pytest.raises(AssertionError, match="attribution drift"):
            collector.collect_episode()


@pytest.mark.timeout(60)
def test_sampled_mode_raises_on_settle_step_drift(monkeypatch):
    """Settle-tick always-check carve-out fires in sampled mode.

    With strict OFF and a long sample window (N=10_000) the only
    firing ticks on the synthetic 2-race day are the two settle
    ticks. Patching ``np.isclose=False`` then makes the assert raise
    on the first settle tick.
    """
    from training_v2.discrete_ppo import rollout as rollout_mod

    monkeypatch.setattr(rollout_mod, "_STRICT_ATTRIBUTION", False)
    monkeypatch.setattr(
        rollout_mod, "_SAMPLED_ATTRIBUTION_EVERY_N", 10_000,
    )

    _env, _shim, _policy, collector = _build_collector(seed=42)

    with mock.patch.object(rollout_mod.np, "isclose", return_value=False):
        with pytest.raises(AssertionError, match="attribution drift"):
            collector.collect_episode()


@pytest.mark.timeout(60)
def test_sampled_mode_misses_drift_on_non_sample_non_settle_tick(
    monkeypatch,
):
    """Sampled mode does NOT call np.isclose on non-firing ticks.

    The explicit trade-off documented in the session prompt: "an
    injected drift on a tick that's neither a sample nor a settle is
    not caught." We verify by injecting a sentinel side-effect on
    ``np.isclose`` that records the calls AND returns False, then
    only counting "drifts caught" by counting AssertionError raises
    on non-settle non-sample ticks.

    The cleanest construction: configure sampled mode with a long
    sample window so the only firings are settle ticks. Patch
    ``np.isclose`` to ALWAYS return False — so any firing tick
    raises. Then catch the resulting AssertionError, locate WHICH
    tick fired, and assert the tick was a settle tick (not a non-
    settle non-sample tick). The "miss" property is implied: had a
    non-settle non-sample tick fired the check, it would have raised
    earlier.

    A complementary, sharper formulation: count np.isclose calls in
    sampled mode with a very long sample window — they should all be
    settle ticks (the count equals n_settle_ticks, never more). This
    is a clean structural assertion.
    """
    from training_v2.discrete_ppo import rollout as rollout_mod

    monkeypatch.setattr(rollout_mod, "_STRICT_ATTRIBUTION", False)
    # Sample window larger than the episode so sample-rate firings
    # never happen — the only checks are settle ticks.
    monkeypatch.setattr(
        rollout_mod, "_SAMPLED_ATTRIBUTION_EVERY_N", 10_000,
    )

    _env, _shim, _policy, collector = _build_collector(seed=42)

    spy, calls = _make_isclose_spy()
    with mock.patch.object(rollout_mod.np, "isclose", side_effect=spy):
        transitions = collector.collect_episode()

    n_steps = len(transitions)
    n_settle_ticks = 2  # 2 races × 1 settle tick each
    assert len(calls) == n_settle_ticks, (
        f"sampled-mode (N=10_000) np.isclose call count "
        f"({len(calls)}) should equal n_settle_ticks "
        f"({n_settle_ticks}) since the sample window is larger than "
        f"the episode (n_steps={n_steps}); a higher count means a "
        "non-settle non-sample tick fired the check (regression: "
        "drift WOULD have been caught on a tick the session prompt "
        "documents as 'missed by design')"
    )


@pytest.mark.timeout(60)
def test_attribution_outputs_unchanged_across_modes(monkeypatch):
    """Per-tick ``per_runner_reward`` is byte-identical across modes.

    The Phase 4 Session 05 contract: only the assertion *frequency*
    changes; the per-tick output is the same. Run the same fixed-
    seed rollout in strict and sampled mode and assert byte-equal
    arrays. Catches a future regression where the assert path
    mutates state (it shouldn't — ``per_runner.sum()`` is a side-
    effect-free read).
    """
    from training_v2.discrete_ppo import rollout as rollout_mod

    # Run 1: strict mode.
    monkeypatch.setattr(rollout_mod, "_STRICT_ATTRIBUTION", True)
    _env, _shim, _policy, collector = _build_collector(seed=42)
    transitions_strict = collector.collect_episode()

    # Run 2: sampled mode (settle-only, since N is huge).
    monkeypatch.setattr(rollout_mod, "_STRICT_ATTRIBUTION", False)
    monkeypatch.setattr(
        rollout_mod, "_SAMPLED_ATTRIBUTION_EVERY_N", 10_000,
    )
    _env2, _shim2, _policy2, collector2 = _build_collector(seed=42)
    transitions_sampled = collector2.collect_episode()

    assert len(transitions_strict) == len(transitions_sampled), (
        f"transition count diverged across modes "
        f"(strict={len(transitions_strict)} vs "
        f"sampled={len(transitions_sampled)}) — the gating logic "
        "should not affect rollout length"
    )

    drift_ticks: list[int] = []
    for i, (a, b) in enumerate(
        zip(transitions_strict, transitions_sampled),
    ):
        if not np.array_equal(a.per_runner_reward, b.per_runner_reward):
            drift_ticks.append(i)

    if drift_ticks:
        i = drift_ticks[0]
        raise AssertionError(
            f"per_runner_reward diverged across modes on tick {i}: "
            f"strict={transitions_strict[i].per_runner_reward!r} vs "
            f"sampled={transitions_sampled[i].per_runner_reward!r} "
            f"(total drifting ticks: {len(drift_ticks)})"
        )
