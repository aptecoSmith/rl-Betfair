"""Phase 4 Session 03 regression guards for the per-tick distribution
construction path in ``training_v2/discrete_ppo/rollout.py``.

The session prompt
(``plans/rewrite/phase-4-restore-speed/session_prompts/
03_distribution_objects.md``) specifies four load-bearing tests:

1. **Bit-identity vs the pre-Session-03 state** — at fixed RNG state,
   ``Beta(α, β, validate_args=True).sample()`` byte-equals
   ``Beta(α, β, validate_args=False).sample()`` for a sweep of α / β
   values. This is the strictest guard the session prompt called for
   (``Beta.sample`` is an RNG-consuming op; any reordering shows up
   immediately on a fixed seed). The same shape is checked on
   ``log_prob`` so the PPO update consumer's reconstruction stays
   bit-identical.

2. **``log_prob_stake`` bit-identity at PPO-update time** — the
   consumer side reconstructs the wrapper from ``stake_alpha`` /
   ``stake_beta`` stored on the transition. We verify the
   reconstruction produces the same numbers under the disabled-
   validation regime.

3. **Validation disabled globally after collector module import** —
   asserts ``torch.distributions.Distribution._validate_args`` is
   ``False`` after importing
   ``training_v2.discrete_ppo.rollout``. Catches a future regression
   where someone imports the collector but the toggle doesn't take
   effect (e.g. the toggle line gets deleted, or moved into a
   conditional that never fires).

4. **Full-episode stake_unit + log_prob_stake bit-identity across the
   toggle** — runs a small synthetic episode at fixed seed under
   each setting and asserts byte-equal ``[tr.stake_unit ...]`` and
   ``[tr.log_prob_stake ...]`` arrays. Belt-and-braces: even if
   ``Beta.sample``'s internal path were to skip a validation-time
   tensor allocation that consumed RNG, the full-episode signature
   would catch it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributions


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


# ── Cheap, no-scorer guards (always run) ──────────────────────────────


def test_distribution_validation_disabled_globally():
    """``Distribution._validate_args`` is ``False`` after the rollout
    module has been imported.

    Load-bearing pin against a regression where the toggle line
    (``set_default_validate_args(False)``) is deleted, moved into a
    conditional, or accidentally re-toggled by a downstream import.
    The session prompt's #3 deliverable.
    """
    # Importing the module should set the toggle. Even if a previous
    # test in the session has flipped it back on, importing again
    # re-runs the module-level statement (cached imports don't re-
    # execute, but the toggle is sticky once disabled and we restore
    # it at end of any test that flips it). To make the test robust
    # against test-order, force-toggle to True first then re-import.
    torch.distributions.Distribution.set_default_validate_args(True)
    assert torch.distributions.Distribution._validate_args is True

    import importlib

    import training_v2.discrete_ppo.rollout as rollout_mod
    importlib.reload(rollout_mod)

    assert torch.distributions.Distribution._validate_args is False, (
        "importing training_v2.discrete_ppo.rollout did not disable "
        "Distribution validation — Session 03 toggle missing or "
        "broken"
    )


@pytest.mark.parametrize(
    "alpha,beta",
    [
        (0.5, 0.5),
        (1.0, 1.0),
        (2.0, 5.0),
        (10.0, 1.0),
        (0.1, 100.0),
        (50.0, 50.0),
    ],
)
def test_beta_sample_bit_identical_across_validation_toggle(alpha, beta):
    """``Beta(α, β).sample()`` byte-equals across validate_args=True
    vs False at a fixed RNG state.

    The session prompt's #1 strictest guard, in unit-test form. We
    set the same ``torch.manual_seed`` before each branch and assert
    strict equality. Any RNG-consuming op gated on
    ``_validate_args`` would surface here on the very first sample.
    """
    a = torch.tensor([alpha], dtype=torch.float32)
    b = torch.tensor([beta], dtype=torch.float32)

    torch.manual_seed(42)
    sample_validated = torch.distributions.Beta(
        a, b, validate_args=True,
    ).sample()

    torch.manual_seed(42)
    sample_unvalidated = torch.distributions.Beta(
        a, b, validate_args=False,
    ).sample()

    assert torch.equal(sample_validated, sample_unvalidated), (
        f"Beta sample drift across validation toggle at α={alpha}, "
        f"β={beta}: validated={sample_validated.tolist()!r} vs "
        f"unvalidated={sample_unvalidated.tolist()!r}"
    )


@pytest.mark.parametrize(
    "alpha,beta",
    [
        (0.5, 0.5),
        (2.0, 5.0),
        (10.0, 1.0),
    ],
)
def test_beta_log_prob_bit_identical_across_validation_toggle(alpha, beta):
    """``Beta.log_prob`` byte-equals across the validation toggle.

    Session prompt's #2 deliverable. The PPO update side
    reconstructs ``Beta(stored_α, stored_β).log_prob(stored_sample)``
    once per mini-batch; if that reconstruction's numbers drift
    relative to the rollout-time log-probs, the policy ratio
    ``exp(new_lp - old_lp)`` is wrong on every update.
    """
    a = torch.tensor([alpha], dtype=torch.float32)
    b = torch.tensor([beta], dtype=torch.float32)

    # Sample once outside the toggle — we only need a value in (0, 1)
    # to evaluate log_prob on.
    torch.manual_seed(7)
    sample = torch.distributions.Beta(a, b).sample()
    # Sample is in (0, 1) so log_prob is well-defined under either
    # branch. (The validation branch would only reject a sample
    # outside the support; we don't trigger that here on purpose —
    # the bit-identity claim is for the happy path the rollout
    # actually exercises.)

    lp_validated = torch.distributions.Beta(
        a, b, validate_args=True,
    ).log_prob(sample)
    lp_unvalidated = torch.distributions.Beta(
        a, b, validate_args=False,
    ).log_prob(sample)

    assert torch.equal(lp_validated, lp_unvalidated), (
        f"Beta log_prob drift across validation toggle at α={alpha}, "
        f"β={beta}, x={sample.tolist()!r}: "
        f"validated={lp_validated.tolist()!r} vs "
        f"unvalidated={lp_unvalidated.tolist()!r}"
    )


# ── Full-episode integration tests (need scorer artefacts) ────────────


pytestmark = []  # set per-test below; the cheap tests above stay always-on


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
    from agents_v2.discrete_policy import DiscreteLSTMPolicy
    from agents_v2.env_shim import DiscreteActionShim
    from env.betfair_env import BetfairEnv
    from tests.test_betfair_env import _make_day
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


@pytest.mark.slow
@pytest.mark.skipif(not _runtime_ok, reason=_runtime_reason)
@pytest.mark.timeout(120)
def test_full_episode_stake_unit_and_log_prob_stake_unchanged_by_toggle():
    """Running a full episode under each setting at fixed seed
    produces byte-equal ``stake_unit`` and ``log_prob_stake``
    arrays.

    Belt-and-braces complement to the unit tests above — even if
    ``Beta.sample``'s internal path skips a validation-time tensor
    allocation that consumed RNG, the full-episode signature would
    catch it. The session prompt's #1 deliverable in
    end-to-end form.

    We run the rollout TWICE under matching seeds — once with the
    toggle disabled (the post-Session-03 default), once after
    re-enabling it — and assert strict equality on the per-tick
    stake_unit + log_prob_stake arrays.
    """
    # The toggle is set False by the rollout module's import. Force
    # the post-Session-03 production state in case a prior test in
    # the same process flipped it.
    import training_v2.discrete_ppo.rollout  # noqa: F401  (toggle SE)

    torch.distributions.Distribution.set_default_validate_args(False)
    assert torch.distributions.Distribution._validate_args is False
    from training_v2.discrete_ppo.transition import (
        rollout_batch_to_transitions,
    )
    _env, _shim, _policy, collector_off = _build_collector(seed=42)
    transitions_off = rollout_batch_to_transitions(
        collector_off.collect_episode()
    )

    # Re-enable validation and re-run with the same seed.
    torch.distributions.Distribution.set_default_validate_args(True)
    try:
        _env2, _shim2, _policy2, collector_on = _build_collector(seed=42)
        transitions_on = rollout_batch_to_transitions(
            collector_on.collect_episode()
        )
    finally:
        # Restore the post-Session-03 default so subsequent tests in
        # the session see the production state.
        torch.distributions.Distribution.set_default_validate_args(False)

    assert len(transitions_off) == len(transitions_on), (
        f"transition count drift: off={len(transitions_off)} vs "
        f"on={len(transitions_on)} — RNG sequence diverged"
    )
    assert len(transitions_off) > 0

    stake_off = np.array(
        [tr.stake_unit for tr in transitions_off], dtype=np.float64,
    )
    stake_on = np.array(
        [tr.stake_unit for tr in transitions_on], dtype=np.float64,
    )
    np.testing.assert_array_equal(
        stake_off, stake_on,
        err_msg="stake_unit drift across validation toggle on full "
        "episode — Beta.sample's RNG sequence was perturbed by the "
        "toggle, contradicting the bit-identity claim",
    )

    lp_off = np.array(
        [tr.log_prob_stake for tr in transitions_off], dtype=np.float64,
    )
    lp_on = np.array(
        [tr.log_prob_stake for tr in transitions_on], dtype=np.float64,
    )
    np.testing.assert_array_equal(
        lp_off, lp_on,
        err_msg="log_prob_stake drift across validation toggle on "
        "full episode",
    )
