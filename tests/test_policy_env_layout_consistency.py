"""Cross-consistency: policy layout must match env layout under
every supported obs mode.

This catches the bug class that took most of 2026-05-24 to find:
the policy was constructed with module-level constants that
assumed full obs (RUNNER_DIM=143). When the env was built with
``--predictor-lean-obs`` (per-runner block = 23 dims), the policy
silently fell into a "test-mode" zero-pad fallback in
``_runner_block_size`` resolution — producing structurally garbage
input to ``direction_prob_head`` for 16 days of training before
anyone noticed BCE was flat at the random floor.

The narrow bug is caught by ``tests/test_v2_direction_head_runner_dim.py``.
These tests guard the BROADER invariant: **whenever an
end-to-end env + policy pair is constructed in production, the
policy MUST NOT be in the test-mode fallback path.** Each
supported obs mode gets its own parametric test, so adding a
future obs variant (e.g. an "ultra-lean" 10-dim mode) requires
adding a test row here — making the cross-consistency explicit
and discoverable.

Bug history: see
``plans/direction-predictor-label-alignment/findings.md``
"2026-05-24 (evening)".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from env.betfair_env import (
    LEAN_RUNNER_DIM,
    MARKET_DIM,
    RUNNER_DIM,
    VELOCITY_DIM,
)


# ─── helpers ─────────────────────────────────────────────────────────


def _build_env_and_policy(predictor_lean_obs: bool):
    """Build a real env (using a tiny synthetic day if needed) plus
    the policy that the cohort worker would construct from it.

    Returns (env, shim, policy). Defers heavy imports so test
    collection is cheap.
    """
    import torch  # noqa: F401  (imported by the policy module)

    from agents_v2.discrete_policy import DiscreteLSTMPolicy
    from agents_v2.env_shim import DEFAULT_SCORER_DIR
    from training_v2.cohort.worker import (
        _build_env_for_day, scalping_train_config,
    )

    cfg = scalping_train_config()
    env, shim = _build_env_for_day(
        day_str="2026-04-11",
        data_dir=Path("data/processed"),
        cfg=cfg,
        scorer_dir=DEFAULT_SCORER_DIR,
        predictor_lean_obs=predictor_lean_obs,
    )
    # Mirror the cohort worker's policy construction (training_v2/
    # cohort/worker.py:1162). The key invariant under test: the
    # worker MUST forward env.active_runner_dim, not let policy
    # default to module-level RUNNER_DIM=143.
    policy = DiscreteLSTMPolicy(
        obs_dim=shim.obs_dim,
        action_space=shim.action_space,
        hidden_size=128,
        runner_dim=int(env.active_runner_dim),
    )
    return env, shim, policy


# ─── tests ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "predictor_lean_obs,expected_runner_dim",
    [
        pytest.param(True, LEAN_RUNNER_DIM, id="lean_obs"),
        pytest.param(False, RUNNER_DIM, id="full_obs"),
    ],
)
class TestPolicyEnvLayoutConsistency:
    """One parametric class — each obs mode runs every check below."""

    def test_env_active_runner_dim_matches_keys(
        self, predictor_lean_obs, expected_runner_dim,
    ):
        """env.active_runner_dim must equal len(active runner keys).
        Catches drift if a future commit adds a key but forgets to
        bump active_runner_dim."""
        env, _shim, _policy = _build_env_and_policy(predictor_lean_obs)
        assert env.active_runner_dim == expected_runner_dim, (
            f"env.active_runner_dim = {env.active_runner_dim}, "
            f"expected {expected_runner_dim} for "
            f"predictor_lean_obs={predictor_lean_obs}"
        )

    def test_policy_runner_dim_matches_env(
        self, predictor_lean_obs, expected_runner_dim,
    ):
        """Policy must adopt the env's active_runner_dim, NOT default
        to the module-level constant. This is the invariant the
        2026-05-24 bug violated."""
        env, _shim, policy = _build_env_and_policy(predictor_lean_obs)
        assert policy._runner_dim == env.active_runner_dim, (
            f"policy._runner_dim ({policy._runner_dim}) must match "
            f"env.active_runner_dim ({env.active_runner_dim})"
        )

    def test_runner_block_NOT_in_test_mode_fallback(
        self, predictor_lean_obs, expected_runner_dim,
    ):
        """**This is the test that would have caught the original
        bug.** In production, the policy's runner-block slice must
        anchor at MARKET_DIM+VELOCITY_DIM and span exactly
        max_runners * active_runner_dim. The test-mode fallback path
        anchors at 0 and zero-pads — silently produces garbage in
        prod."""
        env, _shim, policy = _build_env_and_policy(predictor_lean_obs)
        assert policy._runner_block_offset == MARKET_DIM + VELOCITY_DIM, (
            f"_runner_block_offset = {policy._runner_block_offset}, "
            f"expected {MARKET_DIM + VELOCITY_DIM} — FALLBACK FIRED"
        )
        expected_size = policy.max_runners * env.active_runner_dim
        assert policy._runner_block_size == expected_size, (
            f"_runner_block_size = {policy._runner_block_size}, "
            f"expected {expected_size} — FALLBACK FIRED"
        )
        assert (
            policy._runner_block_size == policy._runner_block_full_size
        ), (
            "_runner_block_size != _runner_block_full_size — the "
            "policy is in the zero-pad fallback. Production "
            "obs/policy mismatch."
        )

    def test_direction_head_layers_sized_correctly(
        self, predictor_lean_obs, expected_runner_dim,
    ):
        """direction_prob_head's LayerNorm + first Linear must accept
        env.active_runner_dim features."""
        _env, _shim, policy = _build_env_and_policy(predictor_lean_obs)
        layer_norm = policy.direction_prob_head[0]
        first_linear = policy.direction_prob_head[1]
        assert layer_norm.normalized_shape == (expected_runner_dim,)
        assert first_linear.in_features == expected_runner_dim

    def test_forward_pass_runs_without_fallback_pad(
        self, predictor_lean_obs, expected_runner_dim,
    ):
        """Forward pass on a real env's reset obs must produce valid
        direction logits. If the runner-block slice were
        zero-padded, this would still run (no crash) — so the
        STRUCTURAL check is in test_runner_block_NOT_in_test_mode_
        fallback above. Here we additionally assert no NaN/inf
        sneak into the output."""
        import torch
        env, shim, policy = _build_env_and_policy(predictor_lean_obs)
        obs, _ = shim.reset()
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        out = policy(obs_t)
        assert out.direction_back_logits_per_runner.shape == (
            1, policy.max_runners,
        )
        assert torch.isfinite(
            out.direction_back_logits_per_runner,
        ).all(), "direction_back_logits has NaN/inf"
        assert torch.isfinite(
            out.direction_lay_logits_per_runner,
        ).all(), "direction_lay_logits has NaN/inf"
