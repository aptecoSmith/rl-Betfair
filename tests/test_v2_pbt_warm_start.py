"""Forward-match gate for PBT warm-start weight inheritance.

``plans/pbt-breeding`` Step 1 (HC#5). The entire PBT promotion ladder
rests on one claim: a warm-started child loads the PARENT'S ACTUAL
TRAINED WEIGHTS, so its gen-0 forward reproduces the parent's final
forward on a fixed obs BEFORE any new gradient step. If that's false,
"inheritance" is theatre and selection stays ~half noise (the gene-only
GA's measured failure — purpose.md).

These tests exercise the EXACT code path ``train_one_agent(
init_weights_path=...)`` runs: ``worker.load_warm_start_weights``. They
build the policy the way the worker does (``input_norm=True`` so the
registered ``obs_mean`` / ``obs_std`` buffers are part of the round-trip
— the thing most likely to be silently dropped), perturb the parent to
simulate a trained agent, save through both the real ``ModelStore``
envelope and a bare ``state_dict``, then assert the loaded child's
forward is BIT-identical to the parent's.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agents_v2.action_space import DiscreteActionSpace
from agents_v2.discrete_policy import DiscreteLSTMPolicy
from env.betfair_env import ACTION_SCHEMA_VERSION, OBS_SCHEMA_VERSION
from registry.model_store import ModelStore
from training_v2.cohort.worker import load_warm_start_weights


# obs_dim deliberately < 256 so the direction head uses its documented
# test-mode runner-block fallback WITHOUT emitting the production-mismatch
# RuntimeWarning (discrete_policy.py _PROD_OBS_DIM_THRESHOLD). The
# forward-match is indifferent to obs semantics — parent and child use the
# identical deterministic fallback, so a faithful round-trip still matches.
OBS_DIM = 64
MAX_RUNNERS = 14


def _build(hidden_size: int = 64) -> DiscreteLSTMPolicy:
    """A policy built the way the cohort worker builds it (input_norm=True)."""
    space = DiscreteActionSpace(max_runners=MAX_RUNNERS)
    return DiscreteLSTMPolicy(
        obs_dim=OBS_DIM,
        action_space=space,
        hidden_size=hidden_size,
        input_norm=True,
    )


def _perturb_(policy: DiscreteLSTMPolicy, seed: int) -> None:
    """Overwrite every weight + the input-norm buffers — simulate a
    trained agent whose state is nowhere near the deterministic init."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in policy.parameters():
            p.copy_(torch.randn(p.shape, generator=g))
    # Non-default input_norm stats (a real agent's BC-fitted values).
    mean = torch.randn(policy.obs_dim, generator=g).numpy()
    std = (torch.rand(policy.obs_dim, generator=g) + 0.5).numpy()
    policy.set_input_norm_stats(mean, std)


def _forward(policy: DiscreteLSTMPolicy, obs: torch.Tensor, hidden):
    policy.eval()
    with torch.no_grad():
        return policy(obs, hidden_state=hidden)


def _fixed_obs(batch: int = 3) -> torch.Tensor:
    return torch.randn(
        batch, OBS_DIM, generator=torch.Generator().manual_seed(99),
    )


# Every always-present per-runner / scalar output tensor the forward
# produces. If warm-start drops ANY parameter or buffer, at least one of
# these diverges.
_COMPARE_FIELDS = (
    "logits",
    "masked_logits",
    "value_per_runner",
    "stake_alpha",
    "stake_beta",
    "fill_prob_per_runner",
    "mature_prob_per_runner",
    "predicted_locked_pnl_per_runner",
    "predicted_locked_log_var_per_runner",
    "direction_back_prob_per_runner",
    "direction_lay_prob_per_runner",
)


def _assert_forward_identical(a, b) -> None:
    for f in _COMPARE_FIELDS:
        ta, tb = getattr(a, f), getattr(b, f)
        assert torch.equal(ta, tb), f"{f} diverged after warm-start"


class TestWarmStartForwardMatch:
    """HC#5: child's gen-0 forward == parent's final forward, pre-training."""

    def test_child_reproduces_parent_forward_via_model_store_envelope(
        self, tmp_path,
    ):
        """The faithful path: save through ``ModelStore.save_weights``
        (the wrapped ``{"weights": ..., "obs_schema_version": ...}``
        envelope train_one_agent actually writes) and warm-start from it."""
        parent = _build()
        _perturb_(parent, seed=1)
        obs, hidden = _fixed_obs(), _build().init_hidden(3)
        parent_out = _forward(parent, obs, hidden)

        store = ModelStore(
            db_path=str(tmp_path / "reg.db"),
            weights_dir=str(tmp_path / "weights"),
        )
        mid = store.create_model(
            generation=0,
            architecture_name="v2_discrete_ppo_lstm_h64",
            architecture_description="warm-start gate parent",
            hyperparameters={},
        )
        wpath = store.save_weights(
            model_id=mid,
            state_dict=parent.state_dict(),
            obs_schema_version=OBS_SCHEMA_VERSION,
            action_schema_version=ACTION_SCHEMA_VERSION,
        )

        # Fresh child, DIFFERENT random init.
        child = _build()
        _perturb_(child, seed=2)
        child_before = _forward(child, obs, hidden)
        # Sanity: pre-load the child genuinely differs — proves this test
        # can detect a no-op load (a load that does nothing would still
        # "pass" the post-load equality if the child happened to match).
        assert not torch.equal(child_before.logits, parent_out.logits)

        # Warm-start through THE production helper.
        load_warm_start_weights(child, wpath)
        child_after = _forward(child, obs, hidden)

        _assert_forward_identical(child_after, parent_out)

    def test_child_reproduces_parent_forward_via_bare_state_dict(
        self, tmp_path,
    ):
        """The helper also tolerates a bare ``state_dict`` (no envelope)."""
        parent = _build()
        _perturb_(parent, seed=7)
        obs, hidden = _fixed_obs(), _build().init_hidden(3)
        parent_out = _forward(parent, obs, hidden)

        path = tmp_path / "bare.pt"
        torch.save(parent.state_dict(), str(path))

        child = _build()
        _perturb_(child, seed=8)
        load_warm_start_weights(child, path)
        _assert_forward_identical(_forward(child, obs, hidden), parent_out)

    def test_input_norm_buffers_are_inherited(self, tmp_path):
        """The registered ``obs_mean`` / ``obs_std`` buffers — not just the
        nn.Parameters — must transfer, or the normalised input (and every
        downstream logit) differs. Explicit buffer-equality guard."""
        parent = _build()
        _perturb_(parent, seed=11)
        path = tmp_path / "p.pt"
        torch.save(parent.state_dict(), str(path))

        child = _build()
        _perturb_(child, seed=12)
        # Pre-load the buffers genuinely differ.
        assert not torch.equal(child.obs_mean, parent.obs_mean)
        assert not torch.equal(child.obs_std, parent.obs_std)

        load_warm_start_weights(child, path)
        assert torch.equal(child.obs_mean, parent.obs_mean)
        assert torch.equal(child.obs_std, parent.obs_std)


class TestWarmStartContract:
    """The strict-load guard (HC#10) and basic input validation."""

    def test_structural_mismatch_raises(self, tmp_path):
        """A child whose structural genes differ from the parent (here a
        different ``hidden_size``) CANNOT load the inherited weights — the
        shapes differ. Strict load must raise LOUD, not silently truncate.
        This is the guarantee the breed step's structural-gene freeze
        relies on (HC#10)."""
        parent = _build(hidden_size=64)
        _perturb_(parent, seed=3)
        path = tmp_path / "h64.pt"
        torch.save(parent.state_dict(), str(path))

        child = _build(hidden_size=128)  # structural mismatch
        with pytest.raises(RuntimeError):
            load_warm_start_weights(child, path)

    def test_missing_path_raises_filenotfound(self, tmp_path):
        child = _build()
        with pytest.raises(FileNotFoundError):
            load_warm_start_weights(child, tmp_path / "nope.pt")
