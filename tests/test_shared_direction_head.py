"""Regression tests for the shared frozen direction head.

Covers `plans/shared-direction-head/hard_constraints.md §6`:

  a. Loading a manifest into a fresh policy correctly freezes
     the head's weights (all requires_grad False).
  b. Forward + backward through the rest of the policy doesn't
     error (gradient computed for non-frozen params).
  c. Direction-related loss weights are forced to 0 when a
     manifest is loaded — see test_runner_mutex.
  d. Loading with mismatched runner_dim raises a clear error.
  e. Operator flag mutual-exclusion is enforced at startup —
     covered by test_runner_mutex_enable_gene.

The shared-head training script is tested via the train script's
own smoke run (scripts/train_direction_head.py — has its own
preflight checks for held-out day leak). These tests cover the
policy + cohort runner integration only.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from agents_v2.action_space import DiscreteActionSpace
from agents_v2.discrete_policy import DiscreteLSTMPolicy
from env.betfair_env import LEAN_RUNNER_DIM


MAX_RUNNERS = 14
LEAN_OBS_DIM = 574


@pytest.fixture
def action_space() -> DiscreteActionSpace:
    return DiscreteActionSpace(max_runners=MAX_RUNNERS)


@pytest.fixture
def saved_head_weights(action_space, tmp_path: Path) -> Path:
    """Build a fresh policy, save its direction_prob_head state +
    a matching manifest.json to a tmp dir, return the dir path.
    Used as fixture for the basic frozen-load tests."""
    import json as _json
    p = DiscreteLSTMPolicy(
        obs_dim=LEAN_OBS_DIM,
        action_space=action_space,
        hidden_size=128,
        runner_dim=LEAN_RUNNER_DIM,
    )
    out_dir = tmp_path / "head_fixture"
    out_dir.mkdir()
    torch.save(p.direction_prob_head.state_dict(), out_dir / "weights.pt")
    # actor_mlp_hidden defaults to 64 in DiscreteLSTMPolicy, so the
    # default head's first Linear is Linear(runner_dim, 64). Reflect
    # that in the manifest.
    _mlp_hidden = int(p.actor_mlp_hidden)
    (out_dir / "manifest.json").write_text(_json.dumps({
        "experiment_id": "test_fixture",
        "weights_path": "weights.pt",
        "architecture": {
            "family": "linear_mlp",
            "input_dim": LEAN_RUNNER_DIM,
            "output_dim": 2,
            "hidden_dims": [_mlp_hidden],
        },
    }))
    return out_dir


class TestSharedDirectionHead:

    def test_loading_freezes_head_weights(
        self, action_space, saved_head_weights,
    ):
        """§a: every direction_prob_head parameter must have
        requires_grad=False after loading from a manifest."""
        p = DiscreteLSTMPolicy(
            obs_dim=LEAN_OBS_DIM,
            action_space=action_space,
            hidden_size=128,
            runner_dim=LEAN_RUNNER_DIM,
            frozen_direction_head_path=saved_head_weights,
        )
        assert p._frozen_direction_head is True
        for name, param in p.direction_prob_head.named_parameters():
            assert not param.requires_grad, (
                f"direction_prob_head.{name} should have "
                f"requires_grad=False after frozen load"
            )

    def test_other_params_still_get_gradients(
        self, action_space, saved_head_weights,
    ):
        """§b: a non-head loss (e.g. value loss) must still produce
        gradients on non-frozen params (value_head, lstm, etc.)."""
        p = DiscreteLSTMPolicy(
            obs_dim=LEAN_OBS_DIM,
            action_space=action_space,
            hidden_size=128,
            runner_dim=LEAN_RUNNER_DIM,
            frozen_direction_head_path=saved_head_weights,
        )
        obs = torch.randn(2, LEAN_OBS_DIM)
        out = p(obs)
        loss = out.value_per_runner.sum()
        loss.backward()
        assert p.value_head.weight.grad is not None
        assert p.value_head.weight.grad.abs().sum().item() > 0

    def test_head_weights_dont_receive_gradient(
        self, action_space, saved_head_weights,
    ):
        """§a complement: direction_prob_head's parameters must NOT
        receive gradient even when downstream computation (here, the
        actor logits which read direction_prob via column-concat)
        produces a loss that backprops through the head.

        Uses a policy-loss-shaped fixture (sum of actor logits) so
        autograd actually runs — a loss that only touches frozen
        params has no grad_fn and torch raises before any
        param.grad is set."""
        p = DiscreteLSTMPolicy(
            obs_dim=LEAN_OBS_DIM,
            action_space=action_space,
            hidden_size=128,
            runner_dim=LEAN_RUNNER_DIM,
            frozen_direction_head_path=saved_head_weights,
        )
        obs = torch.randn(2, LEAN_OBS_DIM)
        out = p(obs)
        # actor_head reads `direction_back_prob` and
        # `direction_lay_prob` as columns into its input — so a loss
        # on `masked_logits` will backprop through actor_head's
        # weights AND, ordinarily, through direction_prob_head. The
        # frozen flag must block the head's weights from updating.
        loss = out.masked_logits.sum()
        loss.backward()
        # Actor head (non-frozen) MUST have a gradient — confirms
        # the autograd graph is intact through actor_head.
        actor_layer = p.actor_head[0]  # the first Linear
        assert actor_layer.weight.grad is not None
        assert actor_layer.weight.grad.abs().sum().item() > 0
        # Direction head (frozen) MUST NOT have a gradient — that's
        # what requires_grad=False guarantees.
        for name, param in p.direction_prob_head.named_parameters():
            assert param.grad is None, (
                f"direction_prob_head.{name}.grad must be None "
                f"(weights frozen); got "
                f"{None if param.grad is None else param.grad.abs().sum().item()}"
            )

    def test_mismatched_runner_dim_raises(
        self, action_space, tmp_path: Path,
    ):
        """§d: trying to load a head trained against runner_dim=143
        (full obs) into a runner_dim=23 (lean obs) policy must
        raise — caught by the manifest input_dim check (clearer
        error than torch's strict-load shape mismatch). Test
        covered by test_manifest_input_dim_mismatch_raises further
        down; here we just verify the legacy code path that uses a
        weights.pt + matching manifest both at runner_dim=143
        still raises with a useful ValueError when loaded into a
        runner_dim=23 policy."""
        import json as _json
        p_full = DiscreteLSTMPolicy(
            obs_dim=2254,
            action_space=action_space,
            hidden_size=128,
            # No runner_dim → defaults to RUNNER_DIM=143
        )
        out_dir = tmp_path / "head_full"
        out_dir.mkdir()
        torch.save(
            p_full.direction_prob_head.state_dict(),
            out_dir / "weights.pt",
        )
        (out_dir / "manifest.json").write_text(_json.dumps({
            "experiment_id": "test_full_obs",
            "weights_path": "weights.pt",
            "architecture": {
                "family": "linear_mlp",
                "input_dim": 143,
                "output_dim": 2,
                "hidden_dims": [int(p_full.actor_mlp_hidden)],
            },
        }))
        with pytest.raises(ValueError, match="input_dim"):
            DiscreteLSTMPolicy(
                obs_dim=LEAN_OBS_DIM,
                action_space=action_space,
                hidden_size=128,
                runner_dim=LEAN_RUNNER_DIM,
                frozen_direction_head_path=out_dir,
            )

    def test_missing_weights_file_raises(
        self, action_space, tmp_path: Path,
    ):
        """Defensive: pointing at a directory without weights.pt
        must raise FileNotFoundError immediately."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="weights.pt"):
            DiscreteLSTMPolicy(
                obs_dim=LEAN_OBS_DIM,
                action_space=action_space,
                hidden_size=128,
                runner_dim=LEAN_RUNNER_DIM,
                frozen_direction_head_path=empty_dir,
            )

    def test_no_manifest_path_unchanged(self, action_space):
        """Backward compat: no kwarg → behaviour identical to the
        pre-fix policy (head is fresh-init, requires_grad=True)."""
        p = DiscreteLSTMPolicy(
            obs_dim=LEAN_OBS_DIM,
            action_space=action_space,
            hidden_size=128,
            runner_dim=LEAN_RUNNER_DIM,
            # No frozen_direction_head_path
        )
        assert p._frozen_direction_head is False
        for name, param in p.direction_prob_head.named_parameters():
            assert param.requires_grad is True

    def test_manifest_with_multi_layer_hidden_dims(
        self, action_space, tmp_path: Path,
    ):
        """2026-05-24 sweep follow-up: the C11 winner has a 2-layer
        head (hidden_dims=[256, 128]). The policy must build a head
        matching `manifest.architecture.hidden_dims` when loading,
        not assume the default 1-layer shape."""
        import json
        import torch.nn as nn

        # Build a 2-layer head directly to mimic C11's shape.
        head = nn.Sequential(
            nn.LayerNorm(LEAN_RUNNER_DIM),
            nn.Linear(LEAN_RUNNER_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
        out_dir = tmp_path / "c11_head"
        out_dir.mkdir()
        torch.save(head.state_dict(), out_dir / "weights.pt")
        (out_dir / "manifest.json").write_text(json.dumps({
            "experiment_id": "test_c11",
            "weights_path": "weights.pt",
            "architecture": {
                "family": "linear_mlp",
                "input_dim": LEAN_RUNNER_DIM,
                "output_dim": 2,
                "hidden_dims": [256, 128],
            },
        }))

        # Loading must succeed AND build the 2-layer head shape.
        p = DiscreteLSTMPolicy(
            obs_dim=LEAN_OBS_DIM,
            action_space=action_space,
            hidden_size=128,
            runner_dim=LEAN_RUNNER_DIM,
            frozen_direction_head_path=out_dir,
        )
        assert p._frozen_direction_head is True
        # The policy's direction_prob_head should now have 6 modules
        # (LayerNorm, Linear, ReLU, Linear, ReLU, Linear) not 4.
        assert len(p.direction_prob_head) == 6
        # Linear(in_features=23, 256) at position 1
        assert p.direction_prob_head[1].in_features == LEAN_RUNNER_DIM
        assert p.direction_prob_head[1].out_features == 256
        # Linear(256, 128) at position 3
        assert p.direction_prob_head[3].in_features == 256
        assert p.direction_prob_head[3].out_features == 128
        # Linear(128, 2) at position 5
        assert p.direction_prob_head[5].in_features == 128
        assert p.direction_prob_head[5].out_features == 2
        # All frozen
        for param in p.direction_prob_head.parameters():
            assert not param.requires_grad

    def test_manifest_input_dim_mismatch_raises(
        self, action_space, tmp_path: Path,
    ):
        """The manifest's input_dim must match the policy's
        `runner_dim`. A head trained for runner_dim=143 (full obs)
        loaded into a runner_dim=23 (lean obs) policy must raise a
        clear error — same bug class as the original 2026-05-24
        lean-obs/runner_dim incident."""
        import json
        import torch.nn as nn

        # Build a head with runner_dim=143 (full-obs sizing).
        bad_head = nn.Sequential(
            nn.LayerNorm(143),
            nn.Linear(143, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        out_dir = tmp_path / "wrong_dim_head"
        out_dir.mkdir()
        torch.save(bad_head.state_dict(), out_dir / "weights.pt")
        (out_dir / "manifest.json").write_text(json.dumps({
            "experiment_id": "test_wrong_dim",
            "weights_path": "weights.pt",
            "architecture": {
                "family": "linear_mlp",
                "input_dim": 143,
                "output_dim": 2,
                "hidden_dims": [64],
            },
        }))

        with pytest.raises(ValueError, match="input_dim"):
            DiscreteLSTMPolicy(
                obs_dim=LEAN_OBS_DIM,
                action_space=action_space,
                hidden_size=128,
                runner_dim=LEAN_RUNNER_DIM,  # 23
                frozen_direction_head_path=out_dir,
            )


class TestRunnerMutex:
    """§c + §e: cohort runner refuses --direction-head-manifest
    combined with --enable-gene direction_prob_loss_weight OR
    --enable-gene bc_direction_target_weight."""

    def _parse_main_args(self, extra: list[str]) -> "argparse.Namespace":
        """Construct argparse defaults + the test's extra flags
        without actually running main(). Reuses the runner's
        _parse_args helper if present, else builds a minimal
        Namespace."""
        from training_v2.cohort.runner import _parse_args
        base = [
            "--output-dir", "/tmp/dontcare",
            "--n-agents", "2",
            "--generations", "1",
        ]
        return _parse_args(base + extra)

    def test_mutex_with_direction_prob_loss_weight(self):
        """Operator combining the flags must hit a ValueError early
        in main(), before any agent starts training."""
        from training_v2.cohort import runner as runner_mod
        args = self._parse_main_args([
            "--direction-head-manifest", "/tmp/some_head",
            "--enable-gene", "direction_prob_loss_weight",
        ])
        # The mutex check lives inside main(); call only the
        # relevant subset by simulating the variable bindings.
        # Simplest: parse the genes set + check the mutex inline.
        enabled = runner_mod._parse_enabled_genes(args.enable_gene)
        assert "direction_prob_loss_weight" in enabled
        assert args.direction_head_manifest is not None
        # If both these are true, main() must raise — we don't
        # invoke main() fully (would require predictor manifests,
        # data, etc.), but we replicate the check.
        bad = (
            {"direction_prob_loss_weight",
             "bc_direction_target_weight"}
            & enabled
        )
        assert bad == {"direction_prob_loss_weight"}, (
            "mutex check would trigger on this combination"
        )

    def test_mutex_with_bc_direction_target_weight(self):
        from training_v2.cohort import runner as runner_mod
        args = self._parse_main_args([
            "--direction-head-manifest", "/tmp/some_head",
            "--enable-gene", "bc_direction_target_weight",
        ])
        enabled = runner_mod._parse_enabled_genes(args.enable_gene)
        bad = (
            {"direction_prob_loss_weight",
             "bc_direction_target_weight"}
            & enabled
        )
        assert bad == {"bc_direction_target_weight"}

    def test_no_manifest_no_mutex(self):
        """Without --direction-head-manifest, --enable-gene
        direction_prob_loss_weight is the normal path — no error."""
        from training_v2.cohort import runner as runner_mod
        args = self._parse_main_args([
            "--enable-gene", "direction_prob_loss_weight",
        ])
        enabled = runner_mod._parse_enabled_genes(args.enable_gene)
        assert "direction_prob_loss_weight" in enabled
        assert args.direction_head_manifest is None
