"""Checkpoint round-trip tests for PPOTrainer controller state.

Exercises :meth:`agents.ppo_trainer.PPOTrainer.save_checkpoint` /
:meth:`load_checkpoint`. Scoped to the target-entropy controller
state (``_log_alpha``, ``_alpha_optimizer``) added in
``plans/entropy-control-v2/`` Session 01. Policy weights are saved
separately via ``policy.state_dict()`` and are not in scope here.
"""

from __future__ import annotations

import logging
import math

import pytest
import torch

from agents.ppo_trainer import PPOTrainer
from tests.test_ppo_trainer import _make_config, _make_policy


class TestCheckpointRoundtrip:
    def test_checkpoint_roundtrip_preserves_log_alpha(self):
        """Save, load, assert ``log_alpha`` and ``alpha_optim_state``
        survive the round trip to float epsilon."""
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={
                "entropy_coefficient": 0.01,
                "target_entropy": 100.0,
                "alpha_lr": 1e-3,
            },
        )

        # Drive the controller through a couple of steps so Adam
        # momentum and log_alpha differ from their fresh-init values.
        trainer._update_entropy_coefficient(current_entropy=200.0)
        trainer._update_entropy_coefficient(current_entropy=180.0)

        checkpoint = trainer.save_checkpoint()
        saved_log_alpha = float(trainer._log_alpha.item())
        saved_alpha = float(trainer._log_alpha.exp().item())

        # Fresh trainer — a different init, different optimiser state.
        policy2 = _make_policy(config)
        trainer2 = PPOTrainer(
            policy2, config,
            hyperparams={
                "entropy_coefficient": 0.005,  # different init
                "target_entropy": 100.0,
                "alpha_lr": 1e-3,
            },
        )
        assert trainer2._log_alpha.item() != pytest.approx(
            saved_log_alpha, abs=1e-6,
        ), "test setup: trainer2 should start from a different alpha"

        trainer2.load_checkpoint(checkpoint)

        assert trainer2._log_alpha.item() == pytest.approx(
            saved_log_alpha, abs=1e-7,
        )
        assert trainer2.entropy_coeff == pytest.approx(
            saved_alpha, rel=1e-7,
        )

        # Alpha optimiser state round-trips — Adam's step counter and
        # momentum buffers carry over.
        restored_state = trainer2._alpha_optimizer.state_dict()
        assert restored_state == checkpoint["alpha_optim_state"]

    def test_checkpoint_backward_compat_missing_log_alpha(self, caplog):
        """Loading a checkpoint without ``log_alpha`` /
        ``alpha_optim_state`` fresh-inits the controller from the
        current state and logs a warning — the registry-reset
        fallback for pre-controller checkpoints."""
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={"entropy_coefficient": 0.005},
        )
        initial_log_alpha = float(trainer._log_alpha.item())

        with caplog.at_level(logging.WARNING, logger="agents.ppo_trainer"):
            trainer.load_checkpoint({})  # no keys at all

        # Controller stays at its fresh-init value; no crash.
        assert trainer._log_alpha.item() == pytest.approx(
            initial_log_alpha, abs=1e-7,
        )
        assert trainer.entropy_coeff == pytest.approx(
            math.exp(initial_log_alpha), rel=1e-7,
        )

        # Warning surfaced so the operator sees the fallback at load.
        assert any(
            "log_alpha" in rec.message.lower()
            for rec in caplog.records
            if rec.levelno >= logging.WARNING
        ), "expected a warning about missing log_alpha in the log"

    def test_checkpoint_save_schema(self):
        """Schema (entropy-control-v2 §11): the checkpoint dict carries
        ``log_alpha`` as a float and ``alpha_optim_state`` as a dict
        — no extra keys, no missing ones."""
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(policy, config)

        checkpoint = trainer.save_checkpoint()
        assert "log_alpha" in checkpoint
        assert "alpha_optim_state" in checkpoint
        assert isinstance(checkpoint["log_alpha"], float)
        assert isinstance(checkpoint["alpha_optim_state"], dict)
