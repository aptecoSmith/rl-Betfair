"""
agents/architecture_registry.py — Architecture registry for RL policy networks.

All architectures are registered by name and selectable via ``config.yaml``.
This allows different agents in the population to use different architectures,
and new architectures to be added without touching training or evaluation code.

Usage::

    from agents.architecture_registry import create_policy, REGISTRY

    policy = create_policy("ppo_lstm_v1", obs_dim=1338, action_dim=28, hyperparams={})
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.policy_network import BasePolicy

# Architecture registry — maps name → class.
# Populated by register_architecture() calls at module level.
REGISTRY: dict[str, type[BasePolicy]] = {}


def register_architecture(cls: type[BasePolicy]) -> type[BasePolicy]:
    """Decorator that registers a policy class by its ``architecture_name``."""
    name = cls.architecture_name
    if name in REGISTRY:
        raise ValueError(f"Architecture '{name}' already registered")
    REGISTRY[name] = cls
    return cls


def create_policy(
    name: str,
    obs_dim: int,
    action_dim: int,
    max_runners: int,
    hyperparams: dict | None = None,
) -> BasePolicy:
    """Instantiate a policy network by architecture name.

    Parameters
    ----------
    name:
        Key in ``REGISTRY`` (e.g. ``"ppo_lstm_v1"``).
    obs_dim:
        Dimension of the flat observation vector.
    action_dim:
        Dimension of the action vector (max_runners * 2).
    max_runners:
        Maximum number of runners the env pads to.
    hyperparams:
        Architecture-specific hyperparameters (hidden sizes, etc.).
    """
    if name not in REGISTRY:
        available = ", ".join(sorted(REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown architecture '{name}'. Available: {available}"
        )
    cls = REGISTRY[name]
    return cls(
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_runners=max_runners,
        hyperparams=hyperparams or {},
    )
