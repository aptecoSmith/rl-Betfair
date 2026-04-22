"""
agents/architecture_registry.py — Architecture registry for RL policy networks.

All architectures are registered by name and selectable via ``config.yaml``.
This allows different agents in the population to use different architectures,
and new architectures to be added without touching training or evaluation code.

Usage::

    from agents.architecture_registry import create_policy, REGISTRY

    policy = create_policy("ppo_lstm_v1", obs_dim=1338, action_dim=42, hyperparams={})
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
        Dimension of the action vector (max_runners * ACTIONS_PER_RUNNER).
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


def infer_arch_hp_from_state_dict(
    name: str, state_dict: dict,
) -> dict:
    """Recover the architecture-shape hyperparams from a saved state dict.

    Background: ``ModelRecord.hyperparameters`` is mutable (e.g. via
    breeding/crossover that rewrites genes on a child without re-
    initialising weights). When that happens the record drifts out of
    sync with the weights on disk, and ``load_state_dict`` fails with
    size-mismatch or missing-key errors. This helper inspects the
    state-dict shapes directly to recover the architecture hyperparams
    the weights were trained under, so the caller can rebuild the policy
    with matching dimensions.

    The returned dict contains ONLY the hyperparams the shapes directly
    encode — callers should merge it over the record's hp (with this
    dict taking precedence) to keep non-shape genes like entropy
    coefficients untouched. Parameters the state dict can't disambiguate
    (e.g. ``transformer_heads`` — affects internal attention projection
    splits but shares the same weight-matrix shape as other head counts)
    are NOT returned; callers fall back to the record's value for those.

    Supports the three shipped architectures:
        - ``ppo_lstm_v1``: ``lstm_hidden_size``, ``lstm_num_layers``,
          ``mlp_hidden_size``
        - ``ppo_time_lstm_v1``: same keys; LSTM hidden inferred from
          ``time_lstm_cells.0.linear_ih.weight``
        - ``ppo_transformer_v1``: ``transformer_ctx_ticks``,
          ``transformer_depth``, ``lstm_hidden_size`` (= d_model),
          ``mlp_hidden_size``

    Raises ``KeyError`` if ``name`` isn't a known architecture. Returns
    an empty dict if the state dict is structurally unrecognisable
    (e.g. an unrelated checkpoint) — the caller's load attempt will
    then fail loudly and the model is genuinely corrupt.
    """
    if name not in REGISTRY:
        raise KeyError(f"Unknown architecture '{name}'")

    inferred: dict = {}

    # Common helpers: both LSTM-family and transformer share a
    # ``runner_encoder`` built via _build_mlp. The first layer's weight
    # tensor is ``runner_encoder.0.weight`` with shape [mlp_hidden,
    # RUNNER_INPUT_DIM]. If present, that pins mlp_hidden_size.
    enc0 = state_dict.get("runner_encoder.0.weight")
    if enc0 is not None and enc0.ndim == 2:
        inferred["mlp_hidden_size"] = int(enc0.shape[0])

    if name == "ppo_lstm_v1":
        # LSTM weight_ih_l0 is [4 * hidden_size, input_size]. Divide by
        # 4 to recover hidden.
        ih0 = state_dict.get("lstm.weight_ih_l0")
        if ih0 is not None and ih0.ndim == 2:
            inferred["lstm_hidden_size"] = int(ih0.shape[0]) // 4
        n_layers = 0
        while f"lstm.weight_ih_l{n_layers}" in state_dict:
            n_layers += 1
        if n_layers > 0:
            inferred["lstm_num_layers"] = n_layers

    elif name == "ppo_time_lstm_v1":
        # Custom TimeLSTM: per-layer ``linear_ih.weight`` shape
        # [4 * hidden, input]. Layer index starts at 0.
        ih0 = state_dict.get("time_lstm_cells.0.linear_ih.weight")
        if ih0 is not None and ih0.ndim == 2:
            inferred["lstm_hidden_size"] = int(ih0.shape[0]) // 4
        n_layers = 0
        while f"time_lstm_cells.{n_layers}.linear_ih.weight" in state_dict:
            n_layers += 1
        if n_layers > 0:
            inferred["lstm_num_layers"] = n_layers

    elif name == "ppo_transformer_v1":
        # Position embedding: [ctx_ticks, d_model]. Both dimensions
        # matter — d_model is aliased to ``lstm_hidden_size`` in the
        # transformer init so the rest of the code can stay generic.
        pe = state_dict.get("position_embedding.weight")
        if pe is not None and pe.ndim == 2:
            inferred["transformer_ctx_ticks"] = int(pe.shape[0])
            inferred["lstm_hidden_size"] = int(pe.shape[1])
        # Transformer depth = count of encoder layers by their
        # distinctive ``self_attn.in_proj_weight`` key.
        depth = 0
        while (
            f"transformer_encoder.layers.{depth}.self_attn.in_proj_weight"
            in state_dict
        ):
            depth += 1
        if depth > 0:
            inferred["transformer_depth"] = depth

    return inferred
