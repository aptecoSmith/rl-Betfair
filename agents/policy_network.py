"""
agents/policy_network.py — PPO + LSTM policy network (architecture v1).

Architecture (from PLAN.md)::

    Input: flat observation vector
      │
      ├── Runner feature encoder: per-runner MLP (shared weights)
      │     → permutation-invariant runner embeddings
      │
      ├── Market feature encoder: MLP (market + velocity + agent state)
      │     → market-level embedding
      │
      ├── Concatenate: [pooled_runners, market_emb, per_runner_embs...]
      │
      ├── LSTM — hidden state carries across ticks AND across races
      │
      ├── Actor head: per-runner (action_signal + stake_fraction)
      │
      └── Critic head: scalar V(s)

The LSTM hidden state persists across the entire day episode, allowing
the agent to carry context from earlier races into later ones.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from env.betfair_env import (
    AGENT_STATE_DIM,
    MARKET_DIM,
    POSITION_DIM,
    RUNNER_DIM,
    VELOCITY_DIM,
)

# ── Observation layout constants ────────────────────────────────────────────

MARKET_TOTAL_DIM = MARKET_DIM + VELOCITY_DIM + AGENT_STATE_DIM  # 37 + 11 + 6 = 54
RUNNER_INPUT_DIM = RUNNER_DIM + POSITION_DIM  # 110 + 3 = 113 (per-runner features + position)


# ── Checkpoint migration (scalping-active-management session 01) ──────────


def migrate_scalping_action_head(
    state_dict: dict,
    max_runners: int,
    old_per_runner: int,
    new_per_runner: int,
) -> dict:
    """Pad a pre-Session-01 scalping state_dict to the new per-runner dim.

    Scalping-active-management session 01 bumped
    :data:`~env.betfair_env.SCALPING_ACTIONS_PER_RUNNER` from 5 to 6 by
    appending a ``requote_signal`` dim. Old checkpoints remain loadable:
    the existing ``old_per_runner`` rows of the ``actor_head`` final
    linear layer, its bias, and the ``action_log_std`` parameter are
    preserved; the new row(s) are initialised fresh (small orthogonal
    for the weight, zeros for bias / log-std — matching the fresh-init
    rules in :class:`PPOLSTMPolicy._init_weights`).

    The function is shape-based and architecture-agnostic: it looks for
    parameter tensors whose output dim equals ``old_per_runner`` (or, for
    ``action_log_std``, ``max_runners * old_per_runner``) and widens them
    to the corresponding ``new_per_runner`` shape. Other tensors are
    returned unchanged, so running this against an already-migrated or
    non-scalping state_dict is a no-op.

    Parameters
    ----------
    state_dict:
        PyTorch state-dict mapping parameter names to tensors.
    max_runners:
        Number of runner slots the network was built for.
    old_per_runner:
        Per-runner action dim saved into ``state_dict`` (typically 5 for
        a pre-Session-01 scalping checkpoint).
    new_per_runner:
        Per-runner action dim the loading network expects (typically 6
        for a post-Session-01 scalping checkpoint).
    """
    if new_per_runner == old_per_runner:
        return dict(state_dict)
    if new_per_runner < old_per_runner:
        raise ValueError(
            f"Cannot shrink action head ({old_per_runner} → {new_per_runner})"
        )

    old_log_std_size = max_runners * old_per_runner
    new_log_std_size = max_runners * new_per_runner

    migrated: dict = {}
    for key, val in state_dict.items():
        if not isinstance(val, torch.Tensor):
            migrated[key] = val
            continue

        # ── Actor-head final linear weight: (old_per_runner, hidden) ──
        if (
            val.ndim == 2
            and val.shape[0] == old_per_runner
            and "actor_head" in key
            and key.endswith(".weight")
        ):
            new_weight = torch.empty(
                new_per_runner, val.shape[1], dtype=val.dtype,
            )
            nn.init.orthogonal_(new_weight, gain=0.01)
            new_weight[:old_per_runner] = val
            migrated[key] = new_weight
            continue

        # ── Actor-head final linear bias: (old_per_runner,) ──────────
        if (
            val.ndim == 1
            and val.shape[0] == old_per_runner
            and "actor_head" in key
            and key.endswith(".bias")
        ):
            new_bias = torch.zeros(new_per_runner, dtype=val.dtype)
            new_bias[:old_per_runner] = val
            migrated[key] = new_bias
            continue

        # ── Action log-std: flat (max_runners * old_per_runner,) ─────
        if (
            key == "action_log_std"
            and val.ndim == 1
            and val.shape[0] == old_log_std_size
        ):
            new_log_std = torch.zeros(new_log_std_size, dtype=val.dtype)
            new_log_std[:old_log_std_size] = val
            migrated[key] = new_log_std
            continue

        migrated[key] = val

    return migrated


# ── Checkpoint migration (scalping-close-signal session 01) ──────────────


def migrate_scalping_action_head_v3_to_v4(
    state_dict: dict, max_runners: int,
) -> dict:
    """Pad a v3 (post-requote) scalping state_dict for the v4 close_signal dim.

    Scalping-close-signal session 01 bumped
    :data:`~env.betfair_env.SCALPING_ACTIONS_PER_RUNNER` from 6 to 7 by
    appending a ``close_signal`` dim. Old v3 checkpoints remain loadable:
    the actor head's final linear, its bias, and ``action_log_std`` are
    widened by one row per runner, zero-initialised so the unmigrated
    agent outputs ``close_signal = 0`` identically to pre-plan behaviour.

    Thin wrapper over :func:`migrate_scalping_action_head` — reuses the
    same shape-based widening logic with ``old_per_runner=6`` /
    ``new_per_runner=7`` so there's only one copy of the migration
    code to maintain.

    NOTE: the new row of ``actor_head``'s weight matrix is initialised
    via the same small-orthogonal as ``_init_weights``. Combined with a
    zero bias and zero log-std, the head's sampled ``close_signal``
    stays centered at 0 for the first rollout — i.e. it fires on some
    fraction of ticks due to exploration noise rather than being a
    hard silent channel. This matches the v1→v2 behaviour and keeps
    migration policy consistent across schema bumps.
    """
    return migrate_scalping_action_head(
        state_dict,
        max_runners=max_runners,
        old_per_runner=6,
        new_per_runner=7,
    )


# ── Checkpoint migration (scalping-active-management session 02) ──────────


def migrate_fill_prob_head(
    state_dict: dict, fresh_policy: nn.Module,
) -> dict:
    """Extend a pre-Session-02 ``state_dict`` with the fill-prob head weights.

    Scalping-active-management session 02 added a per-runner
    fill-probability head (``fill_prob_head.weight`` / ``.bias``) sharing
    the backbone with policy + value. Older checkpoints don't carry these
    parameters, so a strict ``load_state_dict`` would fail with a
    "Missing key(s)" error.

    This helper injects fresh weights from ``fresh_policy.state_dict()``
    for every ``fill_prob_head.*`` key not already present in the input
    dict. The caller then calls ``load_state_dict(..., strict=True)``
    normally. Chose this approach over the alternative (load with
    ``strict=False``) because strict=False silently swallows unrelated
    missing-key errors too — making the migration explicit keeps the
    audit trail.

    Parameters
    ----------
    state_dict:
        PyTorch state-dict mapping parameter names to tensors. Typically
        loaded from a pre-Session-02 checkpoint.
    fresh_policy:
        A freshly-constructed policy module with the new head in place.
        Its ``fill_prob_head.*`` parameters are used as the source of
        fresh weights for the missing keys.

    Returns
    -------
    dict
        A new dict: input contents plus ``fill_prob_head.*`` keys copied
        from ``fresh_policy.state_dict()`` for any that were missing.
    """
    migrated = dict(state_dict)
    fresh = fresh_policy.state_dict()
    for key, value in fresh.items():
        if key.startswith("fill_prob_head.") and key not in migrated:
            # ``.clone()`` so later mutations on the fresh policy don't
            # alias the migrated weights.
            migrated[key] = value.detach().clone()
    return migrated


# ── Checkpoint migration (scalping-active-management session 03) ──────────


#: Clamp bounds on the risk head's log-var output. Chosen so that
#: ``stddev = exp(0.5 * log_var)`` spans ~£0.02 to ~£7.39 (on £100
#: stakes), covering the realistic realised-locked-pnl stddev range
#: while keeping ``exp(log_var)`` numerically well-behaved inside the
#: Gaussian NLL. Bounds match master_todo.md's session-03 spec.
RISK_LOG_VAR_MIN: float = -8.0
RISK_LOG_VAR_MAX: float = 4.0


def migrate_risk_head(
    state_dict: dict, fresh_policy: nn.Module,
) -> dict:
    """Extend a pre-Session-03 ``state_dict`` with the risk-head weights.

    Scalping-active-management session 03 added a per-runner risk head
    (``risk_head.weight`` / ``.bias``) producing two outputs per runner
    (predicted mean + log-var of locked P&L). Older checkpoints don't
    carry these parameters, so strict ``load_state_dict`` would fail.

    Mirrors :func:`migrate_fill_prob_head` exactly, but keyed on
    ``"risk_head."``. Chose an explicit helper over ``strict=False``
    for the same reason as Session 02 — strict=False silently swallows
    unrelated missing-key errors too, and keeping the migration
    explicit preserves the audit trail.

    Parameters
    ----------
    state_dict:
        PyTorch state-dict mapping parameter names to tensors. Typically
        loaded from a pre-Session-03 checkpoint.
    fresh_policy:
        A freshly-constructed policy module with the new head in place.
        Its ``risk_head.*`` parameters are used as the source of fresh
        weights for the missing keys.

    Returns
    -------
    dict
        A new dict: input contents plus ``risk_head.*`` keys copied
        from ``fresh_policy.state_dict()`` for any that were missing.
    """
    migrated = dict(state_dict)
    fresh = fresh_policy.state_dict()
    for key, value in fresh.items():
        if key.startswith("risk_head.") and key not in migrated:
            # ``.clone()`` so later mutations on the fresh policy don't
            # alias the migrated weights.
            migrated[key] = value.detach().clone()
    return migrated


# ── Base class ──────────────────────────────────────────────────────────────


class BasePolicy(nn.Module, abc.ABC):
    """Interface that all policy architectures must implement.

    Subclasses must accept ``(obs_dim, action_dim, max_runners, hyperparams)``
    in their ``__init__`` and call ``super().__init__()``.

    Hidden-state protocol (Session 6): ``init_hidden`` returns a 2-tuple
    of tensors so that existing callers (``PPOTrainer`` in particular)
    can move them to device via ``h[0].to(device), h[1].to(device)``
    without knowing which architecture produced them. For recurrent
    architectures the tuple is ``(h, c)``; for the transformer it is
    ``(rolling_buffer, valid_count)``. The precise shapes are private
    to each architecture -- only the 2-tuple-of-tensors contract is
    shared.
    """

    architecture_name: str = ""
    description: str = ""
    # Per-architecture default learning rate
    # (plans/naked-clip-and-stability, Session 02, 2026-04-18).
    # Consulted by ``PPOTrainer.__init__`` when the hp dict omits
    # ``learning_rate``. Subclasses override to tune fresh-init LR to
    # their saturation profile (see ``PPOTransformerPolicy`` which
    # halves this to 1.5e-4). The GA still mutates LR around the
    # sampled gene value when ``learning_rate`` is present in hp.
    default_learning_rate: float = 3e-4

    @abc.abstractmethod
    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        signal_bias: float = 0.0,
    ) -> PolicyOutput: ...

    @abc.abstractmethod
    def init_hidden(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a zero-initialised hidden-state 2-tuple."""
        ...

    # -- Recurrent-PPO hidden-state batching (ppo-kl-fix, 2026-04-24) -----
    #
    # These two methods let ``PPOTrainer._ppo_update`` pass the
    # rollout-time hidden state back into the policy's forward pass
    # rather than evaluating it statelessly. Each architecture
    # knows which tensor dimension its hidden-state carries the
    # batch on (LSTM family: dim 1, transformer: dim 0), so the
    # trainer delegates the concat / slice to the policy subclass
    # rather than hard-coding axes.
    #
    # Default implementations below use dim 0 — correct for the
    # transformer; the LSTM subclasses override both methods.
    def pack_hidden_states(
        self,
        hiddens: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate a list of single-batch hidden states into one
        batched hidden state.

        ``hiddens[i]`` is the 2-tuple the policy's forward expects at
        batch=1. The returned tuple has batch=n, ready to feed back
        into ``forward(obs, hidden)`` where ``obs`` is shape
        ``(n, obs_dim)``.
        """
        h0 = torch.cat([h[0] for h in hiddens], dim=0)
        h1 = torch.cat([h[1] for h in hiddens], dim=0)
        return h0, h1

    def slice_hidden_states(
        self,
        hidden: tuple[torch.Tensor, torch.Tensor],
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice a batched hidden state by integer ``indices``.

        Used by ``PPOTrainer._ppo_update`` to draw the mini-batch's
        hidden states from the full-rollout packed tensor.
        """
        return hidden[0][indices], hidden[1][indices]


# ── Session 3 (arb-improvements) — signal-bias warmup helper ───────────────


def _apply_signal_bias(
    actor_out: torch.Tensor,
    signal_bias: float,
    per_runner_action_dim: int,
) -> torch.Tensor:
    """Add ``signal_bias`` to the per-runner ``signal`` head (index 0).

    ``actor_out`` has shape ``(batch, max_runners, per_runner_action_dim)``.
    Signal is head index 0, matching ``_HEAD_NAMES`` in
    ``agents/ppo_trainer.py``. When ``signal_bias == 0.0`` the tensor is
    returned untouched so the default training path is byte-identical.

    The bias is a *soft prior* on the Normal mean — decays linearly to
    zero by the warmup epoch; never a hard override
    (``hard_constraints.md §Stabilisation``).
    """
    if signal_bias == 0.0:
        return actor_out
    bias_vec = torch.zeros(
        per_runner_action_dim, device=actor_out.device, dtype=actor_out.dtype,
    )
    bias_vec[0] = signal_bias
    # Broadcasts across (batch, max_runners) and adds only to head 0.
    return actor_out + bias_vec


@dataclass
class PolicyOutput:
    """Structured output from a policy forward pass."""

    action_mean: torch.Tensor       # (batch, action_dim) — mean of action distribution
    action_log_std: torch.Tensor    # (batch, action_dim) — log std
    value: torch.Tensor             # (batch, 1) — state value estimate
    hidden_state: tuple[torch.Tensor, torch.Tensor]  # opaque 2-tuple (see BasePolicy)
    # Scalping-active-management session 02 — per-runner fill-probability
    # prediction, shape ``(batch, max_runners)``, every element in [0, 1].
    # The sigmoid is applied inside each architecture's forward pass so
    # ``PolicyOutput`` exposes probabilities directly: this matches the
    # "0.5 = unsure" default below, lets the ``Bet`` record carry a
    # human-readable number, and feeds cleanly into the trainer's masked
    # BCE loss (with an ε-clamp to avoid log(0) on extreme predictions).
    # BCEWithLogits would be marginally more numerically stable at the
    # extremes, but the 0.5-default contract made probabilities the cleaner
    # interface. Default is a same-shape ``0.5`` tensor so stub policies /
    # legacy tests constructing ``PolicyOutput`` with positional args keep
    # working (any consumer that reads the field sees "unsure").
    fill_prob_per_runner: torch.Tensor = field(
        default_factory=lambda: torch.full((1, 1), 0.5)
    )
    # mature-prob-head (2026-04-26) — per-runner probability that the
    # pair will resolve FAVOURABLY WITHOUT FORCE-CLOSE INTERVENTION.
    # Same shape ``(batch, max_runners)`` and same [0, 1] contract as
    # ``fill_prob_per_runner`` above; same "0.5 = unsure" default for
    # stub policies / legacy callers.
    #
    # Distinct from ``fill_prob_per_runner``: fill_prob's BCE label is
    # "≥2 legs in bm.bets by episode end" (which conflates natural
    # maturations, agent-closes, AND env force-closes); mature_prob's
    # BCE label is the strict "matured naturally OR closed by agent
    # signal" — force-closed pairs land in the negative class. The
    # diagnostic for the difference is in
    # plans/per-runner-credit/findings.md.
    mature_prob_per_runner: torch.Tensor = field(
        default_factory=lambda: torch.full((1, 1), 0.5)
    )
    # Scalping-active-management session 03 — per-runner risk head.
    # Two tensors shape ``(batch, max_runners)``: predicted mean of
    # locked_pnl and its (clamped) log-variance. The "stddev" exposed
    # to consumers (UI, parquet) is ``exp(0.5 * log_var)``; the
    # Gaussian NLL operates on log-var directly for numerical stability
    # (gradients stay bounded; ``exp(log_var)`` stays finite given the
    # clamp applied inside each architecture's ``forward``). Defaults
    # are zero-tensors shape ``(1, 1)`` — mean=0, log_var=0 → stddev=1,
    # matching the "unsure" prior used by the fill-prob head default.
    # Stub policies / legacy callers constructing ``PolicyOutput``
    # positionally continue to work with the default zero prior.
    predicted_locked_pnl_per_runner: torch.Tensor = field(
        default_factory=lambda: torch.zeros((1, 1))
    )
    predicted_locked_log_var_per_runner: torch.Tensor = field(
        default_factory=lambda: torch.zeros((1, 1))
    )


# ── Helper: build an MLP stack ──────────────────────────────────────────────


def _build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    n_layers: int,
    activation: type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    """Build a simple MLP: input → [hidden → activation] × n_layers → output."""
    layers: list[nn.Module] = []
    prev = input_dim
    for _ in range(n_layers):
        layers.append(nn.Linear(prev, hidden_dim))
        layers.append(activation())
        prev = hidden_dim
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


# ── PPO + LSTM v1 ───────────────────────────────────────────────────────────

# Import here to avoid circular import — registry needs BasePolicy defined first
from agents.architecture_registry import register_architecture  # noqa: E402


@register_architecture
class PPOLSTMPolicy(BasePolicy):
    """PPO + LSTM policy network (architecture v1).

    Per-runner features are encoded through a shared-weight MLP, producing
    a fixed-size embedding per runner.  Market features (including velocity
    and agent state) go through a separate MLP.  The concatenation of pooled
    runner context + market embedding feeds into an LSTM whose hidden state
    persists across the entire day episode.

    The actor head re-combines the LSTM output with each runner's embedding
    to produce per-runner action parameters.  The critic head maps the LSTM
    output to a scalar V(s).
    """

    architecture_name = "ppo_lstm_v1"
    description = (
        "PPO with LSTM sequence model. Per-runner shared MLP encoder, "
        "market MLP encoder, LSTM for temporal context across ticks and "
        "races within a day episode."
    )

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        max_runners: int,
        hyperparams: dict,
    ) -> None:
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_runners = max_runners

        # Hyperparameters with defaults
        lstm_hidden = hyperparams.get("lstm_hidden_size", 256)
        mlp_hidden = hyperparams.get("mlp_hidden_size", 128)
        mlp_layers = hyperparams.get("mlp_layers", 2)
        runner_embed_dim = mlp_hidden  # runner embedding matches MLP hidden

        # Session 5 — LSTM structural genes. Defaults match the
        # pre-Session-5 architecture so checkpoints saved without these
        # keys load identically.
        lstm_num_layers = int(hyperparams.get("lstm_num_layers", 1))
        lstm_dropout = float(hyperparams.get("lstm_dropout", 0.0))
        lstm_layer_norm = bool(hyperparams.get("lstm_layer_norm", False))

        self.lstm_hidden_size = lstm_hidden
        self.lstm_num_layers = lstm_num_layers
        self.lstm_dropout = lstm_dropout
        self.lstm_layer_norm_enabled = lstm_layer_norm
        self.runner_embed_dim = runner_embed_dim

        # ── Runner encoder (shared weights across all runners) ──────────
        self.runner_encoder = _build_mlp(
            input_dim=RUNNER_INPUT_DIM,
            hidden_dim=mlp_hidden,
            output_dim=runner_embed_dim,
            n_layers=mlp_layers,
        )

        # ── Market encoder ──────────────────────────────────────────────
        self.market_encoder = _build_mlp(
            input_dim=MARKET_TOTAL_DIM,
            hidden_dim=mlp_hidden,
            output_dim=mlp_hidden,
            n_layers=mlp_layers,
        )

        # ── LSTM ────────────────────────────────────────────────────────
        # Input: market_emb + mean-pooled runner_embs + max-pooled runner_embs
        lstm_input_dim = mlp_hidden + runner_embed_dim * 2
        # NOTE: PyTorch's nn.LSTM only applies ``dropout`` BETWEEN stacked
        # layers, and silently ignores it when ``num_layers == 1``. To
        # avoid the warning we only pass dropout in the stacked case.
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_num_layers,
            dropout=(lstm_dropout if lstm_num_layers > 1 else 0.0),
            batch_first=True,
        )

        # Layer norm is applied to the LSTM output (post-recurrence) — a
        # single fixed location chosen so the actor/critic heads always
        # see normalised activations regardless of stacking.
        self.lstm_output_norm: nn.Module = (
            nn.LayerNorm(lstm_hidden) if lstm_layer_norm else nn.Identity()
        )

        # ── Actor head (per-runner) ─────────────────────────────────────
        # For each runner: concat(runner_emb, lstm_output, fill_prob_i,
        # mature_prob_i) → action params. output_dim = action_dim //
        # max_runners (signal + stake + aggression).
        # fill-prob-in-actor (2026-04-26): actor_input_dim was bumped
        # by +1 to accept the per-runner ``fill_prob`` scalar.
        # mature-prob-head (2026-04-26): bumped a further +1 to accept
        # the per-runner ``mature_prob`` scalar — the strict
        # "naturally-matured-or-agent-closed" forecast that excludes
        # force-closed pairs, distinguishing "good open" from
        # "open that will need a bail-out". See
        # plans/per-runner-credit/findings.md.
        actor_input_dim = runner_embed_dim + lstm_hidden + 2
        self._per_runner_action_dim = action_dim // max_runners
        self.actor_head = _build_mlp(
            input_dim=actor_input_dim,
            hidden_dim=mlp_hidden,
            output_dim=self._per_runner_action_dim,
            n_layers=1,
        )

        # Learnable log-std for the action distribution (per action dim)
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))

        # ── Critic head (global) ────────────────────────────────────────
        self.critic_head = _build_mlp(
            input_dim=lstm_hidden,
            hidden_dim=mlp_hidden,
            output_dim=1,
            n_layers=1,
        )

        # ── Fill-probability head (scalping-active-management §02) ─────
        # Auxiliary supervised head — per-runner probability that a paired
        # passive will fill before race-off. Shares the LSTM backbone with
        # policy + value. fill-prob-in-actor (2026-04-26): the head's
        # output is ALSO fed into ``actor_head`` (concat into actor_input
        # alongside the runner embedding and LSTM context), so the
        # surrogate-loss gradient now flows back through this head in
        # addition to the BCE auxiliary. A single linear to logits,
        # sigmoid applied in ``forward``. Initialised with orthogonal
        # gain=0.01 so the pre-training output is ≈0.5 ("unsure") per
        # runner.
        self.fill_prob_head = nn.Linear(lstm_hidden, max_runners)

        # ── Mature-probability head (mature-prob-head, 2026-04-26) ────
        # Auxiliary supervised head — per-runner probability that the
        # pair will MATURE NATURALLY OR BE CLOSED BY AGENT SIGNAL
        # (force-closed pairs are the negative class). Same backbone,
        # same shape, same init pattern as ``fill_prob_head``; the BCE
        # label classifier in ``PPOTrainer._collect_rollout`` is what
        # makes the labels different. Output is fed into ``actor_head``
        # alongside ``fill_prob`` so the action distribution can
        # condition on a forecast that EXCLUDES force-close.
        self.mature_prob_head = nn.Linear(lstm_hidden, max_runners)

        # ── Risk head (scalping-active-management §03) ────────────────
        # Second auxiliary supervised head — per-runner predicted
        # locked-P&L distribution as (mean, log-var). Shares the same
        # LSTM backbone as the fill-prob head; conditions on state only
        # (hard_constraints.md §8). Output layout is ``(batch,
        # max_runners * 2)``: the forward pass reshapes and splits into
        # mean/log-var channels, clamps log-var to
        # [RISK_LOG_VAR_MIN, RISK_LOG_VAR_MAX] before publishing on
        # PolicyOutput so downstream consumers (UI, parquet, NLL) never
        # see an unsafe log-var.
        self.risk_head = nn.Linear(lstm_hidden, max_runners * 2)

        # Orthogonal init for better PPO training stability
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply orthogonal initialisation (standard for PPO)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2**0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Smaller init for action head (encourages exploration early on)
        for module in self.actor_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
        # Smaller init for critic output
        for module in self.critic_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
        # Session-02 fill-prob head: match actor-head small init so the
        # head outputs ≈0 logits → sigmoid ≈ 0.5 ("unsure") at init.
        nn.init.orthogonal_(self.fill_prob_head.weight, gain=0.01)
        nn.init.zeros_(self.fill_prob_head.bias)
        # mature-prob-head (2026-04-26): same small init as fill-prob
        # head. Pre-training output ≈ 0.5 ("unsure") so feeding the
        # untrained head into actor_head is benign — the actor sees a
        # near-constant column until BCE training pulls it apart.
        nn.init.orthogonal_(self.mature_prob_head.weight, gain=0.01)
        nn.init.zeros_(self.mature_prob_head.bias)
        # Session-03 risk head: small orthogonal weight + zero bias → at
        # init the raw mean output ≈ 0 and raw log-var ≈ 0 (i.e. stddev
        # ≈ £1 on a £100 budget — a reasonable "unsure" prior).
        nn.init.orthogonal_(self.risk_head.weight, gain=0.01)
        nn.init.zeros_(self.risk_head.bias)

    def _split_obs(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split flat observation into market features and per-runner features.

        Parameters
        ----------
        obs : (batch, obs_dim)

        Returns
        -------
        market_feats : (batch, MARKET_TOTAL_DIM)
            Market (31) + velocity (11) + agent state (6).
        runner_feats : (batch, max_runners, RUNNER_DIM + POSITION_DIM)
            Per-runner features + per-runner position, reshaped from flat vector.
        """
        # Layout: [market(31) | velocity(11) | runners(max_runners×110) | agent_state(6) | position(max_runners×3)]
        market = obs[:, :MARKET_DIM]
        velocity = obs[:, MARKET_DIM : MARKET_DIM + VELOCITY_DIM]
        runner_start = MARKET_DIM + VELOCITY_DIM
        runner_end = runner_start + self.max_runners * RUNNER_DIM
        runners_flat = obs[:, runner_start:runner_end]
        agent_state = obs[:, runner_end : runner_end + AGENT_STATE_DIM]
        position_start = runner_end + AGENT_STATE_DIM
        position_end = position_start + self.max_runners * POSITION_DIM
        position_flat = obs[:, position_start:position_end]

        market_feats = torch.cat([market, velocity, agent_state], dim=-1)
        runner_feats_raw = runners_flat.view(-1, self.max_runners, RUNNER_DIM)
        position_feats = position_flat.view(-1, self.max_runners, POSITION_DIM)
        runner_feats = torch.cat([runner_feats_raw, position_feats], dim=-1)
        return market_feats, runner_feats

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        signal_bias: float = 0.0,
    ) -> PolicyOutput:
        """Forward pass through the full network.

        Parameters
        ----------
        obs : (batch, obs_dim)  or  (batch, seq_len, obs_dim)
            If 3-D, processes the full sequence through the LSTM.
        hidden_state :
            ``(h, c)`` each of shape ``(1, batch, lstm_hidden_size)``.
            Pass ``None`` on the first tick of an episode (zeros used).
        signal_bias :
            Session 3 (arb-improvements) warmup bias added to the per-runner
            ``signal`` head mean only. Default ``0.0`` → byte-identical.
        """
        # Handle both 2-D (single timestep) and 3-D (sequence) inputs
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (batch, 1, obs_dim)
        batch, seq_len, _ = obs.shape

        # Default hidden state
        if hidden_state is None:
            hidden_state = self.init_hidden(batch)
            # Move to same device as obs
            hidden_state = (
                hidden_state[0].to(obs.device),
                hidden_state[1].to(obs.device),
            )

        # Process each timestep through encoders
        # Reshape to (batch*seq_len, obs_dim) for encoder passes
        obs_flat = obs.reshape(batch * seq_len, -1)
        market_feats, runner_feats = self._split_obs(obs_flat)

        # Market encoding: (batch*seq_len, mlp_hidden)
        market_emb = self.market_encoder(market_feats)

        # Runner encoding: shared weights across all runners
        # (batch*seq_len, max_runners, RUNNER_INPUT_DIM) → (batch*seq_len, max_runners, embed)
        b_s = runner_feats.shape[0]
        runners_flat = runner_feats.reshape(b_s * self.max_runners, RUNNER_INPUT_DIM)
        runner_embs = self.runner_encoder(runners_flat)
        runner_embs = runner_embs.view(b_s, self.max_runners, self.runner_embed_dim)

        # Pool runner embeddings for LSTM input (permutation-invariant summary)
        runner_mean = runner_embs.mean(dim=1)  # (batch*seq_len, embed)
        runner_max = runner_embs.max(dim=1).values  # (batch*seq_len, embed)

        # LSTM input: [market_emb, runner_mean_pool, runner_max_pool]
        lstm_input = torch.cat([market_emb, runner_mean, runner_max], dim=-1)
        lstm_input = lstm_input.view(batch, seq_len, -1)

        # LSTM forward
        lstm_out, new_hidden = self.lstm(lstm_input, hidden_state)
        # lstm_out: (batch, seq_len, lstm_hidden)
        lstm_out = self.lstm_output_norm(lstm_out)

        # Use last timestep for action/value heads
        lstm_last = lstm_out[:, -1, :]  # (batch, lstm_hidden)

        # ── Actor: per-runner action parameters ─────────────────────────
        # Get runner embeddings for the last timestep
        if seq_len > 1:
            # Re-extract runner features for last timestep only
            last_obs = obs[:, -1, :]  # (batch, obs_dim)
            _, last_runner_feats = self._split_obs(last_obs)
            last_runners_flat = last_runner_feats.reshape(
                batch * self.max_runners, RUNNER_INPUT_DIM
            )
            last_runner_embs = self.runner_encoder(last_runners_flat)
            last_runner_embs = last_runner_embs.view(
                batch, self.max_runners, self.runner_embed_dim
            )
        else:
            last_runner_embs = runner_embs.view(
                batch, self.max_runners, self.runner_embed_dim
            )

        # ── Fill-probability head (scalping-active-management §02) ────
        # Conditions on the shared backbone (``lstm_last``), NOT the
        # sampled action (hard_constraints.md §8). Sigmoid here so
        # ``PolicyOutput`` carries a probability in [0, 1].
        # fill-prob-in-actor (2026-04-26): computed BEFORE actor_head so
        # the per-runner scalar can be fed into actor_input. Surrogate
        # loss now flows back through this head.
        fill_prob_logits = self.fill_prob_head(lstm_last)
        fill_prob = torch.sigmoid(fill_prob_logits)  # (batch, max_runners)

        # ── Mature-probability head (mature-prob-head, 2026-04-26) ───
        # Conditions on the same backbone as ``fill_prob`` but trained
        # on a STRICTER label (force-closed pairs → 0). Computed BEFORE
        # actor_head so its per-runner scalar can be fed in alongside
        # fill_prob; the surrogate-loss gradient flows back through
        # this head identically to fill_prob (no detach).
        mature_prob_logits = self.mature_prob_head(lstm_last)
        mature_prob = torch.sigmoid(mature_prob_logits)  # (batch, max_runners)

        # Expand LSTM output to match each runner
        lstm_expanded = lstm_last.unsqueeze(1).expand(
            -1, self.max_runners, -1
        )  # (batch, max_runners, lstm_hidden)

        # Concat runner embedding with LSTM context and per-runner
        # fill_prob + mature_prob scalars (mature-prob-head, 2026-04-26).
        actor_input = torch.cat(
            [
                last_runner_embs,
                lstm_expanded,
                fill_prob.unsqueeze(-1),
                mature_prob.unsqueeze(-1),
            ],
            dim=-1,
        )  # (batch, max_runners, embed + lstm_hidden + 2)

        # Per-runner action params: (batch, max_runners, per_runner_action_dim)
        actor_out = self.actor_head(actor_input)

        # Session 3: optional warmup bias on the signal head (index 0).
        actor_out = _apply_signal_bias(
            actor_out, signal_bias, self._per_runner_action_dim,
        )

        # Flatten to action_dim: [signals..., stakes..., aggression...]
        # Each per-runner dim is scattered into its own contiguous block.
        parts = [actor_out[:, :, i] for i in range(self._per_runner_action_dim)]
        action_mean = torch.cat(parts, dim=-1)  # (batch, max_runners * per_runner_action_dim)

        # ── Critic: scalar V(s) ────────────────────────────────────────
        value = self.critic_head(lstm_last)  # (batch, 1)

        # ── Risk head (scalping-active-management §03) ────────────────
        # Two outputs per runner: mean + log-var. Clamp log-var here so
        # ``PolicyOutput`` consumers (UI, parquet, NLL) can trust the
        # bounds without knowing the clamp values. Per purpose.md §3,
        # clamping at the forward-pass boundary keeps ``exp(log_var)``
        # finite in the NLL regardless of what the raw head emits.
        risk_out = self.risk_head(lstm_last)
        risk_out = risk_out.view(batch, self.max_runners, 2)
        risk_mean = risk_out[..., 0]
        risk_log_var = risk_out[..., 1].clamp(
            RISK_LOG_VAR_MIN, RISK_LOG_VAR_MAX,
        )

        return PolicyOutput(
            action_mean=action_mean,
            action_log_std=self.action_log_std.expand(batch, -1),
            value=value,
            hidden_state=new_hidden,
            fill_prob_per_runner=fill_prob,
            mature_prob_per_runner=mature_prob,
            predicted_locked_pnl_per_runner=risk_mean,
            predicted_locked_log_var_per_runner=risk_log_var,
        )

    def init_hidden(
        self, batch_size: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-initialised LSTM hidden state ``(h_0, c_0)``.

        Shape ``(num_layers, batch, hidden)`` — matches ``nn.LSTM``'s
        expected hidden-state layout for stacked recurrence.
        """
        h = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size)
        c = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size)
        return h, c

    # -- ppo-kl-fix (2026-04-24) — LSTM-shaped hidden-state batching -----
    #
    # ``nn.LSTM`` lays the hidden state out as ``(num_layers, batch,
    # hidden)`` with the batch axis at dim 1 (not dim 0 as the base
    # protocol assumes). Override the pack / slice helpers so the
    # PPO update can batch per-transition hidden states without
    # mangling the layer axis.
    def pack_hidden_states(
        self,
        hiddens: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.cat([h[0] for h in hiddens], dim=1)
        h1 = torch.cat([h[1] for h in hiddens], dim=1)
        return h0, h1

    def slice_hidden_states(
        self,
        hidden: tuple[torch.Tensor, torch.Tensor],
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            hidden[0].index_select(1, indices),
            hidden[1].index_select(1, indices),
        )

    def get_action_distribution(
        self,
        obs: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        signal_bias: float = 0.0,
    ) -> tuple[Normal, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Return the action distribution, value, and new hidden state.

        Convenience method for PPO rollout collection.
        """
        out = self.forward(obs, hidden_state, signal_bias=signal_bias)
        std = out.action_log_std.exp()
        dist = Normal(out.action_mean, std)
        return dist, out.value, out.hidden_state


# ── Time-aware LSTM cell (Session 2.8) ─────────────────────────────────────


class TimeLSTMCell(nn.Module):
    """Custom LSTM cell where the forget gate incorporates a time delta.

    The forget gate is modified so that larger time gaps cause more
    forgetting of short-term state::

        f_t = sigmoid(W_f @ [h, x] + W_dt * delta_t + b_f)

    where ``delta_t`` is the wall-clock time since the previous tick
    (normalised).  A larger delta pushes the forget gate towards 1
    (more forgetting) via a learned positive weight ``W_dt``.

    This lets the LSTM distinguish "prices stable for 3 minutes"
    (high delta → forget more short-term memory) from "prices stable
    for 15 seconds" (low delta → retain short-term memory).
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Standard LSTM gates: input, forget, cell, output
        # All share a single linear layer for efficiency: [i, f, g, o]
        self.linear_ih = nn.Linear(input_size, 4 * hidden_size)
        self.linear_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

        # Time-decay weight for the forget gate (scalar per hidden unit)
        self.W_dt = nn.Parameter(torch.zeros(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        hc: tuple[torch.Tensor, torch.Tensor],
        time_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for one timestep.

        Parameters
        ----------
        x : (batch, input_size)
        hc : (h, c) each (batch, hidden_size)
        time_delta : (batch, 1) or (batch,)
            Normalised seconds since last tick.

        Returns
        -------
        (h_new, c_new) : each (batch, hidden_size)
        """
        h, c = hc
        gates = self.linear_ih(x) + self.linear_hh(h)  # (batch, 4*hidden)

        i, f, g, o = gates.chunk(4, dim=-1)

        # Inject time delta into forget gate
        if time_delta.dim() == 1:
            time_delta = time_delta.unsqueeze(-1)  # (batch, 1)
        f = f + self.W_dt * time_delta  # broadcast: (batch, hidden)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, c_new


# ── PPO + Time-LSTM v1 ─────────────────────────────────────────────────────


# Index of `seconds_since_last_tick` within the velocity section of the
# observation vector.  The velocity section starts after MARKET_DIM and
# the time delta features are at the end of MARKET_VELOCITY_KEYS.
_TIME_DELTA_VEL_INDEX: int = VELOCITY_DIM - 4  # index of seconds_since_last_tick in velocity vec


@register_architecture
class PPOTimeLSTMPolicy(BasePolicy):
    """PPO + Time-aware LSTM policy network (architecture v1t).

    Identical to :class:`PPOLSTMPolicy` except:

    * The standard ``nn.LSTM`` is replaced with :class:`TimeLSTMCell`,
      which modulates the forget gate by ``seconds_since_last_tick``.
    * The ``seconds_since_last_tick`` feature is extracted from the
      observation and fed to the cell at each timestep.

    The time delta is already part of the observation vector (in the
    velocity section), so it also flows through the market encoder.
    The TimeLSTMCell receives it *additionally* as a separate signal
    so it can directly modulate memory retention.
    """

    architecture_name = "ppo_time_lstm_v1"
    description = (
        "PPO with time-aware LSTM. Same structure as ppo_lstm_v1 but "
        "the LSTM forget gate incorporates seconds_since_last_tick so "
        "larger time gaps cause more forgetting of short-term state."
    )

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        max_runners: int,
        hyperparams: dict,
    ) -> None:
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_runners = max_runners

        lstm_hidden = hyperparams.get("lstm_hidden_size", 256)
        mlp_hidden = hyperparams.get("mlp_hidden_size", 128)
        mlp_layers = hyperparams.get("mlp_layers", 2)
        runner_embed_dim = mlp_hidden

        # Session 5 — LSTM structural genes. Defaults match the
        # pre-Session-5 architecture (single layer, no dropout, no
        # layer norm) so checkpoints saved without these keys load
        # identically.
        lstm_num_layers = int(hyperparams.get("lstm_num_layers", 1))
        lstm_dropout = float(hyperparams.get("lstm_dropout", 0.0))
        lstm_layer_norm = bool(hyperparams.get("lstm_layer_norm", False))

        self.lstm_hidden_size = lstm_hidden
        self.lstm_num_layers = lstm_num_layers
        self.lstm_dropout = lstm_dropout
        self.lstm_layer_norm_enabled = lstm_layer_norm
        self.runner_embed_dim = runner_embed_dim

        # Runner encoder (shared weights)
        self.runner_encoder = _build_mlp(
            input_dim=RUNNER_INPUT_DIM,
            hidden_dim=mlp_hidden,
            output_dim=runner_embed_dim,
            n_layers=mlp_layers,
        )

        # Market encoder
        self.market_encoder = _build_mlp(
            input_dim=MARKET_TOTAL_DIM,
            hidden_dim=mlp_hidden,
            output_dim=mlp_hidden,
            n_layers=mlp_layers,
        )

        # Time-aware LSTM cell(s) — stacked for num_layers > 1.
        # Layer 0 takes the fused market+runner input; subsequent layers
        # take the previous layer's hidden state as input. Dropout (when
        # enabled AND num_layers > 1) is applied to the hidden state
        # *between* layers via F.dropout during training only.
        lstm_input_dim = mlp_hidden + runner_embed_dim * 2
        self.time_lstm_cells = nn.ModuleList(
            [
                TimeLSTMCell(
                    input_size=lstm_input_dim if layer_idx == 0 else lstm_hidden,
                    hidden_size=lstm_hidden,
                )
                for layer_idx in range(lstm_num_layers)
            ]
        )

        # Optional layer norm applied to the top-layer hidden state,
        # mirroring the stock-nn.LSTM variant in PPOLSTMPolicy.
        self.lstm_output_norm: nn.Module = (
            nn.LayerNorm(lstm_hidden) if lstm_layer_norm else nn.Identity()
        )

        # Actor head (per-runner). fill-prob-in-actor (2026-04-26)
        # bumped ``actor_input_dim`` by +1 for fill_prob; mature-prob-
        # head (2026-04-26) bumped it a further +1 for mature_prob —
        # the strict "naturally-matured-or-agent-closed" forecast that
        # excludes force-closes. See PPOLSTMPolicy for rationale.
        actor_input_dim = runner_embed_dim + lstm_hidden + 2
        self._per_runner_action_dim = action_dim // max_runners
        self.actor_head = _build_mlp(
            input_dim=actor_input_dim,
            hidden_dim=mlp_hidden,
            output_dim=self._per_runner_action_dim,
            n_layers=1,
        )

        self.action_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head
        self.critic_head = _build_mlp(
            input_dim=lstm_hidden,
            hidden_dim=mlp_hidden,
            output_dim=1,
            n_layers=1,
        )

        # Fill-probability head — see PPOLSTMPolicy for rationale.
        # fill-prob-in-actor (2026-04-26): output also fed into actor_head.
        self.fill_prob_head = nn.Linear(lstm_hidden, max_runners)
        # Mature-probability head — see PPOLSTMPolicy for rationale.
        # mature-prob-head (2026-04-26): trained on the strict
        # "naturally-matured-or-agent-closed" label; output fed into
        # actor_head alongside fill_prob.
        self.mature_prob_head = nn.Linear(lstm_hidden, max_runners)
        # Risk head — see PPOLSTMPolicy for rationale. Two outputs per
        # runner (mean + log-var), clamped at forward time.
        self.risk_head = nn.Linear(lstm_hidden, max_runners * 2)

        self._init_weights()

    def _init_weights(self) -> None:
        """Apply orthogonal initialisation (standard for PPO)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2**0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.actor_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
        for module in self.critic_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
        # Session-02 fill-prob head: small init → sigmoid ≈ 0.5 at start.
        nn.init.orthogonal_(self.fill_prob_head.weight, gain=0.01)
        nn.init.zeros_(self.fill_prob_head.bias)
        # mature-prob-head (2026-04-26): same small init as fill-prob head.
        nn.init.orthogonal_(self.mature_prob_head.weight, gain=0.01)
        nn.init.zeros_(self.mature_prob_head.bias)
        # Session-03 risk head: small init → mean ≈ 0, log_var ≈ 0 at start.
        nn.init.orthogonal_(self.risk_head.weight, gain=0.01)
        nn.init.zeros_(self.risk_head.bias)

    def _split_obs(
        self, obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split flat observation into market features and per-runner features."""
        market = obs[:, :MARKET_DIM]
        velocity = obs[:, MARKET_DIM : MARKET_DIM + VELOCITY_DIM]
        runner_start = MARKET_DIM + VELOCITY_DIM
        runner_end = runner_start + self.max_runners * RUNNER_DIM
        runners_flat = obs[:, runner_start:runner_end]
        agent_state = obs[:, runner_end : runner_end + AGENT_STATE_DIM]
        position_start = runner_end + AGENT_STATE_DIM
        position_end = position_start + self.max_runners * POSITION_DIM
        position_flat = obs[:, position_start:position_end]

        market_feats = torch.cat([market, velocity, agent_state], dim=-1)
        runner_feats_raw = runners_flat.view(-1, self.max_runners, RUNNER_DIM)
        position_feats = position_flat.view(-1, self.max_runners, POSITION_DIM)
        runner_feats = torch.cat([runner_feats_raw, position_feats], dim=-1)
        return market_feats, runner_feats

    def _extract_time_delta(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract ``seconds_since_last_tick`` from the observation vector.

        The feature sits in the velocity section of the observation at
        index ``MARKET_DIM + _TIME_DELTA_VEL_INDEX``.

        Returns
        -------
        (batch,) tensor of normalised time deltas.
        """
        idx = MARKET_DIM + _TIME_DELTA_VEL_INDEX
        return obs[:, idx]

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        signal_bias: float = 0.0,
    ) -> PolicyOutput:
        """Forward pass — identical to PPOLSTMPolicy but using TimeLSTMCell."""
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        batch, seq_len, _ = obs.shape

        if hidden_state is None:
            hidden_state = self.init_hidden(batch)
            hidden_state = (
                hidden_state[0].to(obs.device),
                hidden_state[1].to(obs.device),
            )

        # Flatten for encoder passes
        obs_flat = obs.reshape(batch * seq_len, -1)
        market_feats, runner_feats = self._split_obs(obs_flat)

        market_emb = self.market_encoder(market_feats)

        b_s = runner_feats.shape[0]
        runners_flat = runner_feats.reshape(b_s * self.max_runners, RUNNER_INPUT_DIM)
        runner_embs = self.runner_encoder(runners_flat)
        runner_embs = runner_embs.view(b_s, self.max_runners, self.runner_embed_dim)

        runner_mean = runner_embs.mean(dim=1)
        runner_max = runner_embs.max(dim=1).values

        lstm_input = torch.cat([market_emb, runner_mean, runner_max], dim=-1)
        lstm_input = lstm_input.view(batch, seq_len, -1)

        # Extract time deltas per timestep
        time_deltas = obs.reshape(batch * seq_len, -1)
        time_deltas = self._extract_time_delta(time_deltas)
        time_deltas = time_deltas.view(batch, seq_len)

        # Step through stacked TimeLSTMCells for each timestep.
        # hidden_state comes in as (num_layers, batch, hidden). We keep
        # per-layer h/c as Python lists so we can assign freely without
        # in-place-ing autograd tensors.
        h_layers = [hidden_state[0][i] for i in range(self.lstm_num_layers)]
        c_layers = [hidden_state[1][i] for i in range(self.lstm_num_layers)]

        outputs = []
        for t in range(seq_len):
            layer_input = lstm_input[:, t, :]
            dt = time_deltas[:, t]
            for layer_idx, cell in enumerate(self.time_lstm_cells):
                h_new, c_new = cell(
                    layer_input,
                    (h_layers[layer_idx], c_layers[layer_idx]),
                    dt,
                )
                h_layers[layer_idx] = h_new
                c_layers[layer_idx] = c_new
                # Inter-layer dropout: applied between stacked layers,
                # matching nn.LSTM's dropout semantics. Only active when
                # there is another layer to feed and we're training.
                if (
                    self.lstm_dropout > 0.0
                    and layer_idx < self.lstm_num_layers - 1
                ):
                    layer_input = F.dropout(
                        h_new, p=self.lstm_dropout, training=self.training,
                    )
                else:
                    layer_input = h_new
            outputs.append(h_layers[-1])

        lstm_out = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden)
        lstm_out = self.lstm_output_norm(lstm_out)

        # Restore layer dim for hidden state: (num_layers, batch, hidden)
        new_hidden = (
            torch.stack(h_layers, dim=0),
            torch.stack(c_layers, dim=0),
        )

        lstm_last = lstm_out[:, -1, :]

        # Actor (per-runner)
        if seq_len > 1:
            last_obs = obs[:, -1, :]
            _, last_runner_feats = self._split_obs(last_obs)
            last_runners_flat = last_runner_feats.reshape(
                batch * self.max_runners, RUNNER_INPUT_DIM,
            )
            last_runner_embs = self.runner_encoder(last_runners_flat)
            last_runner_embs = last_runner_embs.view(
                batch, self.max_runners, self.runner_embed_dim,
            )
        else:
            last_runner_embs = runner_embs.view(
                batch, self.max_runners, self.runner_embed_dim,
            )

        # Fill-probability head — see PPOLSTMPolicy for rationale.
        # fill-prob-in-actor (2026-04-26): computed BEFORE actor_head so
        # the per-runner scalar can be fed into actor_input.
        fill_prob_logits = self.fill_prob_head(lstm_last)
        fill_prob = torch.sigmoid(fill_prob_logits)

        # Mature-probability head — see PPOLSTMPolicy for rationale.
        # mature-prob-head (2026-04-26): same backbone as fill_prob;
        # strict label excludes force-closes.
        mature_prob_logits = self.mature_prob_head(lstm_last)
        mature_prob = torch.sigmoid(mature_prob_logits)

        lstm_expanded = lstm_last.unsqueeze(1).expand(
            -1, self.max_runners, -1,
        )
        actor_input = torch.cat(
            [
                last_runner_embs,
                lstm_expanded,
                fill_prob.unsqueeze(-1),
                mature_prob.unsqueeze(-1),
            ],
            dim=-1,
        )
        actor_out = self.actor_head(actor_input)
        # Session 3: optional warmup bias on the signal head (index 0).
        actor_out = _apply_signal_bias(
            actor_out, signal_bias, self._per_runner_action_dim,
        )
        parts = [actor_out[:, :, i] for i in range(self._per_runner_action_dim)]
        action_mean = torch.cat(parts, dim=-1)

        # Critic
        value = self.critic_head(lstm_last)

        # Risk head — see PPOLSTMPolicy for rationale. Log-var clamp
        # lives at the forward-pass boundary so downstream consumers
        # see safe bounds regardless of the raw head output.
        risk_out = self.risk_head(lstm_last)
        risk_out = risk_out.view(batch, self.max_runners, 2)
        risk_mean = risk_out[..., 0]
        risk_log_var = risk_out[..., 1].clamp(
            RISK_LOG_VAR_MIN, RISK_LOG_VAR_MAX,
        )

        return PolicyOutput(
            action_mean=action_mean,
            action_log_std=self.action_log_std.expand(batch, -1),
            value=value,
            hidden_state=new_hidden,
            fill_prob_per_runner=fill_prob,
            mature_prob_per_runner=mature_prob,
            predicted_locked_pnl_per_runner=risk_mean,
            predicted_locked_log_var_per_runner=risk_log_var,
        )

    def init_hidden(
        self, batch_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-initialised hidden state ``(h_0, c_0)``.

        Shape ``(num_layers, batch, hidden)`` — matches the stacked
        TimeLSTMCell layout expected by ``forward``.
        """
        h = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size)
        c = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size)
        return h, c

    # -- ppo-kl-fix (2026-04-24) — same LSTM-family hidden-state layout --
    # ``(num_layers, batch, hidden)`` with the batch axis at dim 1.
    # Mirror the ``PPOLSTMPolicy`` overrides so the TimeLSTM path sees
    # the same batching semantics.
    def pack_hidden_states(
        self,
        hiddens: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.cat([h[0] for h in hiddens], dim=1)
        h1 = torch.cat([h[1] for h in hiddens], dim=1)
        return h0, h1

    def slice_hidden_states(
        self,
        hidden: tuple[torch.Tensor, torch.Tensor],
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            hidden[0].index_select(1, indices),
            hidden[1].index_select(1, indices),
        )

    def get_action_distribution(
        self,
        obs: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        signal_bias: float = 0.0,
    ) -> tuple[Normal, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Return the action distribution, value, and new hidden state."""
        out = self.forward(obs, hidden_state, signal_bias=signal_bias)
        std = out.action_log_std.exp()
        dist = Normal(out.action_mean, std)
        return dist, out.value, out.hidden_state


# ── PPO + Transformer v1 (Session 6) ────────────────────────────────────────


@register_architecture
class PPOTransformerPolicy(BasePolicy):
    """PPO + Transformer-encoder policy network (architecture v1).

    Shares the market and per-runner encoders with the LSTM variants;
    replaces the recurrent sequence model with a small transformer
    encoder that attends over a rolling buffer of the last
    ``transformer_ctx_ticks`` fused embeddings.

    The ``hidden_state`` slot in :class:`BasePolicy`'s protocol is
    repurposed as a *rolling context buffer*: ``(buffer, valid_count)``
    where ``buffer`` has shape ``(batch, ctx_ticks, d_model)`` and
    ``valid_count`` is a ``(batch,)`` long tensor tracking how many of
    those slots have been filled with real data. Unfilled slots stay
    zero and the transformer simply learns to ignore the initial
    warmup window.

    Three structural genes:
    * ``transformer_heads`` ∈ {2, 4, 8}
    * ``transformer_depth`` ∈ {1, 2, 3}
    * ``transformer_ctx_ticks`` ∈ {32, 64, 128, 256}
      (256 added 2026-04-21 per plans/arb-signal-cleanup §14a-§14d;
      covers the full race for the median case (~238 ticks) where
      128 only reaches ~54%.)

    ``lstm_hidden_size`` is reused as the transformer's ``d_model``
    (all valid choices divide evenly by any of the allowed head counts).
    Positional information is injected via a learned
    ``nn.Embedding(ctx_ticks, d_model)`` -- simpler than sinusoidal,
    cleaner gradient, and cheap.

    Causal masking is enforced so that when :meth:`forward` is called
    with a 3-D ``(batch, seq_len, obs_dim)`` tensor, the transformer
    cannot attend to positions in the future relative to the current
    timestep. This matters for sequence-input use (e.g. tests,
    diagnostics) even though the production rollout path feeds one
    tick at a time.
    """

    architecture_name = "ppo_transformer_v1"
    description = (
        "PPO with transformer encoder over a bounded rolling tick-context "
        "window. Shares market / per-runner encoders and actor / critic "
        "heads with the LSTM variants; replaces the LSTM with a causal "
        "transformer and a learned positional embedding."
    )

    # Transformer action heads saturate on the first PPO update at the
    # shared 3e-4 default LR — transformer ``0a8cacd3`` ep-1
    # ``policy_loss = 1.04e17`` regression confirmed the shared LR is
    # too hot for this arch. Halving here gives the warmup +
    # KL-early-stop + ratio-clamp defences headroom to catch any
    # residual instability. Consumed by
    # ``agents/ppo_trainer.py::PPOTrainer.__init__`` when the hp dict
    # omits ``learning_rate``; the GA still mutates LR around the
    # sampled gene value when it's present.
    # See plans/naked-clip-and-stability/purpose.md §2.
    default_learning_rate: float = 1.5e-4

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        max_runners: int,
        hyperparams: dict,
    ) -> None:
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_runners = max_runners

        # Shared-with-LSTM hyperparameters.
        d_model = int(hyperparams.get("lstm_hidden_size", 256))
        mlp_hidden = int(hyperparams.get("mlp_hidden_size", 128))
        mlp_layers = int(hyperparams.get("mlp_layers", 2))
        runner_embed_dim = mlp_hidden

        # Session 6 — transformer structural genes.
        self.transformer_heads = int(hyperparams.get("transformer_heads", 4))
        self.transformer_depth = int(hyperparams.get("transformer_depth", 2))
        self.ctx_ticks = int(hyperparams.get("transformer_ctx_ticks", 32))

        # d_model must divide evenly by nhead. All shipped
        # lstm_hidden_size choices (64, 128, 256, 512, 1024, 2048) are
        # divisible by every allowed head count (2, 4, 8); this guard
        # is defensive against hand-crafted hyperparam dicts.
        if d_model % self.transformer_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by transformer_heads="
                f"{self.transformer_heads}"
            )

        self.d_model = d_model
        self.lstm_hidden_size = d_model  # alias so population/trainer code
        # that reads lstm_hidden_size still works on transformer agents.
        self.runner_embed_dim = runner_embed_dim

        # ── Runner encoder (shared weights across all runners) ──────────
        self.runner_encoder = _build_mlp(
            input_dim=RUNNER_INPUT_DIM,
            hidden_dim=mlp_hidden,
            output_dim=runner_embed_dim,
            n_layers=mlp_layers,
        )

        # ── Market encoder ──────────────────────────────────────────────
        self.market_encoder = _build_mlp(
            input_dim=MARKET_TOTAL_DIM,
            hidden_dim=mlp_hidden,
            output_dim=mlp_hidden,
            n_layers=mlp_layers,
        )

        # ── Fused → d_model projection ─────────────────────────────────
        fused_dim = mlp_hidden + runner_embed_dim * 2
        self.input_projection = nn.Linear(fused_dim, d_model)

        # ── Positional embedding (learned) ──────────────────────────────
        # Fixed length = ctx_ticks. Indices always 0..ctx_ticks-1 because
        # the rolling buffer has a fixed size; the "current" tick is
        # always at position ``ctx_ticks-1``.
        self.position_embedding = nn.Embedding(self.ctx_ticks, d_model)

        # ── Transformer encoder ─────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=self.transformer_heads,
            dim_feedforward=max(d_model * 2, mlp_hidden),
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        # ``enable_nested_tensor=False`` avoids a UserWarning when
        # ``norm_first=True`` (nested-tensor fast path is disabled in
        # that configuration anyway, and we never pass
        # ``src_key_padding_mask`` so nested tensors wouldn't help).
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.transformer_depth,
            enable_nested_tensor=False,
        )
        self.transformer_norm = nn.LayerNorm(d_model)

        # Causal mask — registered as a buffer so .to(device) moves it
        # with the module. ``nn.Transformer.generate_square_subsequent_mask``
        # returns an upper-triangular tensor of 0/-inf suitable for use
        # as an attention mask with ``batch_first=True``.
        causal_mask = torch.triu(
            torch.full((self.ctx_ticks, self.ctx_ticks), float("-inf")),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        # ── Actor head (per-runner) ─────────────────────────────────────
        # fill-prob-in-actor (2026-04-26): ``actor_input_dim`` is bumped
        # by +1 for fill_prob; mature-prob-head (2026-04-26) bumped a
        # further +1 for mature_prob — the strict
        # "naturally-matured-or-agent-closed" forecast that excludes
        # force-closes. See PPOLSTMPolicy for rationale.
        actor_input_dim = runner_embed_dim + d_model + 2
        self._per_runner_action_dim = action_dim // max_runners
        self.actor_head = _build_mlp(
            input_dim=actor_input_dim,
            hidden_dim=mlp_hidden,
            output_dim=self._per_runner_action_dim,
            n_layers=1,
        )

        self.action_log_std = nn.Parameter(torch.zeros(action_dim))

        # ── Critic head ─────────────────────────────────────────────────
        self.critic_head = _build_mlp(
            input_dim=d_model,
            hidden_dim=mlp_hidden,
            output_dim=1,
            n_layers=1,
        )

        # Fill-probability head — see PPOLSTMPolicy for rationale. Takes
        # the final transformer tick output (``out_last``) as backbone.
        # fill-prob-in-actor (2026-04-26): output also fed into actor_head.
        self.fill_prob_head = nn.Linear(d_model, max_runners)
        # Mature-probability head — see PPOLSTMPolicy for rationale.
        # mature-prob-head (2026-04-26): trained on the strict label;
        # output fed into actor_head alongside fill_prob.
        self.mature_prob_head = nn.Linear(d_model, max_runners)
        # Risk head — see PPOLSTMPolicy for rationale. Two outputs per
        # runner (mean + log-var); log-var clamped inside ``forward``.
        self.risk_head = nn.Linear(d_model, max_runners * 2)

        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal init for linear layers (standard PPO)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2**0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.actor_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
        for module in self.critic_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
        # Session-02 fill-prob head: small init → sigmoid ≈ 0.5 at start.
        nn.init.orthogonal_(self.fill_prob_head.weight, gain=0.01)
        nn.init.zeros_(self.fill_prob_head.bias)
        # mature-prob-head (2026-04-26): same small init as fill-prob head.
        nn.init.orthogonal_(self.mature_prob_head.weight, gain=0.01)
        nn.init.zeros_(self.mature_prob_head.bias)
        # Session-03 risk head: small init → mean ≈ 0, log_var ≈ 0 at start.
        nn.init.orthogonal_(self.risk_head.weight, gain=0.01)
        nn.init.zeros_(self.risk_head.bias)

    # ── Shared obs splitter ─────────────────────────────────────────────
    def _split_obs(
        self, obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        market = obs[:, :MARKET_DIM]
        velocity = obs[:, MARKET_DIM : MARKET_DIM + VELOCITY_DIM]
        runner_start = MARKET_DIM + VELOCITY_DIM
        runner_end = runner_start + self.max_runners * RUNNER_DIM
        runners_flat = obs[:, runner_start:runner_end]
        agent_state = obs[:, runner_end : runner_end + AGENT_STATE_DIM]
        position_start = runner_end + AGENT_STATE_DIM
        position_end = position_start + self.max_runners * POSITION_DIM
        position_flat = obs[:, position_start:position_end]

        market_feats = torch.cat([market, velocity, agent_state], dim=-1)
        runner_feats_raw = runners_flat.view(-1, self.max_runners, RUNNER_DIM)
        position_feats = position_flat.view(-1, self.max_runners, POSITION_DIM)
        runner_feats = torch.cat([runner_feats_raw, position_feats], dim=-1)
        return market_feats, runner_feats

    def _encode_ticks(
        self, obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batched tick stream into ``(fused, last_runner_embs)``.

        Parameters
        ----------
        obs : (N, obs_dim) flat tensor.

        Returns
        -------
        fused : (N, d_model) projected fused embedding per tick.
        runner_embs : (N, max_runners, runner_embed_dim).
        """
        market_feats, runner_feats = self._split_obs(obs)
        market_emb = self.market_encoder(market_feats)
        n = runner_feats.shape[0]
        runners_flat = runner_feats.reshape(
            n * self.max_runners, RUNNER_INPUT_DIM,
        )
        runner_embs = self.runner_encoder(runners_flat).view(
            n, self.max_runners, self.runner_embed_dim,
        )
        runner_mean = runner_embs.mean(dim=1)
        runner_max = runner_embs.max(dim=1).values
        fused = torch.cat([market_emb, runner_mean, runner_max], dim=-1)
        fused = self.input_projection(fused)
        return fused, runner_embs

    def _run_encoder(self, buffer: torch.Tensor) -> torch.Tensor:
        """Run positional + causal transformer encoder on ``buffer``.

        Parameters
        ----------
        buffer : (batch, ctx_ticks, d_model)

        Returns
        -------
        encoded : (batch, ctx_ticks, d_model) with final layer norm applied.
        """
        pos_ids = torch.arange(self.ctx_ticks, device=buffer.device)
        pos_emb = self.position_embedding(pos_ids)  # (ctx_ticks, d_model)
        x = buffer + pos_emb.unsqueeze(0)
        encoded = self.transformer_encoder(x, mask=self.causal_mask)
        return self.transformer_norm(encoded)

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        signal_bias: float = 0.0,
    ) -> PolicyOutput:
        """Forward pass.

        Accepts 2-D ``(batch, obs_dim)`` (single timestep) or 3-D
        ``(batch, seq_len, obs_dim)``. In both cases the rolling buffer
        is shifted-and-appended once per timestep in the sequence and
        the transformer is run on the final buffer state. The critic /
        actor heads read the encoder output at the most recent (last)
        position of the buffer.
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (batch, 1, obs_dim)
        batch, seq_len, _ = obs.shape

        if hidden_state is None:
            hidden_state = self.init_hidden(batch)
            hidden_state = (
                hidden_state[0].to(obs.device),
                hidden_state[1].to(obs.device),
            )

        buffer, valid_count = hidden_state
        # ``buffer`` may be a view on storage we shouldn't mutate in
        # place -- ``torch.cat`` below always produces a fresh tensor
        # so we never hit an autograd / aliasing issue.

        # Encode every tick in the provided sequence.
        obs_flat = obs.reshape(batch * seq_len, -1)
        fused_flat, _ = self._encode_ticks(obs_flat)
        fused = fused_flat.view(batch, seq_len, self.d_model)

        # Shift-left and append each fused tick, one at a time. Cheap
        # even at the largest ctx_ticks (128) because seq_len is 1 on
        # the production rollout path.
        for t in range(seq_len):
            buffer = torch.cat(
                [buffer[:, 1:, :], fused[:, t : t + 1, :]], dim=1,
            )
            valid_count = torch.clamp(valid_count + 1, max=self.ctx_ticks)

        encoded = self._run_encoder(buffer)  # (batch, ctx_ticks, d_model)
        out_last = encoded[:, -1, :]  # most recent tick

        # ── Actor: per-runner ──────────────────────────────────────────
        # Re-derive the runner embeddings for the last tick only (we
        # need them concatenated with the transformer output).
        last_obs = obs[:, -1, :]
        _, last_runner_embs = self._encode_ticks(last_obs)

        # Fill-probability head — backbone is the final-tick transformer
        # output (``out_last``). See PPOLSTMPolicy for rationale.
        # fill-prob-in-actor (2026-04-26): computed BEFORE actor_head so
        # the per-runner scalar can be fed into actor_input.
        fill_prob_logits = self.fill_prob_head(out_last)
        fill_prob = torch.sigmoid(fill_prob_logits)

        # Mature-probability head — see PPOLSTMPolicy for rationale.
        # mature-prob-head (2026-04-26): same backbone; strict label.
        mature_prob_logits = self.mature_prob_head(out_last)
        mature_prob = torch.sigmoid(mature_prob_logits)

        out_expanded = out_last.unsqueeze(1).expand(
            -1, self.max_runners, -1,
        )
        actor_input = torch.cat(
            [
                last_runner_embs,
                out_expanded,
                fill_prob.unsqueeze(-1),
                mature_prob.unsqueeze(-1),
            ],
            dim=-1,
        )
        actor_out = self.actor_head(actor_input)
        # Session 3: optional warmup bias on the signal head (index 0).
        actor_out = _apply_signal_bias(
            actor_out, signal_bias, self._per_runner_action_dim,
        )
        parts = [actor_out[:, :, i] for i in range(self._per_runner_action_dim)]
        action_mean = torch.cat(parts, dim=-1)

        # ── Critic ────────────────────────────────────────────────────
        value = self.critic_head(out_last)

        # Risk head — same backbone, two outputs per runner, log-var
        # clamped at forward-pass boundary. See PPOLSTMPolicy for rationale.
        risk_out = self.risk_head(out_last)
        risk_out = risk_out.view(batch, self.max_runners, 2)
        risk_mean = risk_out[..., 0]
        risk_log_var = risk_out[..., 1].clamp(
            RISK_LOG_VAR_MIN, RISK_LOG_VAR_MAX,
        )

        return PolicyOutput(
            action_mean=action_mean,
            action_log_std=self.action_log_std.expand(batch, -1),
            value=value,
            hidden_state=(buffer, valid_count),
            fill_prob_per_runner=fill_prob,
            mature_prob_per_runner=mature_prob,
            predicted_locked_pnl_per_runner=risk_mean,
            predicted_locked_log_var_per_runner=risk_log_var,
        )

    def init_hidden(
        self, batch_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a zero rolling buffer and zero valid-count tensor.

        * ``buffer``: ``(batch, ctx_ticks, d_model)`` float tensor.
        * ``valid_count``: ``(batch,)`` long tensor.

        The 2-tuple shape matches :class:`BasePolicy`'s protocol so
        that :class:`agents.ppo_trainer.PPOTrainer` can move both
        elements to the training device via the standard
        ``h[0].to(device), h[1].to(device)`` idiom without special
        cases for the transformer.
        """
        buffer = torch.zeros(batch_size, self.ctx_ticks, self.d_model)
        valid_count = torch.zeros(batch_size, dtype=torch.long)
        return buffer, valid_count

    def encode_sequence(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the per-position transformer output for ``obs``.

        Utility used by the causal-masking test. Mirrors :meth:`forward`
        up to the point where the encoder output is produced, but
        returns the full ``(batch, ctx_ticks, d_model)`` tensor instead
        of just the last-position slice.

        Accepts 2-D or 3-D ``obs`` just like :meth:`forward`.
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        batch, seq_len, _ = obs.shape

        buffer = torch.zeros(
            batch, self.ctx_ticks, self.d_model, device=obs.device,
        )
        obs_flat = obs.reshape(batch * seq_len, -1)
        fused_flat, _ = self._encode_ticks(obs_flat)
        fused = fused_flat.view(batch, seq_len, self.d_model)
        for t in range(seq_len):
            buffer = torch.cat(
                [buffer[:, 1:, :], fused[:, t : t + 1, :]], dim=1,
            )
        return self._run_encoder(buffer)

    def get_action_distribution(
        self,
        obs: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        signal_bias: float = 0.0,
    ) -> tuple[Normal, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Return the action distribution, value, and new hidden state."""
        out = self.forward(obs, hidden_state, signal_bias=signal_bias)
        std = out.action_log_std.exp()
        dist = Normal(out.action_mean, std)
        return dist, out.value, out.hidden_state
