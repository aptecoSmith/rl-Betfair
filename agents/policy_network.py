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
from dataclasses import dataclass

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


# ── Base class ──────────────────────────────────────────────────────────────


class BasePolicy(nn.Module, abc.ABC):
    """Interface that all policy architectures must implement.

    Subclasses must accept ``(obs_dim, action_dim, max_runners, hyperparams)``
    in their ``__init__`` and call ``super().__init__()``.
    """

    architecture_name: str = ""
    description: str = ""

    @abc.abstractmethod
    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> PolicyOutput: ...

    @abc.abstractmethod
    def init_hidden(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-initialised LSTM hidden state ``(h_0, c_0)``."""
        ...


@dataclass
class PolicyOutput:
    """Structured output from a policy forward pass."""

    action_mean: torch.Tensor       # (batch, action_dim) — mean of action distribution
    action_log_std: torch.Tensor    # (batch, action_dim) — log std
    value: torch.Tensor             # (batch, 1) — state value estimate
    hidden_state: tuple[torch.Tensor, torch.Tensor]  # new LSTM state


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
        # For each runner: concat(runner_emb, lstm_output) → action params
        actor_input_dim = runner_embed_dim + lstm_hidden
        self.actor_head = _build_mlp(
            input_dim=actor_input_dim,
            hidden_dim=mlp_hidden,
            output_dim=2,  # action_signal_mean, stake_fraction_mean
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
    ) -> PolicyOutput:
        """Forward pass through the full network.

        Parameters
        ----------
        obs : (batch, obs_dim)  or  (batch, seq_len, obs_dim)
            If 3-D, processes the full sequence through the LSTM.
        hidden_state :
            ``(h, c)`` each of shape ``(1, batch, lstm_hidden_size)``.
            Pass ``None`` on the first tick of an episode (zeros used).
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

        # Expand LSTM output to match each runner
        lstm_expanded = lstm_last.unsqueeze(1).expand(
            -1, self.max_runners, -1
        )  # (batch, max_runners, lstm_hidden)

        # Concat runner embedding with LSTM context
        actor_input = torch.cat(
            [last_runner_embs, lstm_expanded], dim=-1
        )  # (batch, max_runners, embed + lstm_hidden)

        # Per-runner action params: (batch, max_runners, 2)
        actor_out = self.actor_head(actor_input)
        action_signal = actor_out[:, :, 0]    # (batch, max_runners)
        stake_fraction = actor_out[:, :, 1]   # (batch, max_runners)

        # Flatten to action_dim: [signals..., stakes...]
        action_mean = torch.cat(
            [action_signal, stake_fraction], dim=-1
        )  # (batch, max_runners * 2)

        # ── Critic: scalar V(s) ────────────────────────────────────────
        value = self.critic_head(lstm_last)  # (batch, 1)

        return PolicyOutput(
            action_mean=action_mean,
            action_log_std=self.action_log_std.expand(batch, -1),
            value=value,
            hidden_state=new_hidden,
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

    def get_action_distribution(
        self,
        obs: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[Normal, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Return the action distribution, value, and new hidden state.

        Convenience method for PPO rollout collection.
        """
        out = self.forward(obs, hidden_state)
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

        # Actor head (per-runner)
        actor_input_dim = runner_embed_dim + lstm_hidden
        self.actor_head = _build_mlp(
            input_dim=actor_input_dim,
            hidden_dim=mlp_hidden,
            output_dim=2,
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

        lstm_expanded = lstm_last.unsqueeze(1).expand(
            -1, self.max_runners, -1,
        )
        actor_input = torch.cat([last_runner_embs, lstm_expanded], dim=-1)
        actor_out = self.actor_head(actor_input)
        action_signal = actor_out[:, :, 0]
        stake_fraction = actor_out[:, :, 1]
        action_mean = torch.cat([action_signal, stake_fraction], dim=-1)

        # Critic
        value = self.critic_head(lstm_last)

        return PolicyOutput(
            action_mean=action_mean,
            action_log_std=self.action_log_std.expand(batch, -1),
            value=value,
            hidden_state=new_hidden,
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

    def get_action_distribution(
        self,
        obs: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[Normal, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Return the action distribution, value, and new hidden state."""
        out = self.forward(obs, hidden_state)
        std = out.action_log_std.exp()
        dist = Normal(out.action_mean, std)
        return dist, out.value, out.hidden_state
