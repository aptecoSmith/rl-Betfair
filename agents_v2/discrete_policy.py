"""Discrete-action policy classes for the v2 trainer.

Phase 1, Session 02 deliverable. Drops the v1 70-dim continuous-multi-
head action space in favour of a single masked categorical over
``{NOOP, OPEN_BACK_i, OPEN_LAY_i, CLOSE_i}`` plus a small Beta-sized
``stake`` continuous head and a **per-runner value head**.

Locked design (see
``plans/rewrite/phase-1-policy-and-env-wiring/purpose.md``):

- The categorical sees a ``-inf`` mask on illegal actions; sampling
  from ``Categorical(logits=masked_logits)`` therefore never returns
  a masked index. **No post-hoc rejection.**
- The Beta heads emit ``alpha, beta > 1`` (via ``softplus + 1``) so
  the distribution is unimodal — no near-0/1 hairpin pathologies.
  The action caller draws ``s = Beta(alpha, beta).sample()`` and
  re-scales to ``[MIN_BET_STAKE, max_stake_cap]`` outside the policy.
- The value head is plain linear, no activation, output dim
  ``max_runners`` — Phase 2's GAE consumes per-runner returns directly.
- The hidden-state pack/slice helpers mirror the v1 contract
  (``CLAUDE.md`` §"Recurrent PPO: hidden-state protocol on update")
  so Phase 2's PPO update can batch per-transition states without
  knowing the architecture's batch-axis.

**Hard constraints (Session 02 prompt):**

* No training code lives here — no optimiser, no loss, no GAE.
* No imports from ``agents/`` (parallel tree; rewrite README §3).
* No env edits.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Beta, Categorical

from agents_v2.action_space import DiscreteActionSpace


__all__ = [
    "DiscretePolicyOutput",
    "BaseDiscretePolicy",
    "DiscreteLSTMPolicy",
]


@dataclass(frozen=True)
class DiscretePolicyOutput:
    """Forward-pass result for any :class:`BaseDiscretePolicy` subclass.

    Attributes
    ----------
    logits:
        Raw categorical logits, shape ``(batch, action_space.n)``.
        These are pre-mask — useful for diagnostics that want to see
        the "would-pick" preference order including illegal actions.
    masked_logits:
        ``logits`` with ``-inf`` at every False entry of the mask
        passed to ``forward``. If no mask is supplied this equals
        ``logits``. Shape ``(batch, action_space.n)``.
    action_dist:
        :class:`torch.distributions.Categorical` built from
        ``masked_logits``. Sampling from it never returns a masked
        index (PyTorch handles the ``-inf`` softmax cleanly as long
        as at least one logit is finite — NOOP is always legal under
        the locked layout, so this invariant always holds).
    stake_alpha, stake_beta:
        Beta-distribution parameters for the stake continuous head,
        shape ``(batch,)`` each, both strictly ``> 1`` so the Beta
        is unimodal.
    value_per_runner:
        Per-runner critic, shape ``(batch, max_runners)``. Phase 2's
        GAE uses this directly; the global scalar value can be
        recovered as a sum-over-runners.
    new_hidden_state:
        Architecture-specific tuple. For LSTM-family policies:
        ``(h, c)`` each ``(num_layers, batch, hidden_size)``; for the
        transformer (Phase 1 follow-on): ``(buffer, valid_count)``.
        The exact shape is private to each subclass — only the
        2-tuple-of-tensors contract is shared.
    """

    logits: torch.Tensor
    masked_logits: torch.Tensor
    action_dist: Categorical
    stake_alpha: torch.Tensor
    stake_beta: torch.Tensor
    value_per_runner: torch.Tensor
    new_hidden_state: tuple[torch.Tensor, ...]


class BaseDiscretePolicy(nn.Module, abc.ABC):
    """Abstract base for v2 discrete-action policies.

    The class is intentionally thin: it pins the ``__init__`` signature
    (``obs_dim``, ``action_space``, ``hidden_size``) and the forward
    contract (returns :class:`DiscretePolicyOutput`). Subclasses pick
    a backbone (LSTM, TimeLSTM, Transformer …) and implement
    ``forward`` + ``init_hidden`` + the pack/slice helpers.

    The pack/slice helpers default to ``dim=0`` concat / index-select
    — correct for the transformer's ``(buffer, valid_count)`` hidden
    state. The LSTM subclass overrides them to use ``dim=1`` (the
    ``nn.LSTM`` batch axis).
    """

    def __init__(
        self,
        obs_dim: int,
        action_space: DiscreteActionSpace,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        if obs_dim <= 0:
            raise ValueError(f"obs_dim must be positive, got {obs_dim!r}")
        if hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive, got {hidden_size!r}",
            )
        self.obs_dim = int(obs_dim)
        self.action_space = action_space
        self.max_runners = action_space.max_runners
        self.hidden_size = int(hidden_size)

    # ── Interface ──────────────────────────────────────────────────────────

    @abc.abstractmethod
    def init_hidden(
        self, batch: int = 1,
    ) -> tuple[torch.Tensor, ...]:
        """Return a zero-initialised hidden state for ``batch`` sequences."""

    @abc.abstractmethod
    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: tuple[torch.Tensor, ...] | None = None,
        mask: torch.Tensor | None = None,
    ) -> DiscretePolicyOutput:
        """Run the backbone + heads.

        Parameters
        ----------
        obs:
            ``(batch, obs_dim)`` for a single timestep, or
            ``(batch, ctx, obs_dim)`` for a sequence — subclasses use
            the last timestep's output for the heads.
        hidden_state:
            Architecture-specific tuple, or ``None`` for a fresh
            zero-init state. The default-init path moves the state to
            ``obs.device`` so callers don't have to.
        mask:
            ``(batch, action_space.n)`` boolean. ``True`` = legal,
            ``False`` = masked. Applied at the logits level
            (``-inf``), never as post-hoc rejection.
        """

    # ── Hidden-state batching (PPO update path) ───────────────────────────

    @staticmethod
    def pack_hidden_states(
        states: list[tuple[torch.Tensor, ...]],
    ) -> tuple[torch.Tensor, ...]:
        """Concatenate per-transition hidden states into one batched tuple.

        Default implementation concatenates each tuple slot along
        ``dim=0`` — correct for transformer-style states whose batch
        axis is dim 0. The LSTM subclass overrides this to use
        ``dim=1``.
        """
        if not states:
            raise ValueError("pack_hidden_states: states list is empty")
        n_slots = len(states[0])
        return tuple(
            torch.cat([s[i] for s in states], dim=0) for i in range(n_slots)
        )

    @staticmethod
    def slice_hidden_states(
        packed: tuple[torch.Tensor, ...],
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Slice a packed hidden state by integer ``indices`` along dim 0."""
        return tuple(t.index_select(0, indices) for t in packed)


# ── Discrete LSTM ──────────────────────────────────────────────────────────


class DiscreteLSTMPolicy(BaseDiscretePolicy):
    """Single-layer LSTM baseline for the v2 discrete action space.

    Architecture::

        Linear(obs_dim → hidden) → ReLU
        LSTM(hidden, hidden, num_layers=1, batch_first=True)
            → lstm_last : (batch, hidden)
        Heads (all consume lstm_last):
            logits_head      : Linear(hidden → action_space.n)
            stake_alpha_head : Linear(hidden → 1) → softplus → +1
            stake_beta_head  : Linear(hidden → 1) → softplus → +1
            value_head       : Linear(hidden → max_runners)

    Hidden state ``(h, c)``, each ``(num_layers=1, batch, hidden_size)``
    — the standard ``nn.LSTM`` layout.
    """

    architecture_name = "discrete_lstm_v2"
    description = (
        "v2 discrete-action policy: single-layer LSTM (hidden=128 by "
        "default), masked categorical over {NOOP, OPEN_BACK_i, "
        "OPEN_LAY_i, CLOSE_i}, Beta stake head, per-runner value head."
    )

    def __init__(
        self,
        obs_dim: int,
        action_space: DiscreteActionSpace,
        hidden_size: int = 128,
    ) -> None:
        super().__init__(obs_dim, action_space, hidden_size)
        self.num_layers = 1

        self.input_proj = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_size),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        self.logits_head = nn.Linear(self.hidden_size, action_space.n)
        self.stake_alpha_head = nn.Linear(self.hidden_size, 1)
        self.stake_beta_head = nn.Linear(self.hidden_size, 1)
        self.value_head = nn.Linear(self.hidden_size, self.max_runners)

    # ── Hidden state ──────────────────────────────────────────────────────

    def init_hidden(
        self, batch: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(self.num_layers, batch, self.hidden_size)
        c = torch.zeros(self.num_layers, batch, self.hidden_size)
        return h, c

    @staticmethod
    def pack_hidden_states(
        states: list[tuple[torch.Tensor, ...]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """LSTM hidden state batches along dim 1, not dim 0."""
        if not states:
            raise ValueError("pack_hidden_states: states list is empty")
        h0 = torch.cat([s[0] for s in states], dim=1)
        h1 = torch.cat([s[1] for s in states], dim=1)
        return h0, h1

    @staticmethod
    def slice_hidden_states(
        packed: tuple[torch.Tensor, ...],
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            packed[0].index_select(1, indices),
            packed[1].index_select(1, indices),
        )

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: tuple[torch.Tensor, ...] | None = None,
        mask: torch.Tensor | None = None,
    ) -> DiscretePolicyOutput:
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (batch, 1, obs_dim)
        if obs.dim() != 3:
            raise ValueError(
                f"obs must be (batch, obs_dim) or (batch, ctx, obs_dim), "
                f"got shape {tuple(obs.shape)}",
            )
        batch, _ctx, obs_dim = obs.shape
        if obs_dim != self.obs_dim:
            raise ValueError(
                f"obs_dim mismatch: expected {self.obs_dim}, got {obs_dim}",
            )

        if hidden_state is None:
            h0, c0 = self.init_hidden(batch)
            hidden_state = (h0.to(obs.device), c0.to(obs.device))

        # Backbone
        flat = obs.reshape(batch * obs.shape[1], obs_dim)
        proj = self.input_proj(flat).reshape(batch, obs.shape[1], -1)
        lstm_out, new_hidden = self.lstm(proj, hidden_state)
        lstm_last = lstm_out[:, -1, :]  # (batch, hidden)

        # Categorical head
        logits = self.logits_head(lstm_last)  # (batch, n)
        masked_logits = self._apply_mask(logits, mask)
        dist = Categorical(logits=masked_logits)

        # Stake Beta heads — softplus + 1 keeps alpha, beta > 1 (unimodal).
        # ``squeeze(-1)`` so shape is (batch,), matching the contract.
        stake_alpha = nn.functional.softplus(
            self.stake_alpha_head(lstm_last),
        ).squeeze(-1) + 1.0
        stake_beta = nn.functional.softplus(
            self.stake_beta_head(lstm_last),
        ).squeeze(-1) + 1.0

        value_per_runner = self.value_head(lstm_last)  # (batch, max_runners)

        return DiscretePolicyOutput(
            logits=logits,
            masked_logits=masked_logits,
            action_dist=dist,
            stake_alpha=stake_alpha,
            stake_beta=stake_beta,
            value_per_runner=value_per_runner,
            new_hidden_state=new_hidden,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _apply_mask(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if mask is None:
            return logits
        if mask.shape != logits.shape:
            # Allow a single (n,) mask broadcast across the batch, which
            # is the common rollout-time shape (one mask per env step).
            if mask.dim() == 1 and mask.shape[0] == logits.shape[1]:
                mask = mask.unsqueeze(0).expand_as(logits)
            else:
                raise ValueError(
                    f"mask shape {tuple(mask.shape)} incompatible with "
                    f"logits shape {tuple(logits.shape)}",
                )
        if mask.dtype != torch.bool:
            mask = mask.bool()
        # ``masked_fill`` with ~mask: True (legal) stays as logits,
        # False (illegal) becomes -inf. The Categorical's softmax then
        # routes zero probability to masked indices.
        neg_inf = torch.tensor(
            float("-inf"), dtype=logits.dtype, device=logits.device,
        )
        return torch.where(mask, logits, neg_inf)


def make_stake_distribution(
    stake_alpha: torch.Tensor,
    stake_beta: torch.Tensor,
) -> Beta:
    """Build a :class:`Beta` from the policy's stake parameters.

    Helper for the smoke driver and tests so the ``alpha, beta``
    contract is exercised in the same place every time.
    """
    return Beta(stake_alpha, stake_beta)
