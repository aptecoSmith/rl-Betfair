"""Per-agent behavioural cloning from arb oracle samples.

Hard contracts (plans/arb-curriculum/hard_constraints.md s16-s20):
- Per-agent; never share weights across the population.
- Only actor_head trains; all other parameters are frozen
  (bit-identical before/after BC completes).
- Separate Adam optimiser from PPO's Adam — no shared state.
- Empty oracle cache -> skip cleanly (return empty BCLossHistory).
- Schema version must match the running env; load_samples raises on
  mismatch and the caller must handle it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from env.betfair_env import MAX_ARB_TICKS
from training.arb_oracle import OracleSample

# Index of signal head within per-runner action dims (layout:
# signal, stake, aggression, cancel, arb_spread, requote, close).
_SIGNAL_DIM: int = 0
_ARB_SPREAD_DIM: int = 4


@dataclass
class BCLossHistory:
    signal_losses: list[float] = field(default_factory=list)
    arb_spread_losses: list[float] = field(default_factory=list)
    total_losses: list[float] = field(default_factory=list)
    final_signal_loss: float = 0.0
    final_arb_spread_loss: float = 0.0


class BCPretrainer:
    def __init__(
        self,
        lr: float = 3e-4,
        batch_size: int = 64,
        signal_weight: float = 1.0,
        arb_spread_weight: float = 0.1,
    ) -> None:
        self.lr = lr
        self.batch_size = batch_size
        self.signal_weight = signal_weight
        self.arb_spread_weight = arb_spread_weight

    def pretrain(
        self,
        policy,
        samples: list[OracleSample],
        n_steps: int,
    ) -> BCLossHistory:
        """Pretrain signal + arb_spread heads only. Returns loss history.

        All non-actor_head parameters are frozen (requires_grad=False)
        during training and restored to requires_grad=True on exit.
        The BC Adam optimiser is a separate instance — PPO's optimiser
        state is untouched.
        """
        if not samples or n_steps <= 0:
            return BCLossHistory()

        max_runners: int = getattr(policy, "max_runners", 1)
        device = next(policy.parameters()).device

        frozen = [
            p for name, p in policy.named_parameters()
            if not _is_bc_target_head(name)
        ]
        for p in frozen:
            p.requires_grad_(False)

        target_params = [
            p for name, p in policy.named_parameters()
            if _is_bc_target_head(name)
        ]
        opt = torch.optim.Adam(target_params, lr=self.lr)

        history = BCLossHistory()
        for _ in range(n_steps):
            batch = _sample_batch(samples, self.batch_size)

            obs_t = torch.tensor(
                np.stack([s.obs for s in batch], axis=0),
                dtype=torch.float32,
                device=device,
            )
            out = policy(obs_t)
            action_mean = out.action_mean  # (batch_size, action_dim)

            # Signal target: push to +1.0 at position runner_idx.
            signal_indices = torch.tensor(
                [_SIGNAL_DIM * max_runners + s.runner_idx for s in batch],
                dtype=torch.long,
                device=device,
            )
            arange = torch.arange(len(batch), device=device)
            signal_pred = action_mean[arange, signal_indices]
            signal_target = torch.ones(len(batch), dtype=torch.float32, device=device)
            signal_loss = F.mse_loss(signal_pred, signal_target)

            # Arb_spread target: normalize arb_spread_ticks by MAX_ARB_TICKS.
            spread_indices = torch.tensor(
                [_ARB_SPREAD_DIM * max_runners + s.runner_idx for s in batch],
                dtype=torch.long,
                device=device,
            )
            spread_pred = action_mean[arange, spread_indices]
            spread_target = torch.tensor(
                [s.arb_spread_ticks / max(MAX_ARB_TICKS, 1) for s in batch],
                dtype=torch.float32,
                device=device,
            )
            arb_spread_loss = F.mse_loss(spread_pred, spread_target)

            loss = (
                self.signal_weight * signal_loss
                + self.arb_spread_weight * arb_spread_loss
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            history.signal_losses.append(float(signal_loss.item()))
            history.arb_spread_losses.append(float(arb_spread_loss.item()))
            history.total_losses.append(float(loss.item()))

        for p in frozen:
            p.requires_grad_(True)

        if history.signal_losses:
            history.final_signal_loss = history.signal_losses[-1]
            history.final_arb_spread_loss = history.arb_spread_losses[-1]

        return history


def measure_entropy(policy, samples: list[OracleSample]) -> float:
    """Return mean action-distribution entropy on the given oracle samples.

    Called post-BC to seed the controller warmup handshake in
    ppo_trainer._effective_target_entropy(). Returns 0.0 when samples is
    empty (no BC ran, so the caller should not set _post_bc_entropy at all).
    """
    if not samples:
        return 0.0
    device = next(policy.parameters()).device
    batch = samples[:256]
    obs_t = torch.tensor(
        np.stack([s.obs for s in batch], axis=0),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        out = policy(obs_t)
        std = out.action_log_std.exp()
        dist = Normal(out.action_mean, std)
        entropy = dist.entropy().sum(dim=-1).mean().item()
    return float(entropy)


def _is_bc_target_head(name: str) -> bool:
    """True for actor_head parameters only.

    All three policy architectures (ppo_lstm_v1, ppo_time_lstm_v1,
    ppo_transformer_v1) use a single actor_head MLP module. BC trains
    only those parameters; value_head, LSTM, and feature encoders are
    frozen and restored to requires_grad=True after BC.
    """
    return "actor_head" in name


def _sample_batch(
    samples: list[OracleSample],
    batch_size: int,
) -> list[OracleSample]:
    """Draw a random batch, sampling with replacement when pool < batch_size."""
    if len(samples) <= batch_size:
        return random.choices(samples, k=batch_size)
    return random.sample(samples, batch_size)
