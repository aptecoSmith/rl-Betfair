"""Behavioural cloning pretrain for the v2 discrete-action policy.

Phase 8 Session 02 deliverable. Mirrors v1's
``agents/bc_pretrainer.py`` shape (freeze/unfreeze pattern, separate
Adam optimiser over actor_head only, inert when ``n_steps == 0``) but
swaps the per-runner MSE loss on a continuous action vector for a
single cross-entropy term over the discrete action logits.

Hard constraints (``plans/rewrite/phase-8-oracle-bc-pretrain/
hard_constraints.md``):

- §5 Per-agent. The pretrainer takes a policy by reference and trains
  it in-place; never share weights across agents in the cohort.
- §6 ``actor_head`` only. All other parameters are frozen during
  ``pretrain`` and restored to ``requires_grad=True`` on exit. PPO's
  optimiser state is untouched (separate Adam instance on actor_head
  parameters only).
- §7 ``n_steps == 0`` is byte-identical to no-BC: empty samples or
  ``n_steps <= 0`` short-circuit before constructing the optimiser, so
  no parameter change is observable.
- §9 The loss is cross-entropy on the discrete action logits. Target
  for an oracle sample on runner R is the action index
  ``space.encode(ActionType.OPEN_BACK, R)``.

The pretrainer assumes the policy returns a
:class:`DiscretePolicyOutput` whose ``logits`` field is the raw
pre-mask categorical logits over ``space.n`` actions. BC does NOT
apply the legality mask — at training time obs come from the env and
the mask varies; at BC time we want the policy to learn an unmasked
preference for OPEN_BACK on the oracle-identified runner.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from agents_v2.action_space import ActionType, DiscreteActionSpace
from training_v2.arb_oracle import OracleSample, load_samples


logger = logging.getLogger(__name__)


__all__ = [
    "BCLossHistory",
    "DiscreteBCPretrainer",
    "load_oracle_samples_for_dates",
    "measure_post_bc_entropy",
]


def load_oracle_samples_for_dates(
    dates: list[str],
    data_dir,
    expected_obs_dim: int,
) -> list[OracleSample]:
    """Concatenate cached oracle samples across multiple training dates.

    Used by the cohort worker to assemble a per-agent BC pool. Days
    with no cache emit a warning and contribute zero samples — empty
    pool is a valid outcome (the BC step then returns an empty
    ``BCLossHistory`` and the warmup handshake stays inactive, which
    matches §7's "zero steps = byte-identical" intent).

    The strict ``expected_obs_dim`` check guards against silently
    feeding a pre-shim cache (or a cache produced under a different
    ``max_runners``) into BC and producing a shape mismatch deep in
    the training loop. Mismatch raises ``ValueError`` from
    ``load_samples`` — callers choosing to swallow it must do so
    explicitly.
    """
    from pathlib import Path
    out: list[OracleSample] = []
    for d in dates:
        try:
            samples = load_samples(
                str(d), Path(data_dir),
                strict=True, expected_obs_dim=int(expected_obs_dim),
            )
        except FileNotFoundError:
            logger.warning(
                "Oracle cache missing for %s; skipping (BC pool will "
                "be smaller). Run `python -m training_v2.oracle_cli "
                "scan --date %s` to populate.", d, d,
            )
            continue
        out.extend(samples)
    return out


@dataclass
class BCLossHistory:
    """Per-step CE loss trace returned by :meth:`DiscreteBCPretrainer.pretrain`.

    Empty when ``n_steps <= 0`` or ``samples`` is empty (the §7
    no-op contract).
    """

    ce_losses: list[float] = field(default_factory=list)
    final_ce_loss: float = 0.0


def _is_bc_target_param(name: str) -> bool:
    """True for ``actor_head`` parameters only.

    The v2 :class:`DiscreteLSTMPolicy` exposes the per-runner head as
    ``self.actor_head`` (``nn.Sequential``). Other heads
    (``noop_head``, ``stake_alpha_head``, ``stake_beta_head``,
    ``value_head``, ``fill_prob_head``, ``mature_prob_head``,
    ``risk_head``) live alongside it under different attribute names
    and stay frozen during BC.
    """
    return "actor_head" in name


def _sample_batch(
    samples: list[OracleSample],
    batch_size: int,
    rng: random.Random,
) -> list[OracleSample]:
    """Draw a random batch, sampling with replacement when pool < batch_size."""
    if len(samples) <= batch_size:
        return [rng.choice(samples) for _ in range(batch_size)]
    return rng.sample(samples, batch_size)


class DiscreteBCPretrainer:
    """Per-agent BC pretrainer for v2 discrete policies.

    Parameters
    ----------
    lr:
        Learning rate for the BC Adam optimiser. The PPO optimiser is
        untouched; this is a fresh instance over ``actor_head``
        parameters only.
    batch_size:
        Per-step mini-batch size. Drawn with replacement from the
        oracle sample pool (matches v1's ``random.choices`` semantics
        when the pool is smaller than the batch size).
    seed:
        Optional integer seed for the per-call sampler RNG. ``None``
        leaves the sampler unseeded (Python's default ``random``
        instance is not used so cross-test interactions can't leak in).
    """

    def __init__(
        self,
        lr: float = 3e-4,
        batch_size: int = 64,
        seed: int | None = None,
    ) -> None:
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self._seed = seed

    def pretrain(
        self,
        policy,
        samples: list[OracleSample],
        n_steps: int,
    ) -> BCLossHistory:
        """Run ``n_steps`` BC mini-batches against ``samples``.

        The policy is trained in-place. All non-actor_head parameters
        are frozen on entry and restored to ``requires_grad=True`` on
        exit (even when an exception fires mid-training — the restore
        is in a ``try / finally``).

        Samples whose ``runner_idx`` is out of the policy's action
        space are silently dropped (defensive guard; the v2 oracle
        already filters via ``runner_map.get`` so this should be
        empty on a healthy cache).
        """
        if not samples or n_steps <= 0:
            return BCLossHistory()

        action_space: DiscreteActionSpace = policy.action_space
        max_runners = int(action_space.max_runners)
        device = next(policy.parameters()).device

        valid = [s for s in samples if 0 <= int(s.runner_idx) < max_runners]
        if not valid:
            logger.warning(
                "DiscreteBCPretrainer.pretrain: no samples have "
                "runner_idx in [0, %d); skipping (returning empty "
                "history).",
                max_runners,
            )
            return BCLossHistory()

        rng = random.Random(self._seed)

        # Freeze all non-actor_head params; collect actor_head params
        # for the optimiser.
        frozen: list[torch.nn.Parameter] = []
        target_params: list[torch.nn.Parameter] = []
        for name, p in policy.named_parameters():
            if _is_bc_target_param(name):
                target_params.append(p)
            else:
                frozen.append(p)

        if not target_params:
            logger.warning(
                "DiscreteBCPretrainer.pretrain: policy has no "
                "actor_head parameters; nothing to train.",
            )
            return BCLossHistory()

        for p in frozen:
            p.requires_grad_(False)

        history = BCLossHistory()
        try:
            opt = torch.optim.Adam(target_params, lr=self.lr)

            for _ in range(int(n_steps)):
                batch = _sample_batch(valid, self.batch_size, rng)

                obs_t = torch.tensor(
                    np.stack([s.obs for s in batch], axis=0),
                    dtype=torch.float32,
                    device=device,
                )
                target_actions = torch.tensor(
                    [
                        action_space.encode(
                            ActionType.OPEN_BACK, int(s.runner_idx),
                        )
                        for s in batch
                    ],
                    dtype=torch.long,
                    device=device,
                )

                out = policy(obs_t)
                # Use raw (unmasked) logits — at BC time we want the
                # policy to learn a preference for OPEN_BACK on the
                # oracle's chosen runner regardless of the env-side
                # mask, which is unavailable / irrelevant here.
                loss = F.cross_entropy(out.logits, target_actions)

                opt.zero_grad()
                loss.backward()
                opt.step()

                history.ce_losses.append(float(loss.item()))

            if history.ce_losses:
                history.final_ce_loss = history.ce_losses[-1]
        finally:
            # Restore the freeze regardless of exception state.
            for p in frozen:
                p.requires_grad_(True)

        return history


def measure_post_bc_entropy(
    policy,
    samples: list[OracleSample],
    *,
    max_eval_samples: int = 256,
) -> float:
    """Mean Categorical entropy of the policy's action distribution.

    Called immediately after BC to seed the trainer's warmup
    handshake. Uses ``Categorical(logits=out.logits)`` — NOT the
    Normal distribution v1 measured (v1's policy emitted continuous
    action means; v2 emits a categorical over a finite action set).

    Returns 0.0 when ``samples`` is empty (nothing was BC'd, so the
    caller should leave ``trainer._post_bc_entropy`` unset and the
    warmup handshake stays inactive).
    """
    if not samples:
        return 0.0
    device = next(policy.parameters()).device
    batch = samples[:max_eval_samples]
    obs_t = torch.tensor(
        np.stack([s.obs for s in batch], axis=0),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        out = policy(obs_t)
        dist = Categorical(logits=out.logits)
        entropy = dist.entropy().mean().item()
    return float(entropy)
