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
from training_v2.direction_label_scan import (
    DirectionLabel,
    load_labels as _load_direction_labels,
)


logger = logging.getLogger(__name__)


__all__ = [
    "BCLossHistory",
    "DiscreteBCPretrainer",
    "build_direction_target_map",
    "load_direction_labels_for_dates",
    "load_oracle_samples_for_dates",
    "measure_post_bc_entropy",
]


def load_direction_labels_for_dates(
    dates: list[str],
    data_dir,
    *,
    direction_horizon_ticks: int,
    direction_threshold_ticks: int,
    force_close_before_off_seconds: float,
    strict: bool = True,
) -> dict[str, list[DirectionLabel]]:
    """Concatenate direction-label caches across multiple dates.

    Returns a map ``{date: [DirectionLabel, ...]}``. Days with no
    cache emit a warning and contribute an empty list — direction
    BC then has no per-(tick, runner) target for that day's tick
    indices and the BC loss term collapses to oracle-only on that
    day.

    Phase-13 S05 deliverable. The strict header check (matching the
    three label-defining knobs) catches mismatched cache invocations
    early — cohorts launched at one threshold cannot accidentally
    consume labels generated at another.
    """
    from pathlib import Path
    out: dict[str, list[DirectionLabel]] = {}
    for d in dates:
        try:
            labels = _load_direction_labels(
                str(d),
                Path(data_dir),
                direction_horizon_ticks=direction_horizon_ticks,
                direction_threshold_ticks=direction_threshold_ticks,
                force_close_before_off_seconds=(
                    force_close_before_off_seconds
                ),
                strict=strict,
            )
        except FileNotFoundError:
            logger.warning(
                "Direction-label cache missing for %s; direction BC "
                "loss will collapse to oracle-only on this day. Run "
                "`python -m training_v2.direction_label_cli scan "
                "--date %s ...` to populate.", d, d,
            )
            out[d] = []
            continue
        out[d] = labels
    return out


def build_direction_target_map(
    labels: list[DirectionLabel],
    action_space: DiscreteActionSpace,
) -> dict[tuple[int, int], int]:
    """Project a list of :class:`DirectionLabel` rows onto a map
    ``{(tick_index, runner_idx): target_action_idx}``.

    The target action is determined by the label tuple
    ``(label_back, label_lay)`` per phase-13 S05 D2:

    - ``(1, 0)`` → ``OPEN_BACK`` at runner_idx (back-first scalp).
    - ``(0, 1)`` → ``OPEN_LAY`` at runner_idx (lay-first scalp).
    - ``(0, 0)`` or ``(1, 1)`` → entry omitted (no direction
      pressure — ambiguous or no signal).

    Rows whose ``runner_idx`` is outside the policy's action space
    are silently dropped (defensive guard).
    """
    out: dict[tuple[int, int], int] = {}
    max_runners = int(action_space.max_runners)
    for r in labels:
        slot = int(r.runner_idx)
        if slot < 0 or slot >= max_runners:
            continue
        b = float(r.label_back) > 0.5
        l = float(r.label_lay) > 0.5
        if b and not l:
            out[(int(r.tick_index), slot)] = action_space.encode(
                ActionType.OPEN_BACK, slot,
            )
        elif l and not b:
            out[(int(r.tick_index), slot)] = action_space.encode(
                ActionType.OPEN_LAY, slot,
            )
        # else: ambiguous or no signal — entry omitted.
    return out


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
        *,
        direction_target_map: dict[tuple[int, int], int] | None = None,
        direction_target_weight: float = 0.0,
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

        # Phase-13 S05 — direction-targeted BC layered with the
        # oracle target. ``direction_target_weight`` interpolates
        # between the oracle CE (alpha = 1 - w) and the direction
        # CE (alpha = w). When the (tick, runner) has no direction
        # entry — both labels 0 OR both 1 — the direction CE is
        # skipped for that sample and the layered loss collapses to
        # oracle-only on that row.
        dir_active = (
            direction_target_map is not None
            and direction_target_weight > 0.0
        )
        dir_w = float(direction_target_weight) if dir_active else 0.0
        oracle_w = 1.0 - dir_w if dir_active else 1.0

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
                oracle_target_actions = torch.tensor(
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
                oracle_ce = F.cross_entropy(
                    out.logits, oracle_target_actions,
                )
                if dir_active:
                    # Build a per-row direction-target tensor + a
                    # boolean mask of rows where the cache had an
                    # unambiguous direction signal. Rows without a
                    # direction entry contribute zero to the
                    # direction CE term (mask out before reduction).
                    dir_targets_list: list[int] = []
                    dir_mask_list: list[bool] = []
                    for s in batch:
                        key = (int(s.tick_index), int(s.runner_idx))
                        tgt = direction_target_map.get(key)
                        if tgt is None:
                            # No direction signal — placeholder that
                            # gets masked out before reduction.
                            dir_targets_list.append(
                                int(oracle_target_actions[
                                    len(dir_targets_list)
                                ].item())
                            )
                            dir_mask_list.append(False)
                        else:
                            dir_targets_list.append(int(tgt))
                            dir_mask_list.append(True)
                    dir_targets = torch.tensor(
                        dir_targets_list,
                        dtype=torch.long,
                        device=device,
                    )
                    dir_mask = torch.tensor(
                        dir_mask_list,
                        dtype=torch.float32,
                        device=device,
                    )
                    if float(dir_mask.sum().item()) > 0.0:
                        per_row_ce = F.cross_entropy(
                            out.logits, dir_targets, reduction="none",
                        )
                        direction_ce = (
                            (per_row_ce * dir_mask).sum()
                            / dir_mask.sum().clamp(min=1.0)
                        )
                    else:
                        # Empty match in this mini-batch — no direction
                        # gradient contribution this step. ``zeros_like``
                        # keeps the autograd graph attached so the
                        # combined loss can backprop cleanly.
                        direction_ce = oracle_ce.detach() * 0.0
                    loss = oracle_w * oracle_ce + dir_w * direction_ce
                else:
                    loss = oracle_ce

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
