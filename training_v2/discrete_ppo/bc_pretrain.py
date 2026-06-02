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
from training_v2.arb_oracle import (
    CloseHoldSample,
    NegativeOracleSample,
    OracleSample,
    TARGET_CLASS_CLOSE,
    TARGET_CLASS_HOLD,
    load_close_hold_samples,
    load_negative_samples,
    load_samples,
)
from training_v2.direction_label_scan import (
    DirectionLabel,
    load_labels as _load_direction_labels,
)


logger = logging.getLogger(__name__)


__all__ = [
    "BCLossHistory",
    "DiscreteBCPretrainer",
    "build_direction_bce_label_map",
    "build_direction_target_map",
    "load_close_hold_samples_for_dates",
    "load_direction_labels_for_dates",
    "load_negative_samples_for_dates",
    "load_oracle_samples_for_dates",
    "measure_post_bc_entropy",
]


# Phase-15 S02 amendment (2026-05-08): the BC pretrainer was
# training ONLY ``actor_head``. ``direction_prob_head`` and its
# LayerNorm were frozen, so the cached direction labels never
# pulled the predictor toward calibration — the head sat at the
# balanced no-skill baseline (BCE ~1.05) regardless of how
# aggressively we set ``direction_prob_loss_weight`` during PPO.
# This name list expands the BC-trainable set to include the
# direction predictor + its LayerNorm so direct supervised
# BCE-with-logits gradient lands on those weights during BC.
# Pre-amendment behaviour is preserved when
# ``direction_bce_weight == 0`` — only actor_head sees gradient.
_BC_TARGET_NAMES: tuple[str, ...] = (
    "actor_head",
    "direction_prob_head",
)


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


def build_direction_bce_label_map(
    labels: list[DirectionLabel],
    action_space: DiscreteActionSpace,
) -> dict[tuple[int, int], tuple[float, float]]:
    """Project DirectionLabel rows onto a map
    ``{(tick_index, runner_idx): (label_back, label_lay)}``.

    Phase-15 S02 amendment (2026-05-08). Whereas
    :func:`build_direction_target_map` collapses the binary labels
    into a single action choice (and silently drops ambiguous
    rows), this function preserves the raw 2-channel labels
    needed to compute BCE-with-logits on
    ``direction_back_logits_per_runner`` and
    ``direction_lay_logits_per_runner`` separately. ALL rows
    contribute (including ambiguous ``(0, 0)`` and ``(1, 1)`` —
    those are valid targets for the predictor, just not for an
    actor action choice).

    Rows whose ``runner_idx`` is outside the policy's action
    space are silently dropped (defensive guard).
    """
    out: dict[tuple[int, int], tuple[float, float]] = {}
    max_runners = int(action_space.max_runners)
    for r in labels:
        slot = int(r.runner_idx)
        if slot < 0 or slot >= max_runners:
            continue
        out[(int(r.tick_index), slot)] = (
            float(r.label_back), float(r.label_lay),
        )
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


def load_negative_samples_for_dates(
    dates: list[str],
    data_dir,
    expected_obs_dim: int,
) -> list[NegativeOracleSample]:
    """Concatenate cached negative-open oracle samples across dates.

    BC label augmentation Phase A. Mirrors
    :func:`load_oracle_samples_for_dates` precisely. Missing-cache-
    per-day emits a warning (via the underlying ``load_negative_
    samples``) and contributes zero negatives — the BC pretrain code
    handles an empty negative pool as "treat exactly like the
    pre-plan positive-only path".

    The strict ``expected_obs_dim`` check guards against silently
    feeding a stale or mis-sized cache into BC.
    """
    from pathlib import Path
    out: list[NegativeOracleSample] = []
    for d in dates:
        # load_negative_samples returns [] on missing cache (does NOT
        # raise) so the same try/except pattern as the positive loader
        # isn't strictly needed — but keep the symmetric shape so the
        # callsite reads identically.
        samples = load_negative_samples(
            str(d), Path(data_dir),
            strict=True, expected_obs_dim=int(expected_obs_dim),
        )
        out.extend(samples)
    return out


def load_close_hold_samples_for_dates(
    dates: list[str],
    data_dir,
    expected_obs_dim: int,
) -> list[CloseHoldSample]:
    """Concatenate cached Phase B close/hold samples across dates.

    BC label augmentation Phase B. Mirrors
    :func:`load_negative_samples_for_dates` precisely. Missing-cache-
    per-day emits a warning (via the underlying
    ``load_close_hold_samples``) and contributes zero samples.

    The strict ``expected_obs_dim`` check guards against silently
    feeding a stale or mis-sized cache into BC.
    """
    from pathlib import Path
    out: list[CloseHoldSample] = []
    for d in dates:
        samples = load_close_hold_samples(
            str(d), Path(data_dir),
            strict=True, expected_obs_dim=int(expected_obs_dim),
        )
        out.extend(samples)
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
    """True for parameters BC is allowed to update.

    Phase-8 S02 (original): ``actor_head`` only — all other heads
    stayed frozen during BC. Phase-15 S02 amendment expands the
    target set to also cover ``direction_prob_head`` (LayerNorm +
    2-layer MLP) so direct BCE-with-logits supervised gradient on
    the cached direction labels can calibrate the predictor before
    PPO starts. Other heads (``noop_head``, ``stake_alpha_head``,
    ``stake_beta_head``, ``value_head``, ``fill_prob_head``,
    ``mature_prob_head``, ``risk_head``) and the LSTM backbone /
    ``input_proj`` / ``runner_slot_embedding`` stay frozen.
    """
    return any(t in name for t in _BC_TARGET_NAMES)


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
        direction_bce_label_map: (
            dict[tuple[int, int], tuple[float, float]] | None
        ) = None,
        direction_bce_weight: float = 0.0,
        direction_bce_use_pos_weight: bool = True,
        negative_samples: list[NegativeOracleSample] | None = None,
        positive_weight: float = 1.0,
        close_hold_samples: list[CloseHoldSample] | None = None,
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

        # BC label augmentation Phase A: filter the negative pool by
        # the same defensive runner_idx guard. When ``negative_samples``
        # is None or empty, ``neg_active`` is False and every code
        # path that touches negatives is gated off — the loss surface,
        # the optimiser sequence, and the per-step CE history are
        # byte-identical to pre-plan.
        valid_negatives: list[NegativeOracleSample] = []
        if negative_samples:
            valid_negatives = [
                s for s in negative_samples
                if 0 <= int(s.runner_idx) < max_runners
            ]
        neg_active = bool(valid_negatives)
        pos_weight = float(positive_weight) if positive_weight is not None else 1.0
        # Per-step negative draw: match the positive batch size so the
        # CE on negatives is the same order of magnitude as the CE on
        # positives. The pool ratio (negatives:positives) only matters
        # for which subset of the negative cache the policy sees; each
        # mini-batch is balanced regardless.
        n_neg_per_batch = self.batch_size if neg_active else 0
        noop_idx = int(action_space.encode(ActionType.NOOP, None))

        # BC label augmentation Phase B: filter the close/hold pool by
        # the same defensive runner_idx guard as positives + negatives.
        # When ``close_hold_samples`` is None/empty, ``ch_active`` is
        # False and every code path that touches close/hold samples is
        # gated off — byte-identical to pre-plan (and to Phase A when
        # negatives are also off).
        valid_close_hold: list[CloseHoldSample] = []
        if close_hold_samples:
            valid_close_hold = [
                s for s in close_hold_samples
                if 0 <= int(s.runner_idx) < max_runners
            ]
        ch_active = bool(valid_close_hold)
        n_ch_per_batch = self.batch_size if ch_active else 0

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

        # Phase-15 S02 amendment: direct BCE-with-logits on
        # direction_prob_head against the cached binary labels.
        # Trains the predictor itself (independent of any actor-CE
        # signal). Default off (weight=0) → byte-identical to
        # phase-13 S05.
        dir_bce_active = (
            direction_bce_label_map is not None
            and direction_bce_weight > 0.0
        )
        dir_bce_w = (
            float(direction_bce_weight) if dir_bce_active else 0.0
        )

        # Phase-15 S02 amendment 2: per-class pos_weight to rebalance
        # the imbalanced labels. The PPO-time aux BCE in trainer.py
        # already uses pos_weight; the BC pretrain didn't. Standard
        # BCE on ~22% positive labels biases the predictor toward
        # "always 0" because the loss is dominated by easy negatives.
        # ``pos_weight = N_neg / N_pos`` rebalances. Computed once
        # over the entire label pool to avoid per-step recomputation.
        dir_bce_pos_weight_back: torch.Tensor | None = None
        dir_bce_pos_weight_lay: torch.Tensor | None = None
        if dir_bce_active and direction_bce_use_pos_weight:
            n_pos_back = 0
            n_neg_back = 0
            n_pos_lay = 0
            n_neg_lay = 0
            for _back, _lay in direction_bce_label_map.values():
                if _back > 0.5:
                    n_pos_back += 1
                else:
                    n_neg_back += 1
                if _lay > 0.5:
                    n_pos_lay += 1
                else:
                    n_neg_lay += 1
            # Cap pos_weight to avoid extreme values when one class is
            # almost empty (e.g. ratio 100+ would over-amplify a
            # handful of positives into a noisy loss).
            pw_back = min(
                10.0,
                float(n_neg_back) / max(float(n_pos_back), 1.0),
            )
            pw_lay = min(
                10.0,
                float(n_neg_lay) / max(float(n_pos_lay), 1.0),
            )
            dir_bce_pos_weight_back = torch.tensor(
                pw_back, dtype=torch.float32, device=device,
            )
            dir_bce_pos_weight_lay = torch.tensor(
                pw_lay, dtype=torch.float32, device=device,
            )
            logger.info(
                "BC direction-BCE pos_weight: back=%.2f (pos=%d/neg=%d) "
                "lay=%.2f (pos=%d/neg=%d)",
                pw_back, n_pos_back, n_neg_back,
                pw_lay, n_pos_lay, n_neg_lay,
            )
        elif dir_bce_active:
            logger.info(
                "BC direction-BCE pos_weight DISABLED (vanilla BCE).",
            )

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

                # Phase-15 S02: direct supervised BCE on
                # direction_prob_head against the cached binary
                # direction labels. Layered ON TOP of oracle / direction
                # action CE — they pull actor_head; this term pulls
                # direction_prob_head.
                if dir_bce_active:
                    bb_logits = out.direction_back_logits_per_runner  # (mb, R)
                    bl_logits = out.direction_lay_logits_per_runner   # (mb, R)
                    label_back_list: list[float] = []
                    label_lay_list: list[float] = []
                    bce_mask_list: list[bool] = []
                    sample_runner_idx: list[int] = []
                    for s in batch:
                        key = (int(s.tick_index), int(s.runner_idx))
                        labels = direction_bce_label_map.get(key)
                        if labels is None:
                            label_back_list.append(0.0)
                            label_lay_list.append(0.0)
                            bce_mask_list.append(False)
                        else:
                            label_back_list.append(float(labels[0]))
                            label_lay_list.append(float(labels[1]))
                            bce_mask_list.append(True)
                        sample_runner_idx.append(int(s.runner_idx))
                    bce_mask_t = torch.tensor(
                        bce_mask_list, dtype=torch.float32, device=device,
                    )
                    n_bce = float(bce_mask_t.sum().item())
                    if n_bce > 0.0:
                        runner_idx_t = torch.tensor(
                            sample_runner_idx, dtype=torch.long,
                            device=device,
                        )
                        rows = torch.arange(len(batch), device=device)
                        # Pick the per-sample runner's logits.
                        bb_per_sample = bb_logits[rows, runner_idx_t]  # (mb,)
                        bl_per_sample = bl_logits[rows, runner_idx_t]  # (mb,)
                        target_back = torch.tensor(
                            label_back_list, dtype=torch.float32,
                            device=device,
                        )
                        target_lay = torch.tensor(
                            label_lay_list, dtype=torch.float32,
                            device=device,
                        )
                        bce_back = F.binary_cross_entropy_with_logits(
                            bb_per_sample, target_back, reduction="none",
                            pos_weight=dir_bce_pos_weight_back,
                        )
                        bce_lay = F.binary_cross_entropy_with_logits(
                            bl_per_sample, target_lay, reduction="none",
                            pos_weight=dir_bce_pos_weight_lay,
                        )
                        denom = bce_mask_t.sum().clamp(min=1.0)
                        dir_bce_term = (
                            ((bce_back + bce_lay) * bce_mask_t).sum()
                            / denom
                        )
                        loss = loss + dir_bce_w * dir_bce_term

                # BC label augmentation Phase A. When negative samples
                # are present, draw a balanced negative mini-batch this
                # step, target NOOP for each row, and add a NOOP-CE
                # term to the loss. When ``negative_samples`` was None
                # or empty on entry, ``neg_active`` is False and this
                # block is a no-op — the loss surface and the
                # optimiser step are byte-identical to pre-plan. The
                # ``positive_weight`` parameter scales the positive
                # (oracle + direction + dir_bce) loss; default 1.0
                # preserves byte-identity when neg_active is False.
                if neg_active:
                    if pos_weight != 1.0:
                        loss = pos_weight * loss
                    neg_batch = _sample_batch(
                        valid_negatives, n_neg_per_batch, rng,
                    )
                    neg_obs_t = torch.tensor(
                        np.stack([s.obs for s in neg_batch], axis=0),
                        dtype=torch.float32,
                        device=device,
                    )
                    neg_targets = torch.full(
                        (len(neg_batch),), noop_idx,
                        dtype=torch.long, device=device,
                    )
                    neg_out = policy(neg_obs_t)
                    neg_ce = F.cross_entropy(neg_out.logits, neg_targets)
                    loss = loss + neg_ce

                # BC label augmentation Phase B. When close/hold samples
                # are present, draw a balanced close/hold mini-batch
                # this step. Per-row target = CLOSE on the sample's
                # runner_idx if target_action_class == TARGET_CLASS_
                # CLOSE, else NOOP (slot-less). The CE runs over the
                # mixed mini-batch with per-row targets. When
                # ``close_hold_samples`` was None or empty on entry,
                # ``ch_active`` is False and this block is a no-op —
                # byte-identical to pre-plan (Phase A behavior).
                #
                # Note: ``pos_weight`` was already applied to ``loss``
                # in the negative-sample block above when both phases
                # are active; we do NOT apply it a second time here.
                # When only Phase B is active (negatives off, close/
                # hold on), we apply pos_weight here so the positive
                # CE has the same weighting contract as Phase A.
                if ch_active:
                    if not neg_active and pos_weight != 1.0:
                        loss = pos_weight * loss
                    ch_batch = _sample_batch(
                        valid_close_hold, n_ch_per_batch, rng,
                    )
                    ch_obs_t = torch.tensor(
                        np.stack([s.obs for s in ch_batch], axis=0),
                        dtype=torch.float32,
                        device=device,
                    )
                    ch_targets_list: list[int] = []
                    for s in ch_batch:
                        if int(s.target_action_class) == int(
                            TARGET_CLASS_CLOSE,
                        ):
                            ch_targets_list.append(
                                int(action_space.encode(
                                    ActionType.CLOSE, int(s.runner_idx),
                                ))
                            )
                        else:
                            ch_targets_list.append(noop_idx)
                    ch_targets = torch.tensor(
                        ch_targets_list, dtype=torch.long, device=device,
                    )
                    ch_out = policy(ch_obs_t)
                    ch_ce = F.cross_entropy(ch_out.logits, ch_targets)
                    loss = loss + ch_ce

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
