"""``DiscretePPOTrainer`` — Phase 2, Session 02 deliverable.

Wraps Session 01's :class:`RolloutCollector` + :func:`compute_per_runner_gae`
into a self-contained PPO update path for the v2 discrete-action policy.

What this owns:

* The PPO surrogate loss (clip-ratio on the joint
  ``Categorical × Beta`` log-prob, with the stake log-prob masked out
  for NOOP / CLOSE actions).
* Per-runner value loss against the per-runner GAE return.
* Entropy bonus (categorical only — Beta entropy is intentionally not
  added; the stake head's role is sizing, not exploration).
* Per-mini-batch KL early-stop (CLAUDE.md §"Per-mini-batch KL check").
* Hidden-state pack / slice handshake with the policy
  (CLAUDE.md §"Recurrent PPO: hidden-state protocol on update").

What this does NOT own:

* Real-day training, multi-day epochs, GA / cohort, frontend events
  — Phase 2 Session 03 + Phase 3.
* Reward shaping. The trainer consumes whatever ``shim`` /
  ``BetfairEnv`` produced; per-runner attribution lives in
  :mod:`training_v2.discrete_ppo.rollout`.
* v1 stabilisers (advantage normalisation, LR warmup, reward
  centering, entropy controller) — rewrite hard constraint §6.

Hard constraints honoured here (rewrite README §3, phase-2 purpose
§"Hard constraints", session prompt §"Hard constraints"):

- No env edits.
- No re-import of ``agents/`` (v1) — read for reference, never import.
- The PPO update reads transitions from Session 01's
  :class:`Transition` verbatim; no shape changes, no field additions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Beta

from agents_v2.action_space import ActionType, DiscreteActionSpace
from agents_v2.discrete_policy import BaseDiscretePolicy
from agents_v2.env_shim import DiscreteActionShim
from training_v2.discrete_ppo.gae import compute_per_runner_gae
from training_v2.discrete_ppo.rollout import RolloutCollector
from training_v2.discrete_ppo.transition import (
    RolloutBatch,
    Transition,
    action_uses_stake,
    transitions_to_rollout_batch,
)

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


__all__ = [
    "DiscretePPOTrainer",
    "EpisodeStats",
    "UpdateLog",
    "build_chosen_advantage",
    "build_uses_stake_mask",
]


def _move_to_device(
    tensor: torch.Tensor,
    device: torch.device,
    non_blocking: bool = True,
) -> torch.Tensor:
    """Pin + non-blocking copy on CUDA, plain ``.to`` on CPU.

    Phase 3 Session 01 (GPU pathway). Mirrors v1's pinned-memory
    pattern in ``agents/ppo_trainer.py:2131-2174`` — pin on the CPU
    side, then transfer with ``non_blocking=True`` so the copy
    overlaps with downstream compute. ``pin_memory()`` errors on
    CPU-only torch builds, so the CPU branch short-circuits to a
    plain ``.to(device)``.
    """
    if device.type == "cuda":
        return tensor.pin_memory().to(device, non_blocking=non_blocking)
    return tensor.to(device)


# ── Public dataclasses ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class UpdateLog:
    """Per-update diagnostics returned by :meth:`DiscretePPOTrainer._ppo_update`.

    All means are over mini-batches that actually ran (i.e. excluding
    those skipped by the per-mini-batch KL early-stop).
    """

    n_updates_run: int
    policy_loss_mean: float
    value_loss_mean: float
    entropy_mean: float
    approx_kl_mean: float
    approx_kl_max: float
    mini_batches_skipped: int
    kl_early_stopped: bool


@dataclass(frozen=True)
class EpisodeStats:
    """Per-episode summary the trainer logs after one rollout + update."""

    total_reward: float
    n_steps: int
    n_updates_run: int
    policy_loss_mean: float
    value_loss_mean: float
    entropy_mean: float
    approx_kl_mean: float
    approx_kl_max: float
    mini_batches_skipped: int
    kl_early_stopped: bool
    wall_time_sec: float
    # Diagnostics added in Session 03 — populated when the trainer
    # observes them; defaults preserve construction-by-name in tests.
    action_histogram: dict[str, int] | None = None
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    advantage_max_abs: float = 0.0
    day_pnl: float = 0.0


# ── Helpers (exported for tests) ───────────────────────────────────────────


def build_uses_stake_mask(
    action_space: DiscreteActionSpace,
    action_idxs: np.ndarray,
) -> np.ndarray:
    """Per-step ``1.0`` iff the chosen action uses the Beta stake head.

    Same semantics as :func:`training_v2.discrete_ppo.transition.
    action_uses_stake` but vectorised across a rollout. The PPO
    update multiplies the new-policy stake log-prob by this mask so
    the placeholder ``log_prob_stake = 0.0`` stored on NOOP / CLOSE
    transitions cannot contribute to the surrogate gradient.
    """
    mask = np.zeros(action_idxs.shape[0], dtype=np.float32)
    for t, idx in enumerate(action_idxs):
        if action_uses_stake(action_space, int(idx)):
            mask[t] = 1.0
    return mask


def build_chosen_advantage(
    action_space: DiscreteActionSpace,
    action_idxs: np.ndarray,
    advantages: np.ndarray,
) -> np.ndarray:
    """Map per-step ``(action_idx, advantages)`` → chosen-runner scalar.

    For ``OPEN_BACK_i / OPEN_LAY_i / CLOSE_i`` the chosen-runner
    advantage is ``advantages[t, i]``. For ``NOOP`` we use
    ``advantages[t, :].mean()`` — the policy chose nothing, so the
    gradient signal is the average across runners (Phase 2 purpose
    §"Per-runner credit assignment").

    Returns shape ``(T,)`` float32.
    """
    if advantages.ndim != 2:
        raise ValueError(
            f"advantages must be (T, max_runners), got {advantages.shape}",
        )
    T = action_idxs.shape[0]
    if advantages.shape[0] != T:
        raise ValueError(
            f"advantages T={advantages.shape[0]} != action_idxs T={T}",
        )
    chosen = np.zeros(T, dtype=np.float32)
    for t in range(T):
        kind, runner = action_space.decode(int(action_idxs[t]))
        if kind is ActionType.NOOP:
            chosen[t] = float(advantages[t].mean())
        else:
            chosen[t] = float(advantages[t, int(runner)])
    return chosen


# ── DiscretePPOTrainer ──────────────────────────────────────────────────────


class DiscretePPOTrainer:
    """Drive one (rollout → GAE → PPO update) loop for the v2 policy.

    The trainer owns its own optimiser; callers construct the policy
    + shim + trainer and call :meth:`train_episode` once per episode.
    Multi-episode loops, day curriculums, GA / cohort scaffolding
    live in Session 03 / Phase 3.

    Parameters
    ----------
    policy:
        Phase-1 :class:`BaseDiscretePolicy` subclass.
    shim:
        :class:`DiscreteActionShim` over a constructed ``BetfairEnv``.
        The trainer calls :meth:`RolloutCollector.collect_episode`
        which forwards to ``shim.reset`` / ``shim.step`` — the caller
        should NOT pre-reset the env.
    learning_rate, gamma, gae_lambda, clip_range, entropy_coeff,
    value_coeff, ppo_epochs, mini_batch_size, max_grad_norm,
    kl_early_stop_threshold:
        Standard PPO knobs. Defaults match Phase 2 purpose.md's locked
        hyperparameter table.
    device:
        Torch device for the update path. ``"cpu"`` is the Phase 2
        default; ``"cuda"`` is supported but not exercised by tests.
    """

    def __init__(
        self,
        policy: BaseDiscretePolicy,
        shim: DiscreteActionShim,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
        max_grad_norm: float = 0.5,
        kl_early_stop_threshold: float = 0.15,
        device: str = "cpu",
    ) -> None:
        self.policy = policy
        self.shim = shim
        self.action_space = shim.action_space
        self.max_runners = shim.max_runners
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_range = float(clip_range)
        self.entropy_coeff = float(entropy_coeff)
        self.value_coeff = float(value_coeff)
        self.ppo_epochs = int(ppo_epochs)
        self.mini_batch_size = int(mini_batch_size)
        self.max_grad_norm = float(max_grad_norm)
        self.kl_early_stop_threshold = float(kl_early_stop_threshold)
        self.device = torch.device(device)

        self.policy.to(self.device)
        self.optimiser = torch.optim.Adam(
            self.policy.parameters(), lr=float(learning_rate),
        )

        # Bound to its own collector so the rollout-time forward pass
        # uses the same device as the update.
        self._collector = RolloutCollector(
            shim=self.shim, policy=self.policy, device=device,
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def train_episode(self) -> EpisodeStats:
        """Run one episode → GAE → PPO update; return per-episode stats."""
        t0 = time.perf_counter()
        batch = self._collector.collect_episode()
        last_info = self._collector.last_info
        return self._update_from_batch(
            batch=batch, last_info=last_info, t0=t0,
        )

    def update_from_rollout(
        self,
        transitions: list[Transition],
        last_info: dict,
    ) -> EpisodeStats:
        """Run GAE + PPO update from an externally-collected rollout.

        Used by the batched cohort runner
        (``training_v2.discrete_ppo.batched_rollout.BatchedRolloutCollector``)
        which produces per-agent transition lists in one shot. The
        post-rollout pipeline (GAE → PPO update → stats) is identical
        to :meth:`train_episode` — the only difference is who owns the
        rollout phase.

        Phase 4 Session 06 (2026-05-02): the legacy ``list[Transition]``
        input is adapted into a :class:`RolloutBatch` via
        :func:`transitions_to_rollout_batch` before flowing into the
        shared ``_update_from_batch`` pipeline. The sequential rollout
        path skips this adapter (its collector returns the batch
        directly).
        """
        batch = transitions_to_rollout_batch(transitions)
        return self._update_from_batch(
            batch=batch,
            last_info=last_info,
            t0=time.perf_counter(),
        )

    # ── Internal: post-rollout pipeline ────────────────────────────────────

    def _update_from_batch(
        self,
        batch: RolloutBatch,
        last_info: dict,
        t0: float,
    ) -> EpisodeStats:
        n_steps = int(batch.n_steps)
        if n_steps == 0:
            raise RuntimeError(
                "Rollout produced 0 transitions — the env did not "
                "produce any steps before terminating.",
            )

        # Phase 4 Session 06: pre-stacked arrays come straight from the
        # collector — no per-transition list comprehension here.
        rewards = batch.per_runner_reward
        values = batch.value_per_runner
        dones = batch.done

        total_reward = float(rewards.sum())

        # Bootstrap. Episodes from RolloutCollector always terminate
        # naturally (the loop is ``while not done``), so the final
        # transition's ``done`` is True and the bootstrap is zero. We
        # keep the explicit branch for the truncated-episode case
        # Session 03 may want.
        if bool(dones[-1]):
            bootstrap_value = np.zeros(self.max_runners, dtype=np.float32)
        else:
            bootstrap_value = self._bootstrap_value(batch)

        advantages, returns = compute_per_runner_gae(
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            dones=dones,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        # Diagnostics for Session 03 — computed BEFORE _ppo_update
        # mutates the policy. The action histogram comes from the
        # rollout's chosen action_idx; advantage stats summarise the
        # per-runner GAE output that the surrogate loss will consume.
        action_hist: dict[str, int] = {}
        action_idx_arr = batch.action_idx
        for i in range(n_steps):
            kind, _runner = self.action_space.decode(int(action_idx_arr[i]))
            action_hist[kind.name] = action_hist.get(kind.name, 0) + 1
        adv_mean = float(np.mean(advantages))
        adv_std = float(np.std(advantages))
        adv_max_abs = float(np.max(np.abs(advantages)))

        # Final-step day_pnl from the env (last transition was the
        # terminal step; the shim's info dict carries day_pnl).
        day_pnl = float((last_info or {}).get("day_pnl", 0.0))

        update_log = self._ppo_update(
            batch=batch,
            advantages=advantages,
            returns=returns,
        )

        wall = time.perf_counter() - t0
        stats = EpisodeStats(
            total_reward=total_reward,
            n_steps=n_steps,
            n_updates_run=update_log.n_updates_run,
            policy_loss_mean=update_log.policy_loss_mean,
            value_loss_mean=update_log.value_loss_mean,
            entropy_mean=update_log.entropy_mean,
            approx_kl_mean=update_log.approx_kl_mean,
            approx_kl_max=update_log.approx_kl_max,
            mini_batches_skipped=update_log.mini_batches_skipped,
            kl_early_stopped=update_log.kl_early_stopped,
            wall_time_sec=wall,
            action_histogram=action_hist,
            advantage_mean=adv_mean,
            advantage_std=adv_std,
            advantage_max_abs=adv_max_abs,
            day_pnl=day_pnl,
        )
        logger.info(
            "DiscretePPOTrainer episode: n_steps=%d n_updates=%d "
            "policy_loss=%.4f value_loss=%.4f entropy=%.4f "
            "approx_kl=%.4f total_reward=%.3f wall=%.2fs",
            stats.n_steps, stats.n_updates_run,
            stats.policy_loss_mean, stats.value_loss_mean,
            stats.entropy_mean, stats.approx_kl_mean,
            stats.total_reward, stats.wall_time_sec,
        )
        return stats

    # ── PPO update path ────────────────────────────────────────────────────

    def _ppo_update(
        self,
        batch: RolloutBatch,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> UpdateLog:
        """Run :pyattr:`ppo_epochs` × mini-batch updates over the rollout.

        Per the session prompt §2 the per-mini-batch KL check fires
        AFTER the optimiser step and breaks BOTH the inner mini-batch
        loop and the outer epoch loop. The count of skipped
        mini-batches is logged so the operator can see compute saved.

        Phase 4 Session 06 (2026-05-02): consumes a :class:`RolloutBatch`
        directly. The pre-Session-06 ``np.stack([tr.field for tr in
        transitions])`` pass is gone — the rollout collector already
        emits the per-tick fields as contiguous arrays.
        """
        T = int(batch.n_steps)
        device = self.device

        # ── Source rollout arrays from the batch directly ───────────────
        # ``batch.{obs,mask,...}`` are slice views into the rollout's
        # per-episode buffers (Phase 4 Sessions 02 / 06). We re-read
        # the same slot every PPO update for a single rollout, but the
        # rollout doesn't mutate those slots after collect_episode
        # returns, so the views stay stable.
        action_idx_np = batch.action_idx
        uses_stake_np = build_uses_stake_mask(self.action_space, action_idx_np)
        chosen_adv_np = build_chosen_advantage(
            self.action_space, action_idx_np, advantages,
        )
        # Joint old log-prob — masked stake placeholder is forced to 0
        # by ``uses_stake_np`` so NOOP / CLOSE transitions ratio purely
        # off the categorical log-prob.
        joint_old_lp_np = (
            batch.log_prob_action + uses_stake_np * batch.log_prob_stake
        ).astype(np.float32, copy=False)

        # ``np.ascontiguousarray`` is a no-op when the input already
        # owns contiguous storage; here it covers the slice-view case
        # so ``torch.from_numpy`` doesn't refuse a non-contiguous
        # input. The episode-long buffers are C-contiguous, and
        # ``[:n_steps]`` is contiguous along axis 0, so this is the
        # zero-copy path in practice.
        obs = _move_to_device(
            torch.from_numpy(np.ascontiguousarray(batch.obs)), device,
        )
        masks = _move_to_device(
            torch.from_numpy(np.ascontiguousarray(batch.mask)), device,
        )
        action_idx = _move_to_device(
            torch.from_numpy(np.ascontiguousarray(action_idx_np)), device,
        )
        stake_unit = _move_to_device(
            torch.from_numpy(np.ascontiguousarray(batch.stake_unit)),
            device,
        )
        uses_stake = _move_to_device(torch.from_numpy(uses_stake_np), device)
        chosen_adv = _move_to_device(torch.from_numpy(chosen_adv_np), device)
        joint_old_lp = _move_to_device(torch.from_numpy(joint_old_lp_np), device)
        returns_t = _move_to_device(torch.from_numpy(returns), device)

        # ── Pack per-transition hidden states ──────────────────────────
        # ppo-kl-fix protocol: Phase 1's policy class owns the batch-
        # axis convention via pack_hidden_states / pack_hidden_buffer;
        # we don't peek inside.
        # Phase 3 Session 01b: states arrive from rollout already on
        # the trainer's device (no per-tick CUDA→CPU sync).
        # Phase 4 Session 06: the batch carries pre-stacked
        # ``(n_steps, *element_shape)`` buffers. ``pack_hidden_buffer``
        # converts to the policy-specific packed form via view-only
        # ops (squeeze + permute) — no per-tick concat over N small
        # slices.
        packed_hidden = self.policy.pack_hidden_buffer(batch.hidden_state_in)

        # ── Mini-batch loop ────────────────────────────────────────────
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        approx_kls: list[float] = []
        mini_batches_skipped = 0
        kl_early_stopped = False
        mini_batches_per_epoch = (T + self.mini_batch_size - 1) // self.mini_batch_size

        was_training = self.policy.training
        self.policy.train()
        try:
            for epoch_idx in range(self.ppo_epochs):
                indices = torch.randperm(T, device=device)
                for mb_pos, start in enumerate(
                    range(0, T, self.mini_batch_size),
                ):
                    end = min(start + self.mini_batch_size, T)
                    mb_idx = indices[start:end]

                    mb_obs = obs[mb_idx]
                    mb_mask = masks[mb_idx]
                    mb_action = action_idx[mb_idx]
                    mb_stake = stake_unit[mb_idx]
                    mb_uses_stake = uses_stake[mb_idx]
                    mb_chosen_adv = chosen_adv[mb_idx]
                    mb_old_lp = joint_old_lp[mb_idx]
                    mb_returns = returns_t[mb_idx]

                    mb_hidden = self.policy.slice_hidden_states(
                        packed_hidden, mb_idx,
                    )

                    out = self.policy(
                        mb_obs, hidden_state=mb_hidden, mask=mb_mask,
                    )

                    # Categorical log-prob at the rollout-time action.
                    new_lp_action = out.action_dist.log_prob(mb_action)

                    # Beta log-prob at the rollout-time stake. Multiply
                    # by the uses_stake mask so NOOP / CLOSE rows
                    # contribute zero to the joint — the placeholder
                    # log_prob_stake = 0 stored on those transitions
                    # must NOT push gradient through stake_alpha_head /
                    # stake_beta_head.
                    stake_dist = Beta(out.stake_alpha, out.stake_beta)
                    new_lp_stake = stake_dist.log_prob(mb_stake)
                    new_lp_joint = (
                        new_lp_action + mb_uses_stake * new_lp_stake
                    )

                    ratio = torch.exp(new_lp_joint - mb_old_lp)
                    surr1 = ratio * mb_chosen_adv
                    surr2 = torch.clamp(
                        ratio,
                        1.0 - self.clip_range,
                        1.0 + self.clip_range,
                    ) * mb_chosen_adv
                    surrogate = torch.min(surr1, surr2).mean()
                    policy_loss = -surrogate

                    # Per-runner value loss — mean over (T, R) is the
                    # textbook recipe. value_coeff folds in here so the
                    # total loss assembly below matches the spec exactly.
                    value_mse = ((out.value_per_runner - mb_returns) ** 2).mean()
                    value_loss = self.value_coeff * value_mse

                    # Categorical entropy only — Beta entropy is
                    # intentionally not added; the stake head's role is
                    # sizing, not exploration (purpose.md §"PPO algorithm
                    # shape").
                    entropy = out.action_dist.entropy().mean()

                    total_loss = (
                        policy_loss + value_loss - self.entropy_coeff * entropy
                    )

                    self.optimiser.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.max_grad_norm,
                    )
                    self.optimiser.step()

                    # Per-mini-batch KL check (CLAUDE.md §"Per-mini-
                    # batch KL check"). The forward pass above is
                    # mid-tape but ``new_lp_joint`` is already
                    # computed, so we just detach for the diagnostic.
                    with torch.no_grad():
                        approx_kl = float(
                            (mb_old_lp - new_lp_joint.detach()).mean().item()
                        )

                    policy_losses.append(float(policy_loss.item()))
                    value_losses.append(float(value_loss.item()))
                    entropies.append(float(entropy.item()))
                    approx_kls.append(approx_kl)

                    if approx_kl > self.kl_early_stop_threshold:
                        # Tally remaining mini-batches in this epoch
                        # plus full mini-batches in all skipped epochs.
                        remaining_in_epoch = max(
                            0, mini_batches_per_epoch - (mb_pos + 1),
                        )
                        remaining_epochs = self.ppo_epochs - (epoch_idx + 1)
                        mini_batches_skipped = (
                            remaining_in_epoch
                            + remaining_epochs * mini_batches_per_epoch
                        )
                        kl_early_stopped = True
                        logger.info(
                            "DiscretePPOTrainer KL early-stop "
                            "epoch=%d mb_pos=%d approx_kl=%.4f > "
                            "threshold=%.4f (skipping %d remaining "
                            "mini-batches across %d epoch(s))",
                            epoch_idx, mb_pos, approx_kl,
                            self.kl_early_stop_threshold,
                            mini_batches_skipped,
                            1 + remaining_epochs,
                        )
                        break
                if kl_early_stopped:
                    break
        finally:
            self.policy.train(was_training)

        n_run = len(policy_losses)
        return UpdateLog(
            n_updates_run=n_run,
            policy_loss_mean=float(np.mean(policy_losses)) if n_run else 0.0,
            value_loss_mean=float(np.mean(value_losses)) if n_run else 0.0,
            entropy_mean=float(np.mean(entropies)) if n_run else 0.0,
            approx_kl_mean=float(np.mean(approx_kls)) if n_run else 0.0,
            approx_kl_max=float(np.max(approx_kls)) if n_run else 0.0,
            mini_batches_skipped=mini_batches_skipped,
            kl_early_stopped=kl_early_stopped,
        )

    # ── Internals ──────────────────────────────────────────────────────────

    def _bootstrap_value(self, batch: RolloutBatch) -> np.ndarray:
        """Forward pass on the post-terminal observation for the bootstrap.

        Only called when the last transition has ``done=False`` (i.e.
        a truncated episode). RolloutCollector currently terminates
        only on natural ``done=True``, so this path is dormant in
        Session 02 — kept for the Session 03 truncated-episode case.

        Phase 4 Session 06 (2026-05-02): reads the final tick straight
        from the :class:`RolloutBatch` instead of a per-tick
        ``Transition``.
        """
        device = self.device
        last = int(batch.n_steps) - 1
        obs_t = torch.from_numpy(
            np.ascontiguousarray(batch.obs[last])
        ).to(device, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.from_numpy(
            np.ascontiguousarray(batch.mask[last])
        ).to(device).unsqueeze(0)
        # Phase 3 Session 01b: hidden_state_in is already a tuple of
        # device-resident tensors — no torch.from_numpy / .to(device).
        # Phase 4 Session 06: the batch's hidden_state_in is the
        # pre-stacked ``(n_steps, *element_shape)`` buffer; slicing
        # ``buf[last]`` reproduces the per-tick element shape that
        # the policy.forward expects.
        hidden_in = tuple(buf[last] for buf in batch.hidden_state_in)
        was_training = self.policy.training
        self.policy.eval()
        try:
            with torch.no_grad():
                out = self.policy(obs_t, hidden_state=hidden_in, mask=mask_t)
                return (
                    out.value_per_runner.detach().squeeze(0).cpu().numpy()
                    .astype(np.float32)
                )
        finally:
            self.policy.train(was_training)
