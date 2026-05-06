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
from training_v2.discrete_ppo.aux_labels import (
    assign_per_transition_labels,
)
from training_v2.direction_label_scan import (
    DirectionLabel,
    load_labels as _load_direction_labels,
)
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


def _materialise_direction_grid(
    *,
    day,
    labels: list[DirectionLabel],
    max_runners: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project a list of :class:`DirectionLabel` rows onto a per-env-
    step grid aligned with the env's deterministic tick walk.

    Returns
    -------
    label_grid:
        ``(n_env_steps, max_runners, 2)`` float32. Columns
        ``[..., 0]`` carry ``label_back``; ``[..., 1]`` carry
        ``label_lay``. Zero on rows / runners with no cache row
        (in-play ticks, non-priceable runners).
    mask_grid:
        ``(n_env_steps, max_runners)`` bool. ``True`` only at rows
        the cache emitted (priceable runner at a pre-race tick) so
        the BCE term is computed only on supervised cells.
    env_idx_arr:
        ``(n_env_steps,)`` int32. Diagnostic — pre-race global tick
        index per env step (-1 for in-play steps).
    """
    n_env_steps = sum(len(race.ticks) for race in day.races)
    label_grid = np.zeros(
        (n_env_steps, max_runners, 2), dtype=np.float32,
    )
    mask_grid = np.zeros((n_env_steps, max_runners), dtype=bool)
    env_idx_arr = np.full(n_env_steps, -1, dtype=np.int32)

    # Group labels by global tick for O(1) per-cell lookup.
    by_tick: dict[int, list[DirectionLabel]] = {}
    for r in labels:
        by_tick.setdefault(int(r.tick_index), []).append(r)

    env_step = 0
    global_pre_race = 0
    for race in day.races:
        for tick in race.ticks:
            if not tick.in_play:
                env_idx_arr[env_step] = global_pre_race
                rows = by_tick.get(global_pre_race, [])
                for row in rows:
                    slot = int(row.runner_idx)
                    if 0 <= slot < max_runners:
                        label_grid[env_step, slot, 0] = row.label_back
                        label_grid[env_step, slot, 1] = row.label_lay
                        mask_grid[env_step, slot] = True
                global_pre_race += 1
            env_step += 1
    return label_grid, mask_grid, env_idx_arr


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
    # Phase 7 Session 02 (2026-05-04). Aux head losses, mean over
    # mini-batches that ran. Always populated; zero when the
    # corresponding weight is 0.0 OR the rollout's aux_labels is
    # None (synthetic / legacy batch).
    fill_prob_bce_mean: float = 0.0
    mature_prob_bce_mean: float = 0.0
    risk_nll_mean: float = 0.0
    # Phase 9 Session 02 (2026-05-05). When per-transition credit is
    # active, sums the count of mini-batch entries whose
    # ``mature_mask`` is True across every mini-batch that ran in
    # this update. Zero when the flag is off OR the rollout had no
    # opens (a long all-NOOP rollout). On a healthy rollout this
    # should be roughly ``ppo_epochs * pairs_opened`` because each
    # open transition appears once per epoch's shuffled mini-batch
    # pass — see test_n_mature_targets_nonzero_when_pairs_mature.
    n_mature_targets: int = 0
    # Phase-13 Session 03 (2026-05-06). Per-side direction BCE means.
    # Zero when ``direction_prob_loss_weight == 0`` OR the cache had
    # no priceable rows for this day. ``n_direction_targets`` counts
    # the masked entries that contributed to the BCE term across all
    # mini-batches that ran (mirrors ``n_mature_targets`` semantics).
    direction_back_bce_mean: float = 0.0
    direction_lay_bce_mean: float = 0.0
    n_direction_targets: int = 0


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
    # Phase 7 Session 02 — auxiliary-head loss diagnostics. Zero when
    # the corresponding weight is 0.0 OR the rollout's aux_labels is
    # None.
    fill_prob_bce_mean: float = 0.0
    mature_prob_bce_mean: float = 0.0
    risk_nll_mean: float = 0.0
    # Phase-13 S03 — direction-prob BCE diagnostics.
    direction_back_bce_mean: float = 0.0
    direction_lay_bce_mean: float = 0.0
    n_direction_targets: int = 0
    direction_prob_loss_weight_active: float = 0.0
    # Phase 9 Session 02 — per-transition credit diagnostics. The
    # ``_active`` flag mirrors the trainer's runtime config so JSONL
    # consumers can filter scoreboard rows by which credit-assignment
    # path produced them (hard_constraints.md §5). ``n_mature_targets``
    # is the per-update sum (see ``UpdateLog`` for the exact semantics).
    per_transition_credit_active: bool = False
    n_mature_targets: int = 0
    # Phase 8 Session 02 — BC entropy-warmup diagnostics. ``post_bc_entropy``
    # is the value the worker measured immediately after BC and pushed via
    # :meth:`DiscretePPOTrainer.set_post_bc_entropy`; ``None`` when no BC
    # ran. ``effective_target_entropy`` is the linear-interp output for
    # this episode (== ``entropy_coeff`` when warmup inactive).
    # ``eps_since_bc`` is the post-BC PPO episode counter logged here so
    # JSONL consumers can plot the warmup trajectory without recomputing.
    post_bc_entropy: float | None = None
    effective_target_entropy: float = 0.0
    eps_since_bc: int = 0


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
        hp: dict | None = None,
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

        # Phase 7 Session 02 — auxiliary-head loss weights, read from
        # the per-agent ``hp`` dict ONLY. NO config fallback. The
        # worker pre-merges any ``--reward-overrides`` values into
        # ``hp`` before constructing the trainer (Path A in
        # ``plans/rewrite/phase-7-port-aux-heads/session_prompts/
        # 02_wire_bce_loss_in_trainer.md``). A nested fallback would
        # re-introduce the v1 precedence trap: v2's ``CohortGenes.
        # to_dict`` always populates these keys with their default
        # 0.0, so ``hp.get(name, fallback)`` would return 0.0 and the
        # fallback would never be consulted — silently swallowing the
        # override. See ``lessons_learnt.md`` for the full audit.
        hp = dict(hp or {})
        self.fill_prob_loss_weight = float(
            hp.get("fill_prob_loss_weight", 0.0) or 0.0
        )
        self.mature_prob_loss_weight = float(
            hp.get("mature_prob_loss_weight", 0.0) or 0.0
        )
        self.risk_loss_weight = float(
            hp.get("risk_loss_weight", 0.0) or 0.0
        )
        # Phase 9 Session 02 — per-transition mature_prob credit
        # assignment. Default ``False`` keeps the per-slot path active
        # for byte-identity with Phase 7 baseline runs
        # (hard_constraints.md §1, §6). When ``True`` the trainer
        # replaces the per-slot mature BCE with one that lands the
        # label on the SINGLE step where each pair was opened
        # (purpose.md §"The fix").
        self.per_transition_credit = bool(
            hp.get("per_transition_credit", False)
        )

        # Phase-13 Session 03 (2026-05-06). Direction-prob aux head.
        # Read from ``hp`` ONLY (Path A; Phase 7 lessons-learnt
        # precedence trap). ``direction_prob_loss_weight = 0.0`` is
        # byte-identical to pre-S03: the head is present in the
        # network (architecture-hash break) but contributes no BCE
        # term to total_loss. The three label-defining knobs resolve
        # the offline cache stem and MUST match the values used to
        # scan the labels.
        self.direction_prob_loss_weight = float(
            hp.get("direction_prob_loss_weight", 0.0) or 0.0
        )
        self.direction_horizon_ticks = int(
            hp.get("direction_horizon_ticks", 60),
        )
        self.direction_threshold_ticks = int(
            hp.get("direction_threshold_ticks", 5),
        )
        self.direction_force_close_seconds = float(
            hp.get("direction_force_close_seconds", 60.0),
        )
        # Optional explicit data-dir for the cache lookup. The trainer
        # walks shim.env.day to resolve the date but the cache lives
        # under ``{data_dir.parent}/direction_labels/``; we accept
        # ``data_dir`` as a hint via hp and fall back to
        # ``Path("data/processed")`` on absence.
        self._direction_data_dir = (
            hp.get("direction_data_dir", "data/processed")
        )
        # Lazy per-day label cache (avoid re-loading on the same day
        # across multiple episodes / mini-batches).
        self._direction_label_cache: dict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}

        # Phase 8 Session 02 — BC pretrain warmup handshake. The v2
        # trainer doesn't run an SAC-style alpha controller (one is
        # NOT added in this session — see the session prompt's stop
        # conditions); these fields and ``_effective_target_entropy``
        # are surfaced for diagnostic logging and for tests that
        # validate the linear-interp arithmetic. Per
        # ``hard_constraints.md §8`` the warmup activates only when
        # the worker calls ``set_post_bc_entropy`` after a successful
        # BC pass; absent that call the trainer is byte-identical
        # to pre-S02.
        self._post_bc_entropy: float | None = None
        self._bc_warmup_eps: int = int(
            hp.get("bc_target_entropy_warmup_eps", 5),
        )
        self._eps_since_bc: int = 0

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

    def set_post_bc_entropy(self, entropy: float) -> None:
        """Activate the BC entropy-warmup handshake.

        Called by the worker immediately after a successful BC pass.
        Resets the per-rollout episode counter so the linear interp
        starts at episode 0. Calling this with a sentinel ``None`` (or
        without calling it at all) leaves the trainer in pre-S02
        byte-identical mode (``_effective_target_entropy`` returns the
        constant ``self.entropy_coeff``).
        """
        self._post_bc_entropy = float(entropy)
        self._eps_since_bc = 0

    def _effective_target_entropy(self) -> float:
        """Linear interp from ``_post_bc_entropy`` to ``entropy_coeff``.

        Diagnostic only — no PPO code path consumes this value (the v2
        trainer has no entropy controller in this session). Logged on
        :class:`EpisodeStats` so the per-episode JSONL surfaces the
        warmup trajectory; the coefficient consumed by the surrogate
        loss stays at ``self.entropy_coeff``.
        """
        if self._post_bc_entropy is None:
            return float(self.entropy_coeff)
        if self._bc_warmup_eps <= 0:
            return float(self.entropy_coeff)
        if self._eps_since_bc >= self._bc_warmup_eps:
            return float(self.entropy_coeff)
        frac = float(self._eps_since_bc) / float(self._bc_warmup_eps)
        return (
            float(self._post_bc_entropy)
            + frac * (float(self.entropy_coeff) - float(self._post_bc_entropy))
        )

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
        # Capture warmup state BEFORE the post-episode increment, so
        # the first BC-warmup episode logs ``eps_since_bc = 0`` and
        # ``effective_target_entropy = post_bc_entropy``.
        effective_te = self._effective_target_entropy()
        eps_since_bc_now = int(self._eps_since_bc)
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
            fill_prob_bce_mean=update_log.fill_prob_bce_mean,
            mature_prob_bce_mean=update_log.mature_prob_bce_mean,
            risk_nll_mean=update_log.risk_nll_mean,
            direction_back_bce_mean=update_log.direction_back_bce_mean,
            direction_lay_bce_mean=update_log.direction_lay_bce_mean,
            n_direction_targets=update_log.n_direction_targets,
            direction_prob_loss_weight_active=(
                self.direction_prob_loss_weight
            ),
            per_transition_credit_active=self.per_transition_credit,
            n_mature_targets=update_log.n_mature_targets,
            post_bc_entropy=self._post_bc_entropy,
            effective_target_entropy=effective_te,
            eps_since_bc=eps_since_bc_now,
        )
        # Tick the post-BC counter for the NEXT episode. The increment
        # is gated on ``_post_bc_entropy is not None`` so non-BC runs
        # leave the counter at 0 (cosmetic; nothing reads it on those
        # runs).
        if self._post_bc_entropy is not None:
            self._eps_since_bc += 1
        logger.info(
            "DiscretePPOTrainer episode: n_steps=%d n_updates=%d "
            "policy_loss=%.4f value_loss=%.4f entropy=%.4f "
            "approx_kl=%.4f fill_prob_bce_mean=%.4f "
            "mature_prob_bce_mean=%.4f risk_nll_mean=%.4f "
            "total_reward=%.3f%s wall=%.2fs",
            stats.n_steps, stats.n_updates_run,
            stats.policy_loss_mean, stats.value_loss_mean,
            stats.entropy_mean, stats.approx_kl_mean,
            stats.fill_prob_bce_mean, stats.mature_prob_bce_mean,
            stats.risk_nll_mean,
            stats.total_reward,
            (
                f" n_mature_targets={stats.n_mature_targets}"
                if self.per_transition_credit else ""
            ),
            stats.wall_time_sec,
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

        # ── Phase 7 S02 aux-loss tensors (per-rollout, per-runner) ──────
        # The aux labels live at the rollout scope (not per-tick), so
        # we materialise them ONCE here and broadcast inside the
        # mini-batch loop. ``aux_active`` is False when all three
        # weights are 0 OR the producer didn't compute labels — in
        # that case the loss expression collapses to the pre-S02 form
        # and total_loss is byte-identical (regression test:
        # ``test_all_aux_losses_zero_when_weights_zero``).
        aux_labels = batch.aux_labels
        aux_weights_any = (
            self.fill_prob_loss_weight > 0.0
            or self.mature_prob_loss_weight > 0.0
            or self.risk_loss_weight > 0.0
        )
        aux_active = aux_weights_any and aux_labels is not None
        if aux_active:
            fill_label_t = torch.from_numpy(
                aux_labels.fill_label
            ).to(device)                                  # (R,)
            mature_label_t = torch.from_numpy(
                aux_labels.mature_label
            ).to(device)
            risk_label_t = torch.from_numpy(
                np.nan_to_num(aux_labels.risk_label, nan=0.0)
            ).to(device)
            runner_mask_t = torch.from_numpy(
                aux_labels.runner_mask.astype(np.float32)
            ).to(device)                                  # (R,)
            risk_mask_t = torch.from_numpy(
                aux_labels.risk_mask.astype(np.float32)
            ).to(device)
        else:
            fill_label_t = mature_label_t = risk_label_t = None
            runner_mask_t = risk_mask_t = None

        # ── Phase 9 S02 per-transition mature_prob credit ────────────────
        # When the flag is on AND the rollout collector populated
        # ``pair_open_records`` (sequential rollout path; the batched
        # rollout currently doesn't), build per-step label / mask /
        # runner-slot arrays. The mini-batch loop substitutes the
        # per-transition mature BCE for the per-slot one inside
        # ``_compute_aux_losses``. fill_prob and risk_nll continue on
        # the per-slot path (hard_constraints.md §7).
        per_trans_active = (
            self.per_transition_credit
            and batch.pair_open_records is not None
            and self.mature_prob_loss_weight > 0.0
        )
        if per_trans_active:
            env = getattr(self.shim, "env", None)
            settled = list(getattr(env, "_settled_bets", []) or [])
            live_bm = getattr(env, "bet_manager", None)
            if live_bm is not None:
                settled.extend(live_bm.bets)
            mature_label_np, mature_mask_np = assign_per_transition_labels(
                batch.pair_open_records or [],
                settled,
                n_steps=T,
            )
            # ``runner_slot_at_step`` carries the slot the agent opened
            # at each masked tick. The BCE consumer reads it ONLY at
            # rows where mask is True; off-mask entries default to 0
            # (a valid slot index that will never be consumed).
            slot_np = np.zeros(T, dtype=np.int64)
            for rec in batch.pair_open_records or []:
                idx = int(rec.step_index)
                if 0 <= idx < T:
                    slot_np[idx] = int(rec.runner_slot)
            mature_label_per_step = _move_to_device(
                torch.from_numpy(mature_label_np), device,
            )
            mature_mask_per_step = _move_to_device(
                torch.from_numpy(mature_mask_np), device,
            )
            runner_slot_at_step = _move_to_device(
                torch.from_numpy(slot_np), device,
            )
        else:
            mature_label_per_step = None
            mature_mask_per_step = None
            runner_slot_at_step = None

        # ── Phase-13 S03 direction-prob per-step labels ────────────────
        # Per-(env-step, runner, side) cached labels resolve at trainer
        # init time from the offline scan (training_v2.direction_label_
        # scan). When weight is 0 the helper returns None and the
        # branch below is skipped — byte-identical to pre-S03.
        direction_active = (
            self.direction_prob_loss_weight > 0.0
            and self._build_direction_label_grid(T) is not None
        )
        if direction_active:
            label_grid_np, mask_grid_np = (
                self._build_direction_label_grid(T)
            )
            direction_label_per_step = _move_to_device(
                torch.from_numpy(label_grid_np), device,
            )
            direction_mask_per_step = _move_to_device(
                torch.from_numpy(mask_grid_np), device,
            )
            # Class-balance pos_weights from the grid's positive rate
            # (same intuition as phase-12 / fill_prob path: rare class
            # gets up-weighted).
            mask_f = mask_grid_np.astype(np.float32)
            mask_sum = float(mask_f.sum())
            if mask_sum > 0.0:
                pos_back = float((label_grid_np[..., 0] * mask_f).sum())
                pos_lay = float((label_grid_np[..., 1] * mask_f).sum())
                d_back = max(pos_back / mask_sum, 1e-6)
                d_lay = max(pos_lay / mask_sum, 1e-6)
                pos_w_back_t = torch.tensor(
                    (1.0 - d_back) / d_back,
                    dtype=torch.float32, device=device,
                )
                pos_w_lay_t = torch.tensor(
                    (1.0 - d_lay) / d_lay,
                    dtype=torch.float32, device=device,
                )
            else:
                pos_w_back_t = torch.tensor(
                    1.0, dtype=torch.float32, device=device,
                )
                pos_w_lay_t = torch.tensor(
                    1.0, dtype=torch.float32, device=device,
                )
        else:
            direction_label_per_step = None
            direction_mask_per_step = None
            pos_w_back_t = pos_w_lay_t = None

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
        fill_prob_losses: list[float] = []
        mature_prob_losses: list[float] = []
        risk_losses: list[float] = []
        direction_back_losses: list[float] = []
        direction_lay_losses: list[float] = []
        n_mature_targets_total = 0
        n_direction_targets_total = 0
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

                    # Phase 7 S02 — auxiliary-head loss terms. Skipped
                    # entirely when all weights are 0 OR aux_labels is
                    # None, so total_loss is byte-identical to pre-S02.
                    if aux_active:
                        fill_loss_mb, mature_loss_mb, risk_loss_mb = (
                            self._compute_aux_losses(
                                policy_out=out,
                                fill_label=fill_label_t,
                                mature_label=mature_label_t,
                                risk_label=risk_label_t,
                                runner_mask=runner_mask_t,
                                risk_mask=risk_mask_t,
                            )
                        )
                        # Phase 9 S02 — substitute the per-slot mature
                        # BCE with the per-transition variant when the
                        # flag is on. Fill / risk stay on the per-slot
                        # path (hard_constraints.md §7).
                        if per_trans_active:
                            mature_loss_mb, n_targets_mb = (
                                self._compute_per_transition_mature_loss(
                                    policy_out=out,
                                    mature_label_step=mature_label_per_step[mb_idx],
                                    mature_mask_step=mature_mask_per_step[mb_idx],
                                    runner_slot_step=runner_slot_at_step[mb_idx],
                                )
                            )
                            n_mature_targets_total += int(n_targets_mb)
                        total_loss = (
                            total_loss
                            + self.fill_prob_loss_weight * fill_loss_mb
                            + self.mature_prob_loss_weight * mature_loss_mb
                            + self.risk_loss_weight * risk_loss_mb
                        )
                        fill_prob_losses.append(float(fill_loss_mb.item()))
                        mature_prob_losses.append(float(mature_loss_mb.item()))
                        risk_losses.append(float(risk_loss_mb.item()))
                    else:
                        fill_prob_losses.append(0.0)
                        mature_prob_losses.append(0.0)
                        risk_losses.append(0.0)

                    # Phase-13 S03 — per-side direction BCE-with-logits
                    # on the masked per-step labels, weighted by
                    # direction_prob_loss_weight. When direction_active
                    # is False the branch is skipped and total_loss is
                    # byte-identical to pre-S03.
                    if direction_active:
                        mb_dir_labels = direction_label_per_step[mb_idx]
                        mb_dir_mask = direction_mask_per_step[mb_idx]
                        (
                            dir_back_loss_mb,
                            dir_lay_loss_mb,
                            n_dir_mb,
                        ) = self._compute_direction_loss(
                            policy_out=out,
                            label_per_step=mb_dir_labels,
                            mask_per_step=mb_dir_mask,
                            pos_weight_back=pos_w_back_t,
                            pos_weight_lay=pos_w_lay_t,
                        )
                        total_loss = (
                            total_loss
                            + self.direction_prob_loss_weight
                            * (dir_back_loss_mb + dir_lay_loss_mb)
                        )
                        direction_back_losses.append(
                            float(dir_back_loss_mb.item()),
                        )
                        direction_lay_losses.append(
                            float(dir_lay_loss_mb.item()),
                        )
                        n_direction_targets_total += int(n_dir_mb)
                    else:
                        direction_back_losses.append(0.0)
                        direction_lay_losses.append(0.0)

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
            fill_prob_bce_mean=(
                float(np.mean(fill_prob_losses)) if n_run else 0.0
            ),
            mature_prob_bce_mean=(
                float(np.mean(mature_prob_losses)) if n_run else 0.0
            ),
            risk_nll_mean=(
                float(np.mean(risk_losses)) if n_run else 0.0
            ),
            n_mature_targets=n_mature_targets_total,
            direction_back_bce_mean=(
                float(np.mean(direction_back_losses)) if n_run else 0.0
            ),
            direction_lay_bce_mean=(
                float(np.mean(direction_lay_losses)) if n_run else 0.0
            ),
            n_direction_targets=n_direction_targets_total,
        )

    # ── Phase-13 S03 direction-label resolution ────────────────────────────

    def _build_direction_label_grid(
        self,
        n_steps: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Build per-env-step direction labels for the current day.

        Returns ``(label_per_step, mask_per_step)`` of shapes
        ``(n_steps, max_runners, 2)`` and ``(n_steps, max_runners)``,
        or ``None`` when the cache is missing AND the loss weight is
        0 (byte-identical opt-out path). When weight > 0 and cache is
        missing, raises ``FileNotFoundError`` (hard_constraints §22).

        Determinism: the env walks ``day.races`` in the listed order
        and ticks within each race in tick-index order. This matches
        exactly the iteration order ``training_v2.direction_label_scan
        .scan_day`` uses to assign global pre-race tick indices, so a
        per-env-step lookup by ``(global_pre_race_tick_idx,
        runner_idx)`` is unambiguous.

        Cache: per-date results are memoised on
        ``self._direction_label_cache`` so multi-episode training on
        the same day pays the load cost once.
        """
        if self.direction_prob_loss_weight <= 0.0:
            return None
        env = getattr(self.shim, "env", None)
        if env is None or not hasattr(env, "day"):
            return None
        day = env.day
        date = str(getattr(day, "date", ""))
        if not date:
            return None

        from pathlib import Path as _Path
        data_dir = _Path(self._direction_data_dir)

        cache_key = (
            f"{date}|{self.direction_horizon_ticks}|"
            f"{self.direction_threshold_ticks}|"
            f"{self.direction_force_close_seconds}|"
            f"{self.max_runners}"
        )
        cached = self._direction_label_cache.get(cache_key)
        if cached is not None:
            label_grid, mask_grid, env_idx_arr = cached
        else:
            labels = _load_direction_labels(
                date,
                data_dir,
                direction_horizon_ticks=self.direction_horizon_ticks,
                direction_threshold_ticks=self.direction_threshold_ticks,
                force_close_before_off_seconds=(
                    self.direction_force_close_seconds
                ),
                strict=True,
            )
            label_grid, mask_grid, env_idx_arr = (
                _materialise_direction_grid(
                    day=day,
                    labels=labels,
                    max_runners=self.max_runners,
                )
            )
            self._direction_label_cache[cache_key] = (
                label_grid, mask_grid, env_idx_arr,
            )

        # The grid is indexed by env-step; trim or pad to the rollout's
        # actual ``n_steps``. The env always walks the day's full tick
        # sequence, so n_steps == grid.shape[0] in practice. Defensive
        # branch: if the rollout was truncated mid-day, slice; if it
        # somehow exceeded the grid, raise (a sign that the env / day
        # mapping diverged from the assumption above).
        if int(label_grid.shape[0]) < int(n_steps):
            raise RuntimeError(
                f"Direction-label grid has {label_grid.shape[0]} env "
                f"steps but the rollout produced {n_steps} — env / "
                "day-walk assumption violated. Re-run the offline scan "
                "on the same day data the env was constructed from.",
            )
        return label_grid[:n_steps], mask_grid[:n_steps]

    # ── Phase 7 S02 aux-loss helper ────────────────────────────────────────

    def _compute_aux_losses(
        self,
        *,
        policy_out,
        fill_label: torch.Tensor,
        mature_label: torch.Tensor,
        risk_label: torch.Tensor,
        runner_mask: torch.Tensor,
        risk_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """BCE on fill_prob + mature_prob, Gaussian NLL on risk_head.

        Per-rollout per-runner labels are broadcast across the
        mini-batch's transition axis. Both BCE terms are masked by
        ``runner_mask`` (slot had any matched pair leg). The NLL term
        is masked by ``risk_mask`` (slot had at least one completed
        pair) — naked-only slots' NaN risk labels are replaced with
        zero arithmetic-side and skipped via the mask, so no NaN
        propagates into the gradient.

        Returns the three scalar losses unweighted; the caller
        multiplies by ``self.{fill_prob,mature_prob,risk}_loss_weight``
        and adds to total_loss.
        """
        # Policy outputs: (B, R) each.
        fill_pred = policy_out.fill_prob_per_runner
        mature_pred = policy_out.mature_prob_per_runner
        risk_mean = policy_out.predicted_locked_pnl_per_runner
        risk_log_var = policy_out.predicted_locked_log_var_per_runner

        # Broadcast (R,) labels / masks to (1, R) for broadcasting.
        fill_label_b = fill_label.unsqueeze(0)
        mature_label_b = mature_label.unsqueeze(0)
        runner_mask_b = runner_mask.unsqueeze(0)
        risk_label_b = risk_label.unsqueeze(0)
        risk_mask_b = risk_mask.unsqueeze(0)

        # Numerically-stable BCE on probabilities. Clamping inputs of
        # ``log`` to ``[1e-7, 1 - 1e-7]`` mirrors PyTorch's internal
        # ``BCELoss`` epsilon and keeps gradients finite when the
        # sigmoid output saturates.
        eps = 1e-7
        fill_pred_c = fill_pred.clamp(eps, 1.0 - eps)
        mature_pred_c = mature_pred.clamp(eps, 1.0 - eps)

        fill_bce = -(
            fill_label_b * torch.log(fill_pred_c)
            + (1.0 - fill_label_b) * torch.log(1.0 - fill_pred_c)
        )
        mature_bce = -(
            mature_label_b * torch.log(mature_pred_c)
            + (1.0 - mature_label_b) * torch.log(1.0 - mature_pred_c)
        )

        # Mask sums broadcast to (B, R); .sum() collapses to a scalar
        # weight count. clamp(min=1) keeps the denominator finite if
        # the rollout had no matched pairs at all (all-NOOP day).
        bce_denom = (runner_mask_b.expand_as(fill_bce)).sum().clamp(min=1.0)
        fill_loss = (fill_bce * runner_mask_b).sum() / bce_denom
        mature_loss = (mature_bce * runner_mask_b).sum() / bce_denom

        # Gaussian NLL: 0.5 * ((label - mean)^2 / exp(log_var) + log_var).
        # Masked entries (no completed pair on this slot) contribute 0.
        diff_sq = (risk_label_b - risk_mean) ** 2
        nll = 0.5 * (diff_sq / torch.exp(risk_log_var) + risk_log_var)
        risk_denom = (risk_mask_b.expand_as(nll)).sum()
        if float(risk_denom.item()) > 0.0:
            risk_loss = (nll * risk_mask_b).sum() / risk_denom
        else:
            # No completed pair anywhere in the rollout — skip the
            # term entirely. ``zeros_like`` keeps the dtype / device /
            # autograd-graph-detached property the caller expects so
            # ``total_loss + 0`` flows through unchanged.
            risk_loss = torch.zeros((), device=fill_pred.device)

        return fill_loss, mature_loss, risk_loss

    # ── Phase-13 S03 direction BCE helper ──────────────────────────────────

    def _compute_direction_loss(
        self,
        *,
        policy_out,
        label_per_step: torch.Tensor,
        mask_per_step: torch.Tensor,
        pos_weight_back: torch.Tensor,
        pos_weight_lay: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Per-side direction BCE-with-logits across the mini-batch.

        ``label_per_step`` is ``(mb, R, 2)`` carrying ``label_back`` /
        ``label_lay``; ``mask_per_step`` is ``(mb, R)`` bool. Cells
        outside the mask are EXCLUDED from the loss — they correspond
        to in-play ticks or non-priceable runners where the cache
        emitted no row.

        Returns ``(back_loss, lay_loss, n_supervised_cells)``. Both
        scalars carry the autograd graph back to ``policy_out``'s
        direction logits. ``n_supervised_cells`` is the integer count
        of cells used (== mask.sum()) for the diagnostic.
        """
        back_logits = policy_out.direction_back_logits_per_runner  # (mb, R)
        lay_logits = policy_out.direction_lay_logits_per_runner    # (mb, R)
        label_back = label_per_step[..., 0]
        label_lay = label_per_step[..., 1]
        mask_f = mask_per_step.to(back_logits.dtype)

        # BCE-with-logits with per-element pos_weight. ``reduction
        # ='none'`` so we mask cell-wise; mean over the masked cells.
        back_bce = nn.functional.binary_cross_entropy_with_logits(
            back_logits, label_back,
            pos_weight=pos_weight_back, reduction="none",
        )
        lay_bce = nn.functional.binary_cross_entropy_with_logits(
            lay_logits, label_lay,
            pos_weight=pos_weight_lay, reduction="none",
        )
        denom = mask_f.sum().clamp(min=1.0)
        back_loss = (back_bce * mask_f).sum() / denom
        lay_loss = (lay_bce * mask_f).sum() / denom
        n_supervised = int(mask_f.sum().item())
        return back_loss, lay_loss, n_supervised

    # ── Phase 9 S02 per-transition mature BCE helper ──────────────────────

    def _compute_per_transition_mature_loss(
        self,
        *,
        policy_out,
        mature_label_step: torch.Tensor,
        mature_mask_step: torch.Tensor,
        runner_slot_step: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Per-transition strict-mature BCE for one mini-batch.

        Phase 9 S02. Replaces the per-slot mature BCE
        (``_compute_aux_losses``) when ``per_transition_credit=True``.
        For each step in the mini-batch where ``mature_mask_step`` is
        True, picks the ``mature_prob_per_runner`` column corresponding
        to the runner the agent opened at that step and applies BCE
        against the strict-mature label (1.0 = matured naturally OR
        agent-closed; 0.0 = naked OR force-closed).

        When the mini-batch contains no open steps the loss is a
        zeroed scalar that still produces a valid ``.backward()``
        contribution (no NaN, no detached graph). Returns the loss
        plus the count of masked entries actually consumed (used for
        the ``n_mature_targets`` per-update diagnostic).

        The expected gradient SNR improvement vs. the per-slot path:
        the same total label signal lands on ~200–500 transitions per
        ~11k-transition rollout instead of being broadcast across all
        of them — purpose.md §"The fix" estimates a 20–50× concentration.
        """
        mature_pred = policy_out.mature_prob_per_runner  # (mb, R)
        mb_size = mature_pred.shape[0]
        device = mature_pred.device

        # Gather per-step prediction at the runner the agent opened.
        # ``runner_slot_step`` holds 0 at unmasked rows; harmless because
        # the BCE term is zeroed out by the mask before reduction.
        rows = torch.arange(mb_size, device=device)
        per_step_pred = mature_pred[rows, runner_slot_step]  # (mb,)

        eps = 1e-7
        per_step_pred_c = per_step_pred.clamp(eps, 1.0 - eps)
        bce = -(
            mature_label_step * torch.log(per_step_pred_c)
            + (1.0 - mature_label_step) * torch.log(1.0 - per_step_pred_c)
        )
        # Mask is bool; cast to the BCE dtype for the multiplication.
        mask_f = mature_mask_step.to(bce.dtype)
        masked = bce * mask_f
        denom = mask_f.sum().clamp(min=1.0)
        loss = masked.sum() / denom
        n_targets = int(mature_mask_step.sum().item())
        return loss, n_targets

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
