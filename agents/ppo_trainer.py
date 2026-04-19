"""
agents/ppo_trainer.py -- PPO training loop for a single agent.

Implements proximal policy optimisation (clipped surrogate objective) with
generalised advantage estimation (GAE).  Designed to train one agent on a
sequence of day-episodes produced by the BetfairEnv.

Key features:
- Rollout collection: runs the policy in the env, collects transitions
- GAE advantage estimation with configurable lambda and gamma
- PPO clipped loss + value loss + entropy bonus
- Gradient clipping for stability
- ProgressTracker integration for episode-level and total ETA
- Per-episode logging (reward, P&L, bet count, loss terms) to logs/ dir
- Progress dict published to an optional asyncio.Queue for WebSocket

Usage::

    trainer = PPOTrainer(policy, config)
    stats = trainer.train(days, n_epochs=10)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from agents.architecture_registry import create_policy
from agents.policy_network import BasePolicy, PolicyOutput
from data.episode_builder import Day
from training.perf_log import perf_log
from env.betfair_env import BetfairEnv
from training.progress_tracker import ProgressTracker


# -- Module constants --------------------------------------------------
#
# ``LOG_RATIO_CLAMP`` bounds ``new_logp - old_logp`` before the ``.exp()``
# inside the PPO surrogate. Originally ±20 as a pure numerical backstop
# (prevents float32 overflow on exp(88)). Tightened to ±5 after the
# 2026-04-18 smoke probe showed ep1 policy_loss ~7e+07 on fresh
# transformer/LSTM agents — ``exp(20) ≈ 5e+08`` was large enough that a
# single bad mini-batch with a negative advantage could dominate the
# mean loss. At ±5, ``exp(5) ≈ 148``, which with normalised advantages
# O(1) caps per-mini-batch contribution at ~150 — comfortably below the
# smoke-test ``EP1_POLICY_LOSS_MAX = 100`` in aggregate (mean smooths
# toward 0 because PPO takes the min of the clamped and unclamped
# surrogate). ``|log_ratio| ≪ 5`` in normal PPO updates so the tighter
# clamp remains a no-op in healthy operation.
#
# See plans/naked-clip-and-stability/lessons_learnt.md for the trace
# and the decision log.
LOG_RATIO_CLAMP: float = 5.0

logger = logging.getLogger(__name__)


# -- Per-head action layout ---------------------------------------------------

#: Ordered names of the per-runner action heads, matching the layout
#: produced by ``env.betfair_env`` and consumed by ``BetfairEnv.step``:
#: ``[signal × N | stake × N | aggression × N | cancel × N | arb_spread × N]``
#: where N = ``max_runners``. The 5th head (``arb_spread``) is only present
#: when the env's scalping branch is active; directional runs use the first
#: four. Used by the Session 2 entropy-floor controller to slice the
#: flat Normal distribution into per-head entropies.
_HEAD_NAMES: tuple[str, ...] = (
    "signal",
    "stake",
    "aggression",
    "cancel",
    "arb_spread",
)


# -- Session 3 (arb-improvements) signal-bias warmup --------------------------

#: Magnitude threshold on the sampled ``signal`` action above which we
#: consider the step to have "placed a bet". Matches the env's internal
#: back/lay decision band — anything beyond ±0.33 is taken as an
#: affirmative signal for this diagnostic. Used for ``bet_rate`` only;
#: the env's own threshold is authoritative for actual bet placement.
_BET_SIGNAL_THRESHOLD: float = 0.33

#: Cap on how many per-arb activity-log events are emitted per episode.
#: Scalping episodes can complete dozens of pairs; flooding the WS queue
#: with all of them degrades the training monitor. The overflow is
#: summarised in a single "… plus N more arb pair(s)" line.
_MAX_ARB_EVENTS_PER_EP: int = 20


def _compute_arb_rate(arbs_completed: int, arbs_naked: int) -> float:
    """Fraction of arb attempts that paired (``completed`` / total).

    Returns ``0.0`` when there were no arb attempts at all — avoids
    spurious NaN / divide-by-zero. Always in ``[0.0, 1.0]``.
    """
    total = int(arbs_completed) + int(arbs_naked)
    if total <= 0:
        return 0.0
    return float(arbs_completed) / float(total)


def _compute_fill_prob_bce(
    preds: torch.Tensor, labels: torch.Tensor, *, eps: float = 1e-7,
) -> torch.Tensor:
    """Masked BCE for the scalping-active-management §02 fill-prob head.

    Parameters
    ----------
    preds:
        Probabilities in ``[0, 1]``, shape ``(batch, max_runners)`` — the
        sigmoid-applied output of the fill-prob head.
    labels:
        Labels in ``{0.0, 1.0}`` or NaN. Same shape as ``preds``. NaN
        entries are masked out (no supervision from unresolved pairs).
    eps:
        Clamp bound for ``preds`` to avoid ``log(0)`` at contrived
        extremes (tested predictions of exactly 0 or 1).

    Returns
    -------
    torch.Tensor
        Scalar BCE averaged over non-NaN label cells. Returns a scalar
        zero tensor on the same device as ``preds`` when every label is
        NaN (the "no supervision available" case — the test
        ``test_fill_prob_excluded_from_loss_when_outcome_unresolved``
        pins this behaviour).

    Notes
    -----
    Factored out of :meth:`PPOTrainer._ppo_update` so the invariants
    exercised by the session-02 unit tests (gradient direction, zero
    loss on perfect predictions, mask-out of unresolved samples) can
    be tested without spinning up a full rollout.
    """
    mask = ~torch.isnan(labels)
    if not mask.any():
        return torch.zeros((), device=preds.device, dtype=preds.dtype)
    p = preds.clamp(eps, 1.0 - eps)
    safe_labels = torch.where(mask, labels, torch.zeros_like(labels))
    per_elem_bce = -(
        safe_labels * torch.log(p)
        + (1.0 - safe_labels) * torch.log(1.0 - p)
    )
    mask_f = mask.to(preds.dtype)
    return (per_elem_bce * mask_f).sum() / mask_f.sum()


def _compute_risk_nll(
    means: torch.Tensor,
    log_vars: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Masked Gaussian NLL for the scalping-active-management §03 risk head.

    Parameters
    ----------
    means:
        Predicted locked-P&L means, shape ``(batch, max_runners)``.
    log_vars:
        Predicted locked-P&L log-variances, already clamped inside each
        architecture's ``forward`` so ``exp(log_vars)`` stays finite.
    labels:
        Realised ``locked_pnl`` (float £) per pair, same shape. ``NaN``
        marks "no resolved outcome for this tick-slot" — naked pair or
        unresolved at episode end — and is masked out.

    Returns
    -------
    torch.Tensor
        Scalar NLL averaged over non-NaN label cells. Returns a scalar
        zero tensor on the same device as ``means`` when every label is
        NaN (the "no supervision available" case — defensive zero-return
        mirrors the BCE helper).

    Notes
    -----
    The NLL is the Gaussian constant-free form
    ``0.5 * (log_var + (target - mean)^2 / exp(log_var))``. The
    ``log(2π)/2`` additive constant is omitted — it does not affect the
    gradient and its absence gives a clean analytic target for
    ``test_risk_nll_zero_on_perfect_predictions``. Factored out of
    :meth:`PPOTrainer._ppo_update` so the session-03 unit tests can
    exercise it directly (gradient direction, zero loss on perfect
    predictions, mask-out of unresolved samples).
    """
    mask = ~torch.isnan(labels)
    if not mask.any():
        return torch.zeros((), device=means.device, dtype=means.dtype)
    safe_labels = torch.where(mask, labels, torch.zeros_like(labels))
    inv_var = torch.exp(-log_vars)
    per_elem = 0.5 * (log_vars + (safe_labels - means).pow(2) * inv_var)
    mask_f = mask.to(means.dtype)
    return (per_elem * mask_f).sum() / mask_f.sum()


def _signal_bias_for_epoch(
    epoch: int, warmup: int, magnitude: float,
) -> float:
    """Linearly-decaying additive bias on the signal head mean.

    At epoch ``e``, returns ``magnitude * max(0, 1 - e/warmup)``. Returns
    ``0.0`` when ``warmup <= 0`` or ``magnitude == 0.0`` — both are the
    "bias off" conditions, yielding byte-identical training to the
    unbiased path. At ``e >= warmup`` the return is ``0.0`` regardless
    of ``magnitude``.
    """
    if warmup <= 0 or magnitude == 0.0:
        return 0.0
    frac = 1.0 - float(epoch) / float(warmup)
    if frac <= 0.0:
        return 0.0
    return float(magnitude) * frac


# -- Gene → env reward override mapping ---------------------------------------

#: Maps a hyperparameter (gene) name to the key(s) it overrides inside
#: ``config["reward"]``. Kept here (not in env) because the env doesn't
#: know about the genetic schema — it only speaks "reward config keys".
#:
#: Session 3 split the original ``reward_early_pick_bonus`` scalar into
#: two independent genes (``early_pick_bonus_min`` / ``_max``) plus
#: ``early_pick_min_seconds`` and ``terminal_bonus_weight``. Each new
#: gene is a 1:1 passthrough — its hyperparameter name matches the env
#: reward-config key it overrides, so the map entry below collapses to
#: a single-key tuple.
_REWARD_GENE_MAP: dict[str, tuple[str, ...]] = {
    "early_pick_bonus_min": ("early_pick_bonus_min",),
    "early_pick_bonus_max": ("early_pick_bonus_max",),
    "early_pick_min_seconds": ("early_pick_min_seconds",),
    "terminal_bonus_weight": ("terminal_bonus_weight",),
    "reward_efficiency_penalty": ("efficiency_penalty",),
    "reward_precision_bonus": ("precision_bonus",),
    "reward_drawdown_shaping": ("drawdown_shaping_weight",),
    "reward_spread_cost_weight": ("spread_cost_weight",),
    "inactivity_penalty": ("inactivity_penalty",),
    # Forced-arbitrage (Issue 05, session 3). Only take effect when
    # training.scalping_mode is on; on directional runs they live in
    # the genome but the env's scalping branch is dormant.
    "naked_penalty_weight": ("naked_penalty_weight",),
    "early_lock_bonus_weight": ("early_lock_bonus_weight",),
    # Session 1 (arb-improvements): reward_clip passthrough. Whitelisted
    # in BetfairEnv._REWARD_OVERRIDE_KEYS so it survives the env-side
    # filter even though the env itself doesn't use it — the trainer
    # reads it off self.reward_overrides to clip the per-step reward
    # fed into the advantage/return computation.
    "reward_clip": ("reward_clip",),
    # Scalping-active-management session 02: weight on the BCE fill-prob
    # aux loss. Same passthrough pattern as ``reward_clip``: whitelisted
    # in BetfairEnv._REWARD_OVERRIDE_KEYS so the env's unknown-key debug
    # log stays quiet, but the env never reads it. Defaults to 0.0 →
    # aux loss is plumbed-off and reward scale is unchanged from
    # session 01.
    "fill_prob_loss_weight": ("fill_prob_loss_weight",),
    # Scalping-active-management session 03: weight on the Gaussian NLL
    # risk aux loss. Same plumbing-off-by-default contract as
    # ``fill_prob_loss_weight`` — the env doesn't read it, but the
    # passthrough whitelist keeps the unknown-key debug log quiet.
    "risk_loss_weight": ("risk_loss_weight",),
}


def _reward_overrides_from_hp(hp: dict) -> dict:
    """Extract reward-config overrides from a hyperparameter dict.

    Returns a dict keyed by ``config["reward"]`` key names (not gene
    names). Unknown genes are ignored here — they belong to other
    subsystems. Genes whose values are ``None`` are dropped.
    """
    overrides: dict = {}
    for gene, cfg_keys in _REWARD_GENE_MAP.items():
        if gene not in hp:
            continue
        value = hp[gene]
        if value is None:
            continue
        for cfg_key in cfg_keys:
            overrides[cfg_key] = value
    return overrides


# -- Transition storage -------------------------------------------------------


@dataclass(slots=True)
class Transition:
    """A single (s, a, r, ...) transition from one env step.

    ``reward`` is the raw env reward (telemetry-truth; accumulates into
    EpisodeStats.total_reward and the log line). ``training_reward`` is
    the value actually fed into the advantage/return computation — it
    equals ``reward`` unless Session-1 reward clipping is active, in
    which case it is ``np.clip(reward, -c, +c)``. Keeping them separate
    preserves the raw+shaped≈total_reward invariant (clipping is a
    training-signal transform, not a reward-accumulator transform).

    ``fill_prob_labels`` is the scalping-active-management §02 supervised
    signal: per-slot binary outcome of each paired passive (1.0 =
    completed before race-off, 0.0 = went naked). ``NaN`` means "no
    resolved outcome for this tick-slot" — either no aggressive was
    placed here, the episode ended before resolution, or the data is
    from a pre-Session-02 rollout. The PPO update masks NaNs out of the
    BCE loss so unresolved samples do not contribute (see test
    ``test_fill_prob_excluded_from_loss_when_outcome_unresolved``). The
    1-element NaN default keeps directional (non-scalping) runs cheap:
    the update broadcasts the placeholder to ``(n, max_runners)`` of
    NaN and the mask rejects every element, yielding a zero loss.
    """

    obs: np.ndarray
    action: np.ndarray
    log_prob: float
    value: float
    reward: float
    done: bool
    training_reward: float = 0.0
    fill_prob_labels: np.ndarray = field(
        default_factory=lambda: np.array([np.nan], dtype=np.float32)
    )
    # Scalping-active-management §03 — per-slot realised ``locked_pnl``
    # (float £) of the paired scalp that placed on this tick, parallel
    # to ``fill_prob_labels``. ``NaN`` means "no resolved outcome" —
    # either no aggressive placed here, the pair went naked (no
    # realised locked_pnl to supervise against), or pre-Session-03
    # rollout data. Masked out of the Gaussian NLL the same way the
    # fill-prob BCE masks unresolved pairs. Same 1-element NaN default
    # so directional runs stay cheap.
    risk_labels: np.ndarray = field(
        default_factory=lambda: np.array([np.nan], dtype=np.float32)
    )


@dataclass
class Rollout:
    """A complete rollout (one or more episodes) worth of transitions."""

    transitions: list[Transition] = field(default_factory=list)

    def append(self, t: Transition) -> None:
        self.transitions.append(t)

    def __len__(self) -> int:
        return len(self.transitions)


@dataclass
class EpisodeStats:
    """Summary statistics for one completed episode (one day).

    ``total_reward`` is the full PPO training signal (raw + shaped).
    ``raw_pnl_reward`` is the component tied to actual money (race_pnl
    summed across the day plus the terminal day_pnl/budget bonus).
    ``shaped_bonus`` is everything else (early_pick_bonus, precision,
    efficiency penalty). They should sum to approximately
    ``total_reward`` — divergence indicates a reward-tracking bug.
    """

    day_date: str
    total_reward: float
    total_pnl: float
    bet_count: int
    winning_bets: int
    races_completed: int
    final_budget: float
    n_steps: int
    raw_pnl_reward: float = 0.0
    shaped_bonus: float = 0.0
    # Session 1 (arb-improvements): sum of per-step rewards AFTER the
    # training-signal reward clip. Equals ``total_reward`` when
    # ``reward_clip`` is 0 (off). Divergence indicates an outlier race
    # whose reward was clipped into the advantage buffer — a signal the
    # safety net caught an update that would otherwise have collapsed
    # the policy.
    clipped_reward_total: float = 0.0
    # Forced-arbitrage (scalping) rollups — Issue 05. Always zero on
    # directional runs; non-zero identifies a scalping episode so the
    # activity log / training monitor can surface arb activity.
    arbs_completed: int = 0
    arbs_naked: int = 0
    # Scalping-close-signal session 01 — count of pairs the agent
    # deliberately closed via ``close_signal`` (distinct from
    # ``arbs_completed``, whose passive legs filled naturally).
    arbs_closed: int = 0
    locked_pnl: float = 0.0
    naked_pnl: float = 0.0
    # Session 3 (arb-improvements) — action diagnostics.
    # ``bet_rate`` = fraction of rollout steps where any runner's sampled
    # ``signal`` action crossed the ±0.33 threshold (i.e. the policy
    # indicated a bet). ``arb_rate`` = fraction of completed arbs among
    # all arb attempts (paired + naked). Both are always in ``[0.0, 1.0]``.
    # ``signal_bias`` records the bias actually applied during the rollout
    # (0.0 when warmup is off or past-warmup).
    bet_rate: float = 0.0
    arb_rate: float = 0.0
    signal_bias: float = 0.0
    # Forced-arbitrage — per-pair completion details. Each entry is
    # ``{"selection_id": int, "back_price": float, "lay_price": float,
    # "locked_pnl": float, "race_idx": int}``. Empty on directional runs.
    # Consumed by ``_publish_progress`` to emit one activity-log line
    # per completed pair (Issue 05 — session 3).
    arb_events: list[dict] = field(default_factory=list)
    # Scalping-close-signal session 01 — per-pair close events. Each
    # entry is ``{"selection_id": int, "back_price": float,
    # "lay_price": float, "realised_pnl": float, "race_idx": int}``
    # for pairs the agent deliberately crossed the spread to close.
    # Consumed by ``_publish_progress`` to emit one ``pair_closed``
    # activity-log line per event.
    close_events: list[dict] = field(default_factory=list)


@dataclass
class TrainingStats:
    """Aggregate statistics from a full training run."""

    episodes_completed: int = 0
    total_steps: int = 0
    mean_reward: float = 0.0
    mean_pnl: float = 0.0
    mean_bet_count: float = 0.0
    final_policy_loss: float = 0.0
    final_value_loss: float = 0.0
    final_entropy: float = 0.0
    episode_stats: list[EpisodeStats] = field(default_factory=list)


# -- PPO Trainer ---------------------------------------------------------------


class PPOTrainer:
    """Proximal Policy Optimisation trainer for a single agent.

    Parameters
    ----------
    policy : BasePolicy
        The policy network to train.
    config : dict
        Project config (from config.yaml).
    hyperparams : dict | None
        Agent-specific hyperparameters.  Falls back to config defaults.
    progress_queue : asyncio.Queue | None
        If provided, progress dicts are put() here for WebSocket broadcast.
    device : str
        PyTorch device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        policy: BasePolicy,
        config: dict,
        hyperparams: dict | None = None,
        progress_queue: asyncio.Queue | None = None,
        device: str = "cpu",
        feature_cache: dict[str, list] | None = None,
        model_id: str | None = None,
        architecture_name: str | None = None,
    ) -> None:
        self.policy = policy.to(device)
        self.config = config
        self.device = device
        self.progress_queue = progress_queue
        self.feature_cache = feature_cache
        # Session 9: tag each episode record so a post-run analysis can
        # partition episodes.jsonl by agent and architecture without a
        # separate join. Both default to None so legacy call sites
        # (tests, direct instantiation) stay unchanged.
        self.model_id = model_id
        self.architecture_name = architecture_name

        hp = hyperparams or {}
        self.hyperparams = hp
        # Per-architecture default learning rate
        # (plans/naked-clip-and-stability, Session 02, 2026-04-18).
        # The transformer architecture saturates its action heads on
        # the first PPO update at the shared 3e-4 default — transformer
        # ``0a8cacd3`` ep-1 logged ``policy_loss = 1.04e17`` despite the
        # advantage-normalisation fix (plans/policy-startup-stability).
        # Each architecture class may override ``default_learning_rate``
        # (see agents/policy_network.py::PPOTransformerPolicy at 1.5e-4)
        # so the fresh-init default is appropriate for its saturation
        # profile; the GA still mutates LR around the sampled gene
        # value when ``learning_rate`` is present in ``hp``.
        policy_default_lr = float(
            getattr(type(policy), "default_learning_rate", 3e-4)
        )
        self.lr = hp.get("learning_rate", policy_default_lr)
        self.gamma = hp.get("gamma", 0.99)
        self.gae_lambda = hp.get("gae_lambda", 0.95)
        self.clip_epsilon = hp.get("ppo_clip_epsilon", 0.2)
        # Default halved 2026-04-18 per
        # plans/naked-clip-and-stability/purpose.md §3 — with
        # per-mini-batch advantage normalisation in place, 0.01
        # dominates the surrogate term and flattens the policy
        # under negative-reward pressure (observed rising entropy
        # 139→189 across transformer 0a8cacd3 ep 1–7). Gene range
        # is unchanged; only the fresh-init default halves
        # (hard_constraints.md §13).
        # -- Target-entropy controller (plans/entropy-control-v2) ---------
        # Entropy coefficient is a *learned variable* driven by a small
        # separate Adam optimiser over ``log_alpha = log(entropy_coeff)``.
        # When the policy's forward-pass entropy exceeds ``target_entropy``,
        # gradient descent on ``log_alpha`` drives it DOWN (less entropy
        # bonus → entropy falls toward target); when below, the reverse.
        # Clamped to ``[log(1e-5), log(0.1)]`` to prevent runaway during
        # calibration. See purpose.md for the Baseline-A 2026-04-19
        # entropy drift (139.6 → 201.3 across 64 agents × 15 episodes)
        # that this replaces the fixed-coefficient approach for.
        #
        # The alpha-optimiser is SEPARATE from the policy optimiser —
        # it holds its own momentum state and does NOT share anything
        # with ``self.optimiser``. ``self.entropy_coeff`` (the Python
        # float the surrogate loss reads) is refreshed from
        # ``log_alpha.exp().item()`` after every controller step; the
        # policy's autograd graph stays clean of the controller.
        initial_entropy_coeff = float(
            hp.get("entropy_coefficient", 0.005)
        )
        # ``target_entropy`` default raised 112.0 -> 150.0 on 2026-04-19
        # (Session 06). The original 112.0 target (80% of fresh-init
        # ep-1 entropy 139.6) turned out to sit BELOW the action
        # space's natural entropy floor: even with the Session-05
        # proportional controller aggressively driving alpha down to
        # ~0.002 (2.5x below the old fixed default), the smoke probe
        # still logged entropy slopes +1.47 / +2.90 over eps 1-3,
        # because no alpha value can coax entropy below the floor.
        # Action space is 14 runners x 5 dims = 70 Gaussian action
        # dims; fresh-init differential entropy ~139 corresponds
        # roughly to ``sigma ~= 1.8`` across all dims, and target
        # 112 would require ``sigma ~= 1.2`` on ALL 70 dims
        # simultaneously -- below what the policy naturally holds
        # early in training.
        #
        # Target 150 (~ +8% above fresh-init 139) sits ABOVE the
        # natural floor, so the controller has real authority: when
        # entropy drifts above 150, alpha shrinks; if entropy ever
        # dips below 150, alpha grows. The Session-05 diagnostic --
        # rapid alpha decay despite continuing entropy rise -- is
        # the evidence that the target, not the controller
        # mechanism, was mis-specified. See
        # plans/entropy-control-v2/lessons_learnt.md 2026-04-19.
        self._target_entropy: float = float(
            hp.get("target_entropy", 150.0)
        )
        self._log_alpha_min: float = math.log(1e-5)
        self._log_alpha_max: float = math.log(0.1)
        # float64 on log_alpha preserves ``log(x).exp() == x`` to
        # machine epsilon on the default hp values (0.005 / 0.01 /
        # 0.02); float32 round-trip visibly drifts at the 7th decimal
        # and breaks pre-existing ``entropy_coeff == <literal>`` tests.
        # The alpha optimiser works on a single scalar — dtype cost is
        # negligible.
        self._log_alpha = torch.tensor(
            math.log(max(initial_entropy_coeff, 1e-12)),
            dtype=torch.float64,
            device=self.device,
            requires_grad=True,
        )
        # Optimiser: plain SGD (momentum=0), NOT Adam. Rationale
        # (entropy-control-v2 Session 05, 2026-04-19): Adam
        # normalises the gradient magnitude away, producing ~``lr``-
        # sized steps regardless of how far entropy is from target.
        # On our one-call-per-episode cadence that made the
        # controller too timid to track even a moderate drift
        # (Session-04 post-launch observed entropy 139->192 across
        # 15 eps with Adam lr=3e-2). SGD gives proportional control:
        # ``log_alpha -= lr * grad`` where
        # ``grad = current_entropy - target_entropy`` (see the sign
        # derivation in _update_entropy_coefficient's docstring), so
        # a large error produces a large correction and the
        # controller self-adapts as it approaches target. The
        # ``log_alpha_min`` / ``log_alpha_max`` clamp is still the
        # ultimate safety net against runaway. See
        # plans/entropy-control-v2/lessons_learnt.md 2026-04-19.
        self._alpha_optimizer = torch.optim.SGD(
            [self._log_alpha],
            lr=float(hp.get("alpha_lr", 1e-2)),
            momentum=0.0,
        )
        # Effective coefficient consumed by the surrogate-loss formula.
        # Kept in sync with ``log_alpha`` after every controller step.
        self.entropy_coeff = float(self._log_alpha.exp().item())

        # -- Session 2 (arb-improvements) per-head entropy diagnostics ----
        # The per-head rolling window + progress-event plumbing from the
        # arb-improvements entropy-floor controller is retained for
        # operator visibility. When ``entropy_floor > 0`` the floor
        # controller layers a multiplicative scale-up on top of the
        # SAC-style base (``_entropy_coeff_base``), which now tracks
        # ``log_alpha.exp()`` rather than a fixed constant. With
        # ``entropy_floor == 0`` (default) the floor scaling is a no-op
        # and ``self.entropy_coeff`` equals the controller output
        # directly. See plans/entropy-control-v2/hard_constraints.md §10.
        self._entropy_coeff_base = float(self.entropy_coeff)
        self.entropy_floor = float(hp.get("entropy_floor", 0.0) or 0.0)
        self.entropy_floor_window = int(hp.get("entropy_floor_window", 10) or 10)
        self.entropy_boost_max = float(hp.get("entropy_boost_max", 10.0) or 10.0)
        # Number of consecutive PPO updates a single head must sit below
        # the floor before the ``entropy_collapse`` warning flag fires.
        # Defaults to 5 (per session_2_entropy_floor.md, "sensible default").
        self.entropy_collapse_patience = int(
            hp.get("entropy_collapse_patience", 5) or 5
        )
        # Rolling window of mean-across-heads entropy, used by the
        # coefficient controller.
        self._entropy_window: deque[float] = deque(
            maxlen=max(1, self.entropy_floor_window)
        )
        # Per-head rolling windows (for action_stats progress reporting)
        # and per-head consecutive-below-floor streak (for collapse flag).
        self._per_head_window: dict[str, deque[float]] = {
            h: deque(maxlen=max(1, self.entropy_floor_window))
            for h in _HEAD_NAMES
        }
        self._per_head_below_streak: dict[str, int] = {
            h: 0 for h in _HEAD_NAMES
        }
        self._entropy_collapse: bool = False
        self._entropy_coeff_active: float = self._entropy_coeff_base
        self.value_loss_coeff = hp.get("value_loss_coeff", 0.5)
        self.max_grad_norm = hp.get("max_grad_norm", 0.5)
        self.ppo_epochs = hp.get("ppo_epochs", 4)
        self.mini_batch_size = hp.get("mini_batch_size", 64)

        # KL early-stop threshold
        # (plans/naked-clip-and-stability, Session 02, 2026-04-18).
        # After each full epoch sweep of mini-batches in ``_ppo_update``,
        # approximate KL across the rollout is compared against this
        # threshold; if exceeded, the remaining epochs for the current
        # rollout are skipped. Default ``0.03`` is the literature
        # standard (Andrychowicz et al. 2021, Engstrom et al. 2020) and
        # ships in stable-baselines3 / CleanRL. Exposed here so the GA
        # gene system can mutate it later if useful.
        self.kl_early_stop_threshold = float(
            hp.get("kl_early_stop_threshold", 0.03)
        )
        # Set by ``_ppo_update`` when the threshold fires; diagnostic
        # only (surfaces in the returned loss_info dict).
        self._last_kl_early_stop_epoch: int | None = None
        self._last_approx_kl: float = 0.0

        # -- Reward centering (plans/naked-clip-and-stability §14) --------
        # Running-mean reward baseline, subtracted from per-step
        # training rewards before advantage computation. Lazy-init on
        # the first observed episode reward so the first rollout does
        # not train against a biased zero baseline. A constant
        # translation of returns cancels under the per-mini-batch
        # advantage normalisation that already lives in ``_ppo_update``
        # — centering fixes the "everything negative → explore wider"
        # pressure without changing advantage ordering in expectation.
        self._reward_ema: float = 0.0
        self._reward_ema_alpha: float = 0.01
        self._reward_ema_initialised: bool = False

        # -- Session 1 (arb-improvements) clipping knobs ---------------------
        # Three independent safety nets that default to 0.0 = off. When
        # off, training is byte-identical to pre-session behaviour. When
        # on, they stop a single outlier race from collapsing the policy
        # during epoch 1 (see plans/arb-improvements/purpose.md).
        #
        #  * reward_clip: per-step reward fed into the advantage/return
        #    buffer is clipped to [-c, +c]. Unclipped reward still flows
        #    into EpisodeStats, info["day_pnl"], the log line, and the
        #    monitor progress events — this is a training-signal-only
        #    transform (hard_constraints.md).
        #  * advantage_clip: per-transition advantage magnitude clamped
        #    before the PPO ratio multiplies it.
        #  * value_loss_clip: per-sample value-loss contribution capped
        #    at ``value_loss_clip ** 2`` before the batch mean.
        self.reward_clip = float(hp.get("reward_clip", 0.0) or 0.0)
        self.advantage_clip = float(hp.get("advantage_clip", 0.0) or 0.0)
        self.value_loss_clip = float(hp.get("value_loss_clip", 0.0) or 0.0)

        # Scalping-active-management session 02: aux-head BCE loss weight.
        # Default 0.0 → aux loss term contributes nothing to the total
        # PPO objective, so this session's total loss is byte-identical
        # to session 01 unless someone opts in. Sourced from hp first
        # (per-agent gene), then ``config["reward"]["fill_prob_loss_weight"]``
        # (project-wide default), then 0.0.
        self.fill_prob_loss_weight = float(
            hp.get(
                "fill_prob_loss_weight",
                config.get("reward", {}).get("fill_prob_loss_weight", 0.0),
            )
            or 0.0
        )

        # Scalping-active-management session 03: Gaussian-NLL weight on
        # the risk head. Default 0.0 → aux term contributes nothing and
        # this session's total loss is byte-identical to session 02
        # when ``fill_prob_loss_weight`` is also 0.0. Same precedence as
        # the fill-prob knob: hp first (per-agent gene), then
        # ``config["reward"]["risk_loss_weight"]``, then 0.0.
        self.risk_loss_weight = float(
            hp.get(
                "risk_loss_weight",
                config.get("reward", {}).get("risk_loss_weight", 0.0),
            )
            or 0.0
        )

        # -- Session 3 (arb-improvements) signal-bias warmup ----------------
        # Additive bias on the per-runner ``signal`` head mean during the
        # first ``signal_bias_warmup`` epochs of training. Linear decay:
        # at epoch ``e``, the bias passed into the policy is
        # ``magnitude * max(0, 1 - e/warmup)``; at ``e >= warmup`` it is
        # 0.0 and the policy is byte-identical to the unbiased path.
        #
        # Both knobs default to off: ``warmup=0`` OR ``magnitude=0`` →
        # every rollout runs with ``signal_bias=0.0`` (see
        # ``_current_signal_bias``). Positive ``magnitude`` biases toward
        # "back"; negative toward "lay".
        self.signal_bias_warmup = int(hp.get("signal_bias_warmup", 0) or 0)
        self.signal_bias_magnitude = float(
            hp.get("signal_bias_magnitude", 0.0) or 0.0
        )
        # Tracked by ``train()`` so ``_collect_rollout`` can compute the
        # decayed bias for the current epoch without re-threading args.
        self._current_epoch: int = 0

        # Build per-agent reward overrides from the sampled genes. This is
        # how reward-shaping hyperparameters reach BetfairEnv — previously
        # these genes were sampled but silently dropped here, so every
        # agent trained with identical reward shaping.
        self.reward_overrides = _reward_overrides_from_hp(hp)
        self.market_type_filter = hp.get("market_type_filter", "BOTH")
        # Forced-arbitrage mechanics override (Issue 05, session 3). Only
        # populated when the gene is present in hp — arb_spread_scale is
        # a float in [0.5, 2.0] stretching / compressing the agent's tick
        # spread. Passed to the env separately from reward_overrides
        # because it changes mechanics, not reward.
        self.scalping_overrides: dict = {}
        if "arb_spread_scale" in hp and hp["arb_spread_scale"] is not None:
            self.scalping_overrides["arb_spread_scale"] = float(hp["arb_spread_scale"])
        # Per-agent scalping_mode flag (may be pinned at population init
        # from the run-level toggle). Falls back to the env reading it
        # from config when the gene is missing.
        self._scalping_mode_override: bool | None = (
            bool(hp["scalping_mode"]) if "scalping_mode" in hp else None
        )

        self.optimiser = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        # First-5-update linear LR warmup
        # (plans/policy-startup-stability, Session 01, 2026-04-18).
        # Defence-in-depth on top of per-mini-batch advantage
        # normalisation. The smoke test at the end of Session 01
        # showed a residual 1.48e+12 episode-1 policy_loss spike with
        # normalisation alone — the rollout's policy/value variance
        # on a fresh LSTM still produces a first-update gradient
        # large enough to shift the action heads. Scaling LR linearly
        # over the first 5 PPO updates lets the optimiser "ease in"
        # without adding a new tunable surface (the ramp is fixed,
        # not a gene). ``self._update_count`` counts how many
        # ``_ppo_update`` calls have started; warmup_factor reaches
        # 1.0 on the 5th update and stays there forever after.
        self._base_learning_rate: float = float(self.lr)
        self._update_count: int = 0
        self._lr_warmup_updates: int = 5

        # Logging setup
        log_dir = Path(config.get("paths", {}).get("logs", "logs"))
        self.log_dir = log_dir / "training"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Smoke-test probe tag (Session 04 of
        # plans/naked-clip-and-stability). When True, every row
        # written by ``_log_episode`` carries ``smoke_test: true`` so
        # downstream tooling (learning-curves panel, the gate's
        # assertion evaluator) can distinguish probe rows from real
        # training rows in the shared episodes.jsonl stream.
        # hard_constraints.md §16 makes this flag non-negotiable.
        self.smoke_test_tag: bool = False

    # -- Public API -----------------------------------------------------------

    def train(
        self,
        days: list[Day],
        n_epochs: int = 1,
    ) -> TrainingStats:
        """Train the policy on the given days for n_epochs passes.

        Each day is one episode.  The policy is updated after every episode
        (on-policy).

        Parameters
        ----------
        days : list[Day]
            Training days (each becomes one BetfairEnv episode).
        n_epochs : int
            Number of passes over the full day list.

        Returns
        -------
        TrainingStats
            Aggregate training statistics.
        """
        total_episodes = len(days) * n_epochs
        tracker = ProgressTracker(total=total_episodes, label="Training episodes")
        tracker.reset_timer()

        stats = TrainingStats()
        all_rewards: list[float] = []
        all_pnls: list[float] = []
        all_bets: list[float] = []

        for epoch in range(n_epochs):
            # Session 3: visible to _collect_rollout so it can compute the
            # decayed signal bias for this epoch without extra args.
            self._current_epoch = epoch
            for day in days:
                # Collect rollout
                rollout, ep_stats = self._collect_rollout(day)
                stats.episode_stats.append(ep_stats)
                stats.episodes_completed += 1
                stats.total_steps += ep_stats.n_steps

                all_rewards.append(ep_stats.total_reward)
                all_pnls.append(ep_stats.total_pnl)
                all_bets.append(ep_stats.bet_count)

                # PPO update
                if len(rollout) > 0:
                    loss_info = self._ppo_update(rollout)
                    stats.final_policy_loss = loss_info["policy_loss"]
                    stats.final_value_loss = loss_info["value_loss"]
                    stats.final_entropy = loss_info["entropy"]
                else:
                    loss_info = {
                        "policy_loss": 0.0,
                        "value_loss": 0.0,
                        "entropy": 0.0,
                    }

                tracker.tick()

                # Log and publish progress
                self._log_episode(ep_stats, loss_info, tracker)
                self._publish_progress(ep_stats, loss_info, tracker)

        if all_rewards:
            stats.mean_reward = float(np.mean(all_rewards))
            stats.mean_pnl = float(np.mean(all_pnls))
            stats.mean_bet_count = float(np.mean(all_bets))

        return stats

    # -- Rollout collection ---------------------------------------------------

    def _collect_rollout(self, day: Day) -> tuple[Rollout, EpisodeStats]:
        """Run one episode (one day) and collect transitions.

        Optimised hot loop: pre-allocates GPU tensors for observations,
        keeps action log_std on GPU, and minimises per-step Python overhead.
        """
        rollout_start = time.perf_counter()
        env = BetfairEnv(
            day,
            self.config,
            feature_cache=self.feature_cache,
            reward_overrides=self.reward_overrides,
            market_type_filter=self.market_type_filter,
            scalping_mode=self._scalping_mode_override,
            scalping_overrides=self.scalping_overrides or None,
        )
        obs, info = env.reset()

        rollout = Rollout()
        hidden_state = self.policy.init_hidden(batch_size=1)
        hidden_state = (
            hidden_state[0].to(self.device),
            hidden_state[1].to(self.device),
        )

        total_reward = 0.0
        clipped_reward_total = 0.0
        n_steps = 0
        done = False
        # Session 3: per-rollout diagnostics. ``bet_steps`` tallies steps
        # whose sampled signal magnitude crossed the bet threshold for
        # any runner — the numerator of ``bet_rate``. The signal bias
        # for this rollout is frozen at the start of the epoch so the
        # log line reflects what the policy actually saw.
        current_signal_bias = _signal_bias_for_epoch(
            self._current_epoch,
            self.signal_bias_warmup,
            self.signal_bias_magnitude,
        )
        bet_steps = 0
        # Per-runner signal slice bounds on the flat action vector. Signal
        # is head 0 in the ``[signal × N | stake × N | ...]`` layout, so
        # it lives at indices ``[0, max_runners)``. Falls back to all
        # dims for minimal stub policies without ``max_runners``.
        max_runners = getattr(self.policy, "max_runners", None)

        # Scalping-active-management §02: decision-time capture store for
        # the fill-prob head. Keyed by ``pair_id`` → ``(transition_idx,
        # slot_idx)``. Populated below when the rollout loop observes an
        # ``aggressive_placed=True`` entry in ``info["action_debug"]`` and
        # finds a matching ``Bet`` in ``env.bet_manager.bets``. At episode
        # end we walk ``env.all_settled_bets``, classify each pair
        # (completed / naked), and write 0/1 labels into the corresponding
        # ``Transition.fill_prob_labels[slot_idx]``. Empty on directional
        # runs — every label stays NaN so the BCE mask rejects it.
        pair_to_transition: dict[str, tuple[int, int]] = {}

        # Pre-allocate a reusable GPU tensor for single-step observations
        obs_dim = obs.shape[0]
        obs_buffer = torch.empty(
            1, obs_dim, dtype=torch.float32, device=self.device,
        )

        with torch.no_grad():
            while not done:
                # Copy obs into pre-allocated GPU buffer (avoids tensor creation)
                obs_buffer[0] = torch.as_tensor(obs, dtype=torch.float32)

                # Session 3: pass the current-epoch signal bias into the
                # policy only when armed. Keeping the default-off path on
                # the original two-arg signature means callers with
                # pre-session-3 policy stubs (e.g. the Session 1/2 tests)
                # continue to work unchanged.
                if current_signal_bias != 0.0:
                    out: PolicyOutput = self.policy(
                        obs_buffer, hidden_state,
                        signal_bias=current_signal_bias,
                    )
                else:
                    out = self.policy(obs_buffer, hidden_state)
                hidden_state = out.hidden_state

                # Sample action — keep computation on GPU
                std = out.action_log_std.exp()
                action_mean = out.action_mean
                noise = torch.randn_like(action_mean)
                action = action_mean + std * noise
                log_prob = (
                    -0.5 * ((action - action_mean) / std).pow(2)
                    - std.log()
                    - 0.5 * 1.8378770664093453  # log(2*pi)
                ).sum(dim=-1)
                value = out.value.squeeze(-1)

                action_np = action.squeeze(0).cpu().numpy()
                np.clip(action_np, -1.0, 1.0, out=action_np)

                next_obs, reward, terminated, truncated, next_info = env.step(action_np)
                done = terminated or truncated

                # Session 1: clip the TRAINING SIGNAL only. The raw
                # reward is still what EpisodeStats/info/logs see.
                raw_reward = float(reward)
                if self.reward_clip > 0.0:
                    training_reward = float(
                        np.clip(raw_reward, -self.reward_clip, self.reward_clip)
                    )
                else:
                    training_reward = raw_reward

                # Scalping-active-management §02: snapshot the per-runner
                # fill-probability prediction at this tick. Shape
                # ``(max_runners,)`` when the policy produces the head,
                # scalar default (0.5) otherwise — we fall back to ``None``
                # in that case so no stamping happens. ``.detach().cpu()``
                # before the no-grad block exits avoids retaining
                # computation graphs.
                fp_tensor = getattr(out, "fill_prob_per_runner", None)
                fp_per_runner_t: np.ndarray | None
                if (
                    fp_tensor is not None
                    and max_runners is not None
                    and fp_tensor.numel() >= max_runners
                ):
                    fp_per_runner_t = (
                        fp_tensor.detach().cpu().numpy().reshape(-1)
                    )
                else:
                    fp_per_runner_t = None

                # Scalping-active-management §03: snapshot per-runner
                # risk-head outputs (mean + stddev) using the same
                # detach→cpu→numpy idiom. ``stddev = exp(0.5 * log_var)``
                # is computed once here so the per-tick stamp on ``Bet``
                # (and, later, parquet consumers) don't need to repeat
                # the math. The log-var tensor coming out of the head is
                # already clamped inside ``forward``.
                risk_mean_tensor = getattr(
                    out, "predicted_locked_pnl_per_runner", None,
                )
                risk_log_var_tensor = getattr(
                    out, "predicted_locked_log_var_per_runner", None,
                )
                risk_mean_t: np.ndarray | None
                risk_stddev_t: np.ndarray | None
                if (
                    risk_mean_tensor is not None
                    and risk_log_var_tensor is not None
                    and max_runners is not None
                    and risk_mean_tensor.numel() >= max_runners
                    and risk_log_var_tensor.numel() >= max_runners
                ):
                    risk_mean_t = (
                        risk_mean_tensor.detach().cpu().numpy().reshape(-1)
                    )
                    risk_stddev_t = np.exp(
                        0.5 * risk_log_var_tensor
                        .detach().cpu().numpy().reshape(-1)
                    )
                else:
                    risk_mean_t = None
                    risk_stddev_t = None

                # Per-transition per-slot label array, filled with NaN;
                # episode-end backfill will flip the slots that had a
                # resolved pair outcome.
                if max_runners is not None:
                    labels_arr = np.full(
                        max_runners, np.nan, dtype=np.float32,
                    )
                    risk_labels_arr = np.full(
                        max_runners, np.nan, dtype=np.float32,
                    )
                else:
                    labels_arr = np.array([np.nan], dtype=np.float32)
                    risk_labels_arr = np.array([np.nan], dtype=np.float32)

                rollout.append(Transition(
                    obs=obs,
                    action=action_np,
                    log_prob=float(log_prob.item()),
                    value=float(value.item()),
                    reward=raw_reward,
                    done=done,
                    training_reward=training_reward,
                    fill_prob_labels=labels_arr,
                    risk_labels=risk_labels_arr,
                ))

                # Scalping-active-management §02/§03: stamp the decision-
                # time aux-head predictions onto newly-placed aggressive
                # ``Bet`` objects. The paired passive inherits them later
                # inside ``PassiveOrderBook.on_tick`` via ``pair_id``
                # lookup. Record the pair → (transition, slot) mapping so
                # the episode-end backfill can write the 0/1 fill-prob
                # label and the realised-locked-pnl risk label into the
                # right cells of ``fill_prob_labels`` / ``risk_labels``.
                if fp_per_runner_t is not None:
                    action_debug = next_info.get("action_debug", {})
                    bm = getattr(env, "bet_manager", None)
                    if bm is not None and action_debug:
                        sid_to_slot = env.current_runner_to_slot()
                        transition_idx = len(rollout) - 1
                        for sid, entry in action_debug.items():
                            if not entry.get("aggressive_placed", False):
                                continue
                            slot_idx = sid_to_slot.get(sid)
                            if (
                                slot_idx is None
                                or slot_idx >= fp_per_runner_t.shape[0]
                            ):
                                continue
                            fp_val = float(fp_per_runner_t[slot_idx])
                            risk_mean_val: float | None = (
                                float(risk_mean_t[slot_idx])
                                if risk_mean_t is not None
                                and slot_idx < risk_mean_t.shape[0]
                                else None
                            )
                            risk_stddev_val: float | None = (
                                float(risk_stddev_t[slot_idx])
                                if risk_stddev_t is not None
                                and slot_idx < risk_stddev_t.shape[0]
                                else None
                            )
                            # Newest matching Bet — scan backwards so we
                            # pick up the one just placed this tick.
                            for bet in reversed(bm.bets):
                                if (
                                    bet.selection_id == sid
                                    and bet.fill_prob_at_placement is None
                                ):
                                    bet.fill_prob_at_placement = fp_val
                                    bet.predicted_locked_pnl_at_placement = (
                                        risk_mean_val
                                    )
                                    bet.predicted_locked_stddev_at_placement = (
                                        risk_stddev_val
                                    )
                                    if bet.pair_id is not None:
                                        pair_to_transition[bet.pair_id] = (
                                            transition_idx, slot_idx,
                                        )
                                    break

                # Session 3: bet_rate — count this step if any runner's
                # sampled ``signal`` action magnitude exceeded the bet
                # threshold. Signal occupies the first ``max_runners``
                # dims of the flat action vector.
                if max_runners is not None and action_np.shape[0] >= max_runners:
                    signal_slice = np.abs(action_np[:max_runners])
                    if signal_slice.max() > _BET_SIGNAL_THRESHOLD:
                        bet_steps += 1
                else:
                    # Fallback for stub policies without max_runners —
                    # treat the whole action vector as the "signal" head.
                    if np.abs(action_np).max() > _BET_SIGNAL_THRESHOLD:
                        bet_steps += 1

                total_reward += raw_reward
                clipped_reward_total += training_reward
                n_steps += 1
                obs = next_obs
                info = next_info

        # Scalping-active-management §02/§03: episode-end label backfill.
        # Classify each pair that placed an aggressive leg this episode:
        # a pair with ≥2 settled bets (aggressive + passive) completed
        # before race-off → fill-prob label 1.0 + risk label = realised
        # locked_pnl; a pair with only the aggressive leg went naked →
        # fill-prob label 0.0 + risk label stays NaN (no realised
        # locked_pnl defined for a naked pair — the NLL mask rejects it).
        # Bets appear in ``env.all_settled_bets`` across races, so this
        # is the correct source (per CLAUDE.md "realised_pnl is last-
        # race-only"). When ``pair_to_transition`` is empty (directional
        # run or no aggressives placed), this loop is a no-op.
        if pair_to_transition:
            pair_bets: dict[str, list] = {}
            for b in env.all_settled_bets:
                if b.pair_id is not None:
                    pair_bets.setdefault(b.pair_id, []).append(b)
            for pair_id, (tr_idx, slot_idx) in pair_to_transition.items():
                legs = pair_bets.get(pair_id, [])
                count = len(legs)
                if count <= 0:
                    # Defensive: aggressive never landed in all_settled_bets
                    # (e.g. the race was aborted). Leave labels as NaN so
                    # the BCE / NLL masks reject them — no fake supervision.
                    continue
                if not (0 <= tr_idx < len(rollout.transitions)):
                    continue
                tr = rollout.transitions[tr_idx]
                if slot_idx < tr.fill_prob_labels.shape[0]:
                    tr.fill_prob_labels[slot_idx] = 1.0 if count >= 2 else 0.0
                # Risk label: realised ``locked_pnl`` of the completed
                # pair. Inline the same math
                # ``BetManager.get_paired_positions`` uses so we don't
                # need a per-market BetManager here (races share
                # ``env.all_settled_bets`` but their fresh-per-race
                # BetManagers are gone by the time this backfills). Only
                # completed pairs (both legs) contribute — naked pairs
                # have no realised locked_pnl to supervise against.
                if count >= 2 and slot_idx < tr.risk_labels.shape[0]:
                    from env.bet_manager import BetSide as _BetSide
                    backs = [b for b in legs if b.side is _BetSide.BACK]
                    lays = [b for b in legs if b.side is _BetSide.LAY]
                    if backs and lays:
                        back = max(backs, key=lambda b: b.average_price)
                        lay = min(lays, key=lambda b: b.average_price)
                        commission = 0.05
                        win_pnl = (
                            back.matched_stake
                            * (back.average_price - 1.0)
                            * (1.0 - commission)
                            - lay.matched_stake
                            * (lay.average_price - 1.0)
                        )
                        lose_pnl = (
                            -back.matched_stake
                            + lay.matched_stake * (1.0 - commission)
                        )
                        locked = max(0.0, min(win_pnl, lose_pnl))
                        tr.risk_labels[slot_idx] = float(locked)

        rollout_elapsed = time.perf_counter() - rollout_start
        logger.info(
            "Rollout %s: %d steps in %.2fs (%.0f steps/s)",
            day.date, n_steps, rollout_elapsed,
            n_steps / rollout_elapsed if rollout_elapsed > 0 else 0,
        )

        ep_stats = EpisodeStats(
            day_date=day.date,
            total_reward=total_reward,
            # Use ``day_pnl`` (accumulated across all races in the episode),
            # not ``realised_pnl`` (which is only the LAST race because the
            # env recreates a fresh BetManager per race).
            total_pnl=info.get("day_pnl", 0.0),
            bet_count=info.get("bet_count", 0),
            winning_bets=info.get("winning_bets", 0),
            races_completed=info.get("races_completed", 0),
            final_budget=info.get("budget", 0.0),
            n_steps=n_steps,
            raw_pnl_reward=info.get("raw_pnl_reward", 0.0),
            shaped_bonus=info.get("shaped_bonus", 0.0),
            clipped_reward_total=clipped_reward_total,
            arbs_completed=int(info.get("arbs_completed", 0) or 0),
            arbs_naked=int(info.get("arbs_naked", 0) or 0),
            arbs_closed=int(info.get("arbs_closed", 0) or 0),
            locked_pnl=float(info.get("locked_pnl", 0.0) or 0.0),
            naked_pnl=float(info.get("naked_pnl", 0.0) or 0.0),
            bet_rate=(float(bet_steps) / float(n_steps)) if n_steps > 0 else 0.0,
            arb_rate=_compute_arb_rate(
                int(info.get("arbs_completed", 0) or 0),
                int(info.get("arbs_naked", 0) or 0),
            ),
            signal_bias=float(current_signal_bias),
            arb_events=list(info.get("arb_events", []) or []),
            close_events=list(info.get("close_events", []) or []),
        )

        return rollout, ep_stats

    # -- GAE advantage estimation ---------------------------------------------

    def _compute_advantages(
        self, rollout: Rollout,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and discounted returns.

        Returns
        -------
        advantages : (N,)
        returns : (N,)
        """
        transitions = rollout.transitions
        n = len(transitions)
        advantages = torch.zeros(n, dtype=torch.float32)
        returns = torch.zeros(n, dtype=torch.float32)

        last_gae = 0.0
        last_value = 0.0  # terminal state value = 0

        for t in reversed(range(n)):
            tr = transitions[t]
            if tr.done:
                next_value = 0.0
                last_gae = 0.0
            else:
                next_value = (
                    transitions[t + 1].value if t + 1 < n else last_value
                )

            # Session 1: advantage/return computation uses the clipped
            # training signal (equals tr.reward when reward_clip is off).
            # Reward centering (plans/naked-clip-and-stability §14):
            # subtract the EMA baseline so advantages are not biased by
            # the running reward level. The subtraction is a pure
            # translation of returns that the per-mini-batch advantage
            # normalisation downstream erases in expectation — see
            # ``test_centering_preserves_advantage_ordering``.
            centered_reward = tr.training_reward - self._reward_ema
            delta = centered_reward + self.gamma * next_value - tr.value
            last_gae = delta + self.gamma * self.gae_lambda * last_gae
            advantages[t] = last_gae
            returns[t] = last_gae + tr.value

        return advantages, returns

    # -- Reward-centering baseline --------------------------------------------

    def _update_reward_baseline(self, per_step_mean_reward: float) -> None:
        """EMA-updated reward baseline, in PER-STEP reward units.

        Subtracted from per-step training rewards inside
        ``_compute_advantages`` to prevent the policy from flattening
        under uniformly-negative rewards (see
        plans/naked-clip-and-stability/purpose.md §3). The constant
        subtraction does not change advantage ordering in expectation —
        per-mini-batch normalisation in ``_ppo_update`` erases any
        translation. First call initialises on the observed reward
        rather than blending from zero (a zero-initialised EMA
        produces biased advantages for the first rollout).

        UNITS CONTRACT: callers MUST pass a per-step mean, not an
        episode sum. The subtraction inside ``_compute_advantages`` is
        per-step; feeding an episode sum shifts each step's reward by
        the whole-episode total, exploding returns through the GAE
        accumulator. See
        ``test_reward_baseline_stores_per_step_mean_not_episode_sum``
        and plans/naked-clip-and-stability/lessons_learnt.md.
        """
        if not self._reward_ema_initialised:
            self._reward_ema = float(per_step_mean_reward)
            self._reward_ema_initialised = True
            return
        self._reward_ema = (
            (1.0 - self._reward_ema_alpha) * self._reward_ema
            + self._reward_ema_alpha * float(per_step_mean_reward)
        )

    # -- PPO update -----------------------------------------------------------

    def _ppo_update(self, rollout: Rollout) -> dict[str, float]:
        """Run PPO optimisation epochs on collected rollout data.

        Returns a dict with final loss components.
        """
        ppo_start = time.perf_counter()
        transitions = rollout.transitions
        n = len(transitions)

        # Linear LR warmup over the first ``_lr_warmup_updates`` PPO
        # updates (plans/policy-startup-stability, Session 01,
        # 2026-04-18). On update 0 the optimiser sees lr * 1/5; by
        # update 4 it sees lr * 5/5 = lr; from update 5 onward the
        # factor stays at 1.0. Pairs with the per-mini-batch
        # advantage normalisation in the surrogate-loss branch below;
        # the smoke test showed normalisation alone left a residual
        # ≫100 spike on update 0, and the warmup is the defence-in-
        # depth that brings it down to the bounded regime.
        warmup_factor = min(
            1.0,
            (self._update_count + 1) / float(self._lr_warmup_updates),
        )
        for param_group in self.optimiser.param_groups:
            param_group["lr"] = self._base_learning_rate * warmup_factor
        self._update_count += 1

        # Prepare tensors — build numpy arrays first, then transfer to GPU.
        # Use pinned memory when CUDA is available for faster async transfer.
        obs_np = np.array([t.obs for t in transitions], dtype=np.float32)
        action_np = np.array([t.action for t in transitions], dtype=np.float32)
        lp_np = np.array([t.log_prob for t in transitions], dtype=np.float32)

        # Scalping-active-management §02: build the fill-prob labels
        # batch. Each transition carries either a ``(max_runners,)`` array
        # (populated with NaN + optional 0/1 at resolved slots) or the
        # 1-element NaN default. Pad placeholders to ``max_runners``
        # width so stacking yields a uniform ``(n, max_runners)`` tensor
        # the minibatch loop can slice. When the policy has no
        # ``max_runners`` attribute (stub policies used in unit tests),
        # fall back to width 1 — the BCE loss will see an all-NaN mask
        # and contribute zero.
        fp_max_runners = int(
            getattr(self.policy, "max_runners", 0) or 0
        )
        if fp_max_runners <= 0:
            # Width-1 fallback: every transition's labels fit directly.
            fp_labels_np = np.stack(
                [t.fill_prob_labels[:1] for t in transitions], axis=0,
            ).astype(np.float32)
            # Scalping-active-management §03 — same width-1 fallback for
            # the realised-locked-pnl labels that feed the risk NLL.
            risk_labels_np = np.stack(
                [t.risk_labels[:1] for t in transitions], axis=0,
            ).astype(np.float32)
        else:
            fp_labels_np = np.full(
                (n, fp_max_runners), np.nan, dtype=np.float32,
            )
            risk_labels_np = np.full(
                (n, fp_max_runners), np.nan, dtype=np.float32,
            )
            for i, t in enumerate(transitions):
                src = t.fill_prob_labels
                w = min(src.shape[0], fp_max_runners)
                fp_labels_np[i, :w] = src[:w]
                r_src = t.risk_labels
                r_w = min(r_src.shape[0], fp_max_runners)
                risk_labels_np[i, :r_w] = r_src[:r_w]

        if self.device != "cpu":
            obs_pin = torch.from_numpy(obs_np).pin_memory()
            action_pin = torch.from_numpy(action_np).pin_memory()
            lp_pin = torch.from_numpy(lp_np).pin_memory()
            fp_pin = torch.from_numpy(fp_labels_np).pin_memory()
            risk_pin = torch.from_numpy(risk_labels_np).pin_memory()
            obs_batch = obs_pin.to(self.device, non_blocking=True)
            action_batch = action_pin.to(self.device, non_blocking=True)
            old_log_probs = lp_pin.to(self.device, non_blocking=True)
            fp_labels_batch = fp_pin.to(self.device, non_blocking=True)
            risk_labels_batch = risk_pin.to(self.device, non_blocking=True)
        else:
            obs_batch = torch.from_numpy(obs_np)
            action_batch = torch.from_numpy(action_np)
            old_log_probs = torch.from_numpy(lp_np)
            fp_labels_batch = torch.from_numpy(fp_labels_np)
            risk_labels_batch = torch.from_numpy(risk_labels_np)

        advantages, returns = self._compute_advantages(rollout)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        # Reward centering (plans/naked-clip-and-stability §14): update
        # the EMA baseline with THIS rollout's per-step mean training
        # reward AFTER advantages have been computed, so the current
        # rollout's advantages use the pre-update EMA (no
        # self-referential leakage).
        #
        # UNITS: the EMA is in PER-STEP reward units because
        # ``_compute_advantages`` subtracts ``_reward_ema`` from each
        # ``tr.training_reward`` (per-step). Feeding the episode SUM
        # here would create a units mismatch: a long rollout with a
        # total reward of −1551 would poison ep2+ by shifting every
        # per-step reward by +1551, which GAE then accumulates into
        # returns O(tens of thousands), blowing up the value head
        # (observed 2026-04-18 smoke probe, value_loss 6.8e+08 on
        # ep2 of a fresh transformer). See
        # plans/naked-clip-and-stability/lessons_learnt.md.
        per_step_mean_reward = (
            float(sum(tr.training_reward for tr in transitions))
            / max(1, len(transitions))
        )
        self._update_reward_baseline(per_step_mean_reward)

        # Per-mini-batch advantage normalisation lives inside the loop
        # below (plans/policy-startup-stability, Session 01, 2026-04-18).
        # The prior per-rollout normalisation was dropped in favour of
        # the literature-standard per-batch recipe — see hard_constraints.md §5.

        policy_losses = []
        value_losses = []
        entropies = []
        fill_prob_losses: list[float] = []
        risk_losses: list[float] = []
        # Session 2: accumulate per-head entropy contributions across the
        # mini-batch loop so we can push a single average-per-head sample
        # to the rolling window at the end of this update. Only heads
        # actually present in the policy's action space are populated —
        # a directional (4-head) policy omits ``arb_spread`` so it never
        # trips the collapse detector for a head it can't produce.
        per_head_sums: dict[str, float] = {}
        per_head_count: int = 0

        # Reset KL early-stop diagnostics for this rollout.
        self._last_kl_early_stop_epoch = None
        self._last_approx_kl = 0.0
        epochs_completed = 0

        for epoch_idx in range(self.ppo_epochs):
            # Generate random mini-batch indices
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, self.mini_batch_size):
                end = min(start + self.mini_batch_size, n)
                mb_idx = indices[start:end]

                mb_obs = obs_batch[mb_idx]
                mb_actions = action_batch[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                # Session 1: advantage magnitude clamp. Sits in front of
                # the PPO ratio, not in place of max_grad_norm. Default
                # 0.0 = off.
                if self.advantage_clip > 0.0:
                    mb_advantages = torch.clamp(
                        mb_advantages,
                        -self.advantage_clip,
                        self.advantage_clip,
                    )

                # Forward pass (no LSTM state for mini-batch -- treat each
                # transition independently during optimisation)
                out = self.policy(mb_obs)
                std = out.action_log_std.exp()
                dist = Normal(out.action_mean, std)

                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                per_dim_entropy = dist.entropy()  # (batch, action_dim)
                entropy = per_dim_entropy.sum(dim=-1).mean()
                # Session 2: per-head entropy slices for the floor
                # controller + progress event.
                mb_per_head = self._compute_per_head_entropy(per_dim_entropy)
                for h_name, h_val in mb_per_head.items():
                    per_head_sums[h_name] = (
                        per_head_sums.get(h_name, 0.0) + h_val
                    )
                per_head_count += 1
                values = out.value.squeeze(-1)

                # Per-mini-batch advantage normalisation
                # (plans/policy-startup-stability, Session 01, 2026-04-18).
                # Stabilises the PPO update against large-magnitude
                # rewards that would otherwise produce a first-rollout
                # policy_loss spike and saturate action heads (most
                # notably close_signal — see purpose.md, agent
                # 3e37822e-c9fa). Literature standard: Engstrom et al.
                # 2020, "Implementation Matters in Deep Policy Gradients".
                # Ships in stable-baselines3, CleanRL, RLlib. Applied to
                # the policy-loss branch only; the value loss still uses
                # the un-normalised returns (hard_constraints.md §6).
                # ``eps = 1e-8`` guards the degenerate case where every
                # advantage in the batch is identical (std = 0).
                if mb_advantages.numel() > 1:
                    adv_mean = mb_advantages.mean()
                    adv_std = mb_advantages.std() + 1e-8
                    mb_advantages = (mb_advantages - adv_mean) / adv_std

                # PPO clipped surrogate. ``log_ratio`` is clamped to
                # ``[-LOG_RATIO_CLAMP, +LOG_RATIO_CLAMP]`` before
                # ``.exp()`` — see the module-level constant for the
                # rationale (tightened from ±20 to ±5 on 2026-04-18
                # after the smoke probe showed the looser bound wasn't
                # actually capping policy_loss magnitude).
                # ``|log_ratio| ≪ LOG_RATIO_CLAMP`` in normal PPO
                # updates so the clamp is a no-op in healthy operation;
                # it only bites when an aggressive first-minibatch
                # update has already driven the ratio toward what would
                # otherwise be a ``policy_loss`` spike.
                log_ratio = torch.clamp(
                    new_log_probs - mb_old_log_probs,
                    min=-LOG_RATIO_CLAMP, max=LOG_RATIO_CLAMP,
                )
                ratio = log_ratio.exp()
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss. Session 1 adds an optional per-sample cap:
                # clamp each squared residual at ``value_loss_clip ** 2``
                # before the batch mean, so one outlier return cannot
                # dominate the update. Default 0.0 = off → identical to
                # the previous F.mse_loss(values, mb_returns).
                if self.value_loss_clip > 0.0:
                    sq_err = (values - mb_returns).pow(2)
                    value_loss = torch.clamp(
                        sq_err, max=self.value_loss_clip ** 2
                    ).mean()
                else:
                    value_loss = nn.functional.mse_loss(values, mb_returns)

                # Scalping-active-management §02: auxiliary BCE loss on
                # the fill-probability head. See :func:`_compute_fill_prob_bce`
                # — only slots with a resolved label (non-NaN) contribute;
                # unresolved samples are masked out (see
                # ``test_fill_prob_excluded_from_loss_when_outcome_unresolved``).
                fp_preds = getattr(out, "fill_prob_per_runner", None)
                mb_fp_labels = fp_labels_batch[mb_idx]
                if (
                    fp_preds is not None
                    and fp_preds.shape == mb_fp_labels.shape
                ):
                    fill_prob_loss = _compute_fill_prob_bce(
                        fp_preds, mb_fp_labels,
                    )
                else:
                    # Shape mismatch (e.g. stub policy with a 1-element
                    # default fill_prob_per_runner) — no aux loss.
                    fill_prob_loss = torch.zeros((), device=mb_obs.device)

                # Scalping-active-management §03: auxiliary Gaussian NLL
                # loss on the risk head. Same shape contract as the
                # fill-prob aux loss — unresolved slots (NaN labels) are
                # masked out by :func:`_compute_risk_nll`, and a shape
                # mismatch (stub policy with default zero-tensors) means
                # this session's aux loss contributes zero.
                risk_mean_preds = getattr(
                    out, "predicted_locked_pnl_per_runner", None,
                )
                risk_log_var_preds = getattr(
                    out, "predicted_locked_log_var_per_runner", None,
                )
                mb_risk_labels = risk_labels_batch[mb_idx]
                if (
                    risk_mean_preds is not None
                    and risk_log_var_preds is not None
                    and risk_mean_preds.shape == mb_risk_labels.shape
                    and risk_log_var_preds.shape == mb_risk_labels.shape
                ):
                    risk_loss = _compute_risk_nll(
                        risk_mean_preds, risk_log_var_preds, mb_risk_labels,
                    )
                else:
                    risk_loss = torch.zeros((), device=mb_obs.device)

                # Total loss. Both aux weights default to 0.0 so the
                # added terms are exactly 0 and this session's loss is
                # numerically identical to session 02 when both weights
                # are 0 (verified by
                # ``test_risk_weight_zero_is_noop_on_total_loss``).
                loss = (
                    policy_loss
                    + self.value_loss_coeff * value_loss
                    - self.entropy_coeff * entropy
                    + self.fill_prob_loss_weight * fill_prob_loss
                    + self.risk_loss_weight * risk_loss
                )

                self.optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm,
                )
                self.optimiser.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))
                fill_prob_losses.append(float(fill_prob_loss.item()))
                risk_losses.append(float(risk_loss.item()))

            # KL early-stop at PPO-epoch granularity
            # (plans/naked-clip-and-stability, Session 02, 2026-04-18,
            # hard_constraints.md §9). After each full epoch sweep of
            # mini-batches, compute approximate KL across the whole
            # rollout as evaluated by the current (post-this-epoch)
            # policy: ``mean(old_logp - new_logp)``. If the threshold
            # is exceeded, break out of the remaining epochs for this
            # rollout — prevents runaway updates from continuing to
            # push the policy further after a large KL swing.
            # Literature standard (Andrychowicz et al. 2021, Engstrom
            # et al. 2020); ships in stable-baselines3. Kept at epoch
            # granularity rather than mini-batch (hard_constraints §9)
            # because mid-epoch breaks leave mini-batches unevenly
            # weighted.
            epochs_completed += 1
            with torch.no_grad():
                full_out = self.policy(obs_batch)
                full_std = full_out.action_log_std.exp()
                full_dist = Normal(full_out.action_mean, full_std)
                new_logp_full = full_dist.log_prob(action_batch).sum(dim=-1)
                approx_kl = float(
                    (old_log_probs - new_logp_full).mean().item()
                )
            self._last_approx_kl = approx_kl
            if approx_kl > self.kl_early_stop_threshold:
                self._last_kl_early_stop_epoch = epoch_idx
                logger.info(
                    "PPO KL early-stop after epoch %d: approx_kl=%.4f"
                    " > threshold=%.4f (skipping %d remaining epochs)",
                    epoch_idx, approx_kl,
                    self.kl_early_stop_threshold,
                    self.ppo_epochs - (epoch_idx + 1),
                )
                break

        ppo_elapsed = time.perf_counter() - ppo_start
        n_updates = len(policy_losses)
        logger.info(
            "PPO update: %d transitions, %d mini-batch updates in %.2fs"
            " | device=%s",
            n, n_updates, ppo_elapsed, self.device,
        )

        # Target-entropy controller step (entropy-control-v2 §5–§8).
        # Runs once per ``_ppo_update`` with the mean forward-pass
        # entropy across the update. ``entropies`` was appended per
        # mini-batch above; its mean is the best detached estimate of
        # the current policy's entropy on this rollout. Drives
        # ``self.entropy_coeff`` via ``log_alpha.exp()`` and writes
        # back into ``_entropy_coeff_base`` so the Session-2 floor
        # controller (when armed) scales on top of the fresh base.
        # The ``entropies`` list holds Python floats (already
        # detached) — no autograd leakage into the controller.
        if entropies:
            self._update_entropy_coefficient(
                float(np.mean(entropies)),
            )

        # Session 2: flush this update's per-head entropies into the
        # rolling controller. ``per_head_sums`` only contains heads the
        # policy actually produced during the mini-batch loop.
        per_head_mean: dict[str, float] = {
            h_name: total / per_head_count
            for h_name, total in per_head_sums.items()
        } if per_head_count > 0 else {}
        action_stats = self._update_entropy_controller(per_head_mean)

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            # Scalping-active-management §02 diagnostic: mean per-update
            # BCE on the fill-prob head, pre-weight. Always emitted so
            # operators can see the head's behaviour even when
            # ``fill_prob_loss_weight`` is 0 (plumbing-off mode).
            "fill_prob_loss": (
                float(np.mean(fill_prob_losses))
                if fill_prob_losses else 0.0
            ),
            # Scalping-active-management §03 diagnostic: same pattern —
            # pre-weight Gaussian NLL on the risk head, always emitted.
            "risk_loss": (
                float(np.mean(risk_losses))
                if risk_losses else 0.0
            ),
            # Session 02 KL early-stop diagnostics. ``approx_kl``
            # always reports the last-epoch KL so the smoke test and
            # learning-curves panel can see it. ``epochs_completed``
            # is how many PPO epochs actually ran — < ppo_epochs only
            # when the threshold fired.
            "approx_kl": float(self._last_approx_kl),
            "epochs_completed": int(epochs_completed),
            "kl_early_stop_epoch": (
                int(self._last_kl_early_stop_epoch)
                if self._last_kl_early_stop_epoch is not None
                else -1
            ),
            "action_stats": action_stats,
        }

    # -- Target-entropy controller (entropy-control-v2) ---------------------

    def _update_entropy_coefficient(self, current_entropy: float) -> None:
        """Proportional target-entropy controller step.

        Drives the entropy coefficient to hold ``current_entropy`` at
        ``self._target_entropy``. Uses a separate SGD optimiser
        (momentum=0) over ``_log_alpha``; does NOT backprop through
        the policy.

        Call ONCE per ``_ppo_update``, after the entropy value is
        computed on the current rollout. The argument must be a
        detached Python float — no tensor leakage into the
        controller's autograd graph.

        Mechanism. The loss ``-log_alpha * (target - current)`` has
        gradient ``d/d(log_alpha) = -(target - current) = (current -
        target)`` w.r.t. ``log_alpha``. Under SGD with learning rate
        ``lr`` the update is:

            log_alpha <- log_alpha - lr * (current - target)

        That is literal proportional control with gain ``lr`` — when
        ``current > target`` the coefficient shrinks by an amount
        scaling with the overshoot; when ``current < target`` it
        grows symmetrically. A large error produces a large
        correction; as entropy approaches target the error shrinks
        and the step shrinks too. No Adam adaptive normalisation,
        no momentum — the controller has no internal state beyond
        ``log_alpha`` itself.

        Sign check:
        - ``current > target`` → grad = positive → log_alpha shrinks
          → alpha shrinks → less entropy bonus → entropy falls
          toward target. ✓
        - ``current < target`` → grad = negative → log_alpha grows
          → alpha grows → more entropy bonus → entropy rises toward
          target. ✓

        Reason this replaces the Adam-based Session-01 formulation.
        Adam's adaptive normalisation makes per-update log_alpha
        movement ~``lr``-sized regardless of gradient magnitude. At
        the one-call-per-episode cadence our training loop runs,
        the controller couldn't track even a moderate drift
        (Session-04 post-launch observed entropy 139→192 across 15
        eps while alpha barely moved). SGD's proportional behaviour
        fixes that at the formulation level, not via lr tuning. See
        plans/entropy-control-v2/lessons_learnt.md 2026-04-19.
        """
        alpha_loss = -self._log_alpha * (
            self._target_entropy - float(current_entropy)
        )
        self._alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self._alpha_optimizer.step()
        # Clamp to prevent runaway during calibration. Saturation at
        # either bound is a valid failure signal — surface in the
        # learning-curves panel rather than silently letting log_alpha
        # drift out of the useful range.
        self._log_alpha.data.clamp_(
            self._log_alpha_min, self._log_alpha_max,
        )
        # Refresh the effective coefficient the loss formula reads.
        # Also refresh the base the Session-2 floor controller scales
        # on top of; when ``entropy_floor == 0`` the floor controller
        # is a no-op and these two agree after the controller runs.
        self.entropy_coeff = float(self._log_alpha.exp().item())
        self._entropy_coeff_base = self.entropy_coeff

    # -- Checkpointing the controller state ---------------------------------

    def save_checkpoint(self) -> dict:
        """Return a serialisable dict carrying trainer-level state that
        must survive across process boundaries.

        Schema (entropy-control-v2 §11):
        - ``log_alpha``: float — current value of ``_log_alpha``.
        - ``alpha_optim_state``: dict — Adam momentum state for the
          alpha optimiser.

        Consumers that want the policy weights save those separately
        via ``policy.state_dict()``; this method is scoped to trainer
        controller state only.
        """
        return {
            "log_alpha": float(self._log_alpha.item()),
            "alpha_optim_state": self._alpha_optimizer.state_dict(),
        }

    def load_checkpoint(self, checkpoint: dict) -> None:
        """Restore controller state from a dict produced by
        :meth:`save_checkpoint`.

        Backward-compat: when either key is missing we fresh-init
        from the default (log_alpha from the current value; optimiser
        momentum reset to zero) and log a warning. This matches the
        registry-reset playbook's fallback for pre-controller
        checkpoints.
        """
        if "log_alpha" in checkpoint:
            self._log_alpha.data = torch.tensor(
                float(checkpoint["log_alpha"]),
                dtype=torch.float64,
                device=self.device,
            )
        else:
            logger.warning(
                "Checkpoint missing log_alpha; fresh-initing from "
                "default. Expected for checkpoints saved before "
                "entropy-control-v2."
            )
        if "alpha_optim_state" in checkpoint:
            self._alpha_optimizer.load_state_dict(
                checkpoint["alpha_optim_state"]
            )
        # else: optimiser stays at its fresh-init state (Adam momentum
        # starts at 0 — same fallback behaviour as the log_alpha branch
        # above, no extra warning needed).

        # Refresh the effective coefficient after load.
        self.entropy_coeff = float(self._log_alpha.exp().item())
        self._entropy_coeff_base = self.entropy_coeff

    # -- Entropy-floor controller (Session 2) --------------------------------

    def _update_entropy_controller(
        self, per_head_entropy: dict[str, float],
    ) -> dict[str, float]:
        """Push a batch's per-head mean entropies into the rolling window,
        update the coefficient scaling, and recompute the collapse flag.

        Returns an ``action_stats`` dict suitable for the progress event:
        per-head rolling mean entropies, the currently-active entropy
        coefficient, and a boolean ``entropy_collapse`` warning flag.

        The controller is a no-op (returns the rolling state but never
        changes the coefficient) when ``entropy_floor == 0``.
        """
        # Record the per-head values into the rolling windows (always —
        # the action_stats dict is useful diagnostics regardless of whether
        # the floor is armed).
        for name, value in per_head_entropy.items():
            if name in self._per_head_window:
                self._per_head_window[name].append(float(value))

        # Overall rolling mean is the mean of the per-head values pushed
        # this call. Stored one-per-update so the window reflects batches,
        # not heads.
        if per_head_entropy:
            batch_mean = float(
                sum(per_head_entropy.values()) / len(per_head_entropy)
            )
            self._entropy_window.append(batch_mean)

        # Coefficient controller. Kept off entirely when entropy_floor<=0
        # — byte-identical to pre-session behaviour.
        if self.entropy_floor > 0.0 and self._entropy_window:
            rolling_mean = float(
                sum(self._entropy_window) / len(self._entropy_window)
            )
            if rolling_mean < self.entropy_floor:
                # Guard against zero/near-zero mean entropy blowing the
                # ratio past float range. ``entropy_boost_max`` is the
                # authoritative cap; this just avoids a div-by-zero on
                # the way to it.
                denom = max(rolling_mean, 1e-8)
                multiplier = min(
                    self.entropy_boost_max, self.entropy_floor / denom,
                )
                self._entropy_coeff_active = (
                    multiplier * self._entropy_coeff_base
                )
            else:
                self._entropy_coeff_active = self._entropy_coeff_base
        else:
            self._entropy_coeff_active = self._entropy_coeff_base

        # Per-head collapse streak — driven even when the floor is off
        # so the diagnostic flag is still meaningful for operators. A
        # zero floor means no head is ever "below", so streaks stay at 0
        # and the flag stays False.
        any_collapse = False
        for name in _HEAD_NAMES:
            value = per_head_entropy.get(name)
            if value is None:
                continue
            if self.entropy_floor > 0.0 and value < self.entropy_floor:
                self._per_head_below_streak[name] += 1
            else:
                self._per_head_below_streak[name] = 0
            if (
                self._per_head_below_streak[name]
                > self.entropy_collapse_patience
            ):
                any_collapse = True
        self._entropy_collapse = bool(any_collapse)

        # Apply the active coefficient so the NEXT mini-batch loop uses
        # the new value for the entropy bonus term.
        self.entropy_coeff = self._entropy_coeff_active

        # Build the action_stats dict with one key per head. Heads that
        # haven't been seen yet (e.g. arb_spread on a directional run)
        # report 0.0 so the schema is stable.
        action_stats: dict[str, float] = {}
        for name in _HEAD_NAMES:
            window = self._per_head_window[name]
            if window:
                action_stats[f"mean_entropy_{name}"] = float(
                    sum(window) / len(window)
                )
            else:
                action_stats[f"mean_entropy_{name}"] = 0.0
        action_stats["entropy_collapse"] = self._entropy_collapse
        action_stats["entropy_coeff_active"] = self._entropy_coeff_active
        return action_stats

    def _compute_per_head_entropy(
        self, per_dim_entropy: torch.Tensor,
    ) -> dict[str, float]:
        """Slice a (batch, action_dim) entropy tensor into per-head means.

        Action layout is ``[head_0 × N | head_1 × N | …]`` where
        ``N = max_runners`` and head indices follow ``_HEAD_NAMES``.
        Returns a plain ``{head_name: float}`` dict (one entry per head
        actually present in the policy's action space).
        """
        max_runners = getattr(self.policy, "max_runners", None)
        per_runner_apd = getattr(self.policy, "_per_runner_action_dim", None)
        if max_runners is None or per_runner_apd is None:
            # Fallback for minimal/stub policies used in isolation tests
            # that don't have the multi-head layout: treat the whole
            # distribution as a single "signal" head.
            return {"signal": float(per_dim_entropy.mean().item())}

        per_head: dict[str, float] = {}
        for h_idx in range(min(per_runner_apd, len(_HEAD_NAMES))):
            start = h_idx * max_runners
            end = start + max_runners
            chunk = per_dim_entropy[:, start:end]
            per_head[_HEAD_NAMES[h_idx]] = float(chunk.mean().item())
        return per_head

    # -- Logging & progress ---------------------------------------------------

    def _log_episode(
        self,
        ep: EpisodeStats,
        loss_info: dict[str, float],
        tracker: ProgressTracker,
    ) -> None:
        """Write per-episode metrics to a JSON-lines log file."""
        record = {
            "episode": tracker.completed,
            "model_id": self.model_id,
            "architecture_name": self.architecture_name,
            "day_date": ep.day_date,
            "total_reward": round(ep.total_reward, 4),
            "clipped_reward_total": round(ep.clipped_reward_total, 4),
            "raw_pnl_reward": round(ep.raw_pnl_reward, 4),
            "shaped_bonus": round(ep.shaped_bonus, 4),
            "total_pnl": round(ep.total_pnl, 4),
            "bet_count": ep.bet_count,
            "winning_bets": ep.winning_bets,
            "races_completed": ep.races_completed,
            "final_budget": round(ep.final_budget, 2),
            "n_steps": ep.n_steps,
            "policy_loss": round(loss_info["policy_loss"], 6),
            "value_loss": round(loss_info["value_loss"], 6),
            "entropy": round(loss_info["entropy"], 6),
            # Target-entropy controller trajectory (entropy-control-v2).
            # Lets the learning-curves panel plot controller alpha
            # alongside entropy. Optional keys — pre-controller rows
            # do not have them; downstream readers must tolerate
            # absence.
            "alpha": round(float(self._log_alpha.exp().item()), 8),
            "log_alpha": round(float(self._log_alpha.item()), 6),
            "target_entropy": round(float(self._target_entropy), 4),
            # Forced-arbitrage (scalping) rollups — zero for directional.
            "arbs_completed": ep.arbs_completed,
            "arbs_naked": ep.arbs_naked,
            "arbs_closed": ep.arbs_closed,
            "locked_pnl": round(ep.locked_pnl, 4),
            "naked_pnl": round(ep.naked_pnl, 4),
            # Session 3 — action-diagnostics rollup.
            "bet_rate": round(ep.bet_rate, 6),
            "arb_rate": round(ep.arb_rate, 6),
            "signal_bias": round(ep.signal_bias, 6),
            "timestamp": time.time(),
        }
        # Smoke-test probe rows are tagged so the learning-curves panel
        # can colour them distinctly and the gate's assertion evaluator
        # can pick them out of the shared stream — see agents/smoke_test.py
        # and plans/naked-clip-and-stability/hard_constraints.md §16.
        if self.smoke_test_tag:
            record["smoke_test"] = True

        log_file = self.log_dir / "episodes.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        logger.info(
            "Episode %d/%d [%s] reward=%.3f (raw=%.3f shaped=%+.3f) pnl=%.2f bets=%d loss=%.4f",
            tracker.completed,
            tracker.total,
            ep.day_date,
            ep.total_reward,
            ep.raw_pnl_reward,
            ep.shaped_bonus,
            ep.total_pnl,
            ep.bet_count,
            loss_info["policy_loss"],
        )

    def _publish_progress(
        self,
        ep: EpisodeStats,
        loss_info: dict[str, float],
        tracker: ProgressTracker,
    ) -> None:
        """Publish a progress event to the asyncio queue (if provided)."""
        scalping_detail = ""
        if ep.arbs_completed or ep.arbs_naked or ep.arbs_closed:
            # Activity-log surface for scalping runs: "arb completed: 3
            # arbs, £0.38 locked, naked=1 £-0.02" — issue 05 session 3.
            # Scalping-close-signal session 01 adds a ``closed=N`` term
            # so operators can see close activity at a glance.
            total_pairs = (
                ep.arbs_completed + ep.arbs_closed + ep.arbs_naked
            )
            scalping_detail = (
                f" | arbs={ep.arbs_completed}/{total_pairs}"
                f" closed={ep.arbs_closed}"
                f" locked=£{ep.locked_pnl:+.2f}"
                f" naked=£{ep.naked_pnl:+.2f}"
            )
        progress = {
            "event": "progress",
            "phase": "training",
            "item": tracker.to_dict(),
            "detail": (
                f"Episode {tracker.completed} [{ep.day_date}] | "
                f"reward={ep.total_reward:+.3f} | "
                f"P&L={ep.total_pnl:+.2f} | "
                f"loss={loss_info['policy_loss']:.4f}"
                f"{scalping_detail}"
            ),
            "episode": {
                "day_date": ep.day_date,
                "total_reward": ep.total_reward,
                "clipped_reward_total": ep.clipped_reward_total,
                "total_pnl": ep.total_pnl,
                "bet_count": ep.bet_count,
                "policy_loss": loss_info["policy_loss"],
                "value_loss": loss_info["value_loss"],
                "entropy": loss_info["entropy"],
                "arbs_completed": ep.arbs_completed,
                "arbs_naked": ep.arbs_naked,
                "arbs_closed": ep.arbs_closed,
                "locked_pnl": ep.locked_pnl,
                "naked_pnl": ep.naked_pnl,
            },
            # Session 2 — per-head entropy + floor-controller diagnostics.
            # Present on every progress event; floor-off runs still emit
            # the rolling per-head means so the monitor can sparkline
            # entropy trajectories before the operator decides to arm the
            # floor. See plans/arb-improvements/session_2_entropy_floor.md.
            "action_stats": loss_info.get(
                "action_stats",
                {
                    f"mean_entropy_{h}": 0.0 for h in _HEAD_NAMES
                } | {
                    "entropy_collapse": False,
                    "entropy_coeff_active": self._entropy_coeff_active,
                },
            ),
        }
        # Session 3 — bet-rate / arb-rate / bias-active diagnostics. Merged
        # into the same ``action_stats`` dict as the entropy controller's
        # per-head values so the monitor has a single observation channel.
        action_stats = progress["action_stats"]
        action_stats["bet_rate"] = float(max(0.0, min(1.0, ep.bet_rate)))
        action_stats["arb_rate"] = float(max(0.0, min(1.0, ep.arb_rate)))
        # ``bias_active`` is True while both the warmup window is open
        # AND the magnitude is non-zero. When either knob is off the
        # flag is False and the bias passed to the policy is 0.0.
        action_stats["bias_active"] = bool(
            self.signal_bias_warmup > 0
            and self.signal_bias_magnitude != 0.0
            and self._current_epoch < self.signal_bias_warmup
        )

        if self.progress_queue is not None:
            try:
                self.progress_queue.put_nowait(progress)
            except Exception:
                pass  # drop if consumer is behind

            # Issue 05 — session 3: surface each completed arb pair as a
            # separate activity-log line so scalping runs show per-pair
            # detail, not just the per-episode rollup appended above.
            # Capped to avoid flooding the WS queue on long episodes.
            if ep.arb_events:
                for idx, ev in enumerate(ep.arb_events[:_MAX_ARB_EVENTS_PER_EP]):
                    try:
                        self.progress_queue.put_nowait({
                            "event": "arb_completed",
                            "phase": "training",
                            "detail": (
                                f"Arb completed: Back @ {ev['back_price']:.2f}"
                                f" / Lay @ {ev['lay_price']:.2f}"
                                f" on runner {ev['selection_id']}"
                                f" → locked £{ev['locked_pnl']:+.2f}"
                            ),
                            "arb_event": ev,
                        })
                    except Exception:
                        break
                if len(ep.arb_events) > _MAX_ARB_EVENTS_PER_EP:
                    overflow = len(ep.arb_events) - _MAX_ARB_EVENTS_PER_EP
                    try:
                        self.progress_queue.put_nowait({
                            "event": "arb_completed",
                            "phase": "training",
                            "detail": (
                                f"… plus {overflow} more arb pair(s) this episode"
                            ),
                        })
                    except Exception:
                        pass

            # Scalping-close-signal session 01 — distinct "pair_closed"
            # activity-log line per close event. Same cap and overflow
            # handling as arb_completed; the two lists are independent.
            if ep.close_events:
                for ev in ep.close_events[:_MAX_ARB_EVENTS_PER_EP]:
                    try:
                        self.progress_queue.put_nowait({
                            "event": "pair_closed",
                            "phase": "training",
                            "detail": (
                                f"Pair closed at loss: Back @ {ev['back_price']:.2f}"
                                f" / Lay @ {ev['lay_price']:.2f}"
                                f" on runner {ev['selection_id']}"
                                f" → realised £{ev['realised_pnl']:+.2f}"
                            ),
                            "close_event": ev,
                        })
                    except Exception:
                        break
                if len(ep.close_events) > _MAX_ARB_EVENTS_PER_EP:
                    overflow = len(ep.close_events) - _MAX_ARB_EVENTS_PER_EP
                    try:
                        self.progress_queue.put_nowait({
                            "event": "pair_closed",
                            "phase": "training",
                            "detail": (
                                f"… plus {overflow} more closed pair(s) this episode"
                            ),
                        })
                    except Exception:
                        pass
