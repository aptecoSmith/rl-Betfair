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
import os
import time
from collections import deque
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


def _compute_arb_rate(arbs_completed: int, arbs_naked: int) -> float:
    """Fraction of arb attempts that paired (``completed`` / total).

    Returns ``0.0`` when there were no arb attempts at all — avoids
    spurious NaN / divide-by-zero. Always in ``[0.0, 1.0]``.
    """
    total = int(arbs_completed) + int(arbs_naked)
    if total <= 0:
        return 0.0
    return float(arbs_completed) / float(total)


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
    """

    obs: np.ndarray
    action: np.ndarray
    log_prob: float
    value: float
    reward: float
    done: bool
    training_reward: float = 0.0


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
        self.lr = hp.get("learning_rate", 3e-4)
        self.gamma = hp.get("gamma", 0.99)
        self.gae_lambda = hp.get("gae_lambda", 0.95)
        self.clip_epsilon = hp.get("ppo_clip_epsilon", 0.2)
        self.entropy_coeff = hp.get("entropy_coefficient", 0.01)
        # -- Session 2 (arb-improvements) entropy-floor controller --------
        # Baseline coefficient captured before any adaptive scaling. When
        # ``entropy_floor`` is 0 (default) the controller is a no-op and
        # ``self.entropy_coeff`` stays at the baseline for every update —
        # byte-identical to pre-session behaviour.
        #
        # The controller keeps a rolling window of per-head entropies.
        # Before each PPO update's entropy-bonus term, if the rolling
        # mean entropy (averaged across heads and the window) is below
        # the floor, ``self.entropy_coeff`` is scaled up to
        # ``min(entropy_boost_max, floor / rolling_mean) * base``. When
        # the rolling mean recovers above the floor, it snaps back to
        # baseline. The floor scales the *coefficient*, never the action
        # distribution directly (hard_constraints.md §Stabilisation).
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

        # Logging setup
        log_dir = Path(config.get("paths", {}).get("logs", "logs"))
        self.log_dir = log_dir / "training"
        self.log_dir.mkdir(parents=True, exist_ok=True)

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

                rollout.append(Transition(
                    obs=obs,
                    action=action_np,
                    log_prob=float(log_prob.item()),
                    value=float(value.item()),
                    reward=raw_reward,
                    done=done,
                    training_reward=training_reward,
                ))

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
            locked_pnl=float(info.get("locked_pnl", 0.0) or 0.0),
            naked_pnl=float(info.get("naked_pnl", 0.0) or 0.0),
            bet_rate=(float(bet_steps) / float(n_steps)) if n_steps > 0 else 0.0,
            arb_rate=_compute_arb_rate(
                int(info.get("arbs_completed", 0) or 0),
                int(info.get("arbs_naked", 0) or 0),
            ),
            signal_bias=float(current_signal_bias),
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
            delta = tr.training_reward + self.gamma * next_value - tr.value
            last_gae = delta + self.gamma * self.gae_lambda * last_gae
            advantages[t] = last_gae
            returns[t] = last_gae + tr.value

        return advantages, returns

    # -- PPO update -----------------------------------------------------------

    def _ppo_update(self, rollout: Rollout) -> dict[str, float]:
        """Run PPO optimisation epochs on collected rollout data.

        Returns a dict with final loss components.
        """
        ppo_start = time.perf_counter()
        transitions = rollout.transitions
        n = len(transitions)

        # Prepare tensors — build numpy arrays first, then transfer to GPU.
        # Use pinned memory when CUDA is available for faster async transfer.
        obs_np = np.array([t.obs for t in transitions], dtype=np.float32)
        action_np = np.array([t.action for t in transitions], dtype=np.float32)
        lp_np = np.array([t.log_prob for t in transitions], dtype=np.float32)

        if self.device != "cpu":
            obs_pin = torch.from_numpy(obs_np).pin_memory()
            action_pin = torch.from_numpy(action_np).pin_memory()
            lp_pin = torch.from_numpy(lp_np).pin_memory()
            obs_batch = obs_pin.to(self.device, non_blocking=True)
            action_batch = action_pin.to(self.device, non_blocking=True)
            old_log_probs = lp_pin.to(self.device, non_blocking=True)
        else:
            obs_batch = torch.from_numpy(obs_np)
            action_batch = torch.from_numpy(action_np)
            old_log_probs = torch.from_numpy(lp_np)

        advantages, returns = self._compute_advantages(rollout)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        # Normalise advantages
        if n > 1:
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / adv_std

        policy_losses = []
        value_losses = []
        entropies = []
        # Session 2: accumulate per-head entropy contributions across the
        # mini-batch loop so we can push a single average-per-head sample
        # to the rolling window at the end of this update. Only heads
        # actually present in the policy's action space are populated —
        # a directional (4-head) policy omits ``arb_spread`` so it never
        # trips the collapse detector for a head it can't produce.
        per_head_sums: dict[str, float] = {}
        per_head_count: int = 0

        for _ in range(self.ppo_epochs):
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

                # PPO clipped surrogate
                ratio = (new_log_probs - mb_old_log_probs).exp()
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

                # Total loss
                loss = (
                    policy_loss
                    + self.value_loss_coeff * value_loss
                    - self.entropy_coeff * entropy
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

        ppo_elapsed = time.perf_counter() - ppo_start
        n_updates = len(policy_losses)
        logger.info(
            "PPO update: %d transitions, %d mini-batch updates in %.2fs"
            " | device=%s",
            n, n_updates, ppo_elapsed, self.device,
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
            "action_stats": action_stats,
        }

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
            # Forced-arbitrage (scalping) rollups — zero for directional.
            "arbs_completed": ep.arbs_completed,
            "arbs_naked": ep.arbs_naked,
            "locked_pnl": round(ep.locked_pnl, 4),
            "naked_pnl": round(ep.naked_pnl, 4),
            # Session 3 — action-diagnostics rollup.
            "bet_rate": round(ep.bet_rate, 6),
            "arb_rate": round(ep.arb_rate, 6),
            "signal_bias": round(ep.signal_bias, 6),
            "timestamp": time.time(),
        }

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
        if ep.arbs_completed or ep.arbs_naked:
            # Activity-log surface for scalping runs: "arb completed: 3
            # arbs, £0.38 locked, naked=1 £-0.02" — issue 05 session 3.
            scalping_detail = (
                f" | arbs={ep.arbs_completed}/{ep.arbs_completed + ep.arbs_naked}"
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
