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
    """A single (s, a, r, ...) transition from one env step."""

    obs: np.ndarray
    action: np.ndarray
    log_prob: float
    value: float
    reward: float
    done: bool


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
        self.value_loss_coeff = hp.get("value_loss_coeff", 0.5)
        self.max_grad_norm = hp.get("max_grad_norm", 0.5)
        self.ppo_epochs = hp.get("ppo_epochs", 4)
        self.mini_batch_size = hp.get("mini_batch_size", 64)

        # Build per-agent reward overrides from the sampled genes. This is
        # how reward-shaping hyperparameters reach BetfairEnv — previously
        # these genes were sampled but silently dropped here, so every
        # agent trained with identical reward shaping.
        self.reward_overrides = _reward_overrides_from_hp(hp)

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
        )
        obs, info = env.reset()

        rollout = Rollout()
        hidden_state = self.policy.init_hidden(batch_size=1)
        hidden_state = (
            hidden_state[0].to(self.device),
            hidden_state[1].to(self.device),
        )

        total_reward = 0.0
        n_steps = 0
        done = False

        # Pre-allocate a reusable GPU tensor for single-step observations
        obs_dim = obs.shape[0]
        obs_buffer = torch.empty(
            1, obs_dim, dtype=torch.float32, device=self.device,
        )

        with torch.no_grad():
            while not done:
                # Copy obs into pre-allocated GPU buffer (avoids tensor creation)
                obs_buffer[0] = torch.as_tensor(obs, dtype=torch.float32)

                out: PolicyOutput = self.policy(obs_buffer, hidden_state)
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

                rollout.append(Transition(
                    obs=obs,
                    action=action_np,
                    log_prob=float(log_prob.item()),
                    value=float(value.item()),
                    reward=float(reward),
                    done=done,
                ))

                total_reward += reward
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

            delta = tr.reward + self.gamma * next_value - tr.value
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

                # Forward pass (no LSTM state for mini-batch -- treat each
                # transition independently during optimisation)
                out = self.policy(mb_obs)
                std = out.action_log_std.exp()
                dist = Normal(out.action_mean, std)

                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                values = out.value.squeeze(-1)

                # PPO clipped surrogate
                ratio = (new_log_probs - mb_old_log_probs).exp()
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
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

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
        }

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
        progress = {
            "event": "progress",
            "phase": "training",
            "item": tracker.to_dict(),
            "detail": (
                f"Episode {tracker.completed} [{ep.day_date}] | "
                f"reward={ep.total_reward:+.3f} | "
                f"P&L={ep.total_pnl:+.2f} | "
                f"loss={loss_info['policy_loss']:.4f}"
            ),
            "episode": {
                "day_date": ep.day_date,
                "total_reward": ep.total_reward,
                "total_pnl": ep.total_pnl,
                "bet_count": ep.bet_count,
                "policy_loss": loss_info["policy_loss"],
                "value_loss": loss_info["value_loss"],
                "entropy": loss_info["entropy"],
            },
        }

        if self.progress_queue is not None:
            try:
                self.progress_queue.put_nowait(progress)
            except Exception:
                pass  # drop if consumer is behind
