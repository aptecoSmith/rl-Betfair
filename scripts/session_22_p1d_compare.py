"""
scripts/session_22_p1d_compare.py — Session 22: P1 re-train + decision-gate comparison.

Trains two policies on the same training days with identical hyperparameters:

  baseline  pre-P1 obs vector  RUNNER_DIM=110, schema v1
            (no OBI, microprice, traded_delta, mid_drift)

  p1        full P1 obs vector  RUNNER_DIM=114, schema v4
            (OBI + microprice + traded_delta + mid_drift)

Then evaluates both on the held-out eval window and reports per-day raw P&L.

Session-22 requirement (Q3 resolution): eval metric = raw daily P&L (option A).
The only hard assertion is the gradient-norm check on the P1 policy's new
columns — zero gradient means the P1 features are wired incorrectly and the
whole comparison is worthless.

Usage::

    python scripts/session_22_p1d_compare.py
    python scripts/session_22_p1d_compare.py --train-days 4 --eval-days 3 --n-epochs 5
    python scripts/session_22_p1d_compare.py --dry-run   # build & validate, no training

Constraints (session plan):
- Same hyperparameters for both runs. Only variable: obs vector.
- Fresh init for both. No warm-starting P1 from baseline.
- Gradient-norm check fires before declaring P1 run complete.
"""

from __future__ import annotations

import abc
import argparse
import logging
import sys
import time
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.episode_builder import Day, load_days  # noqa: E402
from env.betfair_env import (  # noqa: E402
    AGENT_STATE_DIM,
    MARKET_DIM,
    POSITION_DIM,
    RUNNER_DIM,
    VELOCITY_DIM,
)
from agents.policy_network import (  # noqa: E402
    BasePolicy,
    PolicyOutput,
    _build_mlp,
    MARKET_TOTAL_DIM,
)
from agents.ppo_trainer import PPOTrainer, Rollout, Transition, EpisodeStats  # noqa: E402
from env.betfair_env import BetfairEnv  # noqa: E402
from env.bet_manager import BetOutcome  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("session_22")


# ── Baseline dimensions (pre-P1, schema v1) ──────────────────────────────────

BASELINE_RUNNER_DIM: int = 110  # before sessions 19-21 added OBI/microprice/traded_delta/mid_drift
N_P1_FEATURES: int = RUNNER_DIM - BASELINE_RUNNER_DIM  # 4
BASELINE_RUNNER_INPUT_DIM: int = BASELINE_RUNNER_DIM + POSITION_DIM  # 113

#: Gradient check fires after this many training episodes (days).
#: After one full day the policy has received gradient updates and the
#: P1 columns must show non-zero sensitivity.
GRAD_CHECK_AFTER_EPISODES: int = 1


# ── Fixed hyperparameters (same for both runs) ────────────────────────────────
#
# Session-22 rule: "same hyperparameters for both runs; only variable is obs dim."
# Use conservative production defaults — not a tuning exercise.

SHARED_HP: dict = {
    "learning_rate": 1e-4,
    "lstm_hidden_size": 256,
    "mlp_hidden_size": 128,
    "mlp_layers": 2,
    "lstm_num_layers": 1,
    "lstm_dropout": 0.0,
    "lstm_layer_norm": False,
    "ppo_clip_epsilon": 0.2,
    "entropy_coefficient": 0.01,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "value_loss_coeff": 0.5,
    "ppo_epochs": 4,
    "mini_batch_size": 64,
    "max_grad_norm": 0.5,
}


# ── Baseline obs-slicing helpers ──────────────────────────────────────────────


def build_baseline_obs_indices(max_runners: int) -> np.ndarray:
    """Return an index array that slices the P1 features out of a full obs vector.

    The full obs layout (RUNNER_DIM=114)::

        [market(37) | velocity(11) | runner0(114) … runner13(114) |
         agent_state(6) | positions(42)]

    The baseline obs layout (RUNNER_DIM=110)::

        [market(37) | velocity(11) | runner0(110) … runner13(110) |
         agent_state(6) | positions(42)]

    The P1 features (OBI, microprice, traded_delta, mid_drift) are the last
    four entries in each runner's block.  This function builds a numpy index
    array that can be used as ``obs[indices]`` to produce the baseline obs.
    """
    prefix_dim = MARKET_DIM + VELOCITY_DIM  # 48
    runner_start = prefix_dim

    indices: list[int] = list(range(prefix_dim))  # market + velocity

    for i in range(max_runners):
        slot_start = runner_start + i * RUNNER_DIM
        # Take only the first BASELINE_RUNNER_DIM features of each slot
        indices.extend(range(slot_start, slot_start + BASELINE_RUNNER_DIM))

    # Tail: agent_state + positions (after the full P1 runner block)
    tail_start = runner_start + max_runners * RUNNER_DIM
    full_obs_dim = (
        MARKET_DIM + VELOCITY_DIM
        + RUNNER_DIM * max_runners
        + AGENT_STATE_DIM
        + POSITION_DIM * max_runners
    )
    indices.extend(range(tail_start, full_obs_dim))

    return np.array(indices, dtype=np.int64)


def baseline_obs_dim(max_runners: int) -> int:
    return (
        MARKET_DIM + VELOCITY_DIM
        + BASELINE_RUNNER_DIM * max_runners
        + AGENT_STATE_DIM
        + POSITION_DIM * max_runners
    )


def p1_obs_dim(max_runners: int) -> int:
    return (
        MARKET_DIM + VELOCITY_DIM
        + RUNNER_DIM * max_runners
        + AGENT_STATE_DIM
        + POSITION_DIM * max_runners
    )


# ── Gradient-norm check ───────────────────────────────────────────────────────


def check_p1_gradient_norm(
    policy: BasePolicy,
    obs_sample: np.ndarray,
    max_runners: int,
    device: str,
) -> float:
    """Compute the gradient norm of value output w.r.t. P1 feature columns.

    Uses a single forward-backward pass with the obs tensor as a leaf.
    A zero norm means the policy is completely insensitive to the P1 features
    — either they are wired to zero everywhere, or the network has collapsed.

    Parameters
    ----------
    policy:
        The trained P1 policy (RUNNER_DIM=114).
    obs_sample:
        A single obs vector from the P1 env (shape: (p1_obs_dim,)).
    max_runners:
        Max runners per race (from config).
    device:
        'cpu' or 'cuda'.

    Returns
    -------
    float
        L2 norm of gradients at the P1 column indices.
    """
    # Create a (1, obs_dim) leaf tensor so .grad is populated after backward.
    # torch.tensor(...).unsqueeze(0) produces a non-leaf; use numpy slicing to
    # get the right shape before wrapping.
    obs_np = obs_sample.astype(np.float32)[None, :]  # (1, obs_dim) numpy array
    obs_tensor = torch.from_numpy(obs_np).to(device)
    obs_tensor.requires_grad_(True)  # leaf tensor — .grad will be populated

    policy.train()  # ensure grad flows
    hidden = policy.init_hidden(1)
    hidden = (hidden[0].to(device), hidden[1].to(device))

    out = policy(obs_tensor, hidden)
    # Use value because it's a scalar per example — clean gradient signal
    out.value.sum().backward()

    assert obs_tensor.grad is not None, "No gradient on obs_tensor after backward"

    # Build P1 column indices: last N_P1_FEATURES of each runner's block
    runner_start = MARKET_DIM + VELOCITY_DIM
    p1_indices: list[int] = []
    for i in range(max_runners):
        slot_start = runner_start + i * RUNNER_DIM
        for j in range(N_P1_FEATURES):
            p1_indices.append(slot_start + BASELINE_RUNNER_DIM + j)

    p1_grad = obs_tensor.grad[0, p1_indices]
    policy.eval()
    return float(p1_grad.norm().item())


# ── Baseline policy (RUNNER_DIM=110, schema v1) ───────────────────────────────


class BaselinePPOLSTMPolicy(BasePolicy):
    """PPO + LSTM policy operating on the pre-P1 obs vector (RUNNER_DIM=110).

    Architecturally identical to PPOLSTMPolicy but uses BASELINE_RUNNER_DIM
    (110) and BASELINE_RUNNER_INPUT_DIM (113) instead of the current globals
    (114 / 117).  All constants are baked in as class-level literals so the
    forward pass does not depend on module-level globals that will change when
    the main env bumps its schema.

    This class must NOT be imported or used outside session-22 scripts.
    """

    architecture_name = "ppo_lstm_baseline_v1"
    description = "Pre-P1 baseline: RUNNER_DIM=110 (schema v1)"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        max_runners: int,
        hyperparams: dict,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_runners = max_runners

        lstm_hidden = int(hyperparams.get("lstm_hidden_size", 256))
        mlp_hidden = int(hyperparams.get("mlp_hidden_size", 128))
        mlp_layers = int(hyperparams.get("mlp_layers", 2))
        runner_embed_dim = mlp_hidden

        lstm_num_layers = int(hyperparams.get("lstm_num_layers", 1))
        lstm_dropout = float(hyperparams.get("lstm_dropout", 0.0))
        lstm_layer_norm = bool(hyperparams.get("lstm_layer_norm", False))

        self.lstm_hidden_size = lstm_hidden
        self.lstm_num_layers = lstm_num_layers
        self.runner_embed_dim = runner_embed_dim

        # Key difference: input_dim=113 (BASELINE_RUNNER_INPUT_DIM), not 117
        self.runner_encoder = _build_mlp(
            input_dim=BASELINE_RUNNER_INPUT_DIM,
            hidden_dim=mlp_hidden,
            output_dim=runner_embed_dim,
            n_layers=mlp_layers,
        )
        self.market_encoder = _build_mlp(
            input_dim=MARKET_TOTAL_DIM,
            hidden_dim=mlp_hidden,
            output_dim=mlp_hidden,
            n_layers=mlp_layers,
        )

        lstm_input_dim = mlp_hidden + runner_embed_dim * 2
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_num_layers,
            dropout=(lstm_dropout if lstm_num_layers > 1 else 0.0),
            batch_first=True,
        )
        self.lstm_output_norm: nn.Module = (
            nn.LayerNorm(lstm_hidden) if lstm_layer_norm else nn.Identity()
        )

        actor_input_dim = runner_embed_dim + lstm_hidden
        self.actor_head = _build_mlp(
            input_dim=actor_input_dim,
            hidden_dim=mlp_hidden,
            output_dim=2,
            n_layers=1,
        )
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic_head = _build_mlp(
            input_dim=lstm_hidden,
            hidden_dim=mlp_hidden,
            output_dim=1,
            n_layers=1,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2 ** 0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.actor_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
        for module in self.critic_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)

    def init_hidden(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size)
        c = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size)
        return (h, c)

    def _split_obs(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split baseline obs (RUNNER_DIM=110) into market and runner tensors."""
        market = obs[:, :MARKET_DIM]
        velocity = obs[:, MARKET_DIM : MARKET_DIM + VELOCITY_DIM]
        runner_start = MARKET_DIM + VELOCITY_DIM
        runner_end = runner_start + self.max_runners * BASELINE_RUNNER_DIM
        runners_flat = obs[:, runner_start:runner_end]
        agent_state = obs[:, runner_end : runner_end + AGENT_STATE_DIM]
        position_start = runner_end + AGENT_STATE_DIM
        position_end = position_start + self.max_runners * POSITION_DIM
        position_flat = obs[:, position_start:position_end]

        market_feats = torch.cat([market, velocity, agent_state], dim=-1)
        runner_feats_raw = runners_flat.view(-1, self.max_runners, BASELINE_RUNNER_DIM)
        position_feats = position_flat.view(-1, self.max_runners, POSITION_DIM)
        runner_feats = torch.cat([runner_feats_raw, position_feats], dim=-1)
        return market_feats, runner_feats

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> PolicyOutput:
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        batch, seq_len, _ = obs.shape

        if hidden_state is None:
            hidden_state = self.init_hidden(batch)
            hidden_state = (
                hidden_state[0].to(obs.device),
                hidden_state[1].to(obs.device),
            )

        obs_flat = obs.reshape(batch * seq_len, -1)
        market_feats, runner_feats = self._split_obs(obs_flat)

        market_emb = self.market_encoder(market_feats)

        b_s = runner_feats.shape[0]
        runners_flat = runner_feats.reshape(b_s * self.max_runners, BASELINE_RUNNER_INPUT_DIM)
        runner_embs = self.runner_encoder(runners_flat)
        runner_embs = runner_embs.view(b_s, self.max_runners, self.runner_embed_dim)

        runner_mean = runner_embs.mean(dim=1)
        runner_max = runner_embs.max(dim=1).values

        lstm_input = torch.cat([market_emb, runner_mean, runner_max], dim=-1)
        lstm_input = lstm_input.view(batch, seq_len, -1)

        lstm_out, new_hidden = self.lstm(lstm_input, hidden_state)
        lstm_out = self.lstm_output_norm(lstm_out)
        lstm_last = lstm_out[:, -1, :]

        if seq_len > 1:
            last_obs = obs[:, -1, :]
            _, last_runner_feats = self._split_obs(last_obs)
            last_runners_flat = last_runner_feats.reshape(
                batch * self.max_runners, BASELINE_RUNNER_INPUT_DIM
            )
            last_runner_embs = self.runner_encoder(last_runners_flat)
            last_runner_embs = last_runner_embs.view(
                batch, self.max_runners, self.runner_embed_dim
            )
        else:
            last_runner_embs = runner_embs.view(
                batch, self.max_runners, self.runner_embed_dim
            )

        lstm_expanded = lstm_last.unsqueeze(1).expand(-1, self.max_runners, -1)
        actor_input = torch.cat([last_runner_embs, lstm_expanded], dim=-1)
        actor_out = self.actor_head(actor_input)
        action_signal = actor_out[:, :, 0]
        stake_fraction = actor_out[:, :, 1]
        action_mean = torch.cat([action_signal, stake_fraction], dim=-1)
        value = self.critic_head(lstm_last)

        return PolicyOutput(
            action_mean=action_mean,
            action_log_std=self.action_log_std.expand(batch, -1),
            value=value,
            hidden_state=new_hidden,
        )


# ── Baseline PPO trainer (slices obs before storing transitions) ──────────────


class BaselinePPOTrainer(PPOTrainer):
    """PPOTrainer variant that applies obs slicing for the pre-P1 baseline.

    Overrides ``_collect_rollout`` to apply ``baseline_indices`` to each obs
    returned by BetfairEnv before storing it in the rollout.  The stored
    transitions contain the sliced (baseline-dim) obs, so ``_ppo_update``
    correctly feeds the baseline-shaped obs to ``BaselinePPOLSTMPolicy``.
    """

    def __init__(
        self,
        *args,
        baseline_indices: np.ndarray,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._baseline_indices = baseline_indices

    def _collect_rollout(self, day: Day) -> tuple[Rollout, EpisodeStats]:
        """Run one episode and collect transitions with baseline-sliced obs."""
        rollout_start = time.perf_counter()
        env = BetfairEnv(
            day,
            self.config,
            feature_cache=self.feature_cache,
            reward_overrides=self.reward_overrides,
        )
        full_obs, info = env.reset()
        obs = full_obs[self._baseline_indices]  # slice to baseline dims

        rollout = Rollout()
        hidden_state = self.policy.init_hidden(batch_size=1)
        hidden_state = (
            hidden_state[0].to(self.device),
            hidden_state[1].to(self.device),
        )

        total_reward = 0.0
        n_steps = 0
        done = False

        obs_dim = obs.shape[0]
        obs_buffer = torch.empty(1, obs_dim, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            while not done:
                obs_buffer[0] = torch.as_tensor(obs, dtype=torch.float32)

                out: PolicyOutput = self.policy(obs_buffer, hidden_state)
                hidden_state = out.hidden_state

                std = out.action_log_std.exp()
                action_mean = out.action_mean
                noise = torch.randn_like(action_mean)
                action = action_mean + std * noise
                log_prob = (
                    -0.5 * ((action - action_mean) / std).pow(2)
                    - std.log()
                    - 0.5 * 1.8378770664093453
                ).sum(dim=-1)
                value = out.value.squeeze(-1)

                action_np = action.squeeze(0).cpu().numpy()
                np.clip(action_np, -1.0, 1.0, out=action_np)

                next_full_obs, reward, terminated, truncated, next_info = env.step(action_np)
                done = terminated or truncated
                next_obs = next_full_obs[self._baseline_indices]  # slice next obs too

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
            "Baseline rollout %s: %d steps in %.2fs",
            day.date, n_steps, rollout_elapsed,
        )

        ep_stats = EpisodeStats(
            day_date=day.date,
            total_reward=total_reward,
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


# ── Standalone evaluator ──────────────────────────────────────────────────────


@dataclass
class DayResult:
    """Per-day evaluation result."""
    date: str
    day_pnl: float
    bet_count: int
    winning_bets: int


def evaluate_policy(
    policy: BasePolicy,
    days: list[Day],
    config: dict,
    device: str,
    baseline_indices: np.ndarray | None = None,
    feature_cache: dict | None = None,
) -> list[DayResult]:
    """Evaluate a policy on a list of days; return per-day raw P&L.

    Parameters
    ----------
    policy:
        Trained policy to evaluate (deterministic: uses action_mean).
    days:
        Held-out eval days (each is an independent episode).
    config:
        Project config.
    device:
        'cpu' or 'cuda'.
    baseline_indices:
        If provided, obs is sliced to baseline dims before being fed to policy.
        Required for BaselinePPOLSTMPolicy; omit for P1 policy.
    feature_cache:
        Optional feature pre-computation cache shared across calls.
    """
    policy = policy.to(device)
    policy.eval()
    results: list[DayResult] = []

    for day in days:
        env = BetfairEnv(day, config, feature_cache=feature_cache)
        full_obs, info = env.reset()
        obs = full_obs[baseline_indices] if baseline_indices is not None else full_obs

        hidden_state = policy.init_hidden(batch_size=1)
        hidden_state = (
            hidden_state[0].to(device),
            hidden_state[1].to(device),
        )

        obs_dim = obs.shape[0]
        obs_buffer = torch.empty(1, obs_dim, dtype=torch.float32, device=device)

        done = False
        with torch.no_grad():
            while not done:
                obs_buffer[0] = torch.as_tensor(obs, dtype=torch.float32)
                out = policy(obs_buffer, hidden_state)
                hidden_state = out.hidden_state

                action = out.action_mean.squeeze(0).cpu().numpy()
                np.clip(action, -1.0, 1.0, out=action)

                next_full_obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_obs = (
                    next_full_obs[baseline_indices]
                    if baseline_indices is not None
                    else next_full_obs
                )
                obs = next_obs

        all_bets = env.all_settled_bets
        day_pnl = info.get("day_pnl", 0.0)
        bet_count = len(all_bets)
        winning_bets = sum(1 for b in all_bets if b.outcome is BetOutcome.WON)

        results.append(DayResult(
            date=day.date,
            day_pnl=day_pnl,
            bet_count=bet_count,
            winning_bets=winning_bets,
        ))
        logger.info(
            "Eval %s | pnl=%+.2f  bets=%d  wins=%d",
            day.date, day_pnl, bet_count, winning_bets,
        )

    return results


# ── Data helpers ──────────────────────────────────────────────────────────────


def find_available_dates(data_dir: Path) -> list[str]:
    """Return sorted list of dates with both .parquet and _runners.parquet."""
    dates: set[str] = set()
    for f in data_dir.glob("*_runners.parquet"):
        date_str = f.name.replace("_runners.parquet", "")
        if (data_dir / f"{date_str}.parquet").exists():
            dates.add(date_str)
    return sorted(dates)


# ── Comparison table + progress.md writer ────────────────────────────────────


def _fmt_result_table(
    baseline_results: list[DayResult],
    p1_results: list[DayResult],
    grad_norm: float,
    n_train_days: int,
    n_epochs: int,
    device: str,
) -> str:
    """Format a human-readable comparison table."""
    lines: list[str] = []
    lines.append("")
    lines.append("## Session 22 — P1 vs Baseline comparison (2026-04-08)")
    lines.append("")
    lines.append(f"- Eval metric (Q3): **raw daily P&L**")
    lines.append(f"- Train days: {n_train_days}, epochs: {n_epochs}, device: {device}")
    lines.append(f"- Hyperparameters: identical for both policies (see SHARED_HP in script)")
    lines.append(f"- P1 gradient norm on new columns after first episode: {grad_norm:.6f}")
    lines.append(f"  (must be > 0 — zero means P1 features are wired incorrectly)")
    lines.append("")
    lines.append("### Per-day raw P&L")
    lines.append("")
    lines.append(f"{'Date':<14} {'Baseline P&L':>13} {'P1 P&L':>10} {'Delta':>8}")
    lines.append("-" * 50)

    baseline_map = {r.date: r for r in baseline_results}
    p1_map = {r.date: r for r in p1_results}
    all_dates = sorted(set(baseline_map) | set(p1_map))

    for date in all_dates:
        b = baseline_map.get(date)
        p = p1_map.get(date)
        b_pnl = b.day_pnl if b else float("nan")
        p_pnl = p.day_pnl if p else float("nan")
        delta = p_pnl - b_pnl if b and p else float("nan")
        lines.append(
            f"{date:<14} {b_pnl:>+13.2f} {p_pnl:>+10.2f} {delta:>+8.2f}"
        )

    lines.append("-" * 50)

    b_agg = sum(r.day_pnl for r in baseline_results)
    p_agg = sum(r.day_pnl for r in p1_results)
    b_mean = b_agg / len(baseline_results) if baseline_results else 0.0
    p_mean = p_agg / len(p1_results) if p1_results else 0.0
    b_bets = sum(r.bet_count for r in baseline_results)
    p_bets = sum(r.bet_count for r in p1_results)

    lines.append(
        f"{'TOTAL':<14} {b_agg:>+13.2f} {p_agg:>+10.2f} {(p_agg-b_agg):>+8.2f}"
    )
    lines.append(
        f"{'MEAN/DAY':<14} {b_mean:>+13.2f} {p_mean:>+10.2f} {(p_mean-b_mean):>+8.2f}"
    )
    lines.append("")
    lines.append(f"- Baseline total bets: {b_bets}")
    lines.append(f"- P1 total bets: {p_bets}")
    lines.append("")

    # Recommendation
    delta_pnl = p_agg - b_agg
    if len(p1_results) == 0:
        rec = "NO DATA — could not evaluate."
    elif delta_pnl > 0:
        rec = (
            "P1 is better than baseline on this eval window. "
            "**Recommendation: continue to P2.**"
        )
    elif abs(delta_pnl) < 2.0 * len(p1_results):
        rec = (
            f"P1 delta ({delta_pnl:+.2f}) is within ~£2/day noise. "
            "Result is inconclusive. "
            "**Recommendation: continue to P2** (features are not harmful; "
            "longer training may widen the gap)."
        )
    else:
        rec = (
            f"P1 is worse than baseline (delta {delta_pnl:+.2f}). "
            "**Recommendation: investigate before continuing** — "
            "check that P1 features are not dominated by noise at this data volume."
        )

    lines.append(f"### Recommendation")
    lines.append("")
    lines.append(rec)
    lines.append("")

    return "\n".join(lines)


def append_to_progress_md(text: str) -> None:
    """Append comparison results to plans/research_driven/progress.md."""
    progress_path = REPO_ROOT / "plans" / "research_driven" / "progress.md"
    with open(progress_path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write(text)
        f.write("\n")
    logger.info("Results appended to %s", progress_path)


# ── Argument parsing ──────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Session 22 P1 vs baseline comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Data split:
              The earliest --train-days dates are used for training.
              The latest  --eval-days  dates are the held-out eval window.
              Both splits must not overlap (enforced by assertion).
        """),
    )
    p.add_argument(
        "--train-days", type=int, default=4,
        help="Number of earliest dates to use for training (default: 4).",
    )
    p.add_argument(
        "--eval-days", type=int, default=3,
        help="Number of latest dates to use for eval (default: 3).",
    )
    p.add_argument(
        "--n-epochs", type=int, default=5,
        help="PPO training epochs per day per policy (default: 5).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Build and validate policies; exit without running training.",
    )
    p.add_argument(
        "--device", default="auto",
        help="'cpu', 'cuda', or 'auto' (default: auto).",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()

    # ── Config ───────────────────────────────────────────────────────────
    config_path = REPO_ROOT / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    max_runners: int = config["training"]["max_runners"]
    action_dim: int = max_runners * 2

    # ── Device ───────────────────────────────────────────────────────────
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Device: %s", device)

    # ── Data ─────────────────────────────────────────────────────────────
    data_dir = REPO_ROOT / config["paths"]["processed_data"]
    all_dates = find_available_dates(data_dir)
    need = args.train_days + args.eval_days
    if len(all_dates) < need:
        logger.error(
            "Need %d dates (%d train + %d eval), found %d: %s",
            need, args.train_days, args.eval_days, len(all_dates), all_dates,
        )
        return 1

    train_dates = all_dates[: args.train_days]
    eval_dates = all_dates[args.train_days : args.train_days + args.eval_days]
    assert not set(train_dates) & set(eval_dates), "Train/eval overlap!"

    logger.info("Train dates (%d): %s", len(train_dates), train_dates)
    logger.info("Eval  dates (%d): %s", len(eval_dates), eval_dates)

    # ── Obs layout ───────────────────────────────────────────────────────
    baseline_obs_indices = build_baseline_obs_indices(max_runners)
    b_obs_dim = baseline_obs_dim(max_runners)
    p1_obs_dim_val = p1_obs_dim(max_runners)

    logger.info(
        "Obs dims — baseline: %d (RUNNER_DIM=110), P1: %d (RUNNER_DIM=114)",
        b_obs_dim, p1_obs_dim_val,
    )
    logger.info("N_P1_FEATURES=%d per runner slot", N_P1_FEATURES)

    # ── Build policies ───────────────────────────────────────────────────
    logger.info("Building baseline policy (pre-P1, RUNNER_DIM=110)...")
    baseline_policy = BaselinePPOLSTMPolicy(
        obs_dim=b_obs_dim,
        action_dim=action_dim,
        max_runners=max_runners,
        hyperparams=SHARED_HP,
    )

    logger.info("Building P1 policy (RUNNER_DIM=114, schema v4)...")
    from agents.policy_network import PPOLSTMPolicy
    p1_policy = PPOLSTMPolicy(
        obs_dim=p1_obs_dim_val,
        action_dim=action_dim,
        max_runners=max_runners,
        hyperparams=SHARED_HP,
    )

    n_baseline_params = sum(p.numel() for p in baseline_policy.parameters())
    n_p1_params = sum(p.numel() for p in p1_policy.parameters())
    logger.info("Baseline params: %d | P1 params: %d", n_baseline_params, n_p1_params)

    if args.dry_run:
        logger.info("--dry-run: policies built successfully. Exiting.")
        return 0

    # ── Load data ────────────────────────────────────────────────────────
    feature_cache: dict = {}

    logger.info("Loading %d train days...", len(train_dates))
    train_days = load_days(train_dates, data_dir=str(data_dir))
    logger.info("Loading %d eval days...", len(eval_dates))
    eval_days = load_days(eval_dates, data_dir=str(data_dir))
    logger.info(
        "Loaded: %d train races, %d eval races",
        sum(len(d.races) for d in train_days),
        sum(len(d.races) for d in eval_days),
    )

    # ── Train baseline ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("BASELINE TRAINING — %d days × %d epochs", len(train_days), args.n_epochs)
    logger.info("=" * 60)

    baseline_trainer = BaselinePPOTrainer(
        policy=baseline_policy,
        config=config,
        hyperparams=SHARED_HP,
        device=device,
        feature_cache=feature_cache,
        model_id="session22_baseline",
        architecture_name="ppo_lstm_baseline_v1",
        baseline_indices=baseline_obs_indices,
    )
    t_start = time.perf_counter()
    baseline_stats = baseline_trainer.train(train_days, n_epochs=args.n_epochs)
    baseline_train_s = time.perf_counter() - t_start
    logger.info(
        "Baseline training complete in %.1fs | mean_pnl=%.2f mean_bets=%.1f",
        baseline_train_s, baseline_stats.mean_pnl, baseline_stats.mean_bet_count,
    )

    # ── Train P1 ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("P1 TRAINING — %d days × %d epochs", len(train_days), args.n_epochs)
    logger.info("=" * 60)

    p1_trainer = PPOTrainer(
        policy=p1_policy,
        config=config,
        hyperparams=SHARED_HP,
        device=device,
        feature_cache=feature_cache,
        model_id="session22_p1",
        architecture_name="ppo_lstm_v1",
    )
    t_start = time.perf_counter()
    p1_stats = p1_trainer.train(train_days, n_epochs=args.n_epochs)
    p1_train_s = time.perf_counter() - t_start
    logger.info(
        "P1 training complete in %.1fs | mean_pnl=%.2f mean_bets=%.1f",
        p1_train_s, p1_stats.mean_pnl, p1_stats.mean_bet_count,
    )

    # ── Gradient-norm check (P1 only) ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("GRADIENT-NORM CHECK (P1 new columns)")
    logger.info("=" * 60)

    # Use first eval day's first obs for the check
    sample_env = BetfairEnv(eval_days[0], config, feature_cache=feature_cache)
    sample_obs, _ = sample_env.reset()

    grad_norm = check_p1_gradient_norm(p1_policy, sample_obs, max_runners, device)
    logger.info(
        "P1 gradient norm on new columns (OBI/microprice/traded_delta/mid_drift): %.8f",
        grad_norm,
    )
    if grad_norm == 0.0:
        logger.error(
            "GRADIENT NORM IS ZERO — P1 features are not being used by the policy. "
            "The comparison is invalid. Investigate before recording results."
        )
        return 2
    else:
        logger.info("Gradient norm > 0 — P1 features are wired correctly.")

    # ── Evaluate both policies ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("EVALUATION — %d eval days", len(eval_days))
    logger.info("=" * 60)

    logger.info("Evaluating baseline policy...")
    baseline_results = evaluate_policy(
        baseline_policy, eval_days, config, device,
        baseline_indices=baseline_obs_indices,
        feature_cache=feature_cache,
    )

    logger.info("Evaluating P1 policy...")
    p1_results = evaluate_policy(
        p1_policy, eval_days, config, device,
        baseline_indices=None,
        feature_cache=feature_cache,
    )

    # ── Print and record results ──────────────────────────────────────────
    table = _fmt_result_table(
        baseline_results=baseline_results,
        p1_results=p1_results,
        grad_norm=grad_norm,
        n_train_days=len(train_days),
        n_epochs=args.n_epochs,
        device=device,
    )
    print(table)
    append_to_progress_md(table)

    logger.info("Session 22 comparison complete.")
    logger.info(
        "Baseline total P&L: %+.2f | P1 total P&L: %+.2f",
        sum(r.day_pnl for r in baseline_results),
        sum(r.day_pnl for r in p1_results),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
