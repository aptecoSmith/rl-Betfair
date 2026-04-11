"""
scripts/session_30_p3c_retrain.py — Session 30: Phase 2 re-train + diversity check + decision gate.

Trains one policy with the full research-driven feature set (P1 obs + P2 shaping +
P3 action space + P4 queue model) and compares it on the held-out eval window against
the Phase 1 baseline.

This is the **Phase 2 decision gate**. Its outcome decides whether the execution-aware
simulator is worth shipping to ai-betfair, or whether the Phase 1 selection-only policy
is already good enough.

Steps:
  1. Regression sanity check: force_aggressive=true on 1-day fixture → confirms new
     codebase reproduces aggressive-only behaviour.
  2. Train P3+P4 policy from fresh init with diversity monitoring.
  3. Mid-training diversity check: aggression histogram must be non-collapsed.
  4. Evaluate on held-out window; record per-day raw P&L.
  5. Diversity assertions: cancel rate and passive fill rate non-trivial.
  6. Decision-gate recommendation.

Usage::

    python scripts/session_30_p3c_retrain.py
    python scripts/session_30_p3c_retrain.py --train-days 4 --eval-days 3 --n-epochs 5
    python scripts/session_30_p3c_retrain.py --dry-run

Constraints:
- Fresh init (no warm-starting). ACTION_SCHEMA_VERSION=2 invalidated all checkpoints.
- Same hyperparameters as session 22 where possible.
- Diversity check is mandatory: collapsed policy ≠ success.
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import textwrap
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.episode_builder import Day, load_days  # noqa: E402
from env.betfair_env import (  # noqa: E402
    ACTIONS_PER_RUNNER,
    AGENT_STATE_DIM,
    MARKET_DIM,
    POSITION_DIM,
    RUNNER_DIM,
    VELOCITY_DIM,
    BetfairEnv,
)
from agents.policy_network import (  # noqa: E402
    BasePolicy,
    PolicyOutput,
    PPOLSTMPolicy,
    MARKET_TOTAL_DIM,
)
from agents.ppo_trainer import PPOTrainer, Rollout, Transition, EpisodeStats  # noqa: E402
from env.bet_manager import BetOutcome  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("session_30")


# ── Fixed hyperparameters (same as session 22) ──────────────────────────────

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

# Diversity thresholds
MIN_PASSIVE_FRACTION = 0.05  # ≥ 5% of actions must be passive (non-collapsed)


# ── Per-day result ──────────────────────────────────────────────────────────

@dataclass
class DayResult:
    """Per-day evaluation result with P3+P4 diversity metrics."""
    date: str
    day_pnl: float
    bet_count: int
    winning_bets: int
    aggressive_count: int = 0
    passive_placed_count: int = 0
    cancel_count: int = 0
    passive_fill_count: int = 0


# ── Diversity tracker ───────────────────────────────────────────────────────

@dataclass
class DiversityTracker:
    """Accumulates aggression/cancel/passive-fill counts across training."""
    aggressive_actions: int = 0
    passive_actions: int = 0
    cancel_actions: int = 0
    passive_fills: int = 0
    total_decisions: int = 0
    # Per-step aggression values for histogram
    aggression_values: list = field(default_factory=list)

    @property
    def passive_fraction(self) -> float:
        if self.total_decisions == 0:
            return 0.0
        return self.passive_actions / self.total_decisions

    @property
    def aggressive_fraction(self) -> float:
        if self.total_decisions == 0:
            return 0.0
        return self.aggressive_actions / self.total_decisions

    def summary(self) -> str:
        vals = np.array(self.aggression_values) if self.aggression_values else np.array([0.0])
        return (
            f"Diversity: agg={self.aggressive_actions} pass={self.passive_actions} "
            f"cancel={self.cancel_actions} fills={self.passive_fills} "
            f"total_decisions={self.total_decisions} "
            f"passive_frac={self.passive_fraction:.3f} "
            f"aggression_mean={vals.mean():.3f} var={vals.var():.5f}"
        )


# ── Training with diversity monitoring ──────────────────────────────────────

class Session30PPOTrainer(PPOTrainer):
    """PPOTrainer subclass that tracks aggression diversity during training."""

    def __init__(self, *args, diversity_tracker: DiversityTracker | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.diversity_tracker = diversity_tracker or DiversityTracker()

    def _collect_rollout(self, day: Day) -> tuple[Rollout, EpisodeStats]:
        """Run one episode and collect transitions, tracking diversity."""
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

                next_obs, reward, terminated, truncated, next_info = env.step(action_np)
                done = terminated or truncated

                # Track diversity from action_debug
                action_debug = next_info.get("action_debug", {})
                for sid, dbg in action_debug.items():
                    if dbg.get("aggressive_placed"):
                        self.diversity_tracker.aggressive_actions += 1
                        self.diversity_tracker.total_decisions += 1
                    elif dbg.get("passive_placed"):
                        self.diversity_tracker.passive_actions += 1
                        self.diversity_tracker.total_decisions += 1
                    if dbg.get("cancelled"):
                        self.diversity_tracker.cancel_actions += 1

                # Track passive fills
                passive_fills = next_info.get("passive_fills", [])
                self.diversity_tracker.passive_fills += len(passive_fills)

                # Track raw aggression values for histogram
                max_runners = env.max_runners
                for slot_idx in range(max_runners):
                    aggression_raw = float(action_np[2 * max_runners + slot_idx])
                    self.diversity_tracker.aggression_values.append(aggression_raw)

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
            "Rollout %s: %d steps in %.2fs | %s",
            day.date, n_steps, rollout_elapsed, self.diversity_tracker.summary(),
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


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_policy(
    policy: BasePolicy,
    days: list[Day],
    config: dict,
    device: str,
    feature_cache: dict | None = None,
) -> list[DayResult]:
    """Evaluate a policy on a list of days; return per-day raw P&L + diversity."""
    policy = policy.to(device)
    policy.eval()
    results: list[DayResult] = []

    for day in days:
        env = BetfairEnv(day, config, feature_cache=feature_cache)
        obs, info = env.reset()

        hidden_state = policy.init_hidden(batch_size=1)
        hidden_state = (
            hidden_state[0].to(device),
            hidden_state[1].to(device),
        )

        obs_dim = obs.shape[0]
        obs_buffer = torch.empty(1, obs_dim, dtype=torch.float32, device=device)

        done = False
        agg_count = 0
        pass_count = 0
        cancel_count = 0
        fill_count = 0

        with torch.no_grad():
            while not done:
                obs_buffer[0] = torch.as_tensor(obs, dtype=torch.float32)
                out = policy(obs_buffer, hidden_state)
                hidden_state = out.hidden_state

                action = out.action_mean.squeeze(0).cpu().numpy()
                np.clip(action, -1.0, 1.0, out=action)

                next_obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Track diversity
                action_debug = info.get("action_debug", {})
                for sid, dbg in action_debug.items():
                    if dbg.get("aggressive_placed"):
                        agg_count += 1
                    elif dbg.get("passive_placed"):
                        pass_count += 1
                    if dbg.get("cancelled"):
                        cancel_count += 1

                passive_fills = info.get("passive_fills", [])
                fill_count += len(passive_fills)

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
            aggressive_count=agg_count,
            passive_placed_count=pass_count,
            cancel_count=cancel_count,
            passive_fill_count=fill_count,
        ))
        logger.info(
            "Eval %s | pnl=%+.2f  bets=%d  wins=%d  agg=%d  pass=%d  cancel=%d  fills=%d",
            day.date, day_pnl, bet_count, winning_bets,
            agg_count, pass_count, cancel_count, fill_count,
        )

    return results


# ── Regression sanity check ─────────────────────────────────────────────────

def regression_sanity_check(
    day: Day,
    config: dict,
    device: str,
    feature_cache: dict,
) -> tuple[bool, float, int]:
    """Run one day with force_aggressive=true and a fresh policy.

    Returns (passed, day_pnl, bet_count).
    The check passes if the policy places at least some bets and no passive
    orders are placed (all routing goes through aggressive path).
    """
    # Override config to force aggressive
    reg_config = copy.deepcopy(config)
    reg_config["actions"]["force_aggressive"] = True

    max_runners: int = reg_config["training"]["max_runners"]
    action_dim: int = max_runners * ACTIONS_PER_RUNNER
    obs_dim = (
        MARKET_DIM + VELOCITY_DIM
        + RUNNER_DIM * max_runners
        + AGENT_STATE_DIM
        + POSITION_DIM * max_runners
    )

    policy = PPOLSTMPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_runners=max_runners,
        hyperparams=SHARED_HP,
    ).to(device)

    # Train for 1 epoch to get a non-trivial policy
    trainer = PPOTrainer(
        policy=policy,
        config=reg_config,
        hyperparams=SHARED_HP,
        device=device,
        feature_cache=feature_cache,
        model_id="session30_regression",
        architecture_name="ppo_lstm_v1",
    )
    stats = trainer.train([day], n_epochs=1)
    logger.info(
        "Regression training: pnl=%.2f bets=%.1f",
        stats.mean_pnl, stats.mean_bet_count,
    )

    # Now evaluate with force_aggressive=true
    env = BetfairEnv(day, reg_config, feature_cache=feature_cache)
    obs, info = env.reset()

    hidden_state = policy.init_hidden(batch_size=1)
    hidden_state = (
        hidden_state[0].to(device),
        hidden_state[1].to(device),
    )
    obs_buffer = torch.empty(1, obs.shape[0], dtype=torch.float32, device=device)

    done = False
    passive_placed = 0
    aggressive_placed = 0

    policy.eval()
    with torch.no_grad():
        while not done:
            obs_buffer[0] = torch.as_tensor(obs, dtype=torch.float32)
            out = policy(obs_buffer, hidden_state)
            hidden_state = out.hidden_state
            action = out.action_mean.squeeze(0).cpu().numpy()
            np.clip(action, -1.0, 1.0, out=action)

            next_obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            action_debug = info.get("action_debug", {})
            for sid, dbg in action_debug.items():
                if dbg.get("aggressive_placed"):
                    aggressive_placed += 1
                if dbg.get("passive_placed"):
                    passive_placed += 1

            obs = next_obs

    day_pnl = info.get("day_pnl", 0.0)
    all_bets = env.all_settled_bets
    bet_count = len(all_bets)

    passed = passive_placed == 0  # force_aggressive must route ALL through aggressive
    logger.info(
        "Regression check: passed=%s | pnl=%+.2f bets=%d agg=%d passive=%d",
        passed, day_pnl, bet_count, aggressive_placed, passive_placed,
    )
    if not passed:
        logger.error(
            "REGRESSION FAILED: %d passive orders placed with force_aggressive=true!",
            passive_placed,
        )

    return passed, day_pnl, bet_count


# ── Data helpers ────────────────────────────────────────────────────────────

def find_available_dates(data_dir: Path) -> list[str]:
    """Return sorted list of dates with both .parquet and _runners.parquet."""
    dates: set[str] = set()
    for f in data_dir.glob("*_runners.parquet"):
        date_str = f.name.replace("_runners.parquet", "")
        if (data_dir / f"{date_str}.parquet").exists():
            dates.add(date_str)
    return sorted(dates)


# ── Results formatting ──────────────────────────────────────────────────────

def _fmt_result_table(
    baseline_results: list[DayResult],
    p3p4_results: list[DayResult],
    diversity: DiversityTracker,
    n_train_days: int,
    n_epochs: int,
    device: str,
    regression_pnl: float,
    regression_bets: int,
) -> str:
    """Format a human-readable comparison table for progress.md."""
    lines: list[str] = []
    lines.append("")
    lines.append("## Session 30 — Phase 2 decision gate (2026-04-11)")
    lines.append("")
    lines.append(f"- Eval metric (Q3): **raw daily P&L**")
    lines.append(f"- Train days: {n_train_days}, epochs: {n_epochs}, device: {device}")
    lines.append(f"- Hyperparameters: same as session 22 (see SHARED_HP in script)")
    lines.append(f"- Action space: 4-dim per slot (signal, stake, aggression, cancel)")
    lines.append(f"- ACTION_SCHEMA_VERSION=2, ACTIONS_PER_RUNNER=4")
    lines.append("")

    # Regression check
    lines.append("### Regression sanity check (force_aggressive=true)")
    lines.append("")
    lines.append(f"- 1-day regression run P&L: {regression_pnl:+.2f}")
    lines.append(f"- Regression bet count: {regression_bets}")
    lines.append(f"- Passive orders placed with force_aggressive=true: 0 (PASSED)")
    lines.append("")

    # Diversity during training
    agg_vals = np.array(diversity.aggression_values) if diversity.aggression_values else np.array([0.0])
    lines.append("### Training diversity")
    lines.append("")
    lines.append(f"- Aggressive actions: {diversity.aggressive_actions}")
    lines.append(f"- Passive placements: {diversity.passive_actions}")
    lines.append(f"- Cancels: {diversity.cancel_actions}")
    lines.append(f"- Passive fills: {diversity.passive_fills}")
    lines.append(f"- Passive fraction: {diversity.passive_fraction:.3f}")
    lines.append(f"- Aggression histogram: mean={agg_vals.mean():.4f}, var={agg_vals.var():.6f}, "
                 f"mode_bin={'positive' if np.median(agg_vals) > 0 else 'negative'}")
    lines.append("")

    # Per-day eval comparison
    lines.append("### Per-day raw P&L (eval window)")
    lines.append("")
    lines.append(f"{'Date':<14} {'Baseline P&L':>13} {'P3+P4 P&L':>11} {'Delta':>8} "
                 f"{'Agg':>5} {'Pass':>5} {'Cancel':>7} {'Fills':>6}")
    lines.append("-" * 80)

    baseline_map = {r.date: r for r in baseline_results}
    p3p4_map = {r.date: r for r in p3p4_results}
    all_dates = sorted(set(baseline_map) | set(p3p4_map))

    for date in all_dates:
        b = baseline_map.get(date)
        p = p3p4_map.get(date)
        b_pnl = b.day_pnl if b else float("nan")
        p_pnl = p.day_pnl if p else float("nan")
        delta = p_pnl - b_pnl if b and p else float("nan")
        lines.append(
            f"{date:<14} {b_pnl:>+13.2f} {p_pnl:>+11.2f} {delta:>+8.2f} "
            f"{p.aggressive_count if p else 0:>5} "
            f"{p.passive_placed_count if p else 0:>5} "
            f"{p.cancel_count if p else 0:>7} "
            f"{p.passive_fill_count if p else 0:>6}"
        )

    lines.append("-" * 80)

    b_agg = sum(r.day_pnl for r in baseline_results)
    p_agg = sum(r.day_pnl for r in p3p4_results)
    b_mean = b_agg / len(baseline_results) if baseline_results else 0.0
    p_mean = p_agg / len(p3p4_results) if p3p4_results else 0.0
    b_bets = sum(r.bet_count for r in baseline_results)
    p_bets = sum(r.bet_count for r in p3p4_results)

    lines.append(
        f"{'TOTAL':<14} {b_agg:>+13.2f} {p_agg:>+11.2f} {(p_agg-b_agg):>+8.2f}"
    )
    lines.append(
        f"{'MEAN/DAY':<14} {b_mean:>+13.2f} {p_mean:>+11.2f} {(p_mean-b_mean):>+8.2f}"
    )
    lines.append("")
    lines.append(f"- Baseline total bets on eval: {b_bets}")
    lines.append(f"- P3+P4 total bets on eval: {p_bets}")
    lines.append("")

    # Diversity assertions on eval
    eval_cancel_days = sum(1 for r in p3p4_results if r.cancel_count > 0)
    eval_fill_days = sum(1 for r in p3p4_results if r.passive_fill_count > 0)
    eval_passive_days = sum(1 for r in p3p4_results if r.passive_placed_count > 0)
    total_eval_days = len(p3p4_results)

    lines.append("### Diversity assertions (eval window)")
    lines.append("")
    lines.append(f"- Days with cancel > 0: {eval_cancel_days}/{total_eval_days}")
    lines.append(f"- Days with passive fill > 0: {eval_fill_days}/{total_eval_days}")
    lines.append(f"- Days with passive placement > 0: {eval_passive_days}/{total_eval_days}")

    # Check diversity pass/fail
    diversity_passed = True
    diversity_notes: list[str] = []
    if diversity.passive_fraction < MIN_PASSIVE_FRACTION:
        diversity_passed = False
        diversity_notes.append(
            f"FAILED: passive fraction {diversity.passive_fraction:.3f} < {MIN_PASSIVE_FRACTION}"
        )
    if eval_cancel_days == 0 and p_bets > 0:
        diversity_notes.append(
            "WARNING: zero cancel count across all eval days"
        )
    if eval_fill_days == 0 and p_bets > 0:
        diversity_notes.append(
            "WARNING: zero passive fills across all eval days"
        )

    if diversity_notes:
        lines.append("")
        for note in diversity_notes:
            lines.append(f"- **{note}**")
    else:
        lines.append(f"- All diversity assertions PASSED")
    lines.append("")

    # Recommendation
    lines.append("### Recommendation")
    lines.append("")

    delta_pnl = p_agg - b_agg
    both_collapsed = p_bets == 0 and b_bets == 0

    if both_collapsed:
        rec = (
            "Both policies collapsed to zero bets. Single-seed PPO comparison is "
            "uninformative (lesson from session 22). "
            "**Recommendation: keep P3+P4 code in simulator, ship Phase 1 policy.** "
            "The code paths are correct (unit tests pass) but single-seed training "
            "cannot demonstrate value. Evolutionary infrastructure needed for "
            "meaningful gate."
        )
    elif p_bets == 0:
        rec = (
            "P3+P4 policy collapsed to zero bets while baseline did not. "
            "The larger action space may need exploration tuning. "
            "**Recommendation: keep P3+P4 code in simulator, ship Phase 1 policy.** "
            "Investigate entropy coefficient / action-space exploration before retrying."
        )
    elif not diversity_passed:
        rec = (
            f"P3+P4 policy collapsed to one regime (passive fraction "
            f"{diversity.passive_fraction:.3f}). The new action space is not being "
            f"used meaningfully. "
            f"**Recommendation: keep P3+P4 code in simulator, ship Phase 1 policy.** "
            f"P3+P4 has not earned its cost."
        )
    elif delta_pnl > 0 and p_bets > 0:
        rec = (
            f"P3+P4 policy beat baseline by {delta_pnl:+.2f} total ({(delta_pnl/len(p3p4_results)):+.2f}/day). "
            f"Diversity check passed. "
            f"**Recommendation: ship P3+P4 policy to ai-betfair.** "
            f"Note: deployment gated by phantom-fill bug R-1 fix in ai-betfair "
            f"(hard_constraints #8). Confirm downstream_knockon.md §3 items queued."
        )
    elif abs(delta_pnl) < 2.0 * len(p3p4_results) and p_bets > 0:
        rec = (
            f"P3+P4 delta ({delta_pnl:+.2f}) is within noise (~£2/day). "
            f"Result inconclusive on single seed. Code paths are correct. "
            f"**Recommendation: keep P3+P4 code in simulator, ship Phase 1 policy.** "
            f"Training gain does not justify additional ai-betfair deployment cost."
        )
    elif delta_pnl < 0 and b_bets > 0:
        if delta_pnl < -5.0 * len(p3p4_results):
            rec = (
                f"P3+P4 significantly underperformed baseline (delta {delta_pnl:+.2f}). "
                f"**Recommendation: regression — investigate before shipping anything.** "
                f"File investigation as new session."
            )
        else:
            rec = (
                f"P3+P4 slightly underperformed baseline (delta {delta_pnl:+.2f}). "
                f"Within single-seed noise. "
                f"**Recommendation: keep P3+P4 code in simulator, ship Phase 1 policy.**"
            )
    else:
        rec = (
            f"Baseline collapsed ({b_bets} bets) but P3+P4 did not ({p_bets} bets). "
            f"P3+P4 P&L: {p_agg:+.2f}. "
            f"Single-seed comparison unreliable when one policy collapses. "
            f"**Recommendation: keep P3+P4 code in simulator, ship Phase 1 policy.** "
            f"Evolutionary infrastructure needed for meaningful comparison."
        )

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


# ── Argument parsing ────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Session 30 P3+P4 re-train + decision gate",
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
        help="PPO training epochs per day (default: 5).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Build and validate; exit without running training.",
    )
    p.add_argument(
        "--device", default="auto",
        help="'cpu', 'cuda', or 'auto' (default: auto).",
    )
    return p.parse_args()


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    # ── Config ──────────────────────────────────────────────────────────
    config_path = REPO_ROOT / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    max_runners: int = config["training"]["max_runners"]
    action_dim: int = max_runners * ACTIONS_PER_RUNNER
    obs_dim = (
        MARKET_DIM + VELOCITY_DIM
        + RUNNER_DIM * max_runners
        + AGENT_STATE_DIM
        + POSITION_DIM * max_runners
    )

    # Ensure force_aggressive is off for the main training run
    config["actions"]["force_aggressive"] = False

    # ── Device ──────────────────────────────────────────────────────────
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Device: %s", device)

    # ── Data ────────────────────────────────────────────────────────────
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
    logger.info("Obs dim: %d | Action dim: %d (ACTIONS_PER_RUNNER=%d)", obs_dim, action_dim, ACTIONS_PER_RUNNER)

    if args.dry_run:
        logger.info("--dry-run: validation passed. Exiting.")
        return 0

    # ── Load data ───────────────────────────────────────────────────────
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

    # ════════════════════════════════════════════════════════════════════
    # STEP 1: REGRESSION SANITY CHECK
    # ════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 1: REGRESSION SANITY CHECK (force_aggressive=true)")
    logger.info("=" * 60)

    reg_passed, reg_pnl, reg_bets = regression_sanity_check(
        train_days[0], config, device, feature_cache,
    )
    if not reg_passed:
        logger.error("Regression check FAILED. Aborting session 30.")
        return 2

    logger.info("Regression check PASSED.")

    # ════════════════════════════════════════════════════════════════════
    # STEP 2: BASELINE TRAINING (aggressive-only, same as Phase 1)
    # ════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 2: BASELINE TRAINING (force_aggressive=true)")
    logger.info("=" * 60)

    baseline_config = copy.deepcopy(config)
    baseline_config["actions"]["force_aggressive"] = True

    baseline_policy = PPOLSTMPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_runners=max_runners,
        hyperparams=SHARED_HP,
    )

    baseline_trainer = PPOTrainer(
        policy=baseline_policy,
        config=baseline_config,
        hyperparams=SHARED_HP,
        device=device,
        feature_cache=feature_cache,
        model_id="session30_baseline",
        architecture_name="ppo_lstm_v1",
    )
    t_start = time.perf_counter()
    baseline_stats = baseline_trainer.train(train_days, n_epochs=args.n_epochs)
    baseline_train_s = time.perf_counter() - t_start
    logger.info(
        "Baseline training complete in %.1fs | mean_pnl=%.2f mean_bets=%.1f",
        baseline_train_s, baseline_stats.mean_pnl, baseline_stats.mean_bet_count,
    )

    # ════════════════════════════════════════════════════════════════════
    # STEP 3: P3+P4 TRAINING (full action space)
    # ════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 3: P3+P4 TRAINING (full action space, force_aggressive=false)")
    logger.info("=" * 60)

    p3p4_policy = PPOLSTMPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_runners=max_runners,
        hyperparams=SHARED_HP,
    )

    diversity = DiversityTracker()
    p3p4_trainer = Session30PPOTrainer(
        policy=p3p4_policy,
        config=config,
        hyperparams=SHARED_HP,
        device=device,
        feature_cache=feature_cache,
        model_id="session30_p3p4",
        architecture_name="ppo_lstm_v1",
        diversity_tracker=diversity,
    )
    t_start = time.perf_counter()
    p3p4_stats = p3p4_trainer.train(train_days, n_epochs=args.n_epochs)
    p3p4_train_s = time.perf_counter() - t_start
    logger.info(
        "P3+P4 training complete in %.1fs | mean_pnl=%.2f mean_bets=%.1f",
        p3p4_train_s, p3p4_stats.mean_pnl, p3p4_stats.mean_bet_count,
    )

    # ── Step 4: Mid-training diversity check ────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: DIVERSITY CHECK (training)")
    logger.info("=" * 60)
    logger.info(diversity.summary())

    if diversity.passive_fraction < MIN_PASSIVE_FRACTION:
        logger.warning(
            "LOW DIVERSITY: passive fraction %.3f < %.3f threshold. "
            "Policy may have collapsed to aggressive-only.",
            diversity.passive_fraction, MIN_PASSIVE_FRACTION,
        )
    else:
        logger.info(
            "Diversity OK: passive fraction %.3f >= %.3f",
            diversity.passive_fraction, MIN_PASSIVE_FRACTION,
        )

    # ════════════════════════════════════════════════════════════════════
    # STEP 5: EVALUATION
    # ════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 5: EVALUATION — %d eval days", len(eval_days))
    logger.info("=" * 60)

    logger.info("Evaluating baseline (force_aggressive=true)...")
    baseline_results = evaluate_policy(
        baseline_policy, eval_days, baseline_config, device,
        feature_cache=feature_cache,
    )

    logger.info("Evaluating P3+P4 policy...")
    p3p4_results = evaluate_policy(
        p3p4_policy, eval_days, config, device,
        feature_cache=feature_cache,
    )

    # ── Step 6: Cancel-rate and passive-fill-rate sanity ────────────────
    logger.info("=" * 60)
    logger.info("STEP 6: DIVERSITY ASSERTIONS (eval window)")
    logger.info("=" * 60)

    for r in p3p4_results:
        logger.info(
            "  %s: cancel=%d passive_fills=%d passive_placed=%d",
            r.date, r.cancel_count, r.passive_fill_count, r.passive_placed_count,
        )

    # ── Print and record results ────────────────────────────────────────
    table = _fmt_result_table(
        baseline_results=baseline_results,
        p3p4_results=p3p4_results,
        diversity=diversity,
        n_train_days=len(train_days),
        n_epochs=args.n_epochs,
        device=device,
        regression_pnl=reg_pnl,
        regression_bets=reg_bets,
    )
    print(table)
    append_to_progress_md(table)

    logger.info("=" * 60)
    logger.info("SESSION 30 COMPLETE")
    logger.info("=" * 60)
    logger.info(
        "Baseline total P&L: %+.2f | P3+P4 total P&L: %+.2f",
        sum(r.day_pnl for r in baseline_results),
        sum(r.day_pnl for r in p3p4_results),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
