"""Tests for the GA early-stop helpers in training_v2/cohort/runner.py
(scalping-tight-naked-variance tnv3, 2026-05-17).
"""

from __future__ import annotations

import math

import pytest

from training_v2.cohort.runner import (
    _early_stop_improved,
    _gen_early_stop_stats,
)


# ── _early_stop_improved ─────────────────────────────────────────────


def test_first_gen_returns_initial_marker() -> None:
    """A single-gen history can't be a stall — there's nothing to compare to."""
    h = [{"median_std": 100.0, "median_composite": 0.5, "beta_med": 0.01}]
    assert _early_stop_improved(h) == ["initial"]


def test_std_improvement_decreases_value() -> None:
    """median_std improves when it DECREASES by ≥ £5/day."""
    h = [
        {"median_std": 100.0, "median_composite": 0.5, "beta_med": 0.02},
        {"median_std": 90.0, "median_composite": 0.5, "beta_med": 0.02},
    ]
    improved = _early_stop_improved(h)
    assert "median_std" in improved


def test_std_no_improvement_below_threshold() -> None:
    """A 4 £/day decrease doesn't count (threshold is 5)."""
    h = [
        {"median_std": 100.0, "median_composite": 0.5, "beta_med": 0.02},
        {"median_std": 96.0, "median_composite": 0.5, "beta_med": 0.02},
    ]
    improved = _early_stop_improved(h)
    assert "median_std" not in improved


def test_composite_improvement_increases_value() -> None:
    """median_composite improves when it INCREASES by ≥ 1 % relative."""
    h = [
        {"median_std": 100.0, "median_composite": 1.0, "beta_med": 0.02},
        {"median_std": 100.0, "median_composite": 1.02, "beta_med": 0.02},
    ]
    improved = _early_stop_improved(h)
    assert "median_composite" in improved


def test_beta_improvement_in_either_direction() -> None:
    """β_med improves when it CHANGES (either up or down) by ≥ 10 %.
    The GA may explore lower-β territory, so we track magnitude only."""
    h_up = [
        {"median_std": 100.0, "median_composite": 0.5, "beta_med": 0.020},
        {"median_std": 100.0, "median_composite": 0.5, "beta_med": 0.024},
    ]
    h_down = [
        {"median_std": 100.0, "median_composite": 0.5, "beta_med": 0.020},
        {"median_std": 100.0, "median_composite": 0.5, "beta_med": 0.016},
    ]
    assert "beta_med" in _early_stop_improved(h_up)
    assert "beta_med" in _early_stop_improved(h_down)


def test_no_improvement_when_all_flat() -> None:
    """Hard stall: every signal within threshold of prior best."""
    h = [
        {"median_std": 100.0, "median_composite": 1.0, "beta_med": 0.020},
        {"median_std": 99.0, "median_composite": 1.005, "beta_med": 0.021},
    ]
    improved = _early_stop_improved(h)
    # std -1 < threshold 5; composite +0.5% < 1%; beta +5% < 10%
    assert improved == []


def test_compare_against_best_across_prior_gens() -> None:
    """The check uses BEST of all prior gens, not just immediately prior."""
    h = [
        {"median_std": 100.0, "median_composite": 1.0, "beta_med": 0.020},
        {"median_std": 50.0, "median_composite": 2.0, "beta_med": 0.030},
        {"median_std": 95.0, "median_composite": 1.5, "beta_med": 0.025},  # this gen
    ]
    improved = _early_stop_improved(h)
    # Current gen's std=95 is worse than the best-prior 50; composite 1.5 < 2.0;
    # beta: compares to immediate prior 0.030, change |0.025-0.030|/0.030 = 16.7%
    assert "median_std" not in improved
    assert "median_composite" not in improved
    assert "beta_med" in improved  # because |0.025-0.030|/0.030 >= 10%


# ── _gen_early_stop_stats ────────────────────────────────────────────

from training_v2.cohort.worker import EvalSummary, TrainSummary  # noqa: E402
from training_v2.cohort.genes import CohortGenes  # noqa: E402

from dataclasses import dataclass


@dataclass
class _FakeAgentResult:
    eval: object
    genes: object


def _make_eval(per_day_nakeds: list[float], locked_pnl: float = 80.0, day_pnl: float = 0.0):
    """Build an EvalSummary with per_day entries that have known naked_pnl values."""
    per_day = []
    for n in per_day_nakeds:
        per_day.append(EvalSummary(
            eval_day="2026-04-30", total_reward=0.0, n_steps=0,
            day_pnl=0.0, bet_count=0, winning_bets=0, bet_precision=0.0,
            pnl_per_bet=0.0, early_picks=0, profitable=False, action_histogram={},
            arbs_completed=0, arbs_naked=0, arbs_closed=0, arbs_force_closed=0,
            arbs_stop_closed=0, arbs_target_pnl_refused=0, pairs_opened=0,
            locked_pnl=locked_pnl, naked_pnl=n, closed_pnl=0.0,
            force_closed_pnl=0.0, stop_closed_pnl=0.0, wall_time_sec=0.0,
        ))
    return EvalSummary(
        eval_day="2026-04-30", total_reward=0.0, n_steps=0,
        day_pnl=day_pnl, bet_count=0, winning_bets=0, bet_precision=0.0,
        pnl_per_bet=0.0, early_picks=0, profitable=False, action_histogram={},
        arbs_completed=0, arbs_naked=0, arbs_closed=0, arbs_force_closed=0,
        arbs_stop_closed=0, arbs_target_pnl_refused=0, pairs_opened=0,
        locked_pnl=locked_pnl, naked_pnl=sum(per_day_nakeds)/len(per_day_nakeds) if per_day_nakeds else 0.0,
        closed_pnl=0.0, force_closed_pnl=0.0, stop_closed_pnl=0.0,
        wall_time_sec=0.0, per_day=per_day,
    )


def _make_genes(beta: float = 0.02) -> CohortGenes:
    return CohortGenes(
        learning_rate=1e-4, entropy_coeff=1e-3, clip_range=0.2, gae_lambda=0.95,
        value_coeff=0.5, mini_batch_size=64, hidden_size=64,
        naked_variance_penalty_beta=beta,
    )


def test_gen_stats_computes_median_std() -> None:
    """median_std is the median across the cohort of per-agent in-sample σ."""
    # 3 agents: per-day nakeds chosen so σ is exactly 50, 100, 150
    # (symmetric around 0)
    agents = [
        _FakeAgentResult(eval=_make_eval([-50.0, +50.0]), genes=_make_genes(0.01)),
        _FakeAgentResult(eval=_make_eval([-100.0, +100.0]), genes=_make_genes(0.02)),
        _FakeAgentResult(eval=_make_eval([-150.0, +150.0]), genes=_make_genes(0.03)),
    ]
    stats = _gen_early_stop_stats(agents, "total_reward", 0.0)
    # Sample std (ddof=1) of [-100, +100] = sqrt((100² + 100²)/1) = 141.42
    expected_middle_std = math.sqrt((100**2 + 100**2) / 1)
    assert stats["median_std"] == pytest.approx(expected_middle_std)
    assert stats["beta_med"] == pytest.approx(0.02)