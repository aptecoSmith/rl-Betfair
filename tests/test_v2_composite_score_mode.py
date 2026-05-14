"""Tests for ``composite_score_mode`` — scalping-locked-fitness-and-age-obs Phase 1.

The plan adds a `locked_weighted` mode to the GA selection-score
formula:

    composite_score = locked_pnl + 0.25 * naked_pnl

The 0.25 weight is locked (plan hard_constraints §9). Default
mode is `total_reward` so a launch without the flag is
byte-identical to pre-plan behaviour.
"""

from __future__ import annotations

from training_v2.cohort.runner import (
    COMPOSITE_SCORE_MODE_LOCKED_WEIGHTED,
    COMPOSITE_SCORE_MODE_TOTAL_REWARD,
    _composite_score,
)
from training_v2.cohort.worker import EvalSummary


def _eval(
    *,
    total_reward: float = 0.0,
    locked_pnl: float = 0.0,
    naked_pnl: float = 0.0,
    arbs_completed: int = 0,
    arbs_closed: int = 0,
) -> EvalSummary:
    """Build a minimal EvalSummary populated with the score-relevant fields."""
    return EvalSummary(
        eval_day="2026-04-30",
        total_reward=float(total_reward),
        n_steps=0,
        day_pnl=0.0,
        bet_count=0,
        winning_bets=0,
        bet_precision=0.0,
        pnl_per_bet=0.0,
        early_picks=0,
        profitable=False,
        action_histogram={},
        arbs_completed=int(arbs_completed),
        arbs_naked=0,
        arbs_closed=int(arbs_closed),
        arbs_force_closed=0,
        arbs_stop_closed=0,
        arbs_target_pnl_refused=0,
        pairs_opened=0,
        locked_pnl=float(locked_pnl),
        naked_pnl=float(naked_pnl),
        closed_pnl=0.0,
        force_closed_pnl=0.0,
        stop_closed_pnl=0.0,
        wall_time_sec=0.0,
    )


def test_locked_weighted_score_formula() -> None:
    """locked=100, naked=200 → 100 + 0.25 * 200 = 150."""
    score = _composite_score(
        _eval(locked_pnl=100.0, naked_pnl=200.0),
        maturation_bonus_weight=0.0,
        composite_score_mode=COMPOSITE_SCORE_MODE_LOCKED_WEIGHTED,
    )
    assert score == 150.0


def test_locked_weighted_handles_negative_naked() -> None:
    """locked=100, naked=-100 → 100 + 0.25 * (-100) = 75."""
    score = _composite_score(
        _eval(locked_pnl=100.0, naked_pnl=-100.0),
        maturation_bonus_weight=0.0,
        composite_score_mode=COMPOSITE_SCORE_MODE_LOCKED_WEIGHTED,
    )
    assert score == 75.0


def test_total_reward_mode_unchanged() -> None:
    """Default mode → score equals total_reward + w * (matured + closed).

    Byte-identity guard: the existing formula is preserved when the
    new mode flag is absent or set to its default.
    """
    eval_stats = _eval(
        total_reward=42.0,
        locked_pnl=999.0,  # ignored in total_reward mode
        naked_pnl=-999.0,
        arbs_completed=3,
        arbs_closed=1,
    )
    # Default kwarg (no mode passed).
    default_score = _composite_score(eval_stats, maturation_bonus_weight=0.0)
    assert default_score == 42.0
    # Explicit total_reward mode matches default.
    explicit_score = _composite_score(
        eval_stats,
        maturation_bonus_weight=0.0,
        composite_score_mode=COMPOSITE_SCORE_MODE_TOTAL_REWARD,
    )
    assert explicit_score == default_score
    # Maturation bonus path still works in total_reward mode.
    with_bonus = _composite_score(
        eval_stats,
        maturation_bonus_weight=5.0,
        composite_score_mode=COMPOSITE_SCORE_MODE_TOTAL_REWARD,
    )
    # total_reward 42 + 5 * (3 completed + 1 closed) = 42 + 20 = 62
    assert with_bonus == 62.0


def test_locked_weighted_ignores_maturation_bonus_weight() -> None:
    """hard_constraints §9: locked_weighted is a single-formula
    replacement, not an additive modification. ``maturation_bonus_weight``
    has no effect in this mode.
    """
    eval_stats = _eval(
        locked_pnl=10.0, naked_pnl=20.0,
        arbs_completed=100, arbs_closed=100,
    )
    base = _composite_score(
        eval_stats,
        maturation_bonus_weight=0.0,
        composite_score_mode=COMPOSITE_SCORE_MODE_LOCKED_WEIGHTED,
    )
    with_bonus = _composite_score(
        eval_stats,
        maturation_bonus_weight=999.0,
        composite_score_mode=COMPOSITE_SCORE_MODE_LOCKED_WEIGHTED,
    )
    assert base == with_bonus == 10.0 + 0.25 * 20.0
