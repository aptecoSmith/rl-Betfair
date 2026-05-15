"""Tests for ``composite_score_mode`` — scalping-locked-fitness-and-age-obs Phase 1.

The plan adds a `locked_weighted` mode to the GA selection-score
formula:

    composite_score = locked_pnl + 0.25 * naked_pnl

The 0.25 weight is locked (plan hard_constraints §9). Default
mode is `total_reward` so a launch without the flag is
byte-identical to pre-plan behaviour.
"""

from __future__ import annotations

import pytest

from training_v2.cohort.runner import (
    COMPOSITE_SCORE_MODE_LOCKED_WEIGHTED,
    COMPOSITE_SCORE_MODE_TIGHT_VARIANCE,
    COMPOSITE_SCORE_MODE_TOTAL_REWARD,
    TIGHT_VARIANCE_NAKED_COEF,
    TIGHT_VARIANCE_VOL_COEF,
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
    per_day: list | None = None,
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
        per_day=list(per_day or []),
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


# ── tight_variance mode (scalping-tight-naked-variance Phase 2A) ──────────


def test_tight_variance_score_formula() -> None:
    """Per-day naked = [-50, 0, +50] across 3 days → sample std = 50.
    locked_pnl=80 (mean), naked_pnl=0 (mean) →
    score = 80 - 0.5*50 + 0.25*0 = 55.
    """
    per_day = [
        _eval(locked_pnl=80.0, naked_pnl=-50.0),
        _eval(locked_pnl=80.0, naked_pnl=0.0),
        _eval(locked_pnl=80.0, naked_pnl=+50.0),
    ]
    eval_stats = _eval(
        locked_pnl=80.0,  # mean-aggregate, matches per_day
        naked_pnl=0.0,    # mean of [-50, 0, +50]
        per_day=per_day,
    )
    score = _composite_score(
        eval_stats,
        maturation_bonus_weight=0.0,
        composite_score_mode=COMPOSITE_SCORE_MODE_TIGHT_VARIANCE,
    )
    # naked_std (ddof=1 sample std of [-50, 0, +50]) = 50
    # score = 80 - 0.5 × 50 + 0.25 × 0 = 80 - 25 + 0 = 55
    assert score == pytest.approx(55.0)


def test_tight_variance_constants_match_plan_spec() -> None:
    """hard_constraints.md §5 — locks both coefficients."""
    assert TIGHT_VARIANCE_VOL_COEF == 0.5
    assert TIGHT_VARIANCE_NAKED_COEF == 0.25


def test_tight_variance_falls_back_to_locked_weighted_when_n_lt_2() -> None:
    """hard_constraints §15. When ``len(per_day) < 2`` σ is undefined →
    the formula falls back to ``locked_weighted`` (i.e.
    ``locked + 0.25 × naked``) and the result equals the
    locked_weighted score on the same eval_stats.
    """
    # No per_day at all (cohort runner ran with n_eval_days=1, no list).
    eval_stats = _eval(locked_pnl=100.0, naked_pnl=-40.0, per_day=[])
    tight = _composite_score(
        eval_stats,
        maturation_bonus_weight=0.0,
        composite_score_mode=COMPOSITE_SCORE_MODE_TIGHT_VARIANCE,
    )
    locked_weighted = _composite_score(
        eval_stats,
        maturation_bonus_weight=0.0,
        composite_score_mode=COMPOSITE_SCORE_MODE_LOCKED_WEIGHTED,
    )
    assert tight == locked_weighted == pytest.approx(100.0 + 0.25 * -40.0)

    # Single-day per_day list also triggers fallback (need >=2 for σ).
    eval_stats_1day = _eval(
        locked_pnl=100.0, naked_pnl=-40.0,
        per_day=[_eval(locked_pnl=100.0, naked_pnl=-40.0)],
    )
    tight_1day = _composite_score(
        eval_stats_1day,
        maturation_bonus_weight=0.0,
        composite_score_mode=COMPOSITE_SCORE_MODE_TIGHT_VARIANCE,
    )
    assert tight_1day == pytest.approx(100.0 + 0.25 * -40.0)


def test_total_reward_and_locked_weighted_unchanged_by_tight_variance() -> None:
    """hard_constraints §14 — adding the new mode doesn't change the
    semantics of the two existing modes.
    """
    per_day = [
        _eval(locked_pnl=10.0, naked_pnl=-5.0),
        _eval(locked_pnl=10.0, naked_pnl=+5.0),
    ]
    eval_stats = _eval(
        total_reward=42.0, locked_pnl=10.0, naked_pnl=0.0,
        arbs_completed=2, arbs_closed=1,
        per_day=per_day,
    )
    # total_reward (default) — per_day is ignored
    assert _composite_score(eval_stats, maturation_bonus_weight=0.0) == 42.0
    # locked_weighted — per_day is ignored
    assert _composite_score(
        eval_stats,
        maturation_bonus_weight=0.0,
        composite_score_mode=COMPOSITE_SCORE_MODE_LOCKED_WEIGHTED,
    ) == pytest.approx(10.0 + 0.25 * 0.0)
