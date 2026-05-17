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


# ── locked_per_std mode (scalping-tight-naked-variance Phase 3 corrective) ──


def _import_locked_per_std():
    """Lazy import so other tests can still run if the constant moves."""
    from training_v2.cohort.runner import (
        COMPOSITE_SCORE_MODE_LOCKED_PER_STD,
        _composite_score,
    )
    return COMPOSITE_SCORE_MODE_LOCKED_PER_STD, _composite_score


def test_locked_per_std_score_formula() -> None:
    """Per-day naked = [-50, 0, +50] → sample std = 50. locked = 100.
    score = 100 / (1 + 50) ≈ 1.961.
    """
    MODE, score_fn = _import_locked_per_std()
    per_day = [
        _eval(locked_pnl=100.0, naked_pnl=-50.0),
        _eval(locked_pnl=100.0, naked_pnl=0.0),
        _eval(locked_pnl=100.0, naked_pnl=+50.0),
    ]
    eval_stats = _eval(locked_pnl=100.0, naked_pnl=0.0, per_day=per_day)
    score = score_fn(
        eval_stats,
        maturation_bonus_weight=0.0,
        composite_score_mode=MODE,
    )
    assert score == pytest.approx(100.0 / (1.0 + 50.0))


def test_locked_per_std_does_not_read_naked_sign() -> None:
    """Hard property: two agents with identical locked AND identical
    σ_naked but OPPOSITE mean_naked must score IDENTICALLY. This is the
    bit the operator's diagnosis identified — tight_variance reads
    naked-sign via its +0.25*mean_naked term; locked_per_std must not.
    """
    MODE, score_fn = _import_locked_per_std()
    # Agent A: naked = [-100, +100] → mean 0, σ 141.42
    per_day_a = [
        _eval(locked_pnl=80.0, naked_pnl=-100.0),
        _eval(locked_pnl=80.0, naked_pnl=+100.0),
    ]
    eval_a = _eval(locked_pnl=80.0, naked_pnl=0.0, per_day=per_day_a)
    # Agent B: naked = [+50, +250] → mean 150, σ 141.42 (SAME spread)
    per_day_b = [
        _eval(locked_pnl=80.0, naked_pnl=+50.0),
        _eval(locked_pnl=80.0, naked_pnl=+250.0),
    ]
    eval_b = _eval(locked_pnl=80.0, naked_pnl=150.0, per_day=per_day_b)

    score_a = score_fn(eval_a, maturation_bonus_weight=0.0, composite_score_mode=MODE)
    score_b = score_fn(eval_b, maturation_bonus_weight=0.0, composite_score_mode=MODE)
    # Both should score identically because mean_naked is NOT read.
    assert score_a == pytest.approx(score_b)


def test_locked_per_std_falls_back_to_locked_weighted_when_n_lt_2() -> None:
    """Same fallback contract as tight_variance: σ undefined → use
    locked + 0.25 × naked formula."""
    MODE, score_fn = _import_locked_per_std()
    eval_stats = _eval(locked_pnl=100.0, naked_pnl=-40.0, per_day=[])
    score = score_fn(eval_stats, maturation_bonus_weight=0.0, composite_score_mode=MODE)
    assert score == pytest.approx(100.0 + 0.25 * -40.0)


def test_locked_per_std_in_modes_tuple() -> None:
    """Cohort runner CLI accepts the new mode."""
    from training_v2.cohort.runner import (
        COMPOSITE_SCORE_MODE_LOCKED_PER_STD,
        COMPOSITE_SCORE_MODES,
    )
    assert COMPOSITE_SCORE_MODE_LOCKED_PER_STD == "locked_per_std"
    assert COMPOSITE_SCORE_MODE_LOCKED_PER_STD in COMPOSITE_SCORE_MODES


# ── day_pnl_per_std mode (scalping-tight-naked-variance tnv3 corrective) ──


def _import_day_pnl_per_std():
    from training_v2.cohort.runner import (
        COMPOSITE_SCORE_MODE_DAY_PNL_PER_STD,
        _composite_score,
    )
    return COMPOSITE_SCORE_MODE_DAY_PNL_PER_STD, _composite_score


def test_day_pnl_per_std_score_formula() -> None:
    """day_pnl=20, naked = [-50, 0, +50] → σ=50. score = 20 / (1+50) ≈ 0.392."""
    MODE, score_fn = _import_day_pnl_per_std()
    per_day = [
        _eval(locked_pnl=80.0, naked_pnl=-50.0),
        _eval(locked_pnl=80.0, naked_pnl=0.0),
        _eval(locked_pnl=80.0, naked_pnl=+50.0),
    ]
    # day_pnl is set directly on the aggregate; per_day is the σ source only
    eval_stats = _eval(
        total_reward=0.0,
        locked_pnl=80.0,
        naked_pnl=0.0,
        per_day=per_day,
    )
    # NB: _eval doesn't set day_pnl; the aggregate's day_pnl defaults to 0.0
    # because EvalSummary's day_pnl is independently populated by the worker.
    # Construct it explicitly:
    eval_stats = type(eval_stats)(
        eval_day=eval_stats.eval_day,
        total_reward=0.0,
        n_steps=0,
        day_pnl=20.0,
        bet_count=0,
        winning_bets=0,
        bet_precision=0.0,
        pnl_per_bet=0.0,
        early_picks=0,
        profitable=True,
        action_histogram={},
        arbs_completed=0, arbs_naked=0, arbs_closed=0, arbs_force_closed=0,
        arbs_stop_closed=0, arbs_target_pnl_refused=0, pairs_opened=0,
        locked_pnl=80.0, naked_pnl=0.0, closed_pnl=0.0,
        force_closed_pnl=-60.0, stop_closed_pnl=0.0, wall_time_sec=0.0,
        per_day=per_day,
    )
    score = score_fn(eval_stats, maturation_bonus_weight=0.0, composite_score_mode=MODE)
    assert score == pytest.approx(20.0 / (1.0 + 50.0))


def test_day_pnl_per_std_penalises_force_close_cost() -> None:
    """Hard property: an agent with high locked AND high fc-cost (low
    day_pnl) scores WORSE than an agent with lower locked but no fc cost
    (high day_pnl). Same σ.
    """
    MODE, score_fn = _import_day_pnl_per_std()
    EvalSummary = type(_eval())
    per_day_a = [
        _eval(locked_pnl=100.0, naked_pnl=-25.0),
        _eval(locked_pnl=100.0, naked_pnl=+25.0),
    ]
    per_day_b = [
        _eval(locked_pnl=50.0, naked_pnl=-25.0),
        _eval(locked_pnl=50.0, naked_pnl=+25.0),
    ]
    # Agent A: locked 100, fc cost -90, day_pnl = 10
    agent_a = EvalSummary(
        eval_day="2026-04-30", total_reward=0.0, n_steps=0,
        day_pnl=10.0, bet_count=0, winning_bets=0, bet_precision=0.0,
        pnl_per_bet=0.0, early_picks=0, profitable=True, action_histogram={},
        arbs_completed=0, arbs_naked=0, arbs_closed=0, arbs_force_closed=0,
        arbs_stop_closed=0, arbs_target_pnl_refused=0, pairs_opened=0,
        locked_pnl=100.0, naked_pnl=0.0, closed_pnl=0.0,
        force_closed_pnl=-90.0, stop_closed_pnl=0.0, wall_time_sec=0.0,
        per_day=per_day_a,
    )
    # Agent B: locked 50, no fc, day_pnl = 50
    agent_b = EvalSummary(
        eval_day="2026-04-30", total_reward=0.0, n_steps=0,
        day_pnl=50.0, bet_count=0, winning_bets=0, bet_precision=0.0,
        pnl_per_bet=0.0, early_picks=0, profitable=True, action_histogram={},
        arbs_completed=0, arbs_naked=0, arbs_closed=0, arbs_force_closed=0,
        arbs_stop_closed=0, arbs_target_pnl_refused=0, pairs_opened=0,
        locked_pnl=50.0, naked_pnl=0.0, closed_pnl=0.0,
        force_closed_pnl=0.0, stop_closed_pnl=0.0, wall_time_sec=0.0,
        per_day=per_day_b,
    )
    score_a = score_fn(agent_a, maturation_bonus_weight=0.0, composite_score_mode=MODE)
    score_b = score_fn(agent_b, maturation_bonus_weight=0.0, composite_score_mode=MODE)
    # B (no fc cost, higher day_pnl) MUST score higher
    assert score_b > score_a, f"B should beat A: A={score_a}, B={score_b}"


def test_day_pnl_per_std_falls_back_to_locked_weighted_when_n_lt_2() -> None:
    MODE, score_fn = _import_day_pnl_per_std()
    eval_stats = _eval(locked_pnl=100.0, naked_pnl=-40.0, per_day=[])
    score = score_fn(eval_stats, maturation_bonus_weight=0.0, composite_score_mode=MODE)
    assert score == pytest.approx(100.0 + 0.25 * -40.0)


def test_day_pnl_per_std_in_modes_tuple() -> None:
    from training_v2.cohort.runner import (
        COMPOSITE_SCORE_MODE_DAY_PNL_PER_STD,
        COMPOSITE_SCORE_MODES,
    )
    assert COMPOSITE_SCORE_MODE_DAY_PNL_PER_STD == "day_pnl_per_std"
    assert COMPOSITE_SCORE_MODE_DAY_PNL_PER_STD in COMPOSITE_SCORE_MODES
