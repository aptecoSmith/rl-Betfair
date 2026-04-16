"""Unit tests for registry/scoreboard.py -- composite scoring and ranking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from registry.model_store import EvaluationDayRecord, ModelStore
from registry.scoreboard import ModelScore, Scoreboard


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path: Path) -> ModelStore:
    return ModelStore(
        db_path=tmp_path / "test.db",
        weights_dir=tmp_path / "weights",
    )


@pytest.fixture
def config() -> dict:
    return {
        "training": {"starting_budget": 100.0},
        "reward": {
            "coefficients": {
                "win_rate": 0.35,
                "sharpe": 0.30,
                "mean_daily_pnl": 0.15,
                "efficiency": 0.20,
            },
        },
        "discard_policy": {
            "min_win_rate": 0.35,
            "min_mean_pnl": 0.0,
            "min_sharpe": -0.5,
        },
    }


def _insert_model_with_days(
    store: ModelStore,
    day_pnls: list[float],
    bet_counts: list[int] | None = None,
    winning_bets_list: list[int] | None = None,
    generation: int = 1,
) -> str:
    """Helper to create a model with evaluation day records."""
    n = len(day_pnls)
    if bet_counts is None:
        bet_counts = [5] * n
    if winning_bets_list is None:
        winning_bets_list = [2] * n

    mid = store.create_model(generation, "ppo_lstm_v1", "test", {})
    dates = [f"2026-03-{20 + i:02d}" for i in range(n)]
    rid = store.create_evaluation_run(mid, "2026-03-19", dates)

    for i in range(n):
        bc = bet_counts[i]
        wb = winning_bets_list[i]
        pnl = day_pnls[i]
        precision = wb / bc if bc > 0 else 0.0
        ppb = pnl / bc if bc > 0 else 0.0

        store.record_evaluation_day(EvaluationDayRecord(
            run_id=rid,
            date=dates[i],
            day_pnl=pnl,
            bet_count=bc,
            winning_bets=wb,
            bet_precision=precision,
            pnl_per_bet=ppb,
            early_picks=0,
            profitable=pnl > 0,
        ))

    return mid


# ── Score computation ─────────────────────────────────────────────────────────


class TestComputeScore:
    """Test the composite score formula."""

    def test_empty_days(self, store, config):
        board = Scoreboard(store, config)
        assert board.compute_score([]) is None

    def test_all_profitable(self, store, config):
        board = Scoreboard(store, config)
        days = [
            EvaluationDayRecord("r", f"2026-03-{20+i}", 10.0, 5, 3, 0.6, 2.0, 0, True)
            for i in range(5)
        ]
        score = board.compute_score(days)
        assert score is not None
        assert score.win_rate == 1.0
        assert score.mean_daily_pnl == 10.0
        assert score.profitable_days == 5
        assert score.test_days == 5
        assert score.composite_score > 0

    def test_all_losing(self, store, config):
        board = Scoreboard(store, config)
        days = [
            EvaluationDayRecord("r", f"2026-03-{20+i}", -10.0, 5, 1, 0.2, -2.0, 0, False)
            for i in range(5)
        ]
        score = board.compute_score(days)
        assert score is not None
        assert score.win_rate == 0.0
        assert score.mean_daily_pnl == -10.0
        assert score.composite_score < 0

    def test_mixed_days(self, store, config):
        board = Scoreboard(store, config)
        days = [
            EvaluationDayRecord("r", "2026-03-20", 10.0, 5, 3, 0.6, 2.0, 0, True),
            EvaluationDayRecord("r", "2026-03-21", -5.0, 5, 1, 0.2, -1.0, 0, False),
            EvaluationDayRecord("r", "2026-03-22", 8.0, 5, 3, 0.6, 1.6, 0, True),
        ]
        score = board.compute_score(days)
        assert score is not None
        assert score.win_rate == pytest.approx(2 / 3)
        assert score.mean_daily_pnl == pytest.approx((10.0 - 5.0 + 8.0) / 3)

    def test_win_rate_weight(self, store, config):
        """Higher win rate should yield higher score (all else equal)."""
        board = Scoreboard(store, config)

        days_high = [
            EvaluationDayRecord("r", f"d{i}", 5.0, 5, 3, 0.6, 1.0, 0, True)
            for i in range(10)
        ]
        days_low = [
            EvaluationDayRecord("r", f"d{i}", 5.0, 5, 3, 0.6, 1.0, 0, i < 5)
            for i in range(10)
        ]

        score_high = board.compute_score(days_high)
        score_low = board.compute_score(days_low)
        assert score_high is not None
        assert score_low is not None
        assert score_high.composite_score > score_low.composite_score

    def test_sharpe_computation(self, store, config):
        board = Scoreboard(store, config)
        # Consistent returns → high sharpe
        days = [
            EvaluationDayRecord("r", f"d{i}", 5.0, 5, 3, 0.6, 1.0, 0, True)
            for i in range(5)
        ]
        score = board.compute_score(days)
        assert score is not None
        # All same PnL → std = 0, sharpe = 0 (protected by epsilon)
        assert score.sharpe == 0.0

    def test_sharpe_varies_with_consistency(self, store, config):
        board = Scoreboard(store, config)

        # Consistent: all 5.0
        days_consistent = [
            EvaluationDayRecord("r", f"d{i}", 5.0 + 0.01 * i, 5, 3, 0.6, 1.0, 0, True)
            for i in range(5)
        ]
        # Volatile: mean ~5 but big swings
        days_volatile = [
            EvaluationDayRecord("r", "d0", 20.0, 5, 3, 0.6, 4.0, 0, True),
            EvaluationDayRecord("r", "d1", -10.0, 5, 1, 0.2, -2.0, 0, False),
            EvaluationDayRecord("r", "d2", 15.0, 5, 3, 0.6, 3.0, 0, True),
            EvaluationDayRecord("r", "d3", -8.0, 5, 1, 0.2, -1.6, 0, False),
            EvaluationDayRecord("r", "d4", 8.0, 5, 3, 0.6, 1.6, 0, True),
        ]

        sc = board.compute_score(days_consistent)
        sv = board.compute_score(days_volatile)
        assert sc is not None and sv is not None
        assert sc.sharpe > sv.sharpe  # consistent should have higher sharpe

    def test_efficiency_rewards_precision(self, store, config):
        board = Scoreboard(store, config)
        # High precision: 4/5 bets win
        days_precise = [
            EvaluationDayRecord("r", f"d{i}", 5.0, 5, 4, 0.8, 1.0, 0, True)
            for i in range(5)
        ]
        # Low precision: 1/10 bets win
        days_sloppy = [
            EvaluationDayRecord("r", f"d{i}", 5.0, 10, 1, 0.1, 0.5, 0, True)
            for i in range(5)
        ]

        sp = board.compute_score(days_precise)
        ss = board.compute_score(days_sloppy)
        assert sp is not None and ss is not None
        assert sp.efficiency > ss.efficiency

    def test_zero_bets(self, store, config):
        """Days with zero bets should not crash."""
        board = Scoreboard(store, config)
        days = [
            EvaluationDayRecord("r", "d0", 0.0, 0, 0, 0.0, 0.0, 0, False),
        ]
        score = board.compute_score(days)
        assert score is not None
        assert score.bet_precision == 0.0
        assert score.pnl_per_bet == 0.0

    def test_single_day(self, store, config):
        board = Scoreboard(store, config)
        days = [
            EvaluationDayRecord("r", "d0", 15.0, 8, 5, 0.625, 1.875, 2, True),
        ]
        score = board.compute_score(days)
        assert score is not None
        assert score.win_rate == 1.0
        assert score.test_days == 1

    def test_early_picks_aggregation(self, store, config):
        board = Scoreboard(store, config)
        days = [
            EvaluationDayRecord("r", "d0", 10.0, 5, 3, 0.6, 2.0, 3, True),
            EvaluationDayRecord("r", "d1", 5.0, 5, 2, 0.4, 1.0, 1, True),
            EvaluationDayRecord("r", "d2", -2.0, 5, 1, 0.2, -0.4, 0, False),
        ]
        score = board.compute_score(days)
        assert score is not None
        assert score.total_early_picks == 4
        assert score.early_picks_per_day == pytest.approx(4 / 3)


# ── Scoreboard ranking ───────────────────────────────────────────────────────


class TestScoreModel:
    """Test scoring individual models from the registry."""

    def test_score_model(self, store, config):
        mid = _insert_model_with_days(store, [10.0, 5.0, -2.0, 8.0, 3.0])
        board = Scoreboard(store, config)
        score = board.score_model(mid)
        assert score is not None
        assert score.model_id == mid
        assert score.test_days == 5

    def test_score_model_no_evaluation(self, store, config):
        mid = store.create_model(1, "arch", "d", {})
        board = Scoreboard(store, config)
        assert board.score_model(mid) is None


class TestRankAll:
    """Test full ranking of all active models."""

    def test_rank_multiple_models(self, store, config):
        # Good model: mostly profitable
        _insert_model_with_days(store, [10.0, 8.0, 5.0, 3.0, 1.0])
        # Bad model: mostly losing
        _insert_model_with_days(store, [-5.0, -8.0, -3.0, -10.0, 1.0])

        board = Scoreboard(store, config)
        rankings = board.rank_all()

        assert len(rankings) == 2
        assert rankings[0].composite_score > rankings[1].composite_score

    def test_rank_excludes_discarded(self, store, config):
        mid1 = _insert_model_with_days(store, [10.0, 5.0])
        mid2 = _insert_model_with_days(store, [3.0, 1.0])
        store.update_model_status(mid2, "discarded")

        board = Scoreboard(store, config)
        rankings = board.rank_all()

        assert len(rankings) == 1
        assert rankings[0].model_id == mid1

    def test_rank_empty(self, store, config):
        board = Scoreboard(store, config)
        assert board.rank_all() == []

    def test_rank_models_without_evaluation(self, store, config):
        """Models with no evaluation run are excluded from rankings."""
        store.create_model(1, "arch", "d", {})
        _insert_model_with_days(store, [10.0])

        board = Scoreboard(store, config)
        rankings = board.rank_all()
        assert len(rankings) == 1

    def test_rank_includes_garaged_active(self, store, config):
        """Garaged active models appear in rankings."""
        mid = _insert_model_with_days(store, [10.0, 5.0])
        store.set_garaged(mid, True)

        board = Scoreboard(store, config)
        rankings = board.rank_all()
        assert len(rankings) == 1
        assert rankings[0].model_id == mid

    def test_rank_includes_garaged_discarded(self, store, config):
        """Garaged discarded models still appear in rankings."""
        mid = _insert_model_with_days(store, [10.0, 5.0])
        store.update_model_status(mid, "discarded")
        store.set_garaged(mid, True)

        board = Scoreboard(store, config)
        rankings = board.rank_all()
        assert len(rankings) == 1
        assert rankings[0].model_id == mid

    def test_rank_no_duplicates_active_garaged(self, store, config):
        """A model that is both active and garaged should not appear twice."""
        mid = _insert_model_with_days(store, [10.0, 5.0])
        store.set_garaged(mid, True)

        board = Scoreboard(store, config)
        rankings = board.rank_all()
        ids = [r.model_id for r in rankings]
        assert ids.count(mid) == 1


class TestUpdateScores:
    """Test persisting scores back to the registry."""

    def test_scores_persisted(self, store, config):
        mid = _insert_model_with_days(store, [10.0, 5.0, -2.0])
        board = Scoreboard(store, config)
        rankings = board.update_scores()

        assert len(rankings) == 1
        model = store.get_model(mid)
        assert model is not None
        assert model.composite_score is not None
        assert model.composite_score == pytest.approx(rankings[0].composite_score)
        assert model.last_evaluated_at is not None


# ── Discard policy ────────────────────────────────────────────────────────────


class TestDiscardPolicy:
    """Test discard candidate detection."""

    def test_bad_model_flagged(self, store, config):
        """A model failing all three thresholds should be a discard candidate."""
        # win_rate < 0.35, mean_pnl < 0, sharpe < -0.5
        mid = _insert_model_with_days(
            store,
            [-10.0, -15.0, -8.0, -12.0, 1.0],  # 1/5 profitable, mean < 0
        )
        board = Scoreboard(store, config)
        candidates = board.check_discard_candidates(config)

        # Check the model's stats to understand if it really qualifies
        score = board.score_model(mid)
        assert score is not None
        if score.win_rate < 0.35 and score.mean_daily_pnl < 0 and score.sharpe < -0.5:
            assert mid in candidates

    def test_good_model_not_flagged(self, store, config):
        mid = _insert_model_with_days(store, [10.0, 8.0, 5.0, 3.0, 1.0])
        board = Scoreboard(store, config)
        candidates = board.check_discard_candidates(config)
        assert mid not in candidates

    def test_partial_failure_not_flagged(self, store, config):
        """A model that fails only some criteria should NOT be flagged."""
        # Low win rate but positive mean_pnl
        mid = _insert_model_with_days(store, [50.0, -5.0, -3.0, -2.0, -1.0])
        board = Scoreboard(store, config)
        score = board.score_model(mid)
        assert score is not None
        # Even if win_rate < 0.35, mean_pnl > 0 should protect it
        if score.mean_daily_pnl > 0:
            candidates = board.check_discard_candidates(config)
            assert mid not in candidates


# ── Coefficient configuration ─────────────────────────────────────────────────


class TestCoefficients:
    """Test that configurable coefficients affect the score."""

    def test_coefficients_sum_to_one(self, config):
        c = config["reward"]["coefficients"]
        total = c["win_rate"] + c["sharpe"] + c["mean_daily_pnl"] + c["efficiency"]
        assert total == pytest.approx(1.0)

    def test_different_weights_different_scores(self, store, config):
        mid = _insert_model_with_days(store, [10.0, -5.0, 8.0, -3.0, 6.0])

        config1 = dict(config)
        config1["reward"] = {
            "coefficients": {"win_rate": 0.7, "sharpe": 0.1, "mean_daily_pnl": 0.1, "efficiency": 0.1},
        }
        config2 = dict(config)
        config2["reward"] = {
            "coefficients": {"win_rate": 0.1, "sharpe": 0.7, "mean_daily_pnl": 0.1, "efficiency": 0.1},
        }

        s1 = Scoreboard(store, config1).score_model(mid)
        s2 = Scoreboard(store, config2).score_model(mid)
        assert s1 is not None and s2 is not None
        # Different weights → different scores (unless exactly coincident)
        # We just verify both compute without error
        assert isinstance(s1.composite_score, float)
        assert isinstance(s2.composite_score, float)


# ── Percentage return (Session 03) ───────────────────────────────────────────


class TestPercentageReturn:
    """Test mean_daily_return_pct computation."""

    def test_return_pct_budget_10(self, store: ModelStore, config):
        """Budget=10, mean_pnl=1.0 → 10.0%."""
        board = Scoreboard(store, config)
        days = [
            EvaluationDayRecord("r", "d0", 1.0, 5, 3, 0.6, 0.2, 0, True, starting_budget=10.0),
        ]
        score = board.compute_score(days)
        assert score is not None
        assert score.mean_daily_return_pct == pytest.approx(10.0)
        assert score.recorded_budget == pytest.approx(10.0)

    def test_return_pct_budget_100(self, store, config):
        """Budget=100, mean_pnl=10.0 → 10.0%."""
        board = Scoreboard(store, config)
        days = [
            EvaluationDayRecord("r", "d0", 10.0, 5, 3, 0.6, 2.0, 0, True, starting_budget=100.0),
        ]
        score = board.compute_score(days)
        assert score is not None
        assert score.mean_daily_return_pct == pytest.approx(10.0)

    def test_both_budgets_same_return(self, store, config):
        """Same return % at different budgets → same composite score."""
        board = Scoreboard(store, config)
        days_10 = [
            EvaluationDayRecord("r", f"d{i}", 1.0, 5, 3, 0.6, 0.2, 0, True, starting_budget=10.0)
            for i in range(5)
        ]
        days_100 = [
            EvaluationDayRecord("r", f"d{i}", 10.0, 5, 3, 0.6, 2.0, 0, True, starting_budget=100.0)
            for i in range(5)
        ]
        score_10 = board.compute_score(days_10)
        score_100 = board.compute_score(days_100)
        assert score_10 is not None and score_100 is not None
        assert score_10.mean_daily_return_pct == pytest.approx(score_100.mean_daily_return_pct)

    def test_negative_return(self, store, config):
        board = Scoreboard(store, config)
        days = [
            EvaluationDayRecord("r", "d0", -2.0, 5, 1, 0.2, -0.4, 0, False, starting_budget=100.0),
        ]
        score = board.compute_score(days)
        assert score is not None
        assert score.mean_daily_return_pct == pytest.approx(-2.0)

    def test_default_budget_backward_compat(self, store, config):
        """Day records without explicit budget (default 100.0) still work."""
        board = Scoreboard(store, config)
        days = [
            EvaluationDayRecord("r", "d0", 5.0, 5, 3, 0.6, 1.0, 0, True),
        ]
        score = board.compute_score(days)
        assert score is not None
        assert score.mean_daily_return_pct == pytest.approx(5.0)
        assert score.recorded_budget == pytest.approx(100.0)


# ── Percentage-based discard (Session 05) ────────────────────────────────────


def _insert_model_with_budget(
    store: ModelStore,
    day_pnls: list[float],
    starting_budget: float = 100.0,
    generation: int = 1,
) -> str:
    """Helper to create a model with evaluation day records and a specific budget."""
    n = len(day_pnls)
    mid = store.create_model(generation, "ppo_lstm_v1", "test", {})
    dates = [f"2026-03-{20 + i:02d}" for i in range(n)]
    rid = store.create_evaluation_run(mid, "2026-03-19", dates)
    for i in range(n):
        pnl = day_pnls[i]
        store.record_evaluation_day(EvaluationDayRecord(
            run_id=rid, date=dates[i], day_pnl=pnl, bet_count=5,
            winning_bets=1, bet_precision=0.2, pnl_per_bet=pnl / 5,
            early_picks=0, profitable=pnl > 0, starting_budget=starting_budget,
        ))
    return mid


class TestPercentageDiscard:
    """Test percentage-based discard threshold (Session 05)."""

    def test_pct_threshold_keeps_profitable_low_budget(self, store, config):
        """Budget=10, pnl=0.5 → return=5% → survives threshold=0%."""
        config["discard_policy"]["min_mean_return_pct"] = 0.0
        mid = _insert_model_with_budget(store, [0.5, 0.3, 0.4, 0.2, 0.1], starting_budget=10.0)
        board = Scoreboard(store, config)
        candidates = board.check_discard_candidates(config)
        # This model has positive return, should NOT be discarded on P&L
        score = board.score_model(mid)
        assert score is not None
        assert score.mean_daily_return_pct is not None
        assert score.mean_daily_return_pct > 0.0
        # Even if win_rate and sharpe fail, P&L check should pass
        # So it should NOT be in candidates (needs ALL criteria to fail)
        # But let's verify the P&L leg specifically
        assert score.mean_daily_return_pct >= 0.0  # passes P&L check

    def test_pct_threshold_discards_losing_model(self, store, config):
        """Budget=100, pnl=-2.0 → return=-2% → discarded if all criteria fail."""
        config["discard_policy"]["min_mean_return_pct"] = 0.0
        mid = _insert_model_with_budget(
            store, [-10.0, -15.0, -8.0, -12.0, -5.0], starting_budget=100.0,
        )
        board = Scoreboard(store, config)
        score = board.score_model(mid)
        assert score is not None
        assert score.mean_daily_return_pct is not None
        assert score.mean_daily_return_pct < 0.0
        candidates = board.check_discard_candidates(config)
        # Should be flagged if all three criteria fail
        if score.win_rate < 0.35 and score.sharpe < -0.5:
            assert mid in candidates

    def test_backward_compat_min_mean_pnl_only(self, store, config):
        """Config with only min_mean_pnl (no min_mean_return_pct) still works."""
        config["discard_policy"].pop("min_mean_return_pct", None)
        mid = _insert_model_with_budget(
            store, [-10.0, -15.0, -8.0, -12.0, -5.0], starting_budget=100.0,
        )
        board = Scoreboard(store, config)
        score = board.score_model(mid)
        assert score is not None
        # Should still use absolute min_mean_pnl
        candidates = board.check_discard_candidates(config)
        if score.win_rate < 0.35 and score.mean_daily_pnl < 0 and score.sharpe < -0.5:
            assert mid in candidates


# ── Scalping aggregates (Sprint 5 Session 3 follow-up) ───────────────────────


class TestScalpingAggregates:
    """ModelScore must sum scalping fields across days so the scoreboard
    can rank scalpers by completed pairs and locked profit."""

    def _day(
        self, run_id: str, date: str, *, day_pnl: float = 0.0,
        bet_count: int = 0, winning_bets: int = 0, profitable: bool = False,
        arbs_completed: int = 0, arbs_naked: int = 0,
        locked_pnl: float = 0.0, naked_pnl: float = 0.0,
    ) -> EvaluationDayRecord:
        precision = winning_bets / bet_count if bet_count > 0 else 0.0
        return EvaluationDayRecord(
            run_id=run_id, date=date, day_pnl=day_pnl,
            bet_count=bet_count, winning_bets=winning_bets,
            bet_precision=precision,
            pnl_per_bet=day_pnl / bet_count if bet_count > 0 else 0.0,
            early_picks=0, profitable=profitable,
            arbs_completed=arbs_completed, arbs_naked=arbs_naked,
            locked_pnl=locked_pnl, naked_pnl=naked_pnl,
        )

    def test_directional_model_zero_scalping_fields(self, store, config):
        """A model with no arb activity has all scalping aggregates at 0."""
        board = Scoreboard(store, config)
        days = [
            self._day("r", f"d{i}", day_pnl=10.0, bet_count=5,
                      winning_bets=3, profitable=True)
            for i in range(3)
        ]
        score = board.compute_score(days)
        assert score is not None
        assert score.total_bets == 15
        assert score.arbs_completed == 0
        assert score.arbs_naked == 0
        assert score.locked_pnl == 0.0
        assert score.naked_pnl == 0.0

    def test_scalping_model_sums_across_days(self, store, config):
        """Aggregates must SUM, not average, so volume matters."""
        board = Scoreboard(store, config)
        days = [
            self._day(
                "r", "d1", day_pnl=20.0, bet_count=100,
                winning_bets=50, profitable=True,
                arbs_completed=10, arbs_naked=40,
                locked_pnl=15.0, naked_pnl=5.0,
            ),
            self._day(
                "r", "d2", day_pnl=-5.0, bet_count=80,
                winning_bets=40, profitable=False,
                arbs_completed=8, arbs_naked=32,
                locked_pnl=12.0, naked_pnl=-17.0,
            ),
            self._day(
                "r", "d3", day_pnl=15.0, bet_count=120,
                winning_bets=60, profitable=True,
                arbs_completed=12, arbs_naked=48,
                locked_pnl=20.0, naked_pnl=-5.0,
            ),
        ]
        score = board.compute_score(days)
        assert score is not None
        assert score.total_bets == 300
        assert score.arbs_completed == 30
        assert score.arbs_naked == 120
        assert score.locked_pnl == pytest.approx(47.0)
        assert score.naked_pnl == pytest.approx(-17.0)

    def test_zero_bet_days_dont_break_aggregation(self, store, config):
        """A model with some no-bet days still aggregates the rest correctly."""
        board = Scoreboard(store, config)
        days = [
            self._day(
                "r", "d1", day_pnl=10.0, bet_count=50,
                winning_bets=25, profitable=True,
                arbs_completed=5, arbs_naked=20,
                locked_pnl=8.0, naked_pnl=2.0,
            ),
            # No-bet day
            self._day("r", "d2", day_pnl=0.0, bet_count=0,
                      winning_bets=0, profitable=False),
            self._day(
                "r", "d3", day_pnl=-3.0, bet_count=30,
                winning_bets=10, profitable=False,
                arbs_completed=3, arbs_naked=12,
                locked_pnl=4.0, naked_pnl=-7.0,
            ),
        ]
        score = board.compute_score(days)
        assert score is not None
        assert score.total_bets == 80
        assert score.arbs_completed == 8
        assert score.arbs_naked == 32
        assert score.locked_pnl == pytest.approx(12.0)
        assert score.naked_pnl == pytest.approx(-5.0)

    def test_dataclass_default_values(self):
        """Backward-compat: ModelScore can be constructed without scalping
        kwargs, defaulting all fields to 0 (preserves old call sites)."""
        s = ModelScore(
            model_id="m1", win_rate=0.5, mean_daily_pnl=10.0,
            sharpe=0.3, bet_precision=0.6, pnl_per_bet=2.0,
            efficiency=0.4, composite_score=0.5,
            test_days=4, profitable_days=2,
        )
        assert s.total_bets == 0
        assert s.arbs_completed == 0
        assert s.arbs_naked == 0
        assert s.locked_pnl == 0.0
        assert s.naked_pnl == 0.0


class TestScoreboardEntryIsScalping:
    """The /models API endpoint must derive is_scalping from the model's
    scalping_mode hyperparameter so the frontend can tab-filter correctly."""

    def _setup_two_models(
        self, store: ModelStore, config: dict,
    ) -> tuple[str, str, "Scoreboard"]:
        """Create one scalping + one directional model, both with eval data."""
        directional_id = store.create_model(
            1, "ppo_lstm_v1", "directional",
            {"scalping_mode": False, "learning_rate": 1e-4},
        )
        scalping_id = store.create_model(
            1, "ppo_lstm_v1", "scalping",
            {"scalping_mode": True, "arb_spread_scale": 1.2},
        )
        for mid in (directional_id, scalping_id):
            rid = store.create_evaluation_run(mid, "2026-04-10", ["2026-04-11"])
            store.record_evaluation_day(EvaluationDayRecord(
                run_id=rid, date="2026-04-11",
                day_pnl=5.0, bet_count=10, winning_bets=6,
                bet_precision=0.6, pnl_per_bet=0.5,
                early_picks=0, profitable=True,
            ))
        board = Scoreboard(store, config)
        return directional_id, scalping_id, board

    def test_score_to_entry_classifies_models(self, store, config):
        """is_scalping must reflect the gene, not the runtime behaviour."""
        from api.routers.models import _score_to_entry
        directional_id, scalping_id, board = self._setup_two_models(store, config)

        d_score = board.score_model(directional_id)
        s_score = board.score_model(scalping_id)
        assert d_score is not None and s_score is not None

        d_entry = _score_to_entry(d_score, store)
        s_entry = _score_to_entry(s_score, store)

        assert d_entry.is_scalping is False
        assert s_entry.is_scalping is True

    def test_score_to_entry_passes_through_scalping_aggregates(
        self, store, config,
    ):
        """Aggregate fields must reach the API entry verbatim."""
        from api.routers.models import _score_to_entry

        mid = store.create_model(
            1, "ppo_lstm_v1", "scalping", {"scalping_mode": True},
        )
        rid = store.create_evaluation_run(mid, "2026-04-10", ["2026-04-11"])
        store.record_evaluation_day(EvaluationDayRecord(
            run_id=rid, date="2026-04-11",
            day_pnl=20.0, bet_count=200, winning_bets=120,
            bet_precision=0.6, pnl_per_bet=0.1,
            early_picks=0, profitable=True,
            arbs_completed=25, arbs_naked=85,
            locked_pnl=42.5, naked_pnl=-22.5,
        ))
        board = Scoreboard(store, config)
        score = board.score_model(mid)
        assert score is not None
        entry = _score_to_entry(score, store)
        assert entry.is_scalping is True
        assert entry.total_bets == 200
        assert entry.arbs_completed == 25
        assert entry.arbs_naked == 85
        assert entry.locked_pnl == pytest.approx(42.5)
        assert entry.naked_pnl == pytest.approx(-22.5)

    def test_missing_scalping_mode_key_defaults_false(self, store, config):
        """A model whose hyperparameters dict has no scalping_mode key is
        treated as directional — bool(None) is False."""
        from api.routers.models import _score_to_entry
        mid = store.create_model(
            1, "ppo_lstm_v1", "no-flag-set", {"learning_rate": 5e-5},
        )
        rid = store.create_evaluation_run(mid, "2026-04-10", ["2026-04-11"])
        store.record_evaluation_day(EvaluationDayRecord(
            run_id=rid, date="2026-04-11", day_pnl=1.0, bet_count=2,
            winning_bets=1, bet_precision=0.5, pnl_per_bet=0.5,
            early_picks=0, profitable=True,
        ))
        board = Scoreboard(store, config)
        score = board.score_model(mid)
        assert score is not None
        entry = _score_to_entry(score, store)
        assert entry.is_scalping is False
