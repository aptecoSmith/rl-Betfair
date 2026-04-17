"""Tests for the shared MACE helper and its scoreboard integration.

Scalping-active-management §06. Covers:

1. ``compute_mace`` math on contrived inputs (empty, sparse,
   perfect, ``min_bucket_size`` override).
2. ``ModelScore.mean_absolute_calibration_error`` is populated
   for scalping runs, None for directional runs, None when the
   parquet read raises.
3. **Ranking invariant** (critical, hard_constraints §14): a
   synthetic set of models whose MACE values would reorder them
   IF MACE fed composite sorts identically with and without the
   MACE field populated. If this test flips, MACE has leaked into
   composite math — revert.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from registry.calibration import (
    MIN_BUCKET_SIZE,
    MIN_BUCKETS_FOR_MACE,
    compute_bucket_outcomes,
    compute_mace,
)
from registry.model_store import (
    EvaluationBetRecord,
    EvaluationDayRecord,
    ModelStore,
)
from registry.scoreboard import Scoreboard


# -- Fixtures & helpers ---------------------------------------------


def _make_store(tmp_dir: str) -> ModelStore:
    return ModelStore(
        db_path=str(Path(tmp_dir) / "test.db"),
        weights_dir=str(Path(tmp_dir) / "weights"),
        bet_logs_dir=str(Path(tmp_dir) / "bet_logs"),
    )


def _pair_records(
    pair_id: str,
    run_id: str,
    date: str,
    fill_prob: float,
    completed: bool,
    aggressive_tick: str,
    passive_tick: str,
) -> list[EvaluationBetRecord]:
    def _rec(side: str, price: float, ts: str) -> EvaluationBetRecord:
        return EvaluationBetRecord(
            run_id=run_id,
            date=date,
            market_id="1.100",
            tick_timestamp=ts,
            seconds_to_off=300.0,
            runner_id=101,
            runner_name="Runner",
            action=side,
            price=price,
            stake=10.0,
            matched_size=10.0,
            outcome="won",
            pnl=0.0,
            pair_id=pair_id,
            fill_prob_at_placement=fill_prob,
        )
    recs = [_rec("back", 5.0, aggressive_tick)]
    if completed:
        recs.append(_rec("lay", 4.0, passive_tick))
    return recs


def _bucket_batch(
    run_id: str, count: int, fill_prob: float,
    completed_count: int, tag: str,
) -> list[EvaluationBetRecord]:
    """``count`` pairs in the bucket for ``fill_prob``, ``completed_count``
    of which are completed. Timestamps are unique per pair."""
    out: list[EvaluationBetRecord] = []
    for i in range(count):
        out.extend(_pair_records(
            pair_id=f"{tag}-{i}",
            run_id=run_id,
            date="2026-04-10",
            fill_prob=fill_prob,
            completed=(i < completed_count),
            aggressive_tick=f"2026-04-10T10:{i:02d}:00",
            passive_tick=f"2026-04-10T10:{i:02d}:05",
        ))
    return out


def _scoreboard_config() -> dict:
    return {
        "reward": {"coefficients": {
            "win_rate": 0.35, "sharpe": 0.30,
            "mean_daily_pnl": 0.15, "efficiency": 0.20,
        }},
        "training": {"starting_budget": 100.0},
    }


def _seed_scalping_model_with_bets(
    store: ModelStore,
    bets: list[EvaluationBetRecord],
    *,
    arbs_completed: int,
    arbs_naked: int,
    model_kwargs: dict | None = None,
) -> str:
    model_id = store.create_model(
        generation=0,
        architecture_name="ppo_lstm_v1",
        architecture_description="Scalping MACE test",
        hyperparameters=(model_kwargs or {"scalping_mode": True}),
    )
    run_id = store.create_evaluation_run(
        model_id=model_id,
        train_cutoff_date="2026-04-09",
        test_days=["2026-04-10"],
    )
    records = [
        EvaluationBetRecord(**{**b.__dict__, "run_id": run_id})
        for b in bets
    ]
    store.record_evaluation_day(EvaluationDayRecord(
        run_id=run_id, date="2026-04-10", day_pnl=0.0,
        bet_count=len(records), winning_bets=0,
        bet_precision=0.0, pnl_per_bet=0.0, early_picks=0,
        profitable=False,
        arbs_completed=arbs_completed, arbs_naked=arbs_naked,
    ))
    store.write_bet_logs_parquet(run_id, "2026-04-10", records)
    return model_id


# -- compute_mace -----------------------------------------------------


class TestComputeMace:
    def test_empty_input_returns_none(self):
        """No bets → no buckets clear threshold → MACE is None."""
        assert compute_mace([]) is None

    def test_all_buckets_sparse_returns_none(self):
        """Every bucket has < MIN_BUCKET_SIZE pairs — not enough
        dense buckets, MACE None."""
        recs: list[EvaluationBetRecord] = []
        for fp, tag in (
            (0.10, "b0"), (0.40, "b1"),
            (0.60, "b2"), (0.90, "b3"),
        ):
            recs.extend(_bucket_batch(
                run_id="R1", count=10, fill_prob=fp,
                completed_count=0, tag=tag,
            ))
        assert compute_mace(recs) is None

    def test_perfect_predictions_yield_zero_mace(self):
        """Every bucket midpoint × count is an exact completed count,
        observed_rate == midpoint, so per-bucket error = 0 → MACE=0."""
        recs: list[EvaluationBetRecord] = []
        for fp, midpoint, tag in (
            (0.10, 0.125, "b0"), (0.40, 0.375, "b1"),
            (0.60, 0.625, "b2"), (0.90, 0.875, "b3"),
        ):
            recs.extend(_bucket_batch(
                run_id="R1", count=40, fill_prob=fp,
                completed_count=int(midpoint * 40), tag=tag,
            ))
        mace = compute_mace(recs)
        assert mace == pytest.approx(0.0, abs=1e-9)

    def test_min_bucket_size_override_changes_result(self):
        """Override ``min_bucket_size`` to 5; sparse buckets that were
        previously excluded now clear the threshold and MACE differs
        from the default-threshold result."""
        recs: list[EvaluationBetRecord] = []
        # Two sparse buckets with 5 pairs each, observed rates
        # deliberately off from the midpoints so they contribute a
        # non-zero error once included.
        recs.extend(_bucket_batch(
            run_id="R1", count=5, fill_prob=0.10,
            completed_count=0, tag="b0",
        ))
        recs.extend(_bucket_batch(
            run_id="R1", count=5, fill_prob=0.90,
            completed_count=5, tag="b3",
        ))
        # Default threshold: fewer than 2 buckets clear 20 → None.
        assert compute_mace(recs) is None
        # Overridden threshold: both buckets clear 5 → MACE = mean of
        # |0.125 - 0| and |0.875 - 1| = (0.125 + 0.125) / 2 = 0.125.
        mace = compute_mace(recs, min_bucket_size=5)
        assert mace == pytest.approx(0.125, abs=1e-9)

    def test_module_constants_match_prompt_contract(self):
        """MIN_BUCKET_SIZE = 20 and MIN_BUCKETS_FOR_MACE = 2 are the
        exported greppable knobs — pin the values so a silent tweak
        here can't silently change the scoreboard's MACE column."""
        assert MIN_BUCKET_SIZE == 20
        assert MIN_BUCKETS_FOR_MACE == 2

    def test_bucket_outcomes_shape(self):
        """``compute_bucket_outcomes`` always returns 4 rows (one per
        bucket), even when every bucket is empty."""
        assert len(compute_bucket_outcomes([])) == 4


# -- ModelScore MACE field ------------------------------------------


class TestModelScoreMaceField:
    def test_populated_for_scalping_run_with_sufficient_bets(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            # Perfect predictions across four buckets → MACE = 0.0.
            recs: list[EvaluationBetRecord] = []
            for fp, midpoint, tag in (
                (0.10, 0.125, "b0"), (0.40, 0.375, "b1"),
                (0.60, 0.625, "b2"), (0.90, 0.875, "b3"),
            ):
                recs.extend(_bucket_batch(
                    run_id="pending", count=40, fill_prob=fp,
                    completed_count=int(midpoint * 40), tag=tag,
                ))
            completed = sum(
                1 for b in recs if b.action == "lay"
            )  # one lay per completed pair
            total_pairs = len({b.pair_id for b in recs})
            model_id = _seed_scalping_model_with_bets(
                store, recs,
                arbs_completed=completed,
                arbs_naked=total_pairs - completed,
            )
            sb = Scoreboard(store, _scoreboard_config())
            score = sb.score_model(model_id)
            assert score is not None
            assert score.mean_absolute_calibration_error is not None
            assert score.mean_absolute_calibration_error == pytest.approx(
                0.0, abs=1e-9,
            )

    def test_none_for_directional_run(self):
        """Directional model has arbs_completed+arbs_naked=0; the MACE
        computation is short-circuited without touching the parquet
        layer at all."""
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            model_id = store.create_model(
                generation=0,
                architecture_name="ppo_lstm_v1",
                architecture_description="Directional",
                hyperparameters={},
            )
            run_id = store.create_evaluation_run(
                model_id=model_id,
                train_cutoff_date="2026-04-09",
                test_days=["2026-04-10"],
            )
            store.record_evaluation_day(EvaluationDayRecord(
                run_id=run_id, date="2026-04-10", day_pnl=1.0,
                bet_count=3, winning_bets=2,
                bet_precision=0.67, pnl_per_bet=0.33, early_picks=0,
                profitable=True,
                arbs_completed=0, arbs_naked=0,
            ))
            sb = Scoreboard(store, _scoreboard_config())
            score = sb.score_model(model_id)
            assert score is not None
            assert score.mean_absolute_calibration_error is None

    def test_none_when_get_evaluation_bets_raises(self):
        """If ``get_evaluation_bets`` blows up (missing parquet, bad
        file), the scoreboard must not crash — the MACE field is
        left as None and scoring continues."""
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            model_id = store.create_model(
                generation=0,
                architecture_name="ppo_lstm_v1",
                architecture_description="Scalping",
                hyperparameters={"scalping_mode": True},
            )
            run_id = store.create_evaluation_run(
                model_id=model_id,
                train_cutoff_date="2026-04-09",
                test_days=["2026-04-10"],
            )
            # Say the run placed arbs but the parquet is unreadable.
            store.record_evaluation_day(EvaluationDayRecord(
                run_id=run_id, date="2026-04-10", day_pnl=0.0,
                bet_count=4, winning_bets=0,
                bet_precision=0.0, pnl_per_bet=0.0, early_picks=0,
                profitable=False,
                arbs_completed=2, arbs_naked=0,
            ))
            sb = Scoreboard(store, _scoreboard_config())
            with patch.object(
                store, "get_evaluation_bets",
                side_effect=FileNotFoundError("parquet gone"),
            ):
                score = sb.score_model(model_id)
            assert score is not None
            assert score.mean_absolute_calibration_error is None


# -- Ranking invariant (critical — hard_constraints §14) ------------


class TestRankingInvariant:
    """MACE is a diagnostic column; it must NOT feed composite score
    or the scoreboard's default ranking. These tests are the tripwire
    that catches any regression that wires MACE into the ranking path.
    """

    def test_rank_all_ordering_unchanged_with_and_without_mace(self):
        """Build the SAME set of models twice — once without bets
        (MACE=None on every row), once with bets producing
        deliberately-perverse MACE values where the worst-MACE model
        has the highest composite. ``rank_all`` must return the same
        composite-score ordering in both cases — the MACE field is
        invisible to the ranking key.

        We compare composite scores rather than model IDs because
        models are created in fresh stores with fresh UUIDs; the
        invariant we care about is "ordering" and composite is the
        stable identity of a row.
        """
        expected_composites = self._ordering_with_no_mace_data()
        observed_composites = (
            self._ordering_with_deliberately_perverse_mace()
        )
        assert observed_composites == expected_composites
        # Sanity: the fixture is constructed so composites are
        # strictly decreasing; the invariant is this decreasing order
        # holds whether MACE is populated or not.
        assert observed_composites == sorted(
            observed_composites, reverse=True,
        )

    @staticmethod
    def _perfect_bucket_recs(
        run_id: str, observed_fraction: float,
    ) -> list[EvaluationBetRecord]:
        """Four dense buckets whose observed rate equals ``observed_fraction``
        in every bucket. That yields a predictable MACE = mean of
        |midpoint − observed_fraction|. The operator overrides
        observed_fraction to control which model ends up with which
        MACE."""
        recs: list[EvaluationBetRecord] = []
        for fp, tag in (
            (0.10, "b0"), (0.40, "b1"),
            (0.60, "b2"), (0.90, "b3"),
        ):
            recs.extend(_bucket_batch(
                run_id=run_id, count=40, fill_prob=fp,
                completed_count=int(observed_fraction * 40), tag=tag,
            ))
        return recs

    def _ordering_with_no_mace_data(self) -> list[float]:
        """Same three scalping models, same composite scores, but NO
        bet parquet for any of them — MACE field is None throughout."""
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            sb = Scoreboard(store, _scoreboard_config())
            for suffix, composite in (
                ("a", 0.9), ("b", 0.5), ("c", 0.2),
            ):
                model_id = store.create_model(
                    generation=0,
                    architecture_name="ppo_lstm_v1",
                    architecture_description="Scalper",
                    hyperparameters={"scalping_mode": True},
                )
                # Rename so we can identify by suffix.
                store.update_composite_score(model_id, composite)
                run_id = store.create_evaluation_run(
                    model_id=model_id,
                    train_cutoff_date="2026-04-09",
                    test_days=["2026-04-10"],
                )
                # One day record with pnl matching composite so that
                # ``compute_score``'s composite is stable across runs.
                store.record_evaluation_day(EvaluationDayRecord(
                    run_id=run_id, date="2026-04-10",
                    day_pnl=composite * 10,
                    bet_count=10, winning_bets=int(composite * 10),
                    bet_precision=composite, pnl_per_bet=composite,
                    early_picks=0, profitable=composite > 0,
                    arbs_completed=0, arbs_naked=0,
                ))
            return [
                round(s.composite_score, 9) for s in sb.rank_all()
            ]

    def _ordering_with_deliberately_perverse_mace(self) -> list[float]:
        """Same three scalping models + composites, but this time each
        model HAS bet-log parquet wired up such that the top-composite
        model has the WORST MACE and the bottom-composite model has
        the BEST MACE. If ``rank_all`` lets MACE influence ordering,
        the returned order flips. It must not flip."""
        with tempfile.TemporaryDirectory() as tmp:
            store = _make_store(tmp)
            sb = Scoreboard(store, _scoreboard_config())
            # Tie composite to a predictable day-pnl, then attach
            # increasingly-well-calibrated buckets to lower-composite
            # models. Observed_fraction == midpoint → MACE = 0 (best).
            # Observed_fraction = 0.0 → MACE = mean(|m − 0|) across
            # the four midpoints = 0.5 (worst).
            setups = [
                ("best_composite_worst_mace", 0.9, 0.0),   # MACE=0.5
                ("mid",                         0.5, 0.5),   # MACE=0.25
                ("worst_composite_best_mace",   0.2, None),  # MACE=0
            ]
            for _label, composite, observed in setups:
                model_id = store.create_model(
                    generation=0,
                    architecture_name="ppo_lstm_v1",
                    architecture_description="Scalper",
                    hyperparameters={"scalping_mode": True},
                )
                run_id = store.create_evaluation_run(
                    model_id=model_id,
                    train_cutoff_date="2026-04-09",
                    test_days=["2026-04-10"],
                )
                store.record_evaluation_day(EvaluationDayRecord(
                    run_id=run_id, date="2026-04-10",
                    day_pnl=composite * 10,
                    bet_count=10, winning_bets=int(composite * 10),
                    bet_precision=composite, pnl_per_bet=composite,
                    early_picks=0, profitable=composite > 0,
                    arbs_completed=80, arbs_naked=80,
                ))
                # ``observed=None`` → use midpoint per bucket so
                # MACE = 0 (the cleanest best-case).
                if observed is None:
                    recs: list[EvaluationBetRecord] = []
                    for fp, midpoint, tag in (
                        (0.10, 0.125, "b0"), (0.40, 0.375, "b1"),
                        (0.60, 0.625, "b2"), (0.90, 0.875, "b3"),
                    ):
                        recs.extend(_bucket_batch(
                            run_id=run_id, count=40, fill_prob=fp,
                            completed_count=int(midpoint * 40),
                            tag=tag,
                        ))
                else:
                    recs = self._perfect_bucket_recs(run_id, observed)
                store.write_bet_logs_parquet(
                    run_id, "2026-04-10", recs,
                )
            ranked = sb.rank_all()
            # Sanity: MACE should be populated (not just None
            # everywhere — that would defeat the test).
            maces = [s.mean_absolute_calibration_error for s in ranked]
            assert all(m is not None for m in maces), maces
            # Sanity: composite should monotonically decrease.
            assert ranked[0].composite_score > ranked[1].composite_score
            assert ranked[1].composite_score > ranked[2].composite_score
            # Sanity: MACE should NOT monotonically decrease — we set
            # it up to be adversarial vs composite.
            assert maces[0] > maces[2]
            return [round(s.composite_score, 9) for s in ranked]
