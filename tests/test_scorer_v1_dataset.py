"""Tests for ``training_v2/scorer/`` — the Phase 0 supervised scorer
dataset pipeline.

Coverage target per the session prompt: ~5-10 tests, one per outcome
class, one per feature group, a few sanity-property tests. New code
only — no env / matcher / bet_manager regression tests (Phase −1's
territory).
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from data.episode_builder import (
    Day,
    PriceSize,
    Race,
    RunnerSnap,
    Tick,
)
from training_v2.scorer.feature_extractor import (
    FEATURE_NAMES,
    FeatureExtractor,
)
from training_v2.scorer.label_generator import (
    LabelGenerator,
    LabelOutcome,
)


# ── Synthetic-tick helpers ─────────────────────────────────────────────


def _runner(
    sid: int = 12345,
    status: str = "ACTIVE",
    ltp: float = 5.0,
    total_matched: float = 1000.0,
    back_levels: tuple[tuple[float, float], ...] = ((4.9, 100.0), (4.8, 50.0), (4.7, 20.0)),
    lay_levels: tuple[tuple[float, float], ...] = ((5.1, 100.0), (5.2, 50.0), (5.3, 20.0)),
    sort_priority: int = 1,
) -> RunnerSnap:
    return RunnerSnap(
        selection_id=sid,
        status=status,
        last_traded_price=ltp,
        total_matched=total_matched,
        starting_price_near=0.0,
        starting_price_far=0.0,
        adjustment_factor=None,
        bsp=None,
        sort_priority=sort_priority,
        removal_date=None,
        available_to_back=[PriceSize(p, s) for p, s in back_levels],
        available_to_lay=[PriceSize(p, s) for p, s in lay_levels],
    )


def _tick(
    timestamp: datetime,
    runners: list[RunnerSnap],
    *,
    market_id: str = "1.111",
    venue: str = "Test",
    start_time: datetime | None = None,
    seq: int = 0,
    traded_volume: float = 5000.0,
    in_play: bool = False,
) -> Tick:
    if start_time is None:
        start_time = timestamp + timedelta(seconds=300)
    return Tick(
        market_id=market_id,
        timestamp=timestamp,
        sequence_number=seq,
        venue=venue,
        market_start_time=start_time,
        number_of_active_runners=sum(1 for r in runners if r.status == "ACTIVE"),
        traded_volume=traded_volume,
        in_play=in_play,
        winner_selection_id=None,
        race_status=None,
        temperature=None,
        precipitation=None,
        wind_speed=None,
        wind_direction=None,
        humidity=None,
        weather_code=None,
        runners=runners,
    )


def _race(
    ticks: list[Tick],
    *,
    market_id: str = "1.111",
    market_type: str = "WIN",
) -> Race:
    return Race(
        market_id=market_id,
        venue=ticks[0].venue,
        market_start_time=ticks[0].market_start_time,
        winner_selection_id=None,
        ticks=ticks,
        runner_metadata={},
        market_name="test",
        market_type=market_type,
        n_runners=len(ticks[0].runners),
    )


# ── Feature extractor tests ─────────────────────────────────────────────


def test_extract_returns_all_locked_feature_names() -> None:
    """The extractor must populate exactly the FEATURE_NAMES set on
    every call — Session 02 / Phase 1 index by these names."""
    base_ts = datetime(2026, 4, 6, 13, 50, 0)
    runners = [_runner(sid=10), _runner(sid=11, ltp=8.0,
                                        back_levels=((7.8, 50.0),),
                                        lay_levels=((8.2, 50.0),))]
    tick = _tick(base_ts, runners,
                 start_time=base_ts + timedelta(seconds=300))
    race = _race([tick])

    fx = FeatureExtractor()
    fx.update_history(race, tick)
    feats = fx.extract(race, 0, 0, "back")

    assert set(feats.keys()) == set(FEATURE_NAMES)
    # Spot-check price + side features
    assert feats["best_back"] == pytest.approx(4.9)
    assert feats["best_lay"] == pytest.approx(5.1)
    assert feats["ltp"] == pytest.approx(5.0)
    assert feats["spread"] == pytest.approx(0.2, abs=1e-6)
    assert feats["side_back"] == 1.0
    assert feats["side_lay"] == 0.0
    assert feats["market_type_win"] == 1.0
    assert feats["market_type_each_way"] == 0.0


def test_extract_velocity_features_populate_after_window() -> None:
    """30s LTP-change feature emits NaN before the window is full and a
    real value once it spans 30s of history."""
    base_ts = datetime(2026, 4, 6, 13, 50, 0)
    fx = FeatureExtractor()
    market_id = "vel.test"
    last_feats: dict[str, float] | None = None
    for i in range(40):
        ts = base_ts + timedelta(seconds=i)
        ltp = 5.0 + i * 0.01  # drift up
        runners = [_runner(sid=99, ltp=ltp,
                           total_matched=1000.0 + 50.0 * i)]
        tick = _tick(ts, runners,
                     market_id=market_id,
                     start_time=base_ts + timedelta(seconds=300),
                     seq=i)
        race = _race([tick], market_id=market_id)
        fx.update_history(race, tick)
        last_feats = fx.extract(race, 0, 0, "back")

    assert last_feats is not None
    # 30 seconds of drift @ 0.01/s → 0.3
    assert last_feats["ltp_change_last_30s"] == pytest.approx(0.3, abs=0.05)
    assert last_feats["traded_volume_last_30s"] > 0.0


def test_extract_handles_missing_ltp_with_nan() -> None:
    """LTP-derived features must NaN-propagate when LTP is absent."""
    base_ts = datetime(2026, 4, 6, 13, 50, 0)
    runner = _runner(sid=10, ltp=0.0)  # LTP <= 1.0 → unpriceable
    runner = replace(runner, last_traded_price=0.0)
    tick = _tick(base_ts, [runner],
                 start_time=base_ts + timedelta(seconds=300))
    race = _race([tick])

    fx = FeatureExtractor()
    fx.update_history(race, tick)
    feats = fx.extract(race, 0, 0, "lay")

    assert math.isnan(feats["ltp"])
    assert feats["side_lay"] == 1.0


# ── Label generator — outcome-class coverage ────────────────────────────


def _build_n_tick_race(
    *,
    n_ticks: int,
    runner_factory,
    seconds_between_ticks: float = 1.0,
    race_off_in: float = 60.0,
    market_id: str = "lbl.test",
    market_type: str = "WIN",
) -> Race:
    """Build a Race with ``n_ticks`` ticks, race-off ``race_off_in``
    seconds after the first tick. ``runner_factory(i)`` returns the
    runners list for tick i.
    """
    base_ts = datetime(2026, 4, 6, 13, 50, 0)
    start_time = base_ts + timedelta(seconds=race_off_in)
    ticks = []
    for i in range(n_ticks):
        ts = base_ts + timedelta(seconds=i * seconds_between_ticks)
        ticks.append(_tick(
            ts, runner_factory(i),
            market_id=market_id,
            start_time=start_time,
            seq=i,
            traded_volume=1000.0 * (i + 1),
        ))
    return _race(ticks, market_id=market_id, market_type=market_type)


def test_label_aggressive_refused_returns_nan() -> None:
    """An aggressive open with empty opposite-side book is refused →
    label NaN, outcome INFEASIBLE_AGG_REFUSED."""
    base_ts = datetime(2026, 4, 6, 13, 50, 0)
    # An aggressive back matches against ``available_to_back`` (the
    # resting lay orders); empty that side and the matcher refuses.
    runner = _runner(sid=10, back_levels=())
    tick = _tick(base_ts, [runner],
                 start_time=base_ts + timedelta(seconds=300))
    race = _race([tick])

    lg = LabelGenerator()
    res = lg.generate(race, 0, 0, "back")
    assert res.label is None
    assert res.outcome is LabelOutcome.INFEASIBLE_AGG_REFUSED


def test_label_inactive_runner_returns_nan() -> None:
    base_ts = datetime(2026, 4, 6, 13, 50, 0)
    runner = _runner(sid=10, status="REMOVED")
    tick = _tick(base_ts, [runner],
                 start_time=base_ts + timedelta(seconds=300))
    race = _race([tick])

    lg = LabelGenerator()
    res = lg.generate(race, 0, 0, "back")
    assert res.label is None
    assert res.outcome is LabelOutcome.INFEASIBLE_INACTIVE


def test_label_force_close_path_returns_zero() -> None:
    """When the passive can't fill before T-N, the relaxed-matcher
    force-close path is exercised — label=0.0, outcome=FORCE_CLOSED.

    Setup: aggressive back fills at 4.9; this test pins
    ``arb_ticks=1`` so the passive lay rests at 4.8 (independent of
    the module-level default). The lay-side book carries an existing
    50-unit queue at 4.8 so ``queue_ahead`` starts non-zero — without
    that, ``PassiveOrderBook.on_tick`` fills the passive on the first
    tick after placement (queue_ahead=0 short-circuits the threshold
    check). LTP stays at 5.0 throughout so no trade crosses below 4.8
    to advance the queue.
    """
    def factory(i: int) -> list[RunnerSnap]:
        return [RunnerSnap(
            selection_id=10, status="ACTIVE", last_traded_price=5.0,
            total_matched=1000.0,
            starting_price_near=0.0, starting_price_far=0.0,
            adjustment_factor=None, bsp=None, sort_priority=1,
            removal_date=None,
            available_to_back=[
                PriceSize(4.9, 100.0), PriceSize(4.7, 50.0),
            ],
            available_to_lay=[
                PriceSize(5.1, 100.0), PriceSize(5.2, 50.0),
                PriceSize(4.8, 50.0),  # queue_ahead for the 1-tick passive
            ],
        )]

    race = _build_n_tick_race(
        n_ticks=50,
        runner_factory=factory,
        seconds_between_ticks=1.0,
        race_off_in=10.0,  # tick 0 already inside the 30s force-close
                            # window — first walker iteration hits the
                            # force-close branch.
    )

    lg = LabelGenerator(force_close_threshold_sec=30.0, arb_ticks=1)
    res = lg.generate(race, 0, 0, "back")
    assert res.label == pytest.approx(0.0)
    assert res.outcome is LabelOutcome.FORCE_CLOSED


def test_label_naked_when_race_runs_out_with_no_close_window() -> None:
    """If the race ends with no fill AND the force-close window never
    opens (race_off_in > total_race_time), outcome=NAKED.

    Same queue_ahead trick as the force-close test (50-unit existing
    queue at the passive price); ticks span less than the 30s window
    to off, so the walker exhausts the race without entering the
    force-close branch.
    """
    def factory(i: int) -> list[RunnerSnap]:
        return [RunnerSnap(
            selection_id=10, status="ACTIVE", last_traded_price=5.0,
            total_matched=1000.0,
            starting_price_near=0.0, starting_price_far=0.0,
            adjustment_factor=None, bsp=None, sort_priority=1,
            removal_date=None,
            available_to_back=[PriceSize(4.9, 100.0)],
            available_to_lay=[
                PriceSize(5.1, 100.0), PriceSize(4.8, 50.0),
            ],
        )]

    # 5 ticks 1s apart, race-off 1000s after t0 → time_to_off > 30
    # throughout, so the walker never enters the force-close branch.
    race = _build_n_tick_race(
        n_ticks=5,
        runner_factory=factory,
        seconds_between_ticks=1.0,
        race_off_in=1000.0,
    )

    lg = LabelGenerator(force_close_threshold_sec=30.0, arb_ticks=1)
    res = lg.generate(race, 0, 0, "back")
    assert res.label == pytest.approx(0.0)
    assert res.outcome is LabelOutcome.NAKED


def test_label_matured_path_returns_one() -> None:
    """Aggressive opens, passive's queue clears via crossable traded
    volume → label=1.0, outcome=MATURED.

    Setup: passive lay at 4.8 (1 tick below agg back at 4.9) with
    queue_ahead=50. LTP starts at 5.0 at the placement tick (so the
    agg fills cleanly), then drops to 4.7 from tick 1 onward. With
    LTP ≤ passive_price the volume gate passes; total_matched advances
    by 100/tick which exceeds the 50-unit queue on tick 1.
    """
    def factory(i: int) -> list[RunnerSnap]:
        ltp = 5.0 if i == 0 else 4.7
        return [RunnerSnap(
            selection_id=10, status="ACTIVE",
            last_traded_price=ltp,
            total_matched=1000.0 + 100.0 * i,
            starting_price_near=0.0, starting_price_far=0.0,
            adjustment_factor=None, bsp=None, sort_priority=1,
            removal_date=None,
            available_to_back=[PriceSize(4.9, 100.0)],
            available_to_lay=[
                PriceSize(5.1, 100.0), PriceSize(4.8, 50.0),
            ],
        )]

    race = _build_n_tick_race(
        n_ticks=10,
        runner_factory=factory,
        seconds_between_ticks=1.0,
        race_off_in=1000.0,
    )

    lg = LabelGenerator(force_close_threshold_sec=30.0, arb_ticks=1)
    res = lg.generate(race, 0, 0, "back")
    assert res.label == pytest.approx(1.0)
    assert res.outcome is LabelOutcome.MATURED


def test_label_in_play_tick_is_infeasible() -> None:
    """In-play ticks are categorically excluded — the env restricts
    placement to pre-race only."""
    base_ts = datetime(2026, 4, 6, 13, 50, 0)
    runner = _runner(sid=10)
    tick = _tick(base_ts, [runner],
                 start_time=base_ts + timedelta(seconds=300),
                 in_play=True)
    race = _race([tick])

    lg = LabelGenerator()
    res = lg.generate(race, 0, 0, "back")
    assert res.label is None
    assert res.outcome is LabelOutcome.INFEASIBLE_IN_PLAY


# ── Property-style sanity tests ─────────────────────────────────────────


def test_feature_names_no_duplicates() -> None:
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))


def test_label_generator_does_not_mutate_input_race() -> None:
    """The simulator runs in a fresh BetManager; iterating it must NOT
    mutate the source Race / Tick / RunnerSnap objects (they are frozen
    dataclasses already, but extra defence-in-depth: number of bets in
    the input data and runner snapshots is unchanged after).
    """
    def factory(i: int) -> list[RunnerSnap]:
        return [RunnerSnap(
            selection_id=10, status="ACTIVE", last_traded_price=5.0,
            total_matched=1000.0,
            starting_price_near=0.0, starting_price_far=0.0,
            adjustment_factor=None, bsp=None, sort_priority=1,
            removal_date=None,
            available_to_back=[PriceSize(4.9, 100.0)],
            available_to_lay=[
                PriceSize(5.1, 100.0), PriceSize(4.8, 50.0),
            ],
        )]

    race = _build_n_tick_race(
        n_ticks=20, runner_factory=factory,
        seconds_between_ticks=1.0, race_off_in=10.0,
    )
    snapshot = [tick.runners[0].total_matched for tick in race.ticks]

    lg = LabelGenerator(arb_ticks=1)
    lg.generate(race, 0, 0, "back")

    after = [tick.runners[0].total_matched for tick in race.ticks]
    assert snapshot == after
