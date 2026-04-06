"""Tests for data/feature_engineer.py — derived features."""

from __future__ import annotations

import json
import math
from datetime import datetime

import pytest

from data.episode_builder import (
    Day,
    PriceSize,
    Race,
    RunnerMeta,
    RunnerSnap,
    Tick,
)
from data.feature_engineer import (
    NaN,
    TickHistory,
    cross_runner_features,
    engineer_day,
    engineer_race,
    engineer_tick,
    log_norm,
    market_tick_features,
    parse_form,
    runner_meta_features,
    runner_tick_features,
    safe_float,
    safe_int,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _snap(
    sid: int = 1000,
    ltp: float = 4.0,
    total_matched: float = 500.0,
    status: str = "ACTIVE",
    backs: list[tuple[float, float]] | None = None,
    lays: list[tuple[float, float]] | None = None,
    spn: float = 3.8,
    spf: float = 4.5,
    af: float | None = 100.0,
    bsp: float | None = None,
) -> RunnerSnap:
    """Build a RunnerSnap for testing."""
    if backs is None:
        backs = [(4.0, 100.0), (3.9, 50.0)]
    if lays is None:
        lays = [(4.1, 80.0), (4.2, 40.0)]
    return RunnerSnap(
        selection_id=sid,
        status=status,
        last_traded_price=ltp,
        total_matched=total_matched,
        starting_price_near=spn,
        starting_price_far=spf,
        adjustment_factor=af,
        bsp=bsp,
        sort_priority=1,
        removal_date=None,
        available_to_back=[PriceSize(p, s) for p, s in backs],
        available_to_lay=[PriceSize(p, s) for p, s in lays],
    )


def _meta(
    sid: int = 1000,
    official_rating: str = "85",
    age: str = "4",
    weight: str = "130",
    stall: str = "3",
    form: str = "1234-21",
    days_since: str = "14",
    jockey_claim: str = "0",
    forecast_num: str = "5",
    forecast_den: str = "2",
    wearing: str = "",
    sex: str = "Gelding",
) -> RunnerMeta:
    return RunnerMeta(
        selection_id=sid,
        runner_name=f"Horse_{sid}",
        sort_priority="1",
        handicap="0.0",
        sire_name="Sire",
        dam_name="Dam",
        damsire_name="Damsire",
        bred="GB",
        official_rating=official_rating,
        adjusted_rating=official_rating,
        age=age,
        sex_type=sex,
        colour_type="Bay",
        weight_value=weight,
        weight_units="LB",
        jockey_name="J. Smith",
        jockey_claim=jockey_claim,
        trainer_name="T. Jones",
        owner_name="Owner",
        stall_draw=stall,
        cloth_number="1",
        form=form,
        days_since_last_run=days_since,
        wearing=wearing,
        forecastprice_numerator=forecast_num,
        forecastprice_denominator=forecast_den,
    )


def _tick(
    runners: list[RunnerSnap] | None = None,
    market_start: str = "2026-03-26 14:00:00",
    timestamp: str = "2026-03-26 13:55:00",
    traded_volume: float = 10000.0,
    n_active: int = 3,
    winner_sid: int | None = 1001,
) -> Tick:
    if runners is None:
        runners = [
            _snap(sid=1000, ltp=3.0),
            _snap(sid=1001, ltp=5.0),
            _snap(sid=1002, ltp=8.0),
        ]
    return Tick(
        market_id="1.234567890",
        timestamp=datetime.fromisoformat(timestamp),
        sequence_number=100,
        venue="Newmarket",
        market_start_time=datetime.fromisoformat(market_start),
        number_of_active_runners=n_active,
        traded_volume=traded_volume,
        in_play=False,
        winner_selection_id=winner_sid,
        race_status=None,
        temperature=15.0,
        precipitation=0.0,
        wind_speed=5.0,
        wind_direction=180.0,
        humidity=65.0,
        weather_code=0,
        runners=runners,
    )


def _race(ticks: list[Tick] | None = None, meta: dict[int, RunnerMeta] | None = None) -> Race:
    if ticks is None:
        ticks = [_tick()]
    if meta is None:
        meta = {
            1000: _meta(sid=1000, official_rating="85"),
            1001: _meta(sid=1001, official_rating="90"),
            1002: _meta(sid=1002, official_rating="75"),
        }
    return Race(
        market_id="1.234567890",
        venue="Newmarket",
        market_start_time=datetime.fromisoformat("2026-03-26 14:00:00"),
        winner_selection_id=1001,
        ticks=ticks,
        runner_metadata=meta,
    )


# ── Tests: safe_float / safe_int / log_norm ──────────────────────────────────


class TestSafeFloat:
    def test_valid(self):
        assert safe_float("3.14") == 3.14

    def test_empty(self):
        assert math.isnan(safe_float(""))

    def test_none(self):
        assert math.isnan(safe_float(None))

    def test_invalid(self):
        assert math.isnan(safe_float("abc"))

    def test_integer_string(self):
        assert safe_float("42") == 42.0


class TestSafeInt:
    def test_valid(self):
        assert safe_int("7") == 7.0

    def test_empty(self):
        assert math.isnan(safe_int(""))

    def test_none(self):
        assert math.isnan(safe_int(None))

    def test_float_string(self):
        # "3.5" should fail int() parsing → NaN
        assert math.isnan(safe_int("3.5"))


class TestLogNorm:
    def test_zero(self):
        assert log_norm(0.0) == 0.0

    def test_positive(self):
        assert log_norm(100.0) == pytest.approx(math.log1p(100.0))

    def test_nan(self):
        assert math.isnan(log_norm(NaN))

    def test_negative(self):
        assert math.isnan(log_norm(-1.0))


# ── Tests: parse_form ────────────────────────────────────────────────────────


class TestParseForm:
    def test_basic_form(self):
        f = parse_form("1234-21")
        assert f["form_runs"] == 6.0
        assert f["form_wins"] == 2.0  # two '1's
        assert f["form_places"] == 5.0  # 1,2,3,2,1 → positions ≤3: 1,2,3,2,1 = 5

    def test_empty_form(self):
        f = parse_form("")
        assert math.isnan(f["form_avg_pos"])
        assert math.isnan(f["form_runs"])

    def test_none_form(self):
        f = parse_form(None)
        assert math.isnan(f["form_avg_pos"])

    def test_form_with_fall(self):
        f = parse_form("1F2")
        assert f["form_runs"] == 3.0
        assert f["form_completion_rate"] == pytest.approx(2 / 3)

    def test_form_zero_means_tenth(self):
        f = parse_form("10")
        assert f["form_worst_pos"] == 10.0

    def test_form_all_wins(self):
        f = parse_form("111")
        assert f["form_wins"] == 3.0
        assert f["form_avg_pos"] == 1.0

    def test_form_truncated_to_six(self):
        f = parse_form("12345678")  # Last 6 chars: "345678"
        assert f["form_runs"] == 6.0

    def test_form_with_separators(self):
        f = parse_form("12-34/56")
        assert f["form_runs"] == 6.0


# ── Tests: runner_meta_features ──────────────────────────────────────────────


class TestRunnerMetaFeatures:
    def test_basic_features(self):
        meta = _meta(official_rating="85", age="4", stall="3")
        feats = runner_meta_features(meta)
        assert feats["official_rating"] == 85.0
        assert feats["age"] == 4.0
        assert feats["stall_draw"] == 3.0

    def test_forecast_price(self):
        meta = _meta(forecast_num="5", forecast_den="2")
        feats = runner_meta_features(meta)
        assert feats["forecast_price"] == pytest.approx(3.5)  # 5/2 + 1

    def test_forecast_implied_prob(self):
        meta = _meta(forecast_num="5", forecast_den="2")
        feats = runner_meta_features(meta)
        assert feats["forecast_implied_prob"] == pytest.approx(1 / 3.5)

    def test_empty_rating(self):
        meta = _meta(official_rating="")
        feats = runner_meta_features(meta)
        assert math.isnan(feats["official_rating"])

    def test_empty_stall(self):
        """Stall draw empty for jumps races."""
        meta = _meta(stall="")
        feats = runner_meta_features(meta)
        assert math.isnan(feats["stall_draw"])

    def test_empty_days_since(self):
        """First-time runner has no days_since_last_run."""
        meta = _meta(days_since="")
        feats = runner_meta_features(meta)
        assert math.isnan(feats["days_since_last_run"])

    def test_sex_encoding(self):
        meta = _meta(sex="Gelding")
        feats = runner_meta_features(meta)
        assert feats["sex_gelding"] == 1.0
        assert feats["sex_mare"] == 0.0

    def test_equipment_flags(self):
        meta = _meta(wearing="Blinkers")
        feats = runner_meta_features(meta)
        assert feats["equip_blinkers"] == 1.0
        assert feats["equip_visor"] == 0.0
        assert feats["has_equipment"] == 1.0

    def test_no_equipment(self):
        meta = _meta(wearing="")
        feats = runner_meta_features(meta)
        assert feats["has_equipment"] == 0.0

    def test_form_features_included(self):
        meta = _meta(form="1234-21")
        feats = runner_meta_features(meta)
        assert "form_avg_pos" in feats
        assert "form_wins" in feats

    def test_jockey_claim(self):
        meta = _meta(jockey_claim="5")
        feats = runner_meta_features(meta)
        assert feats["jockey_claim"] == 5.0

    def test_weight_value(self):
        meta = _meta(weight="132")
        feats = runner_meta_features(meta)
        assert feats["weight_value"] == 132.0

    def test_forecast_zero_denominator(self):
        meta = _meta(forecast_num="5", forecast_den="0")
        feats = runner_meta_features(meta)
        assert math.isnan(feats["forecast_price"])


# ── Tests: runner_tick_features ──────────────────────────────────────────────


class TestRunnerTickFeatures:
    def test_ltp(self):
        snap = _snap(ltp=4.0)
        feats = runner_tick_features(snap)
        assert feats["ltp"] == 4.0

    def test_implied_prob(self):
        snap = _snap(ltp=4.0)
        feats = runner_tick_features(snap)
        assert feats["implied_prob"] == pytest.approx(0.25)

    def test_zero_ltp(self):
        snap = _snap(ltp=0.0)
        feats = runner_tick_features(snap)
        assert math.isnan(feats["ltp"])
        assert math.isnan(feats["implied_prob"])

    def test_spread(self):
        snap = _snap(backs=[(4.0, 100)], lays=[(4.2, 80)])
        feats = runner_tick_features(snap)
        assert feats["spread"] == pytest.approx(0.2)

    def test_spread_pct(self):
        snap = _snap(backs=[(4.0, 100)], lays=[(4.2, 80)])
        feats = runner_tick_features(snap)
        assert feats["spread_pct"] == pytest.approx(0.05)

    def test_mid_price(self):
        snap = _snap(backs=[(4.0, 100)], lays=[(4.2, 80)])
        feats = runner_tick_features(snap)
        assert feats["mid_price"] == pytest.approx(4.1)

    def test_back_prices(self):
        snap = _snap(backs=[(4.0, 100), (3.9, 50)])
        feats = runner_tick_features(snap)
        assert feats["back_price_1"] == 4.0
        assert feats["back_size_1"] == 100.0
        assert feats["back_price_2"] == 3.9

    def test_lay_prices(self):
        snap = _snap(lays=[(4.1, 80), (4.2, 40)])
        feats = runner_tick_features(snap)
        assert feats["lay_price_1"] == 4.1
        assert feats["lay_size_1"] == 80.0

    def test_missing_level_3(self):
        """Only 2 levels available → level 3 is NaN."""
        snap = _snap(backs=[(4.0, 100), (3.9, 50)])
        feats = runner_tick_features(snap)
        assert math.isnan(feats["back_price_3"])
        assert math.isnan(feats["back_size_3"])

    def test_depth(self):
        snap = _snap(
            backs=[(4.0, 100), (3.9, 50)],
            lays=[(4.1, 80), (4.2, 40)],
        )
        feats = runner_tick_features(snap)
        assert feats["back_depth"] == 150.0
        assert feats["lay_depth"] == 120.0
        assert feats["total_depth"] == 270.0

    def test_weight_of_money(self):
        snap = _snap(
            backs=[(4.0, 100)],
            lays=[(4.1, 100)],
        )
        feats = runner_tick_features(snap)
        assert feats["weight_of_money"] == pytest.approx(0.5)

    def test_empty_book(self):
        """No back or lay levels → spreads and depth are NaN/0."""
        snap = _snap(backs=[], lays=[])
        feats = runner_tick_features(snap)
        assert math.isnan(feats["spread"])
        assert feats["back_depth"] == 0.0
        assert math.isnan(feats["weight_of_money"])

    def test_is_active_flag(self):
        snap = _snap(status="ACTIVE")
        feats = runner_tick_features(snap)
        assert feats["is_active"] == 1.0
        assert feats["is_removed"] == 0.0

    def test_is_removed_flag(self):
        snap = _snap(status="REMOVED")
        feats = runner_tick_features(snap)
        assert feats["is_active"] == 0.0
        assert feats["is_removed"] == 1.0

    def test_bsp_none(self):
        snap = _snap(bsp=None)
        feats = runner_tick_features(snap)
        assert math.isnan(feats["bsp"])

    def test_bsp_present(self):
        snap = _snap(bsp=4.5)
        feats = runner_tick_features(snap)
        assert feats["bsp"] == 4.5

    def test_log_normalised_fields(self):
        snap = _snap(total_matched=1000.0)
        feats = runner_tick_features(snap)
        assert feats["runner_total_matched_log"] == pytest.approx(math.log1p(1000.0))


# ── Tests: market_tick_features ──────────────────────────────────────────────


class TestMarketTickFeatures:
    def test_time_to_off(self):
        t = _tick(timestamp="2026-03-26 13:55:00", market_start="2026-03-26 14:00:00")
        feats = market_tick_features(t)
        assert feats["time_to_off_seconds"] == 300.0

    def test_time_to_off_norm(self):
        t = _tick(timestamp="2026-03-26 13:30:00", market_start="2026-03-26 14:00:00")
        feats = market_tick_features(t)
        assert feats["time_to_off_norm"] == pytest.approx(1.0)  # 30 min = 1800s → 1.0

    def test_volume(self):
        t = _tick(traded_volume=50000.0)
        feats = market_tick_features(t)
        assert feats["market_traded_volume"] == 50000.0
        assert feats["market_traded_volume_log"] == pytest.approx(math.log1p(50000.0))

    def test_num_active_runners(self):
        t = _tick(n_active=8)
        feats = market_tick_features(t)
        assert feats["num_active_runners"] == 8.0

    def test_overround(self):
        """Overround = sum(1/best_back) for each active runner."""
        runners = [
            _snap(sid=1, ltp=2.0, backs=[(2.0, 100)]),
            _snap(sid=2, ltp=3.0, backs=[(3.0, 100)]),
            _snap(sid=3, ltp=6.0, backs=[(6.0, 100)]),
        ]
        t = _tick(runners=runners)
        feats = market_tick_features(t)
        # 1/2 + 1/3 + 1/6 = 1.0
        assert feats["overround"] == pytest.approx(1.0)

    def test_overround_pct(self):
        runners = [
            _snap(sid=1, backs=[(2.0, 100)]),
            _snap(sid=2, backs=[(4.0, 100)]),
        ]
        t = _tick(runners=runners)
        feats = market_tick_features(t)
        # 1/2 + 1/4 = 0.75 → (0.75 - 1) * 100 = -25%
        assert feats["overround_pct"] == pytest.approx(-25.0)

    def test_favourite_ltp(self):
        runners = [
            _snap(sid=1, ltp=2.5),
            _snap(sid=2, ltp=5.0),
            _snap(sid=3, ltp=10.0),
        ]
        t = _tick(runners=runners)
        feats = market_tick_features(t)
        assert feats["favourite_ltp"] == 2.5
        assert feats["outsider_ltp"] == 10.0
        assert feats["ltp_range"] == 7.5

    def test_weather_passthrough(self):
        t = _tick()
        feats = market_tick_features(t)
        assert feats["temperature"] == 15.0
        assert feats["precipitation"] == 0.0
        assert feats["wind_speed"] == 5.0
        assert feats["humidity"] == 65.0
        assert feats["weather_code"] == 0.0

    def test_null_weather(self):
        t = Tick(
            market_id="1.1",
            timestamp=datetime(2026, 3, 26, 13, 55),
            sequence_number=1,
            venue="Newmarket",
            market_start_time=datetime(2026, 3, 26, 14, 0),
            number_of_active_runners=1,
            traded_volume=1000,
            in_play=False,
            winner_selection_id=None,
            race_status=None,
            temperature=None,
            precipitation=None,
            wind_speed=None,
            wind_direction=None,
            humidity=None,
            weather_code=None,
            runners=[_snap()],
        )
        feats = market_tick_features(t)
        assert math.isnan(feats["temperature"])
        assert math.isnan(feats["humidity"])

    def test_avg_spread(self):
        runners = [
            _snap(sid=1, backs=[(2.0, 100)], lays=[(2.2, 100)]),  # spread 0.2
            _snap(sid=2, backs=[(4.0, 100)], lays=[(4.4, 100)]),  # spread 0.4
        ]
        t = _tick(runners=runners)
        feats = market_tick_features(t)
        assert feats["avg_spread"] == pytest.approx(0.3)

    def test_no_runners(self):
        t = _tick(runners=[])
        feats = market_tick_features(t)
        assert math.isnan(feats["overround"])
        assert math.isnan(feats["favourite_ltp"])

    def test_removed_runners_excluded_from_overround(self):
        runners = [
            _snap(sid=1, status="ACTIVE", backs=[(2.0, 100)]),
            _snap(sid=2, status="REMOVED", backs=[(3.0, 100)]),
        ]
        t = _tick(runners=runners)
        feats = market_tick_features(t)
        assert feats["overround"] == pytest.approx(0.5)  # only active runner
        assert feats["n_priced_runners"] == 1.0

    def test_market_type_and_ew_defaults_without_race(self):
        """Without a Race, market-type and EW keys exist but are neutral."""
        t = _tick()
        feats = market_tick_features(t)
        # Keys are always present for stable observation shape
        assert feats["market_type_win"] == 0.0
        assert feats["market_type_each_way"] == 0.0
        assert feats["has_each_way_terms"] == 0.0
        assert math.isnan(feats["each_way_divisor"])
        assert math.isnan(feats["place_odds_fraction"])
        assert math.isnan(feats["number_of_each_way_places"])

    def test_market_type_win_one_hot(self):
        t = _tick()
        race = _race(ticks=[t])
        race.market_type = "WIN"
        feats = market_tick_features(t, race)
        assert feats["market_type_win"] == 1.0
        assert feats["market_type_each_way"] == 0.0
        # WIN markets carry no EW terms
        assert feats["has_each_way_terms"] == 0.0
        assert math.isnan(feats["each_way_divisor"])

    def test_each_way_terms_populated(self):
        t = _tick()
        race = _race(ticks=[t])
        race.market_type = "EACH_WAY"
        race.each_way_divisor = 5.0  # 1/5 odds
        race.number_of_each_way_places = 3
        feats = market_tick_features(t, race)
        assert feats["market_type_each_way"] == 1.0
        assert feats["market_type_win"] == 0.0
        assert feats["has_each_way_terms"] == 1.0
        assert feats["each_way_divisor"] == 5.0
        assert feats["place_odds_fraction"] == pytest.approx(0.2)
        assert feats["number_of_each_way_places"] == 3.0

    def test_each_way_divisor_quarter(self):
        t = _tick()
        race = _race(ticks=[t])
        race.market_type = "EACH_WAY"
        race.each_way_divisor = 4.0  # 1/4 odds
        race.number_of_each_way_places = 4
        feats = market_tick_features(t, race)
        assert feats["place_odds_fraction"] == pytest.approx(0.25)
        assert feats["number_of_each_way_places"] == 4.0


# ── Tests: cross_runner_features ─────────────────────────────────────────────


class TestCrossRunnerFeatures:
    def test_ltp_rank(self):
        runners = [
            _snap(sid=1, ltp=3.0),
            _snap(sid=2, ltp=5.0),
            _snap(sid=3, ltp=8.0),
        ]
        t = _tick(runners=runners)
        meta = {
            1: _meta(sid=1, official_rating="90"),
            2: _meta(sid=2, official_rating="85"),
            3: _meta(sid=3, official_rating="80"),
        }
        cross = cross_runner_features(t, meta)
        # Lowest LTP = favourite = rank 1
        assert cross[1]["ltp_rank"] == 1.0
        assert cross[2]["ltp_rank"] == 2.0
        assert cross[3]["ltp_rank"] == 3.0

    def test_gap_to_favourite(self):
        runners = [
            _snap(sid=1, ltp=3.0),
            _snap(sid=2, ltp=5.0),
        ]
        t = _tick(runners=runners)
        cross = cross_runner_features(t, {})
        assert cross[1]["gap_to_favourite"] == pytest.approx(0.0)
        assert cross[2]["gap_to_favourite"] == pytest.approx(2.0)

    def test_vol_proportion(self):
        runners = [
            _snap(sid=1, total_matched=300.0),
            _snap(sid=2, total_matched=700.0),
        ]
        t = _tick(runners=runners)
        cross = cross_runner_features(t, {})
        assert cross[1]["vol_proportion"] == pytest.approx(0.3)
        assert cross[2]["vol_proportion"] == pytest.approx(0.7)

    def test_rating_rank(self):
        runners = [
            _snap(sid=1, ltp=3.0),
            _snap(sid=2, ltp=5.0),
        ]
        t = _tick(runners=runners)
        meta = {
            1: _meta(sid=1, official_rating="80"),
            2: _meta(sid=2, official_rating="90"),
        }
        cross = cross_runner_features(t, meta)
        # Highest rating = rank 1
        assert cross[2]["rating_rank"] == 1.0
        assert cross[1]["rating_rank"] == 2.0

    def test_rating_norm(self):
        runners = [
            _snap(sid=1, ltp=3.0),
            _snap(sid=2, ltp=5.0),
        ]
        t = _tick(runners=runners)
        meta = {
            1: _meta(sid=1, official_rating="80"),
            2: _meta(sid=2, official_rating="90"),
        }
        cross = cross_runner_features(t, meta)
        assert cross[1]["rating_norm"] == pytest.approx(0.0)  # min
        assert cross[2]["rating_norm"] == pytest.approx(1.0)  # max

    def test_empty_runners(self):
        t = _tick(runners=[])
        cross = cross_runner_features(t, {})
        assert cross == {}

    def test_no_metadata(self):
        runners = [_snap(sid=1, ltp=3.0)]
        t = _tick(runners=runners)
        cross = cross_runner_features(t, {})
        assert math.isnan(cross[1]["rating_rank"])
        assert math.isnan(cross[1]["rating_norm"])

    def test_single_runner(self):
        runners = [_snap(sid=1, ltp=3.0)]
        t = _tick(runners=runners)
        meta = {1: _meta(sid=1, official_rating="85")}
        cross = cross_runner_features(t, meta)
        assert cross[1]["ltp_rank"] == 1.0
        assert cross[1]["rating_norm"] == 0.5  # single runner → 0.5


# ── Tests: TickHistory & velocity features ───────────────────────────────────


class TestTickHistory:
    def test_initial_state(self):
        hist = TickHistory()
        feats = hist.runner_velocity_features(1000)
        assert math.isnan(feats["ltp_velocity_3"])
        assert feats["tick_count"] == 0.0

    def test_velocity_after_updates(self):
        hist = TickHistory()
        # Simulate 5 ticks with increasing LTP
        for i in range(5):
            snap = _snap(sid=1000, ltp=3.0 + i * 0.1)
            t = _tick(runners=[snap])
            mkt = market_tick_features(t)
            hist.update(t, mkt)

        feats = hist.runner_velocity_features(1000)
        # velocity_3 = last LTP - LTP 3 ticks ago = 3.4 - 3.2 = 0.2
        assert feats["ltp_velocity_3"] == pytest.approx(0.2, abs=1e-9)
        assert feats["tick_count"] == 5.0

    def test_velocity_5_insufficient(self):
        hist = TickHistory()
        for i in range(3):
            snap = _snap(sid=1, ltp=3.0 + i * 0.1)
            t = _tick(runners=[snap])
            hist.update(t, market_tick_features(t))

        feats = hist.runner_velocity_features(1)
        assert feats["ltp_velocity_3"] == pytest.approx(0.2, abs=1e-9)
        assert math.isnan(feats["ltp_velocity_5"])

    def test_volatility(self):
        hist = TickHistory()
        ltps = [3.0, 3.2, 2.8, 3.1, 2.9]
        for ltp in ltps:
            snap = _snap(sid=1, ltp=ltp)
            t = _tick(runners=[snap])
            hist.update(t, market_tick_features(t))

        feats = hist.runner_velocity_features(1)
        assert feats["ltp_volatility_5"] > 0  # non-zero volatility

    def test_market_velocity(self):
        hist = TickHistory()
        for i in range(5):
            t = _tick(traded_volume=10000 + i * 500)
            hist.update(t, market_tick_features(t))

        mkt_feats = hist.market_velocity_features()
        assert mkt_feats["market_vol_delta_3"] == pytest.approx(1000.0)

    def test_reset(self):
        hist = TickHistory()
        t = _tick()
        hist.update(t, market_tick_features(t))
        hist.reset()
        feats = hist.runner_velocity_features(1000)
        assert feats["tick_count"] == 0.0

    def test_window_cap(self):
        hist = TickHistory(max_window=5)
        for i in range(10):
            snap = _snap(sid=1, ltp=3.0 + i * 0.1)
            t = _tick(runners=[snap])
            hist.update(t, market_tick_features(t))

        feats = hist.runner_velocity_features(1)
        assert feats["tick_count"] == 5.0  # capped by window

    def test_volume_delta(self):
        hist = TickHistory()
        vols = [100, 200, 350, 500, 700]
        for v in vols:
            snap = _snap(sid=1, ltp=3.0, total_matched=v)
            t = _tick(runners=[snap])
            hist.update(t, market_tick_features(t))

        feats = hist.runner_velocity_features(1)
        assert feats["vol_delta_3"] == pytest.approx(350.0)  # 700 - 350


# ── Tests: engineer_tick ─────────────────────────────────────────────────────


class TestEngineerTick:
    def test_returns_all_sections(self):
        race = _race()
        hist = TickHistory()
        result = engineer_tick(race.ticks[0], race, hist)
        assert "market" in result
        assert "runners" in result
        assert "market_velocity" in result

    def test_runners_keyed_by_sid(self):
        race = _race()
        hist = TickHistory()
        result = engineer_tick(race.ticks[0], race, hist)
        assert 1000 in result["runners"]
        assert 1001 in result["runners"]
        assert 1002 in result["runners"]

    def test_metadata_included(self):
        race = _race()
        hist = TickHistory()
        result = engineer_tick(race.ticks[0], race, hist)
        # official_rating comes from metadata
        assert result["runners"][1000]["official_rating"] == 85.0

    def test_cross_features_included(self):
        race = _race()
        hist = TickHistory()
        result = engineer_tick(race.ticks[0], race, hist)
        assert "ltp_rank" in result["runners"][1000]

    def test_velocity_features_included(self):
        race = _race()
        hist = TickHistory()
        result = engineer_tick(race.ticks[0], race, hist)
        assert "ltp_velocity_3" in result["runners"][1000]

    def test_history_updated(self):
        race = _race()
        hist = TickHistory()
        engineer_tick(race.ticks[0], race, hist)
        # After one tick, tick_count should be 1
        feats = hist.runner_velocity_features(1000)
        assert feats["tick_count"] == 1.0


# ── Tests: engineer_race ─────────────────────────────────────────────────────


class TestEngineerRace:
    def test_one_result_per_tick(self):
        ticks = [
            _tick(timestamp=f"2026-03-26 13:{50+i}:00")
            for i in range(5)
        ]
        race = _race(ticks=ticks)
        results = engineer_race(race)
        assert len(results) == 5

    def test_velocity_builds_over_ticks(self):
        ticks = []
        for i in range(5):
            runners = [
                _snap(sid=1000, ltp=3.0 + i * 0.1),
                _snap(sid=1001, ltp=5.0 - i * 0.1),
            ]
            ticks.append(_tick(runners=runners, timestamp=f"2026-03-26 13:{50+i}:00"))
        race = _race(ticks=ticks)
        results = engineer_race(race)

        # First tick: velocity_3 should be NaN
        assert math.isnan(results[0]["runners"][1000]["ltp_velocity_3"])
        # Fourth tick (index 3): velocity_3 should be available
        assert not math.isnan(results[3]["runners"][1000]["ltp_velocity_3"])


# ── Tests: engineer_day ──────────────────────────────────────────────────────


class TestEngineerDay:
    def test_returns_nested_list(self):
        day = Day(date="2026-03-26", races=[_race()])
        results = engineer_day(day)
        assert len(results) == 1
        assert len(results[0]) == 1  # single tick in default race

    def test_multiple_races(self):
        day = Day(date="2026-03-26", races=[_race(), _race()])
        results = engineer_day(day)
        assert len(results) == 2

    def test_velocity_resets_between_races(self):
        """Each race starts with fresh TickHistory."""
        ticks1 = [_tick(timestamp=f"2026-03-26 13:{50+i}:00") for i in range(5)]
        ticks2 = [_tick(timestamp=f"2026-03-26 14:{50+i}:00") for i in range(5)]
        day = Day(
            date="2026-03-26",
            races=[_race(ticks=ticks1), _race(ticks=ticks2)],
        )
        results = engineer_day(day)
        # First tick of second race should have NaN velocity
        assert math.isnan(results[1][0]["runners"][1000]["ltp_velocity_3"])
