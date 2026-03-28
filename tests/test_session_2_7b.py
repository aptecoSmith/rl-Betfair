"""Session 2.7b — RaceCardRunners (PastRacesJson) unit tests.

Tests cover:
- Position parsing (``_parse_position``)
- PastRacesJson parsing (``_parse_past_races_json``)
- Past race feature engineering (``past_race_features``)
- Extractor RaceCardRunners merge (``_merge_racecard_runners``)
- Episode builder backward compatibility
- Environment dimension updates
"""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from data.episode_builder import (
    PastRace,
    RunnerMeta,
    _parse_past_races_json,
    _parse_position,
)
from data.feature_engineer import (
    PAST_RACE_FEATURE_KEYS,
    past_race_features,
    runner_meta_features,
)
from env.betfair_env import RUNNER_DIM, RUNNER_KEYS

NaN = float("nan")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_past_race(**kwargs) -> PastRace:
    """Create a PastRace with sensible defaults."""
    defaults = dict(
        date="2026-01-15",
        course="Kempton",
        distance_yards=3200,
        going="Good to Soft",
        going_abbr="GS",
        bsp=5.0,
        ip_max=100.0,
        ip_min=2.0,
        race_type="Hurdle",
        jockey="J Smith",
        official_rating=90.0,
        position=3,
        field_size=10,
    )
    defaults.update(kwargs)
    return PastRace(**defaults)


def _make_runner_meta(past_races=(), **kwargs) -> RunnerMeta:
    """Create a RunnerMeta with sensible defaults and optional past_races."""
    defaults = dict(
        selection_id=12345,
        runner_name="Test Horse",
        sort_priority="1",
        handicap="0",
        sire_name="", dam_name="", damsire_name="", bred="",
        official_rating="90", adjusted_rating="88",
        age="5", sex_type="GELDING", colour_type="b",
        weight_value="154", weight_units="lbs",
        jockey_name="J Smith", jockey_claim="",
        trainer_name="A Trainer", owner_name="An Owner",
        stall_draw="3", cloth_number="5",
        form="1234", days_since_last_run="14",
        wearing="", forecastprice_numerator="4",
        forecastprice_denominator="1",
        past_races=tuple(past_races),
        timeform_comment="Good horse",
        recent_form="",
    )
    defaults.update(kwargs)
    return RunnerMeta(**defaults)


def _sample_past_races_json() -> str:
    """Return a realistic PastRacesJson string."""
    return json.dumps([
        {
            "date": "2026-03-11T15:38:00Z",
            "countryCode": "GB",
            "course": "Kempton",
            "courseCode": "Kem",
            "distance": 3200,
            "distanceUnit": "yd",
            "going": {"abbr": "GS", "full": "Good to Soft"},
            "inPlayMax": 100,
            "inPlayMin": 3,
            "bsp": 5.5,
            "raceType": {"key": "H", "abbr": "Hdl", "full": "Hurdle"},
            "jockey": "J Smith",
            "officialRating": 90,
            "position": "1/8",
        },
        {
            "date": "2026-02-20T14:00:00Z",
            "course": "Doncaster",
            "distance": 3500,
            "distanceUnit": "yd",
            "going": {"abbr": "Hy", "full": "Heavy"},
            "inPlayMax": 200,
            "inPlayMin": 8,
            "bsp": 8.0,
            "raceType": {"key": "C", "abbr": "Chs", "full": "Chase"},
            "jockey": "A Jockey",
            "position": "3/12",
        },
        {
            "date": "2026-01-05T12:30:00Z",
            "course": "Kempton",
            "distance": 3200,
            "distanceUnit": "yd",
            "going": {"abbr": "GS", "full": "Good to Soft"},
            "inPlayMax": 50,
            "inPlayMin": 2,
            "bsp": 4.0,
            "raceType": {"key": "H", "abbr": "Hdl", "full": "Hurdle"},
            "jockey": "J Smith",
            "officialRating": 88,
            "position": "2/10",
        },
        {
            "date": "2025-12-10T13:00:00Z",
            "course": "Uttoxeter",
            "distance": 4000,
            "distanceUnit": "yd",
            "going": {"abbr": "Gd", "full": "Good"},
            "bsp": 12.0,
            "raceType": {"key": "H", "abbr": "Hdl", "full": "Hurdle"},
            "jockey": "B Rider",
            "position": "U/9",
        },
    ])


# ── Position parsing ─────────────────────────────────────────────────────────


class TestParsePosition:
    def test_normal_position(self):
        assert _parse_position("3/6") == (3, 6)

    def test_first_place(self):
        assert _parse_position("1/8") == (1, 8)

    def test_unseated(self):
        assert _parse_position("U/9") == (None, 9)

    def test_pulled_up(self):
        assert _parse_position("P/15") == (None, 15)

    def test_fell(self):
        assert _parse_position("F/8") == (None, 8)

    def test_empty_string(self):
        assert _parse_position("") == (None, None)

    def test_none(self):
        assert _parse_position(None) == (None, None)

    def test_no_slash(self):
        assert _parse_position("3") == (None, None)

    def test_double_digit(self):
        assert _parse_position("10/12") == (10, 12)


# ── PastRacesJson parsing ────────────────────────────────────────────────────


class TestParsePastRacesJson:
    def test_valid_json(self):
        races = _parse_past_races_json(_sample_past_races_json())
        assert len(races) == 4
        assert races[0].course == "Kempton"
        assert races[0].position == 1
        assert races[0].field_size == 8
        assert races[0].bsp == 5.5
        assert races[0].going == "Good to Soft"
        assert races[0].going_abbr == "GS"
        assert races[0].race_type == "Hurdle"
        assert races[0].distance_yards == 3200
        assert races[0].date == "2026-03-11"

    def test_dnf_position(self):
        races = _parse_past_races_json(_sample_past_races_json())
        assert races[3].position is None  # "U/9"
        assert races[3].field_size == 9

    def test_missing_optional_fields(self):
        """officialRating and other fields can be missing."""
        races = _parse_past_races_json(_sample_past_races_json())
        assert math.isnan(races[1].official_rating)  # Second race has no officialRating
        assert races[0].official_rating == 90.0

    def test_empty_array(self):
        assert _parse_past_races_json("[]") == ()

    def test_null_string(self):
        assert _parse_past_races_json(None) == ()

    def test_empty_string(self):
        assert _parse_past_races_json("") == ()

    def test_malformed_json(self):
        assert _parse_past_races_json("{not json}") == ()

    def test_not_a_list(self):
        assert _parse_past_races_json('{"key": "value"}') == ()


# ── Past race features ───────────────────────────────────────────────────────


class TestPastRaceFeatures:
    def test_all_nan_when_no_history(self):
        meta = _make_runner_meta(past_races=())
        feats = past_race_features(meta, "Kempton")
        assert all(math.isnan(v) for k, v in feats.items()
                   if k not in ("pr_course_runs", "pr_course_wins",
                                "pr_distance_runs", "pr_distance_wins",
                                "pr_going_runs", "pr_going_wins",
                                "pr_runs_count"))

    def test_all_keys_present(self):
        meta = _make_runner_meta(past_races=())
        feats = past_race_features(meta, "Kempton")
        assert set(feats.keys()) == set(PAST_RACE_FEATURE_KEYS)

    def test_course_form(self):
        races = _parse_past_races_json(_sample_past_races_json())
        meta = _make_runner_meta(past_races=races)
        feats = past_race_features(meta, "Kempton")
        assert feats["pr_course_runs"] == 2.0  # Two races at Kempton
        assert feats["pr_course_wins"] == 1.0  # Won 1/8 at Kempton
        assert feats["pr_course_win_rate"] == 0.5

    def test_course_form_case_insensitive(self):
        races = _parse_past_races_json(_sample_past_races_json())
        meta = _make_runner_meta(past_races=races)
        feats = past_race_features(meta, "KEMPTON")
        assert feats["pr_course_runs"] == 2.0

    def test_course_form_no_match(self):
        races = _parse_past_races_json(_sample_past_races_json())
        meta = _make_runner_meta(past_races=races)
        feats = past_race_features(meta, "Ascot")
        assert feats["pr_course_runs"] == 0.0
        assert math.isnan(feats["pr_course_win_rate"])

    def test_distance_form(self):
        races = _parse_past_races_json(_sample_past_races_json())
        meta = _make_runner_meta(past_races=races)
        # 3200 yards — races 0 and 2 are at 3200, race 1 at 3500 (within ±440)
        feats = past_race_features(meta, "Kempton", today_distance_yards=3200)
        assert feats["pr_distance_runs"] == 3.0  # 3200, 3500, 3200 all within ±440yd
        assert feats["pr_distance_wins"] == 1.0  # Only race 0 won

    def test_going_form(self):
        races = _parse_past_races_json(_sample_past_races_json())
        meta = _make_runner_meta(past_races=races)
        feats = past_race_features(meta, "Kempton", today_going_abbr="GS")
        assert feats["pr_going_runs"] == 2.0  # Two races on "GS"
        assert feats["pr_going_wins"] == 1.0

    def test_bsp_features(self):
        races = _parse_past_races_json(_sample_past_races_json())
        meta = _make_runner_meta(past_races=races)
        feats = past_race_features(meta, "Kempton")
        assert not math.isnan(feats["pr_avg_bsp"])
        assert not math.isnan(feats["pr_best_bsp"])
        assert not math.isnan(feats["pr_bsp_trend"])

    def test_best_bsp(self):
        races = _parse_past_races_json(_sample_past_races_json())
        meta = _make_runner_meta(past_races=races)
        feats = past_race_features(meta, "Kempton")
        # BSPs: 5.5, 8.0, 4.0, 12.0 → best (lowest) = 4.0, log_norm'd
        import data.feature_engineer as fe
        assert feats["pr_best_bsp"] == fe.log_norm(4.0)

    def test_avg_position(self):
        races = _parse_past_races_json(_sample_past_races_json())
        meta = _make_runner_meta(past_races=races)
        feats = past_race_features(meta, "Kempton")
        # Positions: 1, 3, 2 (race 4 is DNF) → avg = 2.0
        assert feats["pr_avg_position"] == 2.0
        assert feats["pr_best_position"] == 1.0

    def test_runs_count_and_completion(self):
        races = _parse_past_races_json(_sample_past_races_json())
        meta = _make_runner_meta(past_races=races)
        feats = past_race_features(meta, "Kempton")
        assert feats["pr_runs_count"] == 4.0
        assert feats["pr_completion_rate"] == 0.75  # 3 out of 4 finished

    def test_improving_form_true(self):
        """Last 3 positions descending (improving): 5, 3, 1."""
        races = [
            _make_past_race(position=5, date="2026-03-01"),
            _make_past_race(position=3, date="2026-02-01"),
            _make_past_race(position=1, date="2026-01-01"),
        ]
        meta = _make_runner_meta(past_races=races)
        feats = past_race_features(meta, "Kempton")
        assert feats["pr_improving_form"] == 1.0

    def test_improving_form_false(self):
        """Last 3 positions ascending (worsening): 1, 3, 5."""
        races = [
            _make_past_race(position=1, date="2026-03-01"),
            _make_past_race(position=3, date="2026-02-01"),
            _make_past_race(position=5, date="2026-01-01"),
        ]
        meta = _make_runner_meta(past_races=races)
        feats = past_race_features(meta, "Kempton")
        assert feats["pr_improving_form"] == 0.0

    def test_improving_form_insufficient_data(self):
        """Only 2 completed races — NaN."""
        races = [
            _make_past_race(position=3, date="2026-03-01"),
            _make_past_race(position=1, date="2026-02-01"),
        ]
        meta = _make_runner_meta(past_races=races)
        feats = past_race_features(meta, "Kempton")
        assert math.isnan(feats["pr_improving_form"])

    def test_days_between_runs(self):
        races = [
            _make_past_race(date="2026-03-10"),
            _make_past_race(date="2026-02-10"),
            _make_past_race(date="2026-01-10"),
        ]
        meta = _make_runner_meta(past_races=races)
        feats = past_race_features(meta, "Kempton")
        # Gaps: 28 days, 31 days → avg = 29.5
        assert feats["pr_days_between_runs_avg"] == pytest.approx(29.5, abs=0.5)

    def test_bsp_trend_improving(self):
        """BSP trending down (improving market confidence)."""
        races = [
            _make_past_race(bsp=10.0, date="2026-03-01"),
            _make_past_race(bsp=7.0, date="2026-02-01"),
            _make_past_race(bsp=4.0, date="2026-01-01"),
        ]
        meta = _make_runner_meta(past_races=races)
        feats = past_race_features(meta, "Kempton")
        assert feats["pr_bsp_trend"] < 0  # Negative = improving

    def test_bsp_trend_declining(self):
        """BSP trending up (declining market confidence)."""
        races = [
            _make_past_race(bsp=4.0, date="2026-03-01"),
            _make_past_race(bsp=7.0, date="2026-02-01"),
            _make_past_race(bsp=10.0, date="2026-01-01"),
        ]
        meta = _make_runner_meta(past_races=races)
        feats = past_race_features(meta, "Kempton")
        assert feats["pr_bsp_trend"] > 0  # Positive = declining


# ── Runner meta features — recent_form preference ────────────────────────────


class TestRecentFormPreference:
    def test_recent_form_used_when_available(self):
        meta = _make_runner_meta(form="111111", recent_form="654321")
        feats = runner_meta_features(meta)
        # recent_form "654321" should be used — avg is much higher than "111111"
        assert feats["form_avg_pos"] > 1.5  # Would be 1.0 if form "111111" was used

    def test_fallback_to_form_when_recent_form_empty(self):
        meta = _make_runner_meta(form="111111", recent_form="")
        feats = runner_meta_features(meta)
        assert feats["form_avg_pos"] == 1.0


# ── Extractor merge ──────────────────────────────────────────────────────────


class TestExtractorMerge:
    def test_runners_parquet_has_new_columns(self):
        """Extracted runners Parquet should have the 3 new columns."""
        from data.extractor import RUNNERS_COLUMNS
        assert "timeform_comment" in RUNNERS_COLUMNS
        assert "recent_form" in RUNNERS_COLUMNS
        assert "past_races_json" in RUNNERS_COLUMNS

    def test_runners_columns_count(self):
        from data.extractor import RUNNERS_COLUMNS
        assert len(RUNNERS_COLUMNS) == 40  # 37 original + 3 new


# ── Episode builder backward compatibility ───────────────────────────────────


class TestEpisodeBuilderBackwardCompat:
    def test_runner_meta_has_past_races_field(self):
        meta = _make_runner_meta()
        assert hasattr(meta, "past_races")
        assert meta.past_races == ()

    def test_runner_meta_has_timeform_comment(self):
        meta = _make_runner_meta(timeform_comment="Good horse")
        assert meta.timeform_comment == "Good horse"

    def test_runner_meta_has_recent_form(self):
        meta = _make_runner_meta(recent_form="1234")
        assert meta.recent_form == "1234"

    def test_old_parquet_without_new_columns(self):
        """RunnerMeta should work even if Parquet doesn't have new columns."""
        from data.episode_builder import _build_runner_meta
        # Simulate old Parquet row without new columns
        row = pd.Series({
            "selection_id": 12345,
            "runner_name": "Old Horse",
            "sort_priority": "1",
            "handicap": "0",
            "SIRE_NAME": "", "DAM_NAME": "", "DAMSIRE_NAME": "",
            "BRED": "", "OFFICIAL_RATING": "90", "ADJUSTED_RATING": "88",
            "AGE": "5", "SEX_TYPE": "GELDING", "COLOUR_TYPE": "b",
            "WEIGHT_VALUE": "154", "WEIGHT_UNITS": "lbs",
            "JOCKEY_NAME": "J Smith", "JOCKEY_CLAIM": "",
            "TRAINER_NAME": "A Trainer", "OWNER_NAME": "An Owner",
            "STALL_DRAW": "3", "CLOTH_NUMBER": "5",
            "FORM": "1234", "DAYS_SINCE_LAST_RUN": "14",
            "WEARING": "", "FORECASTPRICE_NUMERATOR": "4",
            "FORECASTPRICE_DENOMINATOR": "1",
            # No past_races_json, timeform_comment, recent_form
        })
        meta = _build_runner_meta(row)
        assert meta.past_races == ()
        assert meta.timeform_comment == ""
        assert meta.recent_form == ""


# ── Environment dimensions ───────────────────────────────────────────────────


class TestEnvDimensions:
    def test_runner_dim(self):
        assert RUNNER_DIM == 110

    def test_runner_keys_count(self):
        assert len(RUNNER_KEYS) == 110

    def test_past_race_keys_in_runner_keys(self):
        for key in PAST_RACE_FEATURE_KEYS:
            assert key in RUNNER_KEYS, f"Missing key: {key}"

    def test_obs_dim(self):
        """obs_dim with max_runners=14 should be 1583."""
        from env.betfair_env import MARKET_DIM, VELOCITY_DIM, AGENT_STATE_DIM
        obs_dim = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * 14) + AGENT_STATE_DIM
        assert obs_dim == 1583

    def test_no_duplicate_runner_keys(self):
        assert len(RUNNER_KEYS) == len(set(RUNNER_KEYS))
