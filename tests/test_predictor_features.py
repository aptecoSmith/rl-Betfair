"""Unit tests for data/predictor_features.py.

Verifies F2 aggregate computation against synthetic past_races
fixtures. The aggregator is the consumer-side replacement for
the predictor repo's `add_f2_aggregates`; cross-checks against the
predictor's training output are not expected to match exactly
(the predictor's training data uses Betfair's `placed_selection_ids`
labels; rl-betfair's `past_races` come from Timeform's position
field) but the SHAPE of the aggregates must match.

Tests in this file are pure-function — no env, no parquet IO.
"""

from __future__ import annotations

import math
import dataclasses
from datetime import date

from data.episode_builder import PastRace, RunnerMeta
from data.predictor_features import (
    F2_AGGREGATE_KEYS,
    _is_placed,
    compute_f2_aggregates,
    compute_f2_aggregates_for_runners,
)


def _empty_meta(sid: int = 1) -> RunnerMeta:
    """Build a RunnerMeta with all string fields blank + zero past_races."""
    return RunnerMeta(
        selection_id=sid,
        runner_name="",
        sort_priority="",
        handicap="",
        sire_name="",
        dam_name="",
        damsire_name="",
        bred="",
        official_rating="",
        adjusted_rating="",
        age="",
        sex_type="",
        colour_type="",
        weight_value="",
        weight_units="",
        jockey_name="",
        jockey_claim="",
        trainer_name="",
        owner_name="",
        stall_draw="",
        cloth_number="",
        form="",
        days_since_last_run="",
        wearing="",
        forecastprice_numerator="",
        forecastprice_denominator="",
    )


def _meta_with_past(
    past: tuple[PastRace, ...], sid: int = 1
) -> RunnerMeta:
    return dataclasses.replace(_empty_meta(sid), past_races=past)


def _past(
    iso_date: str,
    position: int | None,
    field_size: int,
) -> PastRace:
    return PastRace(
        date=iso_date,
        course="",
        distance_yards=1320,
        going="Good",
        going_abbr="G",
        bsp=float("nan"),
        ip_max=float("nan"),
        ip_min=float("nan"),
        race_type="Flat",
        jockey="",
        official_rating=float("nan"),
        position=position,
        field_size=field_size,
    )


# ─── Output contract ─────────────────────────────────────────────────────────


def test_returns_all_six_aggregate_keys():
    aggs = compute_f2_aggregates(_empty_meta(), as_of_date=date(2026, 4, 23))
    assert set(aggs) == set(F2_AGGREGATE_KEYS)
    for v in aggs.values():
        assert isinstance(v, float)


def test_empty_past_races_returns_zeroes_and_nans():
    """Rookie runner: no past races → counts are 0, rates are NaN,
    days-since is NaN."""
    aggs = compute_f2_aggregates(_empty_meta(), as_of_date=date(2026, 4, 23))
    assert aggs["prior_runs"] == 0.0
    assert aggs["prior_wins"] == 0.0
    assert aggs["prior_places"] == 0.0
    assert math.isnan(aggs["prior_win_rate"])
    assert math.isnan(aggs["prior_place_rate"])
    assert math.isnan(aggs["days_since_prior_run"])


# ─── Counting ────────────────────────────────────────────────────────────────


def test_counts_wins_and_places():
    meta = _meta_with_past((
        _past("2026-04-01", position=1, field_size=10),  # win
        _past("2026-04-10", position=2, field_size=10),  # placed (top 3 of 10)
        _past("2026-04-15", position=8, field_size=10),  # not placed
    ))
    aggs = compute_f2_aggregates(meta, as_of_date=date(2026, 4, 23))
    assert aggs["prior_runs"] == 3.0
    assert aggs["prior_wins"] == 1.0
    assert aggs["prior_places"] == 2.0  # win + 2nd both count as placed
    assert aggs["prior_win_rate"] == 1 / 3
    assert aggs["prior_place_rate"] == 2 / 3


def test_strict_as_of_date_filter_excludes_same_day():
    """Hard_constraints §1 (no leakage): a past_race with date ==
    as_of_date is excluded."""
    meta = _meta_with_past((
        _past("2026-04-23", position=1, field_size=10),  # SAME DAY as as_of
        _past("2026-04-22", position=2, field_size=10),  # day before
    ))
    aggs = compute_f2_aggregates(meta, as_of_date=date(2026, 4, 23))
    assert aggs["prior_runs"] == 1.0
    assert aggs["prior_wins"] == 0.0
    assert aggs["prior_places"] == 1.0


def test_dnf_excluded_from_wins_but_counted_in_runs():
    """Position=None (DNF) is a real run but NOT a win or place.
    Place-rate denominator excludes DNFs since their place status
    is unknown."""
    meta = _meta_with_past((
        _past("2026-04-01", position=None, field_size=10),  # DNF
        _past("2026-04-05", position=1, field_size=10),  # win
    ))
    aggs = compute_f2_aggregates(meta, as_of_date=date(2026, 4, 23))
    assert aggs["prior_runs"] == 2.0
    assert aggs["prior_wins"] == 1.0
    assert aggs["prior_places"] == 1.0
    assert aggs["prior_win_rate"] == 0.5
    # Place rate denominator: only the non-DNF row contributes.
    assert aggs["prior_place_rate"] == 1.0


def test_days_since_picks_most_recent_prior():
    meta = _meta_with_past((
        _past("2026-04-01", position=5, field_size=10),
        _past("2026-04-22", position=3, field_size=10),  # most recent prior
        _past("2026-03-15", position=8, field_size=10),  # older
    ))
    aggs = compute_f2_aggregates(meta, as_of_date=date(2026, 4, 23))
    # 23 - 22 = 1 day
    assert aggs["days_since_prior_run"] == 1.0


def test_days_since_unaffected_by_same_day_or_future():
    """Same-day race excluded from prior aggregates → days_since
    falls back to the next-most-recent strictly-prior date."""
    meta = _meta_with_past((
        _past("2026-04-23", position=1, field_size=10),  # same day, excluded
        _past("2026-04-20", position=2, field_size=10),  # 3 days prior
    ))
    aggs = compute_f2_aggregates(meta, as_of_date=date(2026, 4, 23))
    assert aggs["days_since_prior_run"] == 3.0


# ─── Place-count semantics ──────────────────────────────────────────────────


def test_place_count_5_to_7_runners_pays_two():
    # Field of 6: positions 1+2 are placed; position 3 is not.
    assert _is_placed(_past("2026-04-01", position=1, field_size=6)) is True
    assert _is_placed(_past("2026-04-01", position=2, field_size=6)) is True
    assert _is_placed(_past("2026-04-01", position=3, field_size=6)) is False


def test_place_count_8_to_15_runners_pays_three():
    assert _is_placed(_past("2026-04-01", position=3, field_size=10)) is True
    assert _is_placed(_past("2026-04-01", position=4, field_size=10)) is False


def test_place_count_16_plus_runners_pays_four():
    assert _is_placed(_past("2026-04-01", position=4, field_size=20)) is True
    assert _is_placed(_past("2026-04-01", position=5, field_size=20)) is False


def test_place_excluded_when_field_size_too_small():
    """Field_size < 5 is non-EW; place-rate denominator excludes."""
    assert _is_placed(_past("2026-04-01", position=1, field_size=4)) is None
    assert _is_placed(_past("2026-04-01", position=2, field_size=None)) is None


def test_place_excluded_when_position_unknown():
    assert _is_placed(_past("2026-04-01", position=None, field_size=10)) is None


# ─── Convenience wrapper ────────────────────────────────────────────────────


def test_compute_for_runners_keys_by_selection_id():
    metas = [
        _meta_with_past(
            (_past("2026-04-01", position=1, field_size=10),), sid=101,
        ),
        _meta_with_past(
            (_past("2026-04-02", position=5, field_size=10),), sid=202,
        ),
    ]
    out = compute_f2_aggregates_for_runners(metas, as_of_date=date(2026, 4, 23))
    assert set(out.keys()) == {101, 202}
    assert out[101]["prior_wins"] == 1.0
    assert out[202]["prior_wins"] == 0.0


# ─── Iso-date parsing tolerance ─────────────────────────────────────────────


def test_iso_full_timestamp_parses():
    """PastRace.date may be a full ISO timestamp; only the date prefix
    is load-bearing."""
    meta = _meta_with_past((
        _past("2026-04-01T17:45:00Z", position=1, field_size=10),
    ))
    aggs = compute_f2_aggregates(meta, as_of_date=date(2026, 4, 23))
    assert aggs["prior_runs"] == 1.0
    assert aggs["prior_wins"] == 1.0


def test_malformed_date_skipped_silently():
    meta = _meta_with_past((
        _past("not-a-date", position=1, field_size=10),
        _past("2026-04-01", position=2, field_size=10),
    ))
    aggs = compute_f2_aggregates(meta, as_of_date=date(2026, 4, 23))
    # Only the well-formed row is counted
    assert aggs["prior_runs"] == 1.0
