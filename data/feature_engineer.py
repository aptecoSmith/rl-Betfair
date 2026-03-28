"""
data/feature_engineer.py — Derived features for the RL observation vector.

Transforms raw :class:`~data.episode_builder.Tick` /
:class:`~data.episode_builder.Race` objects into numeric feature dicts.
This module owns **all** numeric parsing of ``RunnerMetaData`` string fields
and **all** derived calculations (velocity, implied probability, spreads,
depth, weight of money, form parsing, cross-runner ranks, etc.).

Design principles:

* **Maximum features** — we don't know what the agent will find useful, so
  we expose everything the data can give us.  The agent (or a later feature
  selection step) decides what matters.
* **NaN for missing** — empty strings or unparseable metadata become
  ``float('nan')``.  Downstream code must handle NaN (masking or imputation).
* **No mutation** — functions return new dicts/arrays; input dataclasses are
  never modified (they're frozen anyway).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

from data.episode_builder import (
    Day,
    PastRace,
    PriceSize,
    Race,
    RunnerMeta,
    RunnerSnap,
    Tick,
)

NaN = float("nan")

#: Known race status values from ``RaceStatusEvents``.
RACE_STATUSES: list[str] = [
    "parading", "going down", "going behind",
    "under orders", "at the post", "off",
]


# ── Numeric parsing helpers ──────────────────────────────────────────────────


def safe_float(val: str) -> float:
    """Parse a string to float; return NaN on empty / unparseable."""
    if val is None or val == "":
        return NaN
    try:
        return float(val)
    except (ValueError, TypeError):
        return NaN


def safe_int(val: str) -> float:
    """Parse a string to int-as-float; return NaN on empty / unparseable.

    Returns a float (not int) so NaN is representable in the same array.
    """
    if val is None or val == "":
        return NaN
    try:
        return float(int(val))
    except (ValueError, TypeError):
        return NaN


def log_norm(val: float) -> float:
    """Log-normalise a non-negative value: ``ln(1 + val)``."""
    if math.isnan(val) or val < 0:
        return NaN
    return math.log1p(val)


# ── Form parsing ─────────────────────────────────────────────────────────────


def parse_form(form_str: str) -> dict[str, float]:
    """Parse a Betfair form string into numeric features.

    Form strings look like ``"1234-21"`` where the most recent run is
    rightmost, digits are finishing positions, ``-`` is a separator between
    seasons, ``0`` means 10th+, ``F`` fell, ``P`` pulled up, ``U`` unseated,
    ``R`` refused, ``B`` brought down, ``C`` carried out, ``D`` disqualified.

    Returns a dict of features derived from up to the last 6 runs.
    """
    if not form_str or form_str.strip() == "":
        return {
            "form_avg_pos": NaN,
            "form_best_pos": NaN,
            "form_worst_pos": NaN,
            "form_wins": NaN,
            "form_places": NaN,
            "form_runs": NaN,
            "form_completion_rate": NaN,
        }

    # Strip separators, take last 6 characters
    clean = form_str.replace("-", "").replace("/", "")
    recent = clean[-6:] if len(clean) > 6 else clean

    positions: list[int] = []
    completions = 0
    total = len(recent)

    for ch in recent:
        if ch.isdigit():
            pos = int(ch)
            if pos == 0:
                pos = 10  # '0' means 10th or worse
            positions.append(pos)
            completions += 1
        elif ch in ("F", "P", "U", "R", "B", "C", "D"):
            # Did not finish — counts as a run but not a completion
            pass
        # Other characters ignored

    wins = sum(1 for p in positions if p == 1)
    places = sum(1 for p in positions if p <= 3)

    return {
        "form_avg_pos": (sum(positions) / len(positions)) if positions else NaN,
        "form_best_pos": float(min(positions)) if positions else NaN,
        "form_worst_pos": float(max(positions)) if positions else NaN,
        "form_wins": float(wins),
        "form_places": float(places),
        "form_runs": float(total),
        "form_completion_rate": (completions / total) if total > 0 else NaN,
    }


# ── Per-runner metadata features ─────────────────────────────────────────────


def runner_meta_features(meta: RunnerMeta) -> dict[str, float]:
    """Extract all numeric features from a :class:`RunnerMeta`.

    Parses string fields to floats.  Empty strings → NaN.
    """
    feats: dict[str, float] = {}

    # Direct numeric parses
    feats["official_rating"] = safe_float(meta.official_rating)
    feats["adjusted_rating"] = safe_float(meta.adjusted_rating)
    feats["age"] = safe_int(meta.age)
    feats["weight_value"] = safe_float(meta.weight_value)
    feats["jockey_claim"] = safe_float(meta.jockey_claim)
    feats["stall_draw"] = safe_int(meta.stall_draw)
    feats["cloth_number"] = safe_int(meta.cloth_number)
    feats["days_since_last_run"] = safe_float(meta.days_since_last_run)
    feats["handicap"] = safe_float(meta.handicap)
    feats["sort_priority"] = safe_int(meta.sort_priority)

    # Forecast price: numerator / denominator + 1 (fractional → decimal)
    num = safe_float(meta.forecastprice_numerator)
    den = safe_float(meta.forecastprice_denominator)
    if not math.isnan(num) and not math.isnan(den) and den != 0:
        feats["forecast_price"] = num / den + 1.0
    else:
        feats["forecast_price"] = NaN

    # Implied probability from forecast price
    fp = feats["forecast_price"]
    feats["forecast_implied_prob"] = (1.0 / fp) if (not math.isnan(fp) and fp > 0) else NaN

    # Sex type encoding (one-hot-ish as float)
    sex = meta.sex_type.upper().strip() if meta.sex_type else ""
    for st in ("MARE", "GELDING", "COLT", "FILLY", "HORSE", "RIG"):
        feats[f"sex_{st.lower()}"] = 1.0 if sex == st else 0.0

    # Equipment (wearing) flags
    wearing = meta.wearing.upper().strip() if meta.wearing else ""
    for equip in ("BLINKERS", "VISOR", "CHEEKPIECES", "TONGUE TIE", "HOOD"):
        feats[f"equip_{equip.lower().replace(' ', '_')}"] = (
            1.0 if equip in wearing else 0.0
        )
    feats["has_equipment"] = 1.0 if wearing != "" else 0.0

    # Form features — prefer recent_form from RaceCardRunners (Session 2.7b)
    form_str = meta.recent_form if meta.recent_form else meta.form
    feats.update(parse_form(form_str))

    return feats


# ── Past race features (Session 2.7b) ────────────────────────────────────────

#: All feature keys produced by :func:`past_race_features`.
PAST_RACE_FEATURE_KEYS: list[str] = [
    "pr_course_runs", "pr_course_wins", "pr_course_win_rate",
    "pr_distance_runs", "pr_distance_wins",
    "pr_going_runs", "pr_going_wins", "pr_going_win_rate",
    "pr_avg_bsp", "pr_best_bsp", "pr_bsp_trend",
    "pr_avg_position", "pr_best_position",
    "pr_runs_count", "pr_completion_rate", "pr_improving_form",
    "pr_days_between_runs_avg",
]

#: Distance tolerance for "similar distance" matching (±2 furlongs ≈ 440 yards).
_DISTANCE_TOLERANCE_YARDS = 440


def past_race_features(
    meta: RunnerMeta,
    venue: str,
    today_distance_yards: int = 0,
    today_going_abbr: str = "",
) -> dict[str, float]:
    """Derive features from a runner's past race history.

    Parameters
    ----------
    meta:
        Runner metadata (must have ``past_races`` populated from Session 2.7b).
    venue:
        Today's venue name for course-form matching (case-insensitive).
    today_distance_yards:
        Today's race distance in yards for distance matching.  If 0,
        distance features count all races (no filtering).
    today_going_abbr:
        Today's going abbreviation for going-form matching.  If empty,
        going features count all races (no filtering).

    Returns
    -------
    dict[str, float]
        17 features, all NaN when ``meta.past_races`` is empty.
    """
    races = meta.past_races
    if not races:
        return {k: NaN for k in PAST_RACE_FEATURE_KEYS}

    venue_lower = venue.lower().strip() if venue else ""

    # ── Course form ──────────────────────────────────────────────────────
    course_races = [r for r in races if r.course.lower().strip() == venue_lower] if venue_lower else []
    course_wins = sum(1 for r in course_races if r.position == 1)
    pr_course_runs = float(len(course_races))
    pr_course_wins = float(course_wins)
    pr_course_win_rate = (course_wins / len(course_races)) if course_races else NaN

    # ── Distance form ────────────────────────────────────────────────────
    if today_distance_yards > 0:
        dist_races = [
            r for r in races
            if abs(r.distance_yards - today_distance_yards) <= _DISTANCE_TOLERANCE_YARDS
        ]
    else:
        dist_races = list(races)
    pr_distance_runs = float(len(dist_races))
    pr_distance_wins = float(sum(1 for r in dist_races if r.position == 1))

    # ── Going form ───────────────────────────────────────────────────────
    if today_going_abbr:
        going_lower = today_going_abbr.lower().strip()
        going_races = [r for r in races if r.going_abbr.lower().strip() == going_lower]
    else:
        going_races = []
    going_wins = sum(1 for r in going_races if r.position == 1)
    pr_going_runs = float(len(going_races))
    pr_going_wins = float(going_wins)
    pr_going_win_rate = (going_wins / len(going_races)) if going_races else NaN

    # ── BSP features ─────────────────────────────────────────────────────
    bsps = [r.bsp for r in races if not math.isnan(r.bsp) and r.bsp > 0]
    if bsps:
        pr_avg_bsp = log_norm(sum(bsps) / len(bsps))
        pr_best_bsp = log_norm(min(bsps))
        # BSP trend: linear slope over recent races (oldest first in JSON)
        if len(bsps) >= 2:
            n = len(bsps)
            x_mean = (n - 1) / 2.0
            y_mean = sum(bsps) / n
            num = sum((i - x_mean) * (b - y_mean) for i, b in enumerate(bsps))
            den = sum((i - x_mean) ** 2 for i in range(n))
            pr_bsp_trend = (num / den) if den != 0 else 0.0
        else:
            pr_bsp_trend = 0.0
    else:
        pr_avg_bsp = NaN
        pr_best_bsp = NaN
        pr_bsp_trend = NaN

    # ── Performance features ─────────────────────────────────────────────
    positions = [r.position for r in races if r.position is not None]
    if positions:
        pr_avg_position = sum(positions) / len(positions)
        pr_best_position = float(min(positions))
    else:
        pr_avg_position = NaN
        pr_best_position = NaN

    pr_runs_count = float(len(races))
    completed = sum(1 for r in races if r.position is not None)
    pr_completion_rate = (completed / len(races)) if races else NaN

    # Improving form: last 3 completed positions trending downward (getting better)
    recent_positions = [r.position for r in races if r.position is not None][:3]
    if len(recent_positions) >= 3:
        # All descending means improving (lower pos = better)
        pr_improving_form = 1.0 if (
            recent_positions[0] >= recent_positions[1] >= recent_positions[2]
            and recent_positions[0] > recent_positions[2]
        ) else 0.0
    else:
        pr_improving_form = NaN

    # ── Days between runs ────────────────────────────────────────────────
    dates = [r.date for r in races if r.date]
    if len(dates) >= 2:
        from datetime import datetime as dt
        gaps: list[float] = []
        for i in range(len(dates) - 1):
            try:
                d1 = dt.strptime(dates[i][:10], "%Y-%m-%d")
                d2 = dt.strptime(dates[i + 1][:10], "%Y-%m-%d")
                gap = abs((d1 - d2).days)
                if gap > 0:
                    gaps.append(float(gap))
            except (ValueError, TypeError):
                continue
        pr_days_between_runs_avg = (sum(gaps) / len(gaps)) if gaps else NaN
    else:
        pr_days_between_runs_avg = NaN

    return {
        "pr_course_runs": pr_course_runs,
        "pr_course_wins": pr_course_wins,
        "pr_course_win_rate": pr_course_win_rate,
        "pr_distance_runs": pr_distance_runs,
        "pr_distance_wins": pr_distance_wins,
        "pr_going_runs": pr_going_runs,
        "pr_going_wins": pr_going_wins,
        "pr_going_win_rate": pr_going_win_rate,
        "pr_avg_bsp": pr_avg_bsp,
        "pr_best_bsp": pr_best_bsp,
        "pr_bsp_trend": pr_bsp_trend,
        "pr_avg_position": pr_avg_position,
        "pr_best_position": pr_best_position,
        "pr_runs_count": pr_runs_count,
        "pr_completion_rate": pr_completion_rate,
        "pr_improving_form": pr_improving_form,
        "pr_days_between_runs_avg": pr_days_between_runs_avg,
    }


# ── Per-runner tick features (from order book snapshot) ──────────────────────


def runner_tick_features(snap: RunnerSnap) -> dict[str, float]:
    """Derive features from one runner's order book snapshot.

    Covers prices, spreads, depth, weight of money, LTP, BSP, etc.
    """
    feats: dict[str, float] = {}

    # LTP and implied probability
    ltp = snap.last_traded_price
    feats["ltp"] = ltp if ltp > 0 else NaN
    feats["implied_prob"] = (1.0 / ltp) if ltp > 0 else NaN

    # Volume
    feats["runner_total_matched"] = snap.total_matched
    feats["runner_total_matched_log"] = log_norm(snap.total_matched)

    # BSP indicators
    feats["spn"] = snap.starting_price_near if snap.starting_price_near > 0 else NaN
    feats["spf"] = snap.starting_price_far if snap.starting_price_far > 0 else NaN
    feats["bsp"] = snap.bsp if snap.bsp is not None and snap.bsp > 0 else NaN

    # Adjustment factor
    feats["adjustment_factor"] = snap.adjustment_factor if snap.adjustment_factor is not None else NaN

    # Status encoding
    feats["is_active"] = 1.0 if snap.status == "ACTIVE" else 0.0
    feats["is_removed"] = 1.0 if snap.status == "REMOVED" else 0.0

    # Back prices (best 3 levels)
    backs = snap.available_to_back
    for i in range(3):
        if i < len(backs):
            feats[f"back_price_{i+1}"] = backs[i].price
            feats[f"back_size_{i+1}"] = backs[i].size
            feats[f"back_size_{i+1}_log"] = log_norm(backs[i].size)
        else:
            feats[f"back_price_{i+1}"] = NaN
            feats[f"back_size_{i+1}"] = NaN
            feats[f"back_size_{i+1}_log"] = NaN

    # Lay prices (best 3 levels)
    lays = snap.available_to_lay
    for i in range(3):
        if i < len(lays):
            feats[f"lay_price_{i+1}"] = lays[i].price
            feats[f"lay_size_{i+1}"] = lays[i].size
            feats[f"lay_size_{i+1}_log"] = log_norm(lays[i].size)
        else:
            feats[f"lay_price_{i+1}"] = NaN
            feats[f"lay_size_{i+1}"] = NaN
            feats[f"lay_size_{i+1}_log"] = NaN

    # Spread (best lay - best back)
    bb = feats["back_price_1"]
    bl = feats["lay_price_1"]
    if not math.isnan(bb) and not math.isnan(bl):
        feats["spread"] = bl - bb
        feats["spread_pct"] = (bl - bb) / bb if bb > 0 else NaN
    else:
        feats["spread"] = NaN
        feats["spread_pct"] = NaN

    # Midpoint price
    if not math.isnan(bb) and not math.isnan(bl):
        feats["mid_price"] = (bb + bl) / 2.0
    else:
        feats["mid_price"] = NaN

    # Book depth (total available volume at all 3 levels)
    back_depth = sum(
        backs[i].size for i in range(min(3, len(backs)))
    )
    lay_depth = sum(
        lays[i].size for i in range(min(3, len(lays)))
    )
    feats["back_depth"] = back_depth
    feats["lay_depth"] = lay_depth
    feats["back_depth_log"] = log_norm(back_depth)
    feats["lay_depth_log"] = log_norm(lay_depth)
    feats["total_depth"] = back_depth + lay_depth
    feats["total_depth_log"] = log_norm(back_depth + lay_depth)

    # Weight of money (proportion on back side vs total)
    total = back_depth + lay_depth
    if total > 0:
        feats["weight_of_money"] = back_depth / total
    else:
        feats["weight_of_money"] = NaN

    return feats


# ── Market-level features (per tick) ─────────────────────────────────────────


def market_tick_features(tick: Tick) -> dict[str, float]:
    """Derive market-level features from a single tick.

    Covers overround, total volume, runner count, weather, and
    time-to-off calculations.
    """
    feats: dict[str, float] = {}

    # Time to scheduled off (seconds)
    if tick.market_start_time and tick.timestamp:
        delta = (tick.market_start_time - tick.timestamp).total_seconds()
        feats["time_to_off_seconds"] = delta
        # Normalised: 0 = at the off, 1 = 30 min before
        feats["time_to_off_norm"] = min(max(delta / 1800.0, 0.0), 1.0)
    else:
        feats["time_to_off_seconds"] = NaN
        feats["time_to_off_norm"] = NaN

    # Market volume
    feats["market_traded_volume"] = tick.traded_volume
    feats["market_traded_volume_log"] = log_norm(tick.traded_volume)

    # Active runners
    feats["num_active_runners"] = float(tick.number_of_active_runners or 0)

    # Overround (sum of implied probabilities from best back prices)
    overround = 0.0
    n_priced = 0
    for runner in tick.runners:
        if runner.available_to_back and runner.status == "ACTIVE":
            best_back = runner.available_to_back[0].price
            if best_back > 0:
                overround += 1.0 / best_back
                n_priced += 1
    feats["overround"] = overround if n_priced > 0 else NaN
    feats["overround_pct"] = (overround - 1.0) * 100.0 if n_priced > 0 else NaN
    feats["n_priced_runners"] = float(n_priced)

    # Overround from LTP
    ltp_overround = 0.0
    n_ltp = 0
    for runner in tick.runners:
        if runner.last_traded_price > 0 and runner.status == "ACTIVE":
            ltp_overround += 1.0 / runner.last_traded_price
            n_ltp += 1
    feats["ltp_overround"] = ltp_overround if n_ltp > 0 else NaN

    # Favourite LTP (lowest = most fancied)
    active_ltps = [
        r.last_traded_price
        for r in tick.runners
        if r.last_traded_price > 0 and r.status == "ACTIVE"
    ]
    feats["favourite_ltp"] = min(active_ltps) if active_ltps else NaN
    feats["outsider_ltp"] = max(active_ltps) if active_ltps else NaN
    feats["ltp_range"] = (
        (max(active_ltps) - min(active_ltps)) if len(active_ltps) >= 2 else NaN
    )

    # Total matched volume across all runners
    total_runner_matched = sum(r.total_matched for r in tick.runners)
    feats["total_runner_matched"] = total_runner_matched
    feats["total_runner_matched_log"] = log_norm(total_runner_matched)

    # Total depth across all runners
    total_back_depth = 0.0
    total_lay_depth = 0.0
    for r in tick.runners:
        if r.status == "ACTIVE":
            total_back_depth += sum(ps.size for ps in r.available_to_back[:3])
            total_lay_depth += sum(ps.size for ps in r.available_to_lay[:3])
    feats["market_back_depth"] = total_back_depth
    feats["market_lay_depth"] = total_lay_depth
    feats["market_total_depth"] = total_back_depth + total_lay_depth
    feats["market_total_depth_log"] = log_norm(total_back_depth + total_lay_depth)

    # Average spread across runners
    spreads: list[float] = []
    for r in tick.runners:
        if r.status == "ACTIVE" and r.available_to_back and r.available_to_lay:
            s = r.available_to_lay[0].price - r.available_to_back[0].price
            spreads.append(s)
    feats["avg_spread"] = (sum(spreads) / len(spreads)) if spreads else NaN

    # Weather features (pass through, already numeric)
    feats["temperature"] = tick.temperature if tick.temperature is not None else NaN
    feats["precipitation"] = tick.precipitation if tick.precipitation is not None else NaN
    feats["wind_speed"] = tick.wind_speed if tick.wind_speed is not None else NaN
    feats["wind_direction"] = tick.wind_direction if tick.wind_direction is not None else NaN
    feats["humidity"] = tick.humidity if tick.humidity is not None else NaN
    feats["weather_code"] = float(tick.weather_code) if tick.weather_code is not None else NaN

    # Race status features (Session 2.7a)
    # One-hot encoding for the 6 known statuses
    status = tick.race_status.lower() if tick.race_status else ""
    for s in RACE_STATUSES:
        feats[f"race_status_{s.replace(' ', '_')}"] = 1.0 if status == s else 0.0

    return feats


# ── Cross-runner relative features ───────────────────────────────────────────


def cross_runner_features(
    tick: Tick,
    runner_meta: dict[int, RunnerMeta],
) -> dict[int, dict[str, float]]:
    """Compute features that are relative across runners within a tick.

    Returns a dict keyed by ``selection_id`` → feature dict.
    Features include ranks, gaps to favourite, normalised values.
    """
    result: dict[int, dict[str, float]] = {}

    # Collect active runner data
    active: list[tuple[int, RunnerSnap]] = [
        (r.selection_id, r)
        for r in tick.runners
        if r.status == "ACTIVE"
    ]

    if not active:
        return result

    # LTP values for ranking
    ltps = {sid: snap.last_traded_price for sid, snap in active if snap.last_traded_price > 0}
    # Volume values
    vols = {sid: snap.total_matched for sid, snap in active}
    # Official ratings from metadata
    ratings: dict[int, float] = {}
    for sid, _ in active:
        meta = runner_meta.get(sid)
        if meta:
            r = safe_float(meta.official_rating)
            if not math.isnan(r):
                ratings[sid] = r

    # Ranks (1 = best)
    ltp_sorted = sorted(ltps.items(), key=lambda x: x[1])  # lowest LTP = favourite
    ltp_rank = {sid: rank + 1 for rank, (sid, _) in enumerate(ltp_sorted)}

    vol_sorted = sorted(vols.items(), key=lambda x: x[1], reverse=True)
    vol_rank = {sid: rank + 1 for rank, (sid, _) in enumerate(vol_sorted)}

    rating_sorted = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    rating_rank = {sid: rank + 1 for rank, (sid, _) in enumerate(rating_sorted)}

    # Favourite LTP (minimum)
    fav_ltp = min(ltps.values()) if ltps else NaN
    n_active = len(active)

    # Max rating for normalisation
    max_rating = max(ratings.values()) if ratings else NaN
    min_rating = min(ratings.values()) if ratings else NaN

    # Total volume for proportion
    total_vol = sum(vols.values())

    for sid, snap in active:
        feats: dict[str, float] = {}

        # LTP rank and gap to favourite
        feats["ltp_rank"] = float(ltp_rank.get(sid, NaN))
        feats["ltp_rank_norm"] = (
            float(ltp_rank.get(sid, NaN)) / n_active if n_active > 0 else NaN
        )
        ltp = snap.last_traded_price
        feats["gap_to_favourite"] = (ltp - fav_ltp) if (ltp > 0 and not math.isnan(fav_ltp)) else NaN
        feats["gap_to_favourite_pct"] = (
            (ltp - fav_ltp) / fav_ltp
            if (ltp > 0 and not math.isnan(fav_ltp) and fav_ltp > 0)
            else NaN
        )

        # Volume rank and proportion
        feats["vol_rank"] = float(vol_rank.get(sid, NaN))
        feats["vol_proportion"] = (
            vols.get(sid, 0) / total_vol if total_vol > 0 else NaN
        )

        # Rating rank and normalised rating
        feats["rating_rank"] = float(rating_rank.get(sid, NaN))
        if sid in ratings and not math.isnan(max_rating) and not math.isnan(min_rating):
            if max_rating != min_rating:
                feats["rating_norm"] = (
                    (ratings[sid] - min_rating) / (max_rating - min_rating)
                )
            else:
                feats["rating_norm"] = 0.5
        else:
            feats["rating_norm"] = NaN

        # Implied probability relative to field
        ip = 1.0 / ltp if ltp > 0 else NaN
        feats["implied_prob_relative"] = ip  # raw, for cross-runner comparison

        result[sid] = feats

    return result


# ── Velocity / temporal features ─────────────────────────────────────────────


@dataclass
class TickHistory:
    """Rolling window of recent tick data for velocity calculations.

    Maintains a per-runner LTP and volume history up to ``max_window`` ticks.
    """

    max_window: int = 20
    _ltp_history: dict[int, list[float]] = field(default_factory=dict)
    _vol_history: dict[int, list[float]] = field(default_factory=dict)
    _market_vol_history: list[float] = field(default_factory=list)
    _overround_history: list[float] = field(default_factory=list)
    # Race status tracking (Session 2.7a)
    _last_race_status: str | None = field(default=None)
    _last_status_change_tick: int = field(default=0)
    _tick_counter: int = field(default=0)
    # Timestamp tracking for time delta features (Session 2.8)
    _timestamp_history: list[float] = field(default_factory=list)  # epoch seconds

    def update(self, tick: Tick, market_feats: dict[str, float]) -> None:
        """Record the latest tick's data into the history."""
        # Track race status changes (Session 2.7a)
        self._tick_counter += 1
        current_status = tick.race_status
        if current_status != self._last_race_status:
            self._last_race_status = current_status
            self._last_status_change_tick = self._tick_counter

        # Track timestamps for time delta features (Session 2.8)
        if tick.timestamp is not None:
            self._timestamp_history.append(tick.timestamp.timestamp())
        if len(self._timestamp_history) > self.max_window:
            self._timestamp_history = self._timestamp_history[-self.max_window:]

        for runner in tick.runners:
            sid = runner.selection_id
            if sid not in self._ltp_history:
                self._ltp_history[sid] = []
                self._vol_history[sid] = []
            self._ltp_history[sid].append(runner.last_traded_price)
            self._vol_history[sid].append(runner.total_matched)
            # Trim to window
            if len(self._ltp_history[sid]) > self.max_window:
                self._ltp_history[sid] = self._ltp_history[sid][-self.max_window:]
                self._vol_history[sid] = self._vol_history[sid][-self.max_window:]

        self._market_vol_history.append(tick.traded_volume)
        if len(self._market_vol_history) > self.max_window:
            self._market_vol_history = self._market_vol_history[-self.max_window:]

        ov = market_feats.get("overround", NaN)
        self._overround_history.append(ov)
        if len(self._overround_history) > self.max_window:
            self._overround_history = self._overround_history[-self.max_window:]

    def runner_velocity_features(self, selection_id: int) -> dict[str, float]:
        """Compute velocity / momentum features for one runner.

        Returns features for multiple lookback windows (3, 5, 10 ticks).
        """
        feats: dict[str, float] = {}
        ltps = self._ltp_history.get(selection_id, [])
        vols = self._vol_history.get(selection_id, [])

        for window in (3, 5, 10):
            suffix = f"_{window}"
            if len(ltps) >= window:
                feats[f"ltp_velocity{suffix}"] = ltps[-1] - ltps[-window]
                feats[f"ltp_pct_change{suffix}"] = (
                    (ltps[-1] - ltps[-window]) / ltps[-window]
                    if ltps[-window] > 0
                    else NaN
                )
            else:
                feats[f"ltp_velocity{suffix}"] = NaN
                feats[f"ltp_pct_change{suffix}"] = NaN

            if len(vols) >= window:
                feats[f"vol_delta{suffix}"] = vols[-1] - vols[-window]
                feats[f"vol_delta{suffix}_log"] = log_norm(max(0, vols[-1] - vols[-window]))
            else:
                feats[f"vol_delta{suffix}"] = NaN
                feats[f"vol_delta{suffix}_log"] = NaN

        # Price volatility (std of last N LTPs)
        for window in (5, 10):
            suffix = f"_{window}"
            if len(ltps) >= window:
                recent = ltps[-window:]
                mean = sum(recent) / len(recent)
                var = sum((x - mean) ** 2 for x in recent) / len(recent)
                feats[f"ltp_volatility{suffix}"] = math.sqrt(var)
            else:
                feats[f"ltp_volatility{suffix}"] = NaN

        # Tick count (how many ticks we've seen for this runner)
        feats["tick_count"] = float(len(ltps))

        return feats

    def market_velocity_features(self) -> dict[str, float]:
        """Compute market-level velocity features."""
        feats: dict[str, float] = {}
        vols = self._market_vol_history
        ovs = self._overround_history

        for window in (3, 5, 10):
            suffix = f"_{window}"
            if len(vols) >= window:
                feats[f"market_vol_delta{suffix}"] = vols[-1] - vols[-window]
            else:
                feats[f"market_vol_delta{suffix}"] = NaN

            valid_ovs = [o for o in ovs[-window:] if not math.isnan(o)]
            if len(valid_ovs) >= 2:
                feats[f"overround_delta{suffix}"] = valid_ovs[-1] - valid_ovs[0]
            else:
                feats[f"overround_delta{suffix}"] = NaN

        # Time since last status change (Session 2.7a)
        # Normalised by assuming ~5s ticks: ticks_elapsed * 5 / 1800 (30 min)
        ticks_since = self._tick_counter - self._last_status_change_tick
        feats["time_since_status_change"] = min(ticks_since * 5.0 / 1800.0, 1.0)

        # Time delta features (Session 2.8)
        # seconds_since_last_tick: 0 for first tick, actual delta otherwise
        # Normalised: delta / 300 (5 min max expected gap), clamped to [0, 1]
        ts = self._timestamp_history
        if len(ts) >= 2:
            feats["seconds_since_last_tick"] = min((ts[-1] - ts[-2]) / 300.0, 1.0)
        else:
            feats["seconds_since_last_tick"] = 0.0

        # seconds_spanned_last_N_ticks: wall-clock time covered by velocity window
        # Normalised: span / (N * 60) — assumes ~60s max per tick gap, clamped [0, 1]
        for window in (3, 5, 10):
            key = f"seconds_spanned_{window}"
            if len(ts) >= window:
                span = ts[-1] - ts[-window]
                feats[key] = min(span / (window * 60.0), 1.0)
            else:
                feats[key] = 0.0

        return feats

    def reset(self) -> None:
        """Clear all history (call between races if needed)."""
        self._ltp_history.clear()
        self._vol_history.clear()
        self._market_vol_history.clear()
        self._overround_history.clear()
        self._last_race_status = None
        self._last_status_change_tick = 0
        self._tick_counter = 0
        self._timestamp_history.clear()


# ── High-level feature assembly ──────────────────────────────────────────────


def engineer_tick(
    tick: Tick,
    race: Race,
    tick_history: TickHistory,
) -> dict[str, object]:
    """Compute ALL features for a single tick.

    Returns a dict with:

    * ``"market"`` → dict of market-level features
    * ``"runners"`` → dict of selection_id → dict of per-runner features
      (tick features + metadata features + cross-runner features + velocity)
    * ``"market_velocity"`` → dict of market velocity features

    This function also updates ``tick_history`` with the current tick.
    """
    # Market features
    mkt = market_tick_features(tick)

    # Cross-runner features
    cross = cross_runner_features(tick, race.runner_metadata)

    # Per-runner features
    runners_out: dict[int, dict[str, float]] = {}
    for snap in tick.runners:
        sid = snap.selection_id
        feats = runner_tick_features(snap)

        # Add metadata features
        meta = race.runner_metadata.get(sid)
        if meta:
            feats.update(runner_meta_features(meta))
            feats.update(past_race_features(meta, race.venue))

        # Add cross-runner features
        if sid in cross:
            feats.update(cross[sid])

        # Add velocity features
        feats.update(tick_history.runner_velocity_features(sid))

        runners_out[sid] = feats

    # Update history (after reading velocity — velocity uses prior state)
    tick_history.update(tick, mkt)

    # Market velocity
    mkt_vel = tick_history.market_velocity_features()

    return {
        "market": mkt,
        "runners": runners_out,
        "market_velocity": mkt_vel,
    }


def engineer_race(race: Race) -> list[dict[str, object]]:
    """Compute features for every tick in a race.

    Creates a fresh :class:`TickHistory` per race, so velocity features
    start from zero at the beginning of each race.

    Returns a list of feature dicts (one per tick, in tick order).
    """
    history = TickHistory()
    results: list[dict[str, object]] = []
    for tick in race.ticks:
        results.append(engineer_tick(tick, race, history))
    return results


def engineer_day(day: Day) -> list[list[dict[str, object]]]:
    """Compute features for every tick in every race of a day.

    Returns a nested list: ``day[race_idx][tick_idx] → feature_dict``.
    """
    results = []
    for race in day.races:
        t0 = time.perf_counter()
        feats = engineer_race(race)
        elapsed = time.perf_counter() - t0
        logger.debug(
            "  Race %s: %d ticks engineered in %.3fs",
            race.market_id, len(race.ticks), elapsed,
        )
        results.append(feats)
    return results
