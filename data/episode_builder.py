"""
data/episode_builder.py — Parquet → Episode objects.

Loads a day's ticks and runners Parquet files, parses the raw ``snap_json``
into structured per-runner order book snapshots, groups ticks into races
(by ``market_id``, ordered by ``sequence_number``), and assembles the typed
``Day → [Race → [Tick]]`` hierarchy used by the Gymnasium environment.

Usage::

    from data.episode_builder import load_day, load_days

    day = load_day("2026-03-26", data_dir="data/processed")
    days = load_days(["2026-03-26", "2026-03-27"], data_dir="data/processed")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from training.progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)

# ── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PriceSize:
    """A single level in the order book ladder."""

    price: float
    size: float


@dataclass(frozen=True, slots=True)
class RunnerSnap:
    """One runner's order book state at a single tick.

    Parsed from the ``snap_json`` column in the ticks Parquet.
    """

    selection_id: int
    status: str  # ACTIVE, WINNER, LOSER, REMOVED, PLACED
    last_traded_price: float
    total_matched: float
    starting_price_near: float
    starting_price_far: float
    adjustment_factor: float | None
    bsp: float | None
    sort_priority: int | None
    removal_date: str | None
    available_to_back: list[PriceSize]
    available_to_lay: list[PriceSize]


@dataclass(frozen=True, slots=True)
class RunnerMeta:
    """Static metadata for a runner in a specific market.

    All ``RunnerMetaData`` fields are kept as raw strings — numeric parsing
    is deferred to :mod:`data.feature_engineer`.
    """

    selection_id: int
    runner_name: str
    sort_priority: str
    handicap: str
    sire_name: str
    dam_name: str
    damsire_name: str
    bred: str
    official_rating: str
    adjusted_rating: str
    age: str
    sex_type: str
    colour_type: str
    weight_value: str
    weight_units: str
    jockey_name: str
    jockey_claim: str
    trainer_name: str
    owner_name: str
    stall_draw: str
    cloth_number: str
    form: str
    days_since_last_run: str
    wearing: str
    forecastprice_numerator: str
    forecastprice_denominator: str


@dataclass(frozen=True, slots=True)
class Tick:
    """A single pre-race market snapshot.

    Contains both market-level fields (from the ticks Parquet row) and
    the per-runner order book (parsed from ``snap_json``).
    """

    market_id: str
    timestamp: datetime
    sequence_number: int
    venue: str
    market_start_time: datetime
    number_of_active_runners: int | None
    traded_volume: float
    in_play: bool
    winner_selection_id: int | None
    # Weather (may be None if fetch failed)
    temperature: float | None
    precipitation: float | None
    wind_speed: float | None
    wind_direction: float | None
    humidity: float | None
    weather_code: int | None
    # Per-runner order book
    runners: list[RunnerSnap]


@dataclass(slots=True)
class Race:
    """All ticks for a single market (race), plus runner metadata."""

    market_id: str
    venue: str
    market_start_time: datetime
    winner_selection_id: int | None
    ticks: list[Tick]
    runner_metadata: dict[int, RunnerMeta]  # keyed by selection_id


@dataclass(slots=True)
class Day:
    """One full racing day — the RL episode unit."""

    date: str  # "YYYY-MM-DD"
    races: list[Race]


# ── snap_json parsing ────────────────────────────────────────────────────────


def _parse_price_sizes(raw: list[dict] | None) -> list[PriceSize]:
    """Parse a list of ``{Price, Size}`` dicts from the snap JSON.

    Returns up to 3 levels sorted by best price (level 1 first).
    Missing or null entries produce an empty list.
    """
    if not raw:
        return []
    result = []
    for entry in raw:
        price = entry.get("Price") or entry.get("price")
        size = entry.get("Size") or entry.get("size")
        if price is not None and size is not None:
            result.append(PriceSize(price=float(price), size=float(size)))
    return result


def parse_snap_json(json_str: str) -> list[RunnerSnap]:
    """Parse a ``snap_json`` string into a list of :class:`RunnerSnap`.

    The JSON is expected to have a top-level ``Runners`` array.  Each runner
    object contains PascalCase keys matching the C# EF Core model names from
    StreamRecorder1 (see DATABASE_SCHEMA.md).

    Expected structure::

        {
          "Runners": [
            {
              "SelectionId": 12345678,
              "Status": "ACTIVE",
              "LastTradedPrice": 4.5,
              "TotalMatched": 1234.56,
              "StartingPriceNear": 4.2,
              "StartingPriceFar": 5.0,
              "AdjustmentFactor": 100.0,
              "Bsp": null,
              "SortPriority": 1,
              "RemovalDate": null,
              "AvailableToBack": [
                {"Price": 4.5, "Size": 100.0}, ...
              ],
              "AvailableToLay": [
                {"Price": 4.6, "Size": 150.0}, ...
              ]
            }
          ]
        }
    """
    data = json.loads(json_str)
    runners_raw = data.get("Runners") or data.get("runners") or []
    result: list[RunnerSnap] = []
    for r in runners_raw:
        # Support both PascalCase and camelCase keys
        selection_id = r.get("SelectionId") or r.get("selectionId") or r.get("id") or 0
        result.append(
            RunnerSnap(
                selection_id=int(selection_id),
                status=str(r.get("Status") or r.get("status") or "ACTIVE"),
                last_traded_price=float(r.get("LastTradedPrice") or r.get("ltp") or 0.0),
                total_matched=float(r.get("TotalMatched") or r.get("totalMatched") or r.get("tv") or 0.0),
                starting_price_near=float(r.get("StartingPriceNear") or r.get("spn") or 0.0),
                starting_price_far=float(r.get("StartingPriceFar") or r.get("spf") or 0.0),
                adjustment_factor=_opt_float(r.get("AdjustmentFactor") or r.get("adjustmentFactor")),
                bsp=_opt_float(r.get("Bsp") or r.get("bsp")),
                sort_priority=_opt_int(r.get("SortPriority") or r.get("sortPriority")),
                removal_date=r.get("RemovalDate") or r.get("removalDate"),
                available_to_back=_parse_price_sizes(
                    r.get("AvailableToBack") or r.get("atb")
                ),
                available_to_lay=_parse_price_sizes(
                    r.get("AvailableToLay") or r.get("atl")
                ),
            )
        )
    return result


def _opt_float(val: object) -> float | None:
    """Convert to float if not None/empty, else return None."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (ValueError, TypeError):
        pass
    if val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _opt_int(val: object) -> int | None:
    """Convert to int if not None/empty, else return None."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (ValueError, TypeError):
        pass
    if val == "":
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


# ── Runner metadata parsing ──────────────────────────────────────────────────


def _build_runner_meta(row: pd.Series) -> RunnerMeta:
    """Build a :class:`RunnerMeta` from one row of the runners DataFrame."""
    def s(col: str) -> str:
        """Extract a string value, replacing NaN/None with empty string."""
        v = row.get(col, "")
        if pd.isna(v):
            return ""
        return str(v)

    return RunnerMeta(
        selection_id=int(row["selection_id"]),
        runner_name=s("runner_name"),
        sort_priority=s("sort_priority"),
        handicap=s("handicap"),
        sire_name=s("SIRE_NAME"),
        dam_name=s("DAM_NAME"),
        damsire_name=s("DAMSIRE_NAME"),
        bred=s("BRED"),
        official_rating=s("OFFICIAL_RATING"),
        adjusted_rating=s("ADJUSTED_RATING"),
        age=s("AGE"),
        sex_type=s("SEX_TYPE"),
        colour_type=s("COLOUR_TYPE"),
        weight_value=s("WEIGHT_VALUE"),
        weight_units=s("WEIGHT_UNITS"),
        jockey_name=s("JOCKEY_NAME"),
        jockey_claim=s("JOCKEY_CLAIM"),
        trainer_name=s("TRAINER_NAME"),
        owner_name=s("OWNER_NAME"),
        stall_draw=s("STALL_DRAW"),
        cloth_number=s("CLOTH_NUMBER"),
        form=s("FORM"),
        days_since_last_run=s("DAYS_SINCE_LAST_RUN"),
        wearing=s("WEARING"),
        forecastprice_numerator=s("FORECASTPRICE_NUMERATOR"),
        forecastprice_denominator=s("FORECASTPRICE_DENOMINATOR"),
    )


# ── Building episodes ────────────────────────────────────────────────────────


def _row_to_tick(row: pd.Series) -> Tick:
    """Convert one ticks DataFrame row into a :class:`Tick`."""
    snap_str = row["snap_json"]
    runners = parse_snap_json(snap_str) if pd.notna(snap_str) and snap_str else []

    winner_sid = row.get("winner_selection_id")
    if pd.isna(winner_sid):
        winner_sid = None
    else:
        winner_sid = int(winner_sid)

    return Tick(
        market_id=str(row["market_id"]),
        timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(),
        sequence_number=int(row["sequence_number"]),
        venue=str(row["venue"]),
        market_start_time=pd.Timestamp(row["market_start_time"]).to_pydatetime(),
        number_of_active_runners=_opt_int(row.get("number_of_active_runners")),
        traded_volume=float(row.get("traded_volume", 0.0)),
        in_play=bool(row.get("in_play", False)),
        winner_selection_id=winner_sid,
        temperature=_opt_float(row.get("temperature")),
        precipitation=_opt_float(row.get("precipitation")),
        wind_speed=_opt_float(row.get("wind_speed")),
        wind_direction=_opt_float(row.get("wind_direction")),
        humidity=_opt_float(row.get("humidity")),
        weather_code=_opt_int(row.get("weather_code")),
        runners=runners,
    )


def _build_runner_metadata(
    runners_df: pd.DataFrame, market_id: str
) -> dict[int, RunnerMeta]:
    """Build a dict of RunnerMeta keyed by selection_id for one market."""
    if runners_df.empty or "market_id" not in runners_df.columns:
        return {}
    subset = runners_df[runners_df["market_id"] == market_id]
    result: dict[int, RunnerMeta] = {}
    for _, row in subset.iterrows():
        meta = _build_runner_meta(row)
        result[meta.selection_id] = meta
    return result


def load_day(
    date_str: str,
    data_dir: str | Path = "data/processed",
) -> Day:
    """Load one day's Parquet files and build a :class:`Day` episode.

    Parameters
    ----------
    date_str:
        ISO date string, e.g. ``"2026-03-26"``.
    data_dir:
        Directory containing the Parquet files.

    Returns
    -------
    Day
        Fully-constructed episode with races ordered by ``market_start_time``
        and ticks within each race ordered by ``sequence_number``.

    Raises
    ------
    FileNotFoundError
        If the ticks Parquet file does not exist.
    ValueError
        If any tick has ``in_play=True`` (should never happen if the
        extractor filtered correctly, but we guard against it).
    """
    data_dir = Path(data_dir)
    ticks_path = data_dir / f"{date_str}.parquet"
    runners_path = data_dir / f"{date_str}_runners.parquet"

    if not ticks_path.exists():
        raise FileNotFoundError(f"Ticks file not found: {ticks_path}")

    ticks_df = pd.read_parquet(ticks_path)
    runners_df = (
        pd.read_parquet(runners_path) if runners_path.exists()
        else pd.DataFrame()
    )

    return _build_day(date_str, ticks_df, runners_df)


def _build_day(
    date_str: str,
    ticks_df: pd.DataFrame,
    runners_df: pd.DataFrame,
) -> Day:
    """Core builder: DataFrame → Day.  Factored out for testability."""
    # Guard: no in-play ticks allowed
    if not ticks_df.empty and "in_play" in ticks_df.columns:
        in_play_mask = ticks_df["in_play"].astype(bool)
        if in_play_mask.any():
            n_bad = int(in_play_mask.sum())
            raise ValueError(
                f"{n_bad} in-play tick(s) found in data for {date_str}. "
                "Only pre-race ticks are allowed."
            )

    # Group by market_id → races
    races: list[Race] = []
    if not ticks_df.empty:
        for market_id, group in ticks_df.groupby("market_id", sort=False):
            market_id = str(market_id)
            # Sort ticks by sequence_number ascending within each race
            group = group.sort_values("sequence_number", ascending=True)
            ticks = [_row_to_tick(row) for _, row in group.iterrows()]

            # Derive race-level fields from the first tick
            first = ticks[0]
            runner_meta = _build_runner_metadata(runners_df, market_id)

            races.append(
                Race(
                    market_id=market_id,
                    venue=first.venue,
                    market_start_time=first.market_start_time,
                    winner_selection_id=first.winner_selection_id,
                    ticks=ticks,
                    runner_metadata=runner_meta,
                )
            )

    # Sort races by market_start_time
    races.sort(key=lambda r: r.market_start_time)

    return Day(date=date_str, races=races)


def load_days(
    date_strs: list[str],
    data_dir: str | Path = "data/processed",
) -> list[Day]:
    """Load multiple days with :class:`~training.progress_tracker.ProgressTracker`.

    Parameters
    ----------
    date_strs:
        List of ISO date strings to load.
    data_dir:
        Directory containing the Parquet files.

    Returns
    -------
    list[Day]
        Successfully loaded days (skips dates whose files are missing).
    """
    data_dir = Path(data_dir)
    tracker = ProgressTracker(
        total=len(date_strs),
        label="Building training episodes",
    )
    tracker.reset_timer()
    days: list[Day] = []

    for ds in date_strs:
        try:
            day = load_day(ds, data_dir)
            days.append(day)
        except FileNotFoundError:
            logger.warning("Skipping %s — file not found", ds)
        tracker.tick()
        progress = tracker.to_dict()
        logger.info(
            "[%d/%d] %s — process ETA: %s",
            progress["completed"],
            progress["total"],
            ds,
            progress["process_eta_human"],
        )

    return days
