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

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:
    import orjson

    def _json_loads(s: str | bytes) -> dict:
        if isinstance(s, str):
            return orjson.loads(s.encode("utf-8"))
        return orjson.loads(s)

except ImportError:
    import json

    def _json_loads(s: str | bytes) -> dict:  # type: ignore[misc]
        return json.loads(s)

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
class PastRace:
    """One historical race result from ``RaceCardRunners.PastRacesJson``.

    Session 2.7b — used by the feature engineer to derive course/distance/
    going form, BSP trends, and performance history.
    """

    date: str                # ISO date string, e.g. "2026-03-11"
    course: str              # e.g. "Huntingdon"
    distance_yards: int      # distance in yards
    going: str               # full going description, e.g. "Good to Soft"
    going_abbr: str          # abbreviated going, e.g. "GS"
    bsp: float               # Betfair starting price (NaN if missing)
    ip_max: float            # in-play max price (NaN if missing)
    ip_min: float            # in-play min price (NaN if missing)
    race_type: str           # e.g. "Hurdle", "Flat", "Chase"
    jockey: str
    official_rating: float   # NaN if missing
    position: int | None     # finishing position (None = DNF)
    field_size: int | None   # total runners in the race


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
    # Session 2.7b — RaceCardRunners enrichment
    past_races: tuple[PastRace, ...] = ()
    timeform_comment: str = ""
    recent_form: str = ""


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
    # Race status from RaceStatusEvents (Session 2.7a).
    # One of: "parading", "going down", "going behind", "under orders",
    #         "at the post", "off", or None (not available / legacy data)
    race_status: str | None
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
    # Fields added in Session 1.3 — default values for backward compatibility
    # until the extractor is updated to populate them (Session 2+).
    market_name: str = ""
    market_type: str = ""  # "WIN", "EACH_WAY", etc.
    n_runners: int = 0     # total runners including removed
    # Set of selection IDs that won the market.  For WIN markets this is
    # just {winner_selection_id}.  For EACH_WAY (place) markets this
    # includes WINNER + PLACED runners — all of them pay out on a back bet.
    winning_selection_ids: set[int] = field(default_factory=set)


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

    Supports two JSON layouts:

    **Nested layout** (real ``ResolvedMarketSnaps.SnapJson`` from
    StreamRecorder1) — top-level key ``MarketRunners``, each runner has
    ``RunnerId.SelectionId``, ``Definition.Status``, ``Prices.*``::

        {
          "MarketRunners": [
            {
              "RunnerId": {"SelectionId": 12345678},
              "Definition": {"Status": "ACTIVE", "SortPriority": 1, ...},
              "Prices": {
                "LastTradedPrice": 4.5,
                "TradedVolume": 1234.56,
                "StartingPriceNear": 4.2,
                "StartingPriceFar": 5.0,
                "AvailableToBack": [{"Price": 4.5, "Size": 100.0}],
                "AvailableToLay": [{"Price": 4.6, "Size": 150.0}]
              }
            }
          ]
        }

    **Flat layout** (used by unit tests and older snapshots) — top-level key
    ``Runners``, fields directly on each runner object.
    """
    data = _json_loads(json_str)

    # Detect layout: nested (MarketRunners) vs flat (Runners)
    runners_raw = (
        data.get("MarketRunners")
        or data.get("Runners")
        or data.get("runners")
        or []
    )

    result: list[RunnerSnap] = []
    for r in runners_raw:
        # ── Nested layout: RunnerId / Definition / Prices sub-dicts ──
        runner_id_obj = r.get("RunnerId") or {}
        defn = r.get("Definition") or {}
        prices = r.get("Prices") or {}

        # SelectionId: nested → flat fallback
        selection_id = (
            runner_id_obj.get("SelectionId")
            or r.get("SelectionId")
            or r.get("selectionId")
            or r.get("id")
            or 0
        )

        # Status: nested → flat fallback
        status = (
            defn.get("Status")
            or r.get("Status")
            or r.get("status")
            or "ACTIVE"
        )

        # Numeric fields: nested (Prices) → flat fallback
        ltp = prices.get("LastTradedPrice") or r.get("LastTradedPrice") or r.get("ltp") or 0.0
        tv = prices.get("TradedVolume") or r.get("TotalMatched") or r.get("totalMatched") or r.get("tv") or 0.0
        spn = prices.get("StartingPriceNear") or r.get("StartingPriceNear") or r.get("spn") or 0.0
        spf = prices.get("StartingPriceFar") or r.get("StartingPriceFar") or r.get("spf") or 0.0

        adj = defn.get("AdjustmentFactor") or r.get("AdjustmentFactor") or r.get("adjustmentFactor")
        bsp = defn.get("Bsp") or r.get("Bsp") or r.get("bsp")
        sp = defn.get("SortPriority") or r.get("SortPriority") or r.get("sortPriority")
        rd = defn.get("RemovalDate") or r.get("RemovalDate") or r.get("removalDate")

        # Price ladders: nested (Prices) → flat fallback
        atb_raw = prices.get("AvailableToBack") or r.get("AvailableToBack") or r.get("atb")
        atl_raw = prices.get("AvailableToLay") or r.get("AvailableToLay") or r.get("atl")

        result.append(
            RunnerSnap(
                selection_id=int(selection_id),
                status=str(status),
                last_traded_price=float(ltp),
                total_matched=float(tv),
                starting_price_near=float(spn),
                starting_price_far=float(spf),
                adjustment_factor=_opt_float(adj),
                bsp=_opt_float(bsp),
                sort_priority=_opt_int(sp),
                removal_date=rd,
                available_to_back=_parse_price_sizes(atb_raw),
                available_to_lay=_parse_price_sizes(atl_raw),
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


def _parse_position(pos_str: str) -> tuple[int | None, int | None]:
    """Parse a position string like ``"3/6"`` into (position, field_size).

    Non-numeric positions (``"U/9"``, ``"P/15"``, ``"F/8"``) return
    ``(None, field_size)``.  Invalid strings return ``(None, None)``.
    """
    if not pos_str or "/" not in pos_str:
        return None, None
    parts = pos_str.split("/")
    if len(parts) != 2:
        return None, None
    pos_part, size_part = parts[0].strip(), parts[1].strip()
    try:
        field_size = int(size_part)
    except (ValueError, TypeError):
        field_size = None
    try:
        position = int(pos_part)
    except (ValueError, TypeError):
        position = None
    return position, field_size


def _parse_past_races_json(raw: str | None) -> tuple[PastRace, ...]:
    """Parse a ``PastRacesJson`` string into a tuple of :class:`PastRace`.

    Handles null, empty string, empty array, and malformed JSON gracefully.
    """
    NaN = math.nan
    if not raw or raw.strip() in ("", "[]", "null"):
        return ()
    try:
        entries = _json_loads(raw)
    except (ValueError, TypeError):
        return ()
    if not isinstance(entries, list):
        return ()

    results: list[PastRace] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        going_obj = e.get("going") or {}
        race_type_obj = e.get("raceType") or {}
        pos_str = e.get("position", "")
        position, field_size = _parse_position(pos_str)

        # Extract date — trim to YYYY-MM-DD
        date_str = (e.get("date") or "")[:10]

        results.append(PastRace(
            date=date_str,
            course=e.get("course", ""),
            distance_yards=int(e.get("distance", 0)),
            going=going_obj.get("full", ""),
            going_abbr=going_obj.get("abbr", ""),
            bsp=float(e["bsp"]) if e.get("bsp") is not None else NaN,
            ip_max=float(e["inPlayMax"]) if e.get("inPlayMax") is not None else NaN,
            ip_min=float(e["inPlayMin"]) if e.get("inPlayMin") is not None else NaN,
            race_type=race_type_obj.get("full", ""),
            jockey=e.get("jockey", ""),
            official_rating=float(e["officialRating"]) if e.get("officialRating") is not None else NaN,
            position=position,
            field_size=field_size,
        ))

    return tuple(results)


def _build_runner_meta(row: pd.Series) -> RunnerMeta:
    """Build a :class:`RunnerMeta` from one row of the runners DataFrame."""
    def s(col: str) -> str:
        """Extract a string value, replacing NaN/None with empty string."""
        v = row.get(col, "")
        if pd.isna(v):
            return ""
        return str(v)

    # Session 2.7b — parse PastRacesJson if present (backward-compatible)
    past_races_raw = row.get("past_races_json")
    if past_races_raw is not None and not pd.isna(past_races_raw):
        past_races = _parse_past_races_json(str(past_races_raw))
    else:
        past_races = ()

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
        past_races=past_races,
        timeform_comment=s("timeform_comment"),
        recent_form=s("recent_form"),
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

    # Race status (Session 2.7a) — may be absent in old Parquet files
    race_status_raw = row.get("race_status")
    if race_status_raw is None or (isinstance(race_status_raw, float) and pd.isna(race_status_raw)):
        race_status = None
    elif pd.isna(race_status_raw):
        race_status = None
    else:
        race_status = str(race_status_raw)

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
        race_status=race_status,
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
    t0 = time.perf_counter()
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

    day = _build_day(date_str, ticks_df, runners_df)
    elapsed = time.perf_counter() - t0
    logger.info(
        "Loaded %s: %d ticks, %d races in %.2fs",
        date_str, len(ticks_df), len(day.races), elapsed,
    )
    return day


def _build_day(
    date_str: str,
    ticks_df: pd.DataFrame,
    runners_df: pd.DataFrame,
) -> Day:
    """Core builder: DataFrame → Day.  Factored out for testability.

    Both pre-race and in-play ticks are included.  The agent observes the
    full race — in-play price movement is valuable signal.  Bet placement
    is restricted to pre-race ticks (``in_play == False``) by the
    environment, not the data layer.
    """

    # Group by market_id → races
    races: list[Race] = []
    if not ticks_df.empty:
        # Pre-sort the whole DataFrame once instead of per-group
        ticks_df = ticks_df.sort_values(
            ["market_id", "sequence_number"], ascending=True,
        )
        for market_id, group in ticks_df.groupby("market_id", sort=False):
            market_id = str(market_id)
            # Already sorted by sequence_number from the global sort
            ticks = [_row_to_tick(row) for _, row in group.iterrows()]

            # Derive race-level fields from the first tick / row
            first = ticks[0]
            first_row = group.iloc[0]
            runner_meta = _build_runner_metadata(runners_df, market_id)

            # Build winning_selection_ids from the last tick's runner statuses.
            # For WIN markets: only WINNER.  For EACH_WAY: WINNER + PLACED.
            # Betfair EACH_WAY markets are place markets — the quoted odds
            # already reflect the place fraction, so PLACED pays at full price.
            winning_ids: set[int] = set()
            last_tick = ticks[-1]
            for runner in last_tick.runners:
                if runner.status in ("WINNER", "PLACED"):
                    winning_ids.add(runner.selection_id)

            races.append(
                Race(
                    market_id=market_id,
                    venue=first.venue,
                    market_start_time=first.market_start_time,
                    winner_selection_id=first.winner_selection_id,
                    ticks=ticks,
                    runner_metadata=runner_meta,
                    market_name=str(first_row.get("market_name") or ""),
                    market_type=str(first_row.get("market_type") or ""),
                    n_runners=len(runner_meta),
                    winning_selection_ids=winning_ids,
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
