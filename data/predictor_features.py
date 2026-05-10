"""Predictor-input feature computations from rl-betfair / ai-betfair race data.

Pure functions over `data.episode_builder.RunnerMeta` + `PastRace` that
produce the per-runner aggregates the `betfair-predictors` GBMs expect
as input (F2 / F5 contracts).

Designed as a SHARED MODULE: ai-betfair already imports
`data.episode_builder` from rl-betfair, so once this lands here it
serves both consumers (training rollouts in rl-betfair, live inference
in ai-betfair) without further plumbing.

Why we don't use the predictor repo's
`scripts/outcome_predictor/features/aggregates.py::add_aggregates_for_variant`:
that operates on the predictor's training parquet pipeline (one row
per runner-per-race, joined globally across all training races).
At inference time, the runner's own `past_races` tuple is already
attached to `RunnerMeta` — we just walk it locally. See
`incoming/predictor-integration-data-bridging.md` for the full design
rationale.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date as _date
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data.episode_builder import PastRace, RunnerMeta


# ---------------------------------------------------------------------------
# F2 aggregates — champion's prior-form contract
# ---------------------------------------------------------------------------


F2_AGGREGATE_KEYS: tuple[str, ...] = (
    "prior_runs",
    "prior_wins",
    "prior_places",
    "prior_win_rate",
    "prior_place_rate",
    "days_since_prior_run",
)


def _parse_iso_date(value: str) -> _date | None:
    """Parse the leading YYYY-MM-DD off PastRace.date.

    `PastRace.date` is "YYYY-MM-DD" or sometimes a full ISO timestamp
    (`"2026-04-01T17:45:00Z"`); the date prefix is the load-bearing
    bit. Returns None on parse failure.
    """
    if not value:
        return None
    head = str(value)[:10]
    try:
        return datetime.strptime(head, "%Y-%m-%d").date()
    except ValueError:
        return None


def _is_placed(pr: "PastRace") -> bool | None:
    """Did this runner finish in a paying place position?

    Returns None for DNFs (`position is None`) or unknown
    `field_size` so the caller can exclude the row from
    place-rate denominators (matching the predictor's
    `_cum_placed_known` semantic — only count rows where the
    placed label is determinable).

    Place counts follow the standard UK/IE Betfair-EW
    convention:

    - 5-7 runners → 2 places
    - 8-15 runners → 3 places
    - 16+ runners → 4 places (5 in some big handicaps; we
      use 4 as the conservative default)

    Races with `field_size < 5` are non-EW markets in this
    convention; they return None so the place-rate
    denominator excludes them.
    """
    if pr.position is None:
        return None
    fs = pr.field_size
    if fs is None or fs < 5:
        return None
    if fs < 8:
        n_places = 2
    elif fs < 16:
        n_places = 3
    else:
        n_places = 4
    return pr.position <= n_places


def compute_f2_aggregates(
    runner_meta: "RunnerMeta",
    *,
    as_of_date: _date,
) -> dict[str, float]:
    """Compute the 6 F2 prior-form aggregates from runner_meta.past_races.

    Strict ``< as_of_date`` filter (no result leakage). Returns a dict
    with all 6 ``F2_AGGREGATE_KEYS`` keys populated:

    - ``prior_runs`` — count of past races strictly before as_of_date.
    - ``prior_wins`` — count where ``position == 1``.
    - ``prior_places`` — count where the runner finished in a paying
      place position (per ``_is_placed``).
    - ``prior_win_rate`` — ``prior_wins / prior_runs`` or NaN if
      ``prior_runs == 0``.
    - ``prior_place_rate`` — ``prior_places / prior_known_placed`` or
      NaN if no past races have determinable place status.
    - ``days_since_prior_run`` — days from the most recent strictly-prior
      race to ``as_of_date``, or NaN if no prior runs.

    NaN-safe — a rookie with empty ``past_races`` gets ``prior_runs=0``,
    ``prior_wins=0``, ``prior_places=0``, and NaN for the rate /
    days-since fields.

    Hard_constraints §1 (no leakage): the strict ``<`` comparison on
    ``as_of_date`` excludes any past_race that took place ON the same
    day; the predictor's training dataset uses the same convention.
    """
    nan = float("nan")
    prior_runs = 0
    prior_wins = 0
    prior_places = 0
    prior_known_placed = 0  # rows where placed status is determinable
    most_recent: _date | None = None

    for pr in runner_meta.past_races:
        d = _parse_iso_date(pr.date)
        if d is None:
            continue
        if d >= as_of_date:
            continue
        prior_runs += 1
        if pr.position is not None and pr.position == 1:
            prior_wins += 1
        placed = _is_placed(pr)
        if placed is not None:
            prior_known_placed += 1
            if placed:
                prior_places += 1
        if most_recent is None or d > most_recent:
            most_recent = d

    if prior_runs > 0:
        win_rate = prior_wins / prior_runs
    else:
        win_rate = nan
    if prior_known_placed > 0:
        place_rate = prior_places / prior_known_placed
    else:
        place_rate = nan
    if most_recent is not None:
        days_since = float((as_of_date - most_recent).days)
    else:
        days_since = nan

    return {
        "prior_runs": float(prior_runs),
        "prior_wins": float(prior_wins),
        "prior_places": float(prior_places),
        "prior_win_rate": float(win_rate),
        "prior_place_rate": float(place_rate),
        "days_since_prior_run": float(days_since),
    }


def compute_f2_aggregates_for_runners(
    runner_metas: Iterable["RunnerMeta"],
    *,
    as_of_date: _date,
) -> dict[int, dict[str, float]]:
    """Convenience wrapper: F2 aggregates keyed by selection_id.

    Equivalent to ``{rm.selection_id: compute_f2_aggregates(rm,
    as_of_date=as_of_date) for rm in runner_metas}``.
    """
    return {
        rm.selection_id: compute_f2_aggregates(rm, as_of_date=as_of_date)
        for rm in runner_metas
    }


# ---------------------------------------------------------------------------
# V2 ladder window — direction predictor (per-tick) advisor
# ---------------------------------------------------------------------------


# Column order MUST match the predictor repo's
# `scripts/predictor/datasets.py::feature_columns('V2')` exactly. Order
# is locked: V1_COLS (16) + V2_EXTRA (10) = 26.
DIR_V2_COLS: tuple[str, ...] = (
    # V1 (16): per-tick raw ladder + market state.
    "ltp",
    "back_p1", "back_p2", "back_p3",
    "back_s1", "back_s2", "back_s3",
    "lay_p1", "lay_p2", "lay_p3",
    "lay_s1", "lay_s2", "lay_s3",
    "traded_volume_runner",
    "num_active_runners",
    "time_to_off_sec",
    # V2 extra (10): trailing window stats over LTP series.
    "ltp_lag_1", "ltp_lag_5", "ltp_lag_10", "ltp_lag_30",
    "ltp_w32_mean", "ltp_w32_std", "ltp_w32_min", "ltp_w32_max",
    "ltp_w32_first", "ltp_w32_n",
)
DIR_V2_DIM: int = len(DIR_V2_COLS)  # 26
DIR_WINDOW: int = 32


def build_direction_windows_for_race(
    race: object,
) -> tuple[object, list[tuple[int, int]]]:
    """Build all (32, 26) V2 windows for a race, one per (tick_idx, sid).

    Returns ``(windows, indices)`` where:
    - ``windows`` is a ``np.ndarray`` of shape ``(N, 32, 26)`` float32 — N
      = total active (tick_idx, sid) pairs across the race.
    - ``indices`` is a list of ``(tick_idx, selection_id)`` tuples
      aligned to the first dim of ``windows``.

    Design: precompute per-runner (n_ticks, 26) feature matrices ONCE
    per race (V1 from raw ladder + V2 stats from LTP series), then slice
    32-tick windows from them. Newest tick at index -1; left-pad with
    zeros when fewer than 32 ticks of history are available — matches
    `betfair-predictors/scripts/predictor/datasets.py::build_sequence_examples`.

    Skips runner / tick pairs where the runner has no snapshot at that
    tick (REMOVED, etc.) — caller can map outputs back via the
    ``indices`` list.
    """
    import numpy as np

    n_ticks = len(race.ticks)
    if n_ticks == 0:
        return np.zeros((0, DIR_WINDOW, DIR_V2_DIM), dtype=np.float32), []

    market_start_ts = race.market_start_time

    # Discover active sids across the race.
    all_sids: set[int] = set()
    for tick in race.ticks:
        for r in tick.runners:
            all_sids.add(r.selection_id)
    if not all_sids:
        return np.zeros((0, DIR_WINDOW, DIR_V2_DIM), dtype=np.float32), []

    # Per-runner (n_ticks, 26) feature matrix. Default zero so missing
    # snapshots stay neutral (the predictor's training data also
    # zero-pads).
    per_runner: dict[int, "np.ndarray"] = {}
    # Track which (tick, sid) pairs have a real snapshot — only those
    # become windows.
    has_snap: dict[int, list[bool]] = {sid: [False] * n_ticks for sid in all_sids}

    for sid in all_sids:
        feat = np.zeros((n_ticks, DIR_V2_DIM), dtype=np.float32)
        per_runner[sid] = feat

    for t_idx, tick in enumerate(race.ticks):
        n_active = float(tick.number_of_active_runners or 0)
        # `time_to_off_sec` from market_start_time MINUS current tick
        # timestamp; positive = pre-off, negative = in-play.
        try:
            tto = (market_start_ts - tick.timestamp).total_seconds()
        except Exception:
            tto = 0.0
        for r in tick.runners:
            sid = r.selection_id
            feat = per_runner[sid]
            ltp = float(r.last_traded_price or 0.0)
            feat[t_idx, 0] = ltp
            atb = list(r.available_to_back)[:3]
            atl = list(r.available_to_lay)[:3]
            for i in range(3):
                if i < len(atb):
                    feat[t_idx, 1 + i] = float(atb[i].price)
                    feat[t_idx, 4 + i] = float(atb[i].size)
                if i < len(atl):
                    feat[t_idx, 7 + i] = float(atl[i].price)
                    feat[t_idx, 10 + i] = float(atl[i].size)
            feat[t_idx, 13] = float(getattr(r, "total_matched", 0.0))
            feat[t_idx, 14] = n_active
            feat[t_idx, 15] = float(tto)
            has_snap[sid][t_idx] = True

    # Now compute V2 stats per runner from LTP series.
    for sid, feat in per_runner.items():
        ltps = feat[:, 0]  # (n_ticks,)
        for t_idx in range(n_ticks):
            # lag features — 0 if not enough history (predictor's training
            # path uses left-pad-zero too)
            feat[t_idx, 16] = ltps[t_idx - 1] if t_idx >= 1 else 0.0
            feat[t_idx, 17] = ltps[t_idx - 5] if t_idx >= 5 else 0.0
            feat[t_idx, 18] = ltps[t_idx - 10] if t_idx >= 10 else 0.0
            feat[t_idx, 19] = ltps[t_idx - 30] if t_idx >= 30 else 0.0
            # Window stats over [t-31, t] — 32 ticks max
            lo = max(0, t_idx + 1 - DIR_WINDOW)
            window = ltps[lo:t_idx + 1]
            feat[t_idx, 20] = float(window.mean())
            feat[t_idx, 21] = float(window.std())
            feat[t_idx, 22] = float(window.min())
            feat[t_idx, 23] = float(window.max())
            feat[t_idx, 24] = float(window[0])
            feat[t_idx, 25] = float(len(window))

    # Build (N, 32, 26) windows for each (tick, sid) WHERE the runner has
    # a real snapshot at that tick.
    windows: list = []
    indices: list[tuple[int, int]] = []
    for sid, feat in per_runner.items():
        snap_mask = has_snap[sid]
        for t_idx in range(n_ticks):
            if not snap_mask[t_idx]:
                continue
            lo = max(0, t_idx + 1 - DIR_WINDOW)
            seg = feat[lo:t_idx + 1]
            if seg.shape[0] < DIR_WINDOW:
                pad = np.zeros(
                    (DIR_WINDOW - seg.shape[0], DIR_V2_DIM), dtype=np.float32,
                )
                seg = np.concatenate([pad, seg], axis=0)
            windows.append(seg)
            indices.append((t_idx, sid))

    if not windows:
        return np.zeros((0, DIR_WINDOW, DIR_V2_DIM), dtype=np.float32), []
    arr = np.stack(windows, axis=0).astype(np.float32, copy=False)
    return arr, indices


# ---------------------------------------------------------------------------
# F2 DataFrame stitcher — for bundle.predict_race(df)
# ---------------------------------------------------------------------------


# F1 numeric columns the champion expects, in the order the predictor's
# `numeric_feature_matrix` emits them (see
# `betfair-predictors/scripts/outcome_predictor/datasets.py::F1_NUMERIC`).
_F1_NUMERIC_COLS: tuple[str, ...] = (
    "field_size",
    "draw",
    "weight_lbs",
    "age",
    "days_since_last_run",
    "official_rating",
    "sort_priority",
    "forecast_price",
    "distance_yards",
)

# F1 categorical columns the encoder maps to `<col>_idx` integer columns
# (see `betfair-predictors/scripts/outcome_predictor/datasets.py::F1_CATEGORICAL`).
_F1_CATEGORICAL_COLS: tuple[str, ...] = (
    "course",
    "race_class",
    "race_type",
    "surface",
    "sex",
    "headgear",
)

# F5 columns the ranker expects beyond F2. Until the F5 aggregator lands,
# zero-fill these to mean "no prior jockey/trainer history known"; the
# ranker still runs (column-shape match) but its output is degraded
# vs. fully-populated F5. Order doesn't matter (numeric_feature_matrix
# selects by column name); listing here to keep the contract local.
_F5_ZERO_FILL_COLS: tuple[str, ...] = (
    "jockey_runs", "jockey_wins", "jockey_places",
    "jockey_win_rate", "jockey_place_rate", "jockey_days_since_last",
    "trainer_runs", "trainer_wins", "trainer_places",
    "trainer_win_rate", "trainer_place_rate", "trainer_days_since_last",
    "jockey_trainer_combo_runs", "jockey_trainer_combo_wins",
    "jockey_trainer_combo_win_rate", "jockey_trainer_combo_place_rate",
    "jockey_name_te_win", "jockey_name_te_placed",
    "trainer_name_te_win", "trainer_name_te_placed",
    "course_te_win", "course_te_placed",
)


def _safe_float(value: object, default: float = float("nan")) -> float:
    """Best-effort string -> float, returning ``default`` on parse failure."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return default
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def _forecast_price_decimal(numerator: str, denominator: str) -> float:
    """Convert the racecard's fractional forecast price to decimal form.

    ``"8/1" -> 9.0`` (decimal-form Betfair convention: numerator/denominator
    + 1.0). Returns NaN if either component is missing or unparseable.
    """
    num = _safe_float(numerator)
    den = _safe_float(denominator)
    if num != num or den != den or den == 0.0:  # NaN check
        return float("nan")
    return num / den + 1.0


def build_predict_race_dataframe(
    race: "object",  # data.episode_builder.Race — string-typed to keep this module import-free
    *,
    as_of_date: _date,
    feature_variant: str = "F2",
) -> "object":  # pandas.DataFrame
    """Stitch race-level + per-runner data into the DataFrame the GBM expects.

    Builds one row per runner (active + removed) carrying:

    - The 9 F1 numeric columns + 6 F1 categorical raw values.
    - The 6 F2 prior-form aggregates from
      :func:`compute_f2_aggregates`.
    - ``selection_id`` and ``market_id`` for routing the bundle's
      output dicts.

    Race-level fields rl-betfair / ai-betfair currently have natively
    (``course = race.venue``) get populated. Race-level fields the
    streamrecorder pipeline doesn't yet extract (``race_class``,
    ``race_type``, ``surface``, ``distance_yards``) get an empty
    string / NaN fallback; the encoder's ``<UNKNOWN>`` token absorbs
    them at inference time per the predictor repo's §9 cold-start
    contract. Field-level coverage will improve once the
    streamrecorder coldData extraction lands the missing race-level
    metadata in rl-betfair's parquets.

    Returns a ``pandas.DataFrame``. Column ordering matches what
    ``apply_encoders`` + ``numeric_feature_matrix`` produce so
    downstream wiring just sets `each_way` on the bet, no further
    column shuffling required.

    ``feature_variant`` is currently ``"F2"`` (champion contract).
    The F5 (ranker) variant adds jockey/trainer aggregates; that
    extension lands in the next follow-on iteration.
    """
    if feature_variant not in ("F2", "F5"):
        raise NotImplementedError(
            f"build_predict_race_dataframe only supports F2 / F5; "
            f"got feature_variant={feature_variant!r}"
        )

    import pandas as pd

    n_runners = max(1, getattr(race, "n_runners", 0) or len(race.runner_metadata))

    rows = []
    for sid, rm in race.runner_metadata.items():
        forecast_price = _forecast_price_decimal(
            rm.forecastprice_numerator,
            rm.forecastprice_denominator,
        )
        f2 = compute_f2_aggregates(rm, as_of_date=as_of_date)
        row: dict[str, object] = {
            "market_id": race.market_id,
            "selection_id": int(sid),
            # F1 numerics
            "field_size": int(n_runners),
            "draw": _safe_float(rm.stall_draw),
            "weight_lbs": _safe_float(rm.weight_value),
            "age": _safe_float(rm.age),
            "days_since_last_run": _safe_float(rm.days_since_last_run),
            "official_rating": _safe_float(rm.official_rating),
            "sort_priority": _safe_float(rm.sort_priority),
            "forecast_price": forecast_price,
            # ``distance_yards`` is not yet in rl-betfair's `Race`; leave NaN.
            # See `incoming/predictor-integration-data-bridging.md` for the
            # streamrecorder coldData extension that lands this.
            "distance_yards": float("nan"),
            # F1 categoricals — raw strings; the encoder applies the int mapping.
            # ``course`` derives from ``race.venue``; the other four are not
            # in the day parquet today and route through the encoder's
            # ``<UNKNOWN>`` cold-start token.
            "course": getattr(race, "venue", "") or "",
            "race_class": "",  # TODO(data-bridging): pull from coldData
            "race_type": "",   # TODO(data-bridging): pull from coldData
            "surface": "",     # TODO(data-bridging): pull from coldData
            "sex": rm.sex_type or "",
            "headgear": rm.wearing or "",
        }
        # F2 aggregates
        row.update(f2)

        # F5 ranker columns — zero-fill until the F5 jockey/trainer
        # aggregator lands. Semantically "unknown jockey/trainer history",
        # which makes the ranker effectively rank by F2 features alone.
        # The pipeline runs end-to-end with this fallback; ranker
        # accuracy is degraded versus a fully-populated F5, by design.
        # See `incoming/predictor-integration-data-bridging.md` for the
        # F5 implementation in the next data-bridging iteration.
        for k in _F5_ZERO_FILL_COLS:
            row.setdefault(k, 0.0)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df
