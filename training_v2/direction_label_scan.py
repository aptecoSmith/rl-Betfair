"""Offline direction-label generator — phase-13 S02.

Walks every priceable (pre-race tick × active runner) of a day and
labels whether the runner's LTP made a favourable directional move
within the close horizon. Per-side per-runner: a row carries
``label_back`` (LTP came IN by ≥ N ticks) and ``label_lay`` (LTP
drifted OUT by ≥ N ticks) independently.

Mirrors the shape of :mod:`training_v2.arb_oracle` deliberately — same
data dependencies, same per-race tick walk, same env-matcher rule
checks at the OPEN tick. The DIFFERENCE: dense per-(tick, runner)
labels on FUTURE PRICE MOVEMENT, not on fill mechanics.

Hard constraints (``plans/rewrite/phase-13-directional-scalping/
hard_constraints.md``):

- §1 Offline only. Never invoked inside the training loop.
- §2 Determinism: same data + same config → byte-identical labels.
  Sorted by ``(tick_index, runner_idx)`` before write.
- §3 Match env-matcher priceability rules at the OPEN tick.
- §4 V1 label semantics = threshold-crossing on
  ``last_traded_price``. Magnitude-target labels (V2) require a
  separate cache namespace.
- §5 Cache filename embeds invalidating keys.

Cache layout::

    data/direction_labels/{date}/horizon{H}_thresh{T}_fc{F}.npz
    data/direction_labels/{date}/horizon{H}_thresh{T}_fc{F}_header.json

The ``H`` / ``T`` / ``F`` triple in the filename is the same triple
the trainer uses to resolve the cache; mismatch is caught by the path
resolver before ``load_labels`` is even called.

CLI lives in :mod:`training_v2.direction_label_cli`.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from env.betfair_env import OBS_SCHEMA_VERSION
from env.exchange_matcher import passes_junk_filter, passes_price_cap
from env.tick_ladder import tick_offset

logger = logging.getLogger(__name__)

# Two label-generation modes coexist:
#
#   * "v1_threshold_crossing" (the original 2026-05-06 mode) — labels
#     fire if LTP ever crosses ±threshold ticks at ANY point in a
#     tick-count window forward of T. Tick-count horizon.
#
#   * "v2_time_endpoint_signed_tick" (added 2026-05-24 to align with
#     the betfair-predictors direction model's labels) — labels are
#     the SIGN of (LTP at T + horizon_seconds) − (LTP at T), measured
#     in Betfair ticks. Time horizon, endpoint semantics. This matches
#     extract_labels_prototype.py:HORIZONS_SEC in the predictor repo,
#     so the predictor's quantile outputs in obs become directly
#     informative for the supervised head. See
#     plans/direction-predictor-label-alignment/.
LABEL_VERSION_TICK_CROSSING: str = "v1_threshold_crossing"
LABEL_VERSION_TIME_ENDPOINT: str = "v2_time_endpoint_signed_tick"
# Default mode for back-compat — existing call sites that pass only
# `direction_horizon_ticks` see byte-identical behaviour.
LABEL_VERSION: str = LABEL_VERSION_TICK_CROSSING

_DEFAULT_MAX_DEV_PCT: float = 0.5
_MIN_STAKE: float = 2.0


@dataclass(slots=True)
class DirectionLabel:
    """One (pre-race tick × priceable runner) direction label row.

    ``label_back`` and ``label_lay`` are independent 0/1 floats. Both
    can be 1 if the LTP oscillated through both thresholds within the
    horizon (rare). Both can be 0 if the price stayed within the band.

    Diagnostic fields (``ltp_at_open``, ``threshold_back``,
    ``threshold_lay``, ``first_back_fav_tick``, ``first_lay_fav_tick``)
    are surfaced so cache audits can inspect what the scan saw without
    re-walking the day.
    """

    tick_index: int            # global pre-race tick index across the day
    runner_idx: int            # runner slot (env's sorted-sid index)
    label_back: float          # 0.0 or 1.0
    label_lay: float           # 0.0 or 1.0
    ltp_at_open: float
    threshold_back: float
    threshold_lay: float
    first_back_fav_tick: int   # -1 if label_back == 0
    first_lay_fav_tick: int    # -1 if label_lay == 0


# ── Public API ───────────────────────────────────────────────────────────────


def scan_day(
    date: str,
    data_dir: Path,
    config: dict,
    *,
    direction_horizon_ticks: int | None = None,
    direction_horizon_seconds: float | None = None,
    direction_threshold_ticks: int,
    force_close_before_off_seconds: float,
) -> list[DirectionLabel]:
    """Walk every priceable (pre-race tick, runner) of *date* and emit
    per-side direction labels.

    Exactly ONE of ``direction_horizon_ticks`` /
    ``direction_horizon_seconds`` must be provided:

    * ``direction_horizon_ticks`` → original ``v1_threshold_crossing``
      mode: label fires if LTP ever crosses ±threshold within the
      next N ticks (tick-count horizon, any-crossing semantics).
    * ``direction_horizon_seconds`` → ``v2_time_endpoint_signed_tick``
      mode (2026-05-24): label fires from the SIGN of (LTP at
      T+horizon_seconds) − (LTP at T) in ticks. Time horizon,
      endpoint semantics. Aligns with the betfair-predictors
      direction model's training labels (see
      ``plans/direction-predictor-label-alignment/``).

    Rows where the time horizon falls past in-play or past the
    force-close boundary are SKIPPED in v2 mode (matching the
    predictor's "label missing" handling). In v1 mode the window is
    clipped to whatever ticks are available before the boundary
    (existing pre-2026-05-24 behaviour).

    Returns rows sorted by ``(tick_index, runner_idx)`` for
    determinism.
    """
    n_modes = sum(
        x is not None
        for x in (direction_horizon_ticks, direction_horizon_seconds)
    )
    if n_modes != 1:
        raise ValueError(
            "scan_day: must pass exactly ONE of "
            "direction_horizon_ticks or direction_horizon_seconds, "
            f"got ticks={direction_horizon_ticks!r}, "
            f"seconds={direction_horizon_seconds!r}",
        )
    if direction_horizon_ticks is not None and direction_horizon_ticks <= 0:
        raise ValueError(
            f"direction_horizon_ticks must be > 0, got "
            f"{direction_horizon_ticks!r}",
        )
    if (
        direction_horizon_seconds is not None
        and direction_horizon_seconds <= 0
    ):
        raise ValueError(
            f"direction_horizon_seconds must be > 0, got "
            f"{direction_horizon_seconds!r}",
        )
    if direction_threshold_ticks <= 0:
        raise ValueError(
            f"direction_threshold_ticks must be > 0, got "
            f"{direction_threshold_ticks!r}",
        )
    if force_close_before_off_seconds < 0:
        raise ValueError(
            f"force_close_before_off_seconds must be >= 0, got "
            f"{force_close_before_off_seconds!r}",
        )
    is_time_endpoint = direction_horizon_seconds is not None

    from data.episode_builder import load_day

    day = load_day(date, data_dir)
    if not day.races:
        return []

    betting = config.get("training", {}).get("betting_constraints", {})
    max_back_price: float | None = betting.get("max_back_price")
    max_lay_price: float | None = betting.get("max_lay_price")

    # Build a runner-slot map per race the same way ``BetfairEnv`` does
    # (sorted selection_id → slot index). We reproduce the logic
    # verbatim rather than instantiating a full env to keep this module
    # cheap on memory and free of obs-shim dependencies.
    labels: list[DirectionLabel] = []
    global_tick = 0  # counts pre-race ticks only across the day

    for race in day.races:
        market_start_ts = race.market_start_time.timestamp()
        runner_slot = _runner_slot_map(race)

        # Pre-race ticks only get a global index. We still need the
        # FULL tick list (in_play included) for the close-horizon
        # resolver and the ltp scan window.
        n_ticks = len(race.ticks)

        # Per-(race, runner) ltp array — np.nan for "not active /
        # unpriceable / in-play". Indexed by raw race-tick index.
        ltp_by_sid: dict[int, np.ndarray] = {}
        for sid in runner_slot:
            arr = np.full(n_ticks, np.nan, dtype=np.float64)
            for t, tick in enumerate(race.ticks):
                if tick.in_play:
                    break
                rs = next(
                    (r for r in tick.runners if r.selection_id == sid),
                    None,
                )
                if rs is None or rs.status != "ACTIVE":
                    continue
                ltp = rs.last_traded_price
                if ltp is None or ltp <= 1.0:
                    continue
                arr[t] = float(ltp)
            ltp_by_sid[sid] = arr

        # Per-race tick timestamps array — populated lazily for v2
        # time-endpoint mode. Indexed by raw race-tick index.
        race_tick_ts: np.ndarray | None = None
        if is_time_endpoint:
            race_tick_ts = np.array(
                [t.timestamp.timestamp() for t in race.ticks],
                dtype=np.float64,
            )

        for tick_idx, tick in enumerate(race.ticks):
            if tick.in_play:
                continue

            # Resolve the FORWARD window endpoint per mode.
            if is_time_endpoint:
                # v2: target tick is the LAST pre-race tick whose
                # timestamp ≤ (open_ts + horizon_seconds), bounded by
                # the force-close cutoff. If no such tick exists
                # within the available pre-race window, the row is
                # SKIPPED (matches the predictor's NaN-label handling).
                target_tick_idx = _resolve_endpoint_tick(
                    race,
                    tick_idx,
                    direction_horizon_seconds,
                    force_close_before_off_seconds,
                    market_start_ts=market_start_ts,
                    race_tick_ts=race_tick_ts,
                )
                if target_tick_idx is None:
                    continue
                t_close = target_tick_idx  # for diagnostic consistency
            else:
                t_close = _resolve_close_tick(
                    race,
                    tick_idx,
                    force_close_before_off_seconds,
                    direction_horizon_ticks,
                    market_start_ts=market_start_ts,
                )

            for runner in tick.runners:
                sid = runner.selection_id
                slot = runner_slot.get(sid)
                if slot is None:
                    continue
                if runner.status != "ACTIVE":
                    continue
                ltp_T = runner.last_traded_price
                if ltp_T is None or ltp_T <= 1.0:
                    continue

                # Step 1 — priceability at the OPEN tick.
                priceable_back, priceable_lay = _priceability_at_open(
                    runner.available_to_back,
                    runner.available_to_lay,
                    ltp_T,
                    max_back_price,
                    max_lay_price,
                )
                if not (priceable_back or priceable_lay):
                    continue

                # Step 2 — threshold prices via the ladder helper.
                threshold_back_price = tick_offset(
                    ltp_T, direction_threshold_ticks, direction=-1,
                )
                threshold_lay_price = tick_offset(
                    ltp_T, direction_threshold_ticks, direction=+1,
                )

                # Step 3 + 4 — labels per mode.
                ltp_arr = ltp_by_sid[sid]
                if is_time_endpoint:
                    # v2 endpoint semantics: signed tick distance
                    # between LTP_now and LTP_at_target. If the LTP
                    # at the target tick is unavailable (NaN), skip.
                    label_back = 0.0
                    label_lay = 0.0
                    first_back = -1
                    first_lay = -1
                    ltp_target = ltp_arr[t_close]
                    if not np.isnan(ltp_target) and ltp_target > 1.0:
                        signed = _signed_ticks_between(
                            float(ltp_T), float(ltp_target),
                        )
                        if (
                            priceable_back
                            and threshold_back_price > 1.0
                            and signed <= -int(direction_threshold_ticks)
                        ):
                            label_back = 1.0
                            first_back = t_close
                        if (
                            priceable_lay
                            and signed >= int(direction_threshold_ticks)
                        ):
                            label_lay = 1.0
                            first_lay = t_close
                    else:
                        # Target LTP missing (runner went unpriceable
                        # before the horizon). Skip — predictor's
                        # NaN-label parity.
                        continue
                else:
                    # v1 tick-window any-crossing.
                    window = ltp_arr[tick_idx + 1: t_close + 1]
                    label_back = 0.0
                    first_back = -1
                    if (
                        priceable_back
                        and threshold_back_price > 1.0
                        and window.size > 0
                    ):
                        hits = np.where(window <= threshold_back_price)[0]
                        if hits.size > 0:
                            first_back = int(hits[0]) + tick_idx + 1
                            label_back = 1.0

                    label_lay = 0.0
                    first_lay = -1
                    if priceable_lay and window.size > 0:
                        hits = np.where(window >= threshold_lay_price)[0]
                        if hits.size > 0:
                            first_lay = int(hits[0]) + tick_idx + 1
                            label_lay = 1.0

                labels.append(DirectionLabel(
                    tick_index=global_tick,
                    runner_idx=slot,
                    label_back=label_back,
                    label_lay=label_lay,
                    ltp_at_open=float(ltp_T),
                    threshold_back=float(threshold_back_price),
                    threshold_lay=float(threshold_lay_price),
                    first_back_fav_tick=first_back,
                    first_lay_fav_tick=first_lay,
                ))

            global_tick += 1

    labels.sort(key=lambda r: (r.tick_index, r.runner_idx))
    return labels


def save_labels(
    labels: list[DirectionLabel],
    date: str,
    data_dir: Path,
    config: dict,
    *,
    direction_horizon_ticks: int | None = None,
    direction_horizon_seconds: float | None = None,
    direction_threshold_ticks: int,
    force_close_before_off_seconds: float,
    total_pre_race_ticks: int,
) -> Path:
    """Write the cache for *date*.

    Two layouts, one per label mode (hard_constraints §1):

    v1 ``v1_threshold_crossing``::

        {data_dir.parent}/direction_labels/{date}/
            horizon{H}_thresh{T}_fc{F}.npz
            horizon{H}_thresh{T}_fc{F}_header.json

    v2 ``v2_time_endpoint_signed_tick``::

        {data_dir.parent}/direction_labels/{date}/
            time_horizon{S}s_thresh{T}_fc{F}.npz
            time_horizon{S}s_thresh{T}_fc{F}_header.json

    Returns the .npz path.
    """
    n_modes = sum(
        x is not None
        for x in (direction_horizon_ticks, direction_horizon_seconds)
    )
    if n_modes != 1:
        raise ValueError(
            "save_labels: must pass exactly ONE of "
            "direction_horizon_ticks or direction_horizon_seconds",
        )
    is_time_endpoint = direction_horizon_seconds is not None
    cache_dir = _cache_dir(data_dir, date)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = _cache_stem(
        direction_horizon_ticks=direction_horizon_ticks,
        direction_horizon_seconds=direction_horizon_seconds,
        direction_threshold_ticks=direction_threshold_ticks,
        force_close_before_off_seconds=force_close_before_off_seconds,
    )
    out_path = cache_dir / f"{stem}.npz"
    header_path = cache_dir / f"{stem}_header.json"

    n = len(labels)
    n_back = sum(1 for r in labels if r.label_back > 0.5)
    n_lay = sum(1 for r in labels if r.label_lay > 0.5)
    n_both = sum(
        1 for r in labels if r.label_back > 0.5 and r.label_lay > 0.5
    )
    betting = config.get("training", {}).get("betting_constraints", {})

    label_version = (
        LABEL_VERSION_TIME_ENDPOINT if is_time_endpoint
        else LABEL_VERSION_TICK_CROSSING
    )
    header = {
        "label_version": label_version,
        "obs_schema_version": OBS_SCHEMA_VERSION,
        "direction_horizon_ticks": (
            int(direction_horizon_ticks)
            if direction_horizon_ticks is not None
            else None
        ),
        "direction_horizon_seconds": (
            float(direction_horizon_seconds)
            if direction_horizon_seconds is not None
            else None
        ),
        "direction_threshold_ticks": int(direction_threshold_ticks),
        "force_close_before_off_seconds": float(
            force_close_before_off_seconds,
        ),
        "junk_filter_max_dev_pct": _DEFAULT_MAX_DEV_PCT,
        "max_back_price": betting.get("max_back_price"),
        "max_lay_price": betting.get("max_lay_price"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit_sha": _git_sha(),
        "rows": n,
        "pre_race_ticks": int(total_pre_race_ticks),
        "positive_back": n_back,
        "positive_lay": n_lay,
        "positive_both": n_both,
        "density_back": (n_back / n) if n > 0 else 0.0,
        "density_lay": (n_lay / n) if n > 0 else 0.0,
        "density_both": (n_both / n) if n > 0 else 0.0,
    }
    header_path.write_text(json.dumps(header, indent=2), encoding="utf-8")

    _save_labels_atomic(labels, out_path)
    return out_path


def load_labels(
    date: str,
    data_dir: Path,
    *,
    direction_horizon_ticks: int | None = None,
    direction_horizon_seconds: float | None = None,
    direction_threshold_ticks: int,
    force_close_before_off_seconds: float,
    strict: bool = True,
) -> list[DirectionLabel]:
    """Load + verify the cache for *date*.

    Exactly ONE of ``direction_horizon_ticks`` /
    ``direction_horizon_seconds`` must be provided — the loader
    dispatches the cache filename and label_version verification per
    mode (hard_constraints §1, §5).

    Strict mode raises ``ValueError`` on any header mismatch
    (``label_version``, ``obs_schema_version``, the horizon-defining
    knobs, and the matcher reference values).
    """
    n_modes = sum(
        x is not None
        for x in (direction_horizon_ticks, direction_horizon_seconds)
    )
    if n_modes != 1:
        raise ValueError(
            "load_labels: must pass exactly ONE of "
            "direction_horizon_ticks or direction_horizon_seconds",
        )
    cache_dir = _cache_dir(data_dir, date)
    stem = _cache_stem(
        direction_horizon_ticks=direction_horizon_ticks,
        direction_horizon_seconds=direction_horizon_seconds,
        direction_threshold_ticks=direction_threshold_ticks,
        force_close_before_off_seconds=force_close_before_off_seconds,
    )
    npz_path = cache_dir / f"{stem}.npz"
    header_path = cache_dir / f"{stem}_header.json"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Direction-label cache not found: {npz_path}. "
            "Run `python -m training_v2.direction_label_cli scan ...` "
            "first.",
        )
    if not header_path.exists():
        raise FileNotFoundError(
            f"Direction-label header not found: {header_path}. "
            "Cache is corrupted; re-run the scan.",
        )

    header = json.loads(header_path.read_text(encoding="utf-8"))
    if strict:
        _verify_header(
            header,
            direction_horizon_ticks=direction_horizon_ticks,
            direction_horizon_seconds=direction_horizon_seconds,
            direction_threshold_ticks=direction_threshold_ticks,
            force_close_before_off_seconds=force_close_before_off_seconds,
        )

    data = np.load(npz_path, allow_pickle=False)
    n = int(data["tick_index"].shape[0])
    return [
        DirectionLabel(
            tick_index=int(data["tick_index"][i]),
            runner_idx=int(data["runner_idx"][i]),
            label_back=float(data["label_back"][i]),
            label_lay=float(data["label_lay"][i]),
            ltp_at_open=float(data["ltp_at_open"][i]),
            threshold_back=float(data["threshold_back"][i]),
            threshold_lay=float(data["threshold_lay"][i]),
            first_back_fav_tick=int(data["first_back_fav_tick"][i]),
            first_lay_fav_tick=int(data["first_lay_fav_tick"][i]),
        )
        for i in range(n)
    ]


def density_for_date(
    date: str,
    data_dir: Path,
    *,
    direction_horizon_ticks: int | None = None,
    direction_horizon_seconds: float | None = None,
    direction_threshold_ticks: int,
    force_close_before_off_seconds: float,
) -> tuple[float, float]:
    """Return ``(density_back, density_lay)`` by reading header.json
    only. Returns ``(0.0, 0.0)`` if the cache is missing.
    """
    cache_dir = _cache_dir(data_dir, date)
    stem = _cache_stem(
        direction_horizon_ticks=direction_horizon_ticks,
        direction_horizon_seconds=direction_horizon_seconds,
        direction_threshold_ticks=direction_threshold_ticks,
        force_close_before_off_seconds=force_close_before_off_seconds,
    )
    header_path = cache_dir / f"{stem}_header.json"
    if not header_path.exists():
        return (0.0, 0.0)
    header = json.loads(header_path.read_text(encoding="utf-8"))
    return (
        float(header.get("density_back", 0.0)),
        float(header.get("density_lay", 0.0)),
    )


def count_pre_race_ticks(date: str, data_dir: Path) -> int:
    """Return the total number of pre-race ticks across all races."""
    from data.episode_builder import load_day
    try:
        day = load_day(date, data_dir)
    except FileNotFoundError:
        return 0
    return sum(
        1
        for race in day.races
        for tick in race.ticks
        if not tick.in_play
    )


# ── Internal helpers ─────────────────────────────────────────────────────────


def _runner_slot_map(race) -> dict[int, int]:
    """Sorted-sid → slot index (mirrors BetfairEnv's per-race map)."""
    sids: set[int] = set()
    for tick in race.ticks:
        for r in tick.runners:
            sids.add(r.selection_id)
    return {sid: i for i, sid in enumerate(sorted(sids))}


def _priceability_at_open(
    available_to_back,
    available_to_lay,
    ltp_T: float,
    max_back_price: float | None,
    max_lay_price: float | None,
) -> tuple[bool, bool]:
    """Apply the env-matcher rules at the OPEN tick — junk filter,
    price cap. Returns ``(priceable_back, priceable_lay)``.
    """
    valid_atb = [
        lv for lv in available_to_back
        if lv.size > 0.0
        and passes_junk_filter(lv.price, ltp_T, _DEFAULT_MAX_DEV_PCT)
    ]
    if valid_atb:
        best_back = max(lv.price for lv in valid_atb)
        priceable_back = passes_price_cap(best_back, max_back_price)
    else:
        priceable_back = False

    valid_atl = [
        lv for lv in available_to_lay
        if lv.size > 0.0
        and passes_junk_filter(lv.price, ltp_T, _DEFAULT_MAX_DEV_PCT)
    ]
    if valid_atl:
        best_lay = min(lv.price for lv in valid_atl)
        priceable_lay = passes_price_cap(best_lay, max_lay_price)
    else:
        priceable_lay = False

    return priceable_back, priceable_lay


def _resolve_endpoint_tick(
    race,
    tick_idx: int,
    horizon_seconds: float,
    force_close_seconds: float,
    *,
    market_start_ts: float,
    race_tick_ts: np.ndarray,
) -> int | None:
    """Find the LAST pre-race tick at or before (open_ts +
    horizon_seconds), bounded by the in-play start and the force-
    close cutoff.

    Returns the target tick index, or ``None`` when:

      * the target time falls past in-play (race is already running)
      * the target time falls inside the force-close window
      * the open tick is itself the last pre-race tick available

    Mirrors the betfair-predictors' NaN-label handling: if we can't
    observe the LTP at T+horizon, the label is missing and the row
    is skipped.
    """
    open_ts = float(race_tick_ts[tick_idx])
    target_ts = open_ts + float(horizon_seconds)
    n = len(race.ticks)
    last_valid: int | None = None
    for t in range(tick_idx + 1, n):
        tick = race.ticks[t]
        ts = float(race_tick_ts[t])
        if tick.in_play:
            break
        if (market_start_ts - ts) <= force_close_seconds:
            break
        if ts > target_ts:
            # Past the target — last_valid holds the LATEST tick
            # whose timestamp is ≤ target_ts.
            break
        last_valid = t
    return last_valid


def _signed_ticks_between(p_now: float, p_future: float) -> int:
    """Signed Betfair-tick distance: positive = price drifted UP
    (longer odds, lay-favourable), negative = price came IN (shorter
    odds, back-favourable).

    Mirrors `betfair-predictors/scripts/predictor/extract_labels_
    prototype.py::signed_ticks_between`. Sign convention:
    ``p_future > p_now`` ⇒ price drifted up ⇒ positive.
    """
    if p_future == p_now:
        return 0
    from env.tick_ladder import ticks_between
    n = ticks_between(p_now, p_future)
    return n if p_future > p_now else -n


def _resolve_close_tick(
    race,
    tick_idx: int,
    force_close_seconds: float,
    horizon_ticks: int,
    *,
    market_start_ts: float,
) -> int:
    """Last tick index inclusive at which the fill scan should still
    consider the price.

    Bounded by:
      - the force-close wall-time cutoff
      - the in-play boundary
      - the ``horizon_ticks`` tick-count cap
    """
    horizon_cap = tick_idx + horizon_ticks
    n = len(race.ticks)
    for t in range(tick_idx + 1, n):
        tick = race.ticks[t]
        ts = tick.timestamp.timestamp()
        if tick.in_play:
            return min(t - 1, horizon_cap)
        if (market_start_ts - ts) <= force_close_seconds:
            return min(t - 1, horizon_cap)
    return min(n - 1, horizon_cap)


def _cache_dir(data_dir: Path, date: str) -> Path:
    return data_dir.parent / "direction_labels" / date


def _cache_stem(
    *,
    direction_horizon_ticks: int | None = None,
    direction_horizon_seconds: float | None = None,
    direction_threshold_ticks: int,
    force_close_before_off_seconds: float,
) -> str:
    """Compute the cache filename stem, dispatched by mode.

    v1 (tick crossing): ``horizon{H}_thresh{T}_fc{F}``
    v2 (time endpoint): ``time_horizon{S}s_thresh{T}_fc{F}``

    The two stems are disjoint, so the two cache families coexist on
    disk without collision (hard_constraints §1).
    """
    fc = force_close_before_off_seconds
    fc_token = f"{fc:g}".replace(".", "_")
    if direction_horizon_seconds is not None:
        s_token = f"{float(direction_horizon_seconds):g}".replace(
            ".", "_",
        )
        return (
            f"time_horizon{s_token}s"
            f"_thresh{int(direction_threshold_ticks)}"
            f"_fc{fc_token}"
        )
    if direction_horizon_ticks is None:
        raise ValueError(
            "_cache_stem: must pass exactly ONE of "
            "direction_horizon_ticks or direction_horizon_seconds",
        )
    return (
        f"horizon{int(direction_horizon_ticks)}"
        f"_thresh{int(direction_threshold_ticks)}"
        f"_fc{fc_token}"
    )


def _verify_header(
    header: dict,
    *,
    direction_horizon_ticks: int | None = None,
    direction_horizon_seconds: float | None = None,
    direction_threshold_ticks: int,
    force_close_before_off_seconds: float,
) -> None:
    is_time_endpoint = direction_horizon_seconds is not None
    expected_version = (
        LABEL_VERSION_TIME_ENDPOINT if is_time_endpoint
        else LABEL_VERSION_TICK_CROSSING
    )
    saved_version = header.get("label_version")
    if saved_version != expected_version:
        raise ValueError(
            f"Direction-label cache label_version={saved_version!r}, "
            f"caller expects {expected_version!r}. Re-run scan."
        )
    saved_obs = int(header.get("obs_schema_version", -1))
    if saved_obs != OBS_SCHEMA_VERSION:
        raise ValueError(
            f"Direction-label cache obs_schema_version={saved_obs} but "
            f"env expects OBS_SCHEMA_VERSION={OBS_SCHEMA_VERSION}. "
            "Re-run scan.",
        )
    if is_time_endpoint:
        saved_s = header.get("direction_horizon_seconds")
        if (
            saved_s is None
            or abs(float(saved_s) - float(direction_horizon_seconds)) > 1e-6
        ):
            raise ValueError(
                f"Direction-label cache horizon_seconds={saved_s}, "
                f"caller requested {direction_horizon_seconds}."
            )
    else:
        saved_h = int(header.get("direction_horizon_ticks", -1))
        if saved_h != int(direction_horizon_ticks):
            raise ValueError(
                f"Direction-label cache horizon={saved_h}, caller "
                f"requested {direction_horizon_ticks}.",
            )
    saved_t = int(header.get("direction_threshold_ticks", -1))
    if saved_t != int(direction_threshold_ticks):
        raise ValueError(
            f"Direction-label cache threshold={saved_t}, caller "
            f"requested {direction_threshold_ticks}.",
        )
    saved_fc = float(header.get("force_close_before_off_seconds", -1.0))
    if abs(saved_fc - float(force_close_before_off_seconds)) > 1e-9:
        raise ValueError(
            f"Direction-label cache force_close={saved_fc}, caller "
            f"requested {force_close_before_off_seconds}.",
        )


def _save_labels_atomic(
    labels: list[DirectionLabel], path: Path,
) -> None:
    """Write .npz to a tmp file then rename, mirroring arb_oracle."""
    tmp_stem = path.parent / (path.stem + "_tmp")
    tmp = path.parent / (path.stem + "_tmp.npz")
    n = len(labels)
    if n > 0:
        tick_arr = np.array([r.tick_index for r in labels], dtype=np.int32)
        runner_arr = np.array(
            [r.runner_idx for r in labels], dtype=np.int32,
        )
        lb = np.array([r.label_back for r in labels], dtype=np.float32)
        ll = np.array([r.label_lay for r in labels], dtype=np.float32)
        ltp = np.array([r.ltp_at_open for r in labels], dtype=np.float32)
        thb = np.array(
            [r.threshold_back for r in labels], dtype=np.float32,
        )
        thl = np.array(
            [r.threshold_lay for r in labels], dtype=np.float32,
        )
        fbk = np.array(
            [r.first_back_fav_tick for r in labels], dtype=np.int32,
        )
        flk = np.array(
            [r.first_lay_fav_tick for r in labels], dtype=np.int32,
        )
    else:
        tick_arr = np.empty(0, dtype=np.int32)
        runner_arr = np.empty(0, dtype=np.int32)
        lb = np.empty(0, dtype=np.float32)
        ll = np.empty(0, dtype=np.float32)
        ltp = np.empty(0, dtype=np.float32)
        thb = np.empty(0, dtype=np.float32)
        thl = np.empty(0, dtype=np.float32)
        fbk = np.empty(0, dtype=np.int32)
        flk = np.empty(0, dtype=np.int32)

    np.savez(
        tmp_stem,
        tick_index=tick_arr,
        runner_idx=runner_arr,
        label_back=lb,
        label_lay=ll,
        ltp_at_open=ltp,
        threshold_back=thb,
        threshold_lay=thl,
        first_back_fav_tick=fbk,
        first_lay_fav_tick=flk,
    )
    if path.exists():
        path.unlink()
    tmp.rename(path)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _load_config() -> dict:
    import yaml  # type: ignore[import-untyped]
    with open("config.yaml") as f:
        return yaml.safe_load(f)
