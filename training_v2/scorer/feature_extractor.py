"""Feature extraction for the Phase 0 supervised scorer.

Pure function ``FeatureExtractor.extract`` returns the locked feature set
described in
``plans/rewrite/phase-0-supervised-scorer/purpose.md`` (~25-35 features).
The extractor maintains rolling-window state per (market, runner) so the
velocity features (``traded_volume_last_30s``, ``ltp_change_last_30s``,
``spread_change_last_30s``, ``ltp_rank_change_last_60s``) can be computed
on the fly while iterating ticks.

The feature *order* is locked via ``FEATURE_NAMES``. ``feature_spec.json``
serialises this list so Session 02 (and Phase 1) read the same names in
the same positions. NaN propagates wherever a feature is unavailable
(e.g. no LTP, market just opened so no 30s window). The downstream tree
model handles missingness natively.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

from data.episode_builder import Race, RunnerSnap, Tick

# Declaration order is the contract — Session 02 / Phase 1 indexes into
# the dataset's columns by these names. Append, don't reorder.
FEATURE_NAMES: tuple[str, ...] = (
    # Price features
    "best_back",
    "best_lay",
    "ltp",
    "spread",
    "spread_in_ticks",
    "mid_price",
    # Book depth (top-3 levels each side, plus totals)
    "back_size_l1",
    "back_size_l2",
    "back_size_l3",
    "lay_size_l1",
    "lay_size_l2",
    "lay_size_l3",
    "total_back_size",
    "total_lay_size",
    # Time features
    "time_to_off_seconds",
    "time_since_last_trade_seconds",
    # Velocity (rolling windows)
    "traded_volume_last_30s",
    "ltp_change_last_30s",
    "spread_change_last_30s",
    # Side one-hot
    "side_back",
    "side_lay",
    # Runner attributes
    "favourite_rank",
    "sort_priority",
    "ltp_rank_change_last_60s",
    # Market attributes
    "n_active_runners",
    "total_market_volume",
    "total_market_volume_velocity",
    # Market type one-hot (covers WIN, EACH_WAY; "other" catches the
    # rest — keep small and stable. Append-only if a new type appears.)
    "market_type_win",
    "market_type_each_way",
    "market_type_other",
)


_VELOCITY_LTP_WINDOW_SEC = 30.0
_VELOCITY_SPREAD_WINDOW_SEC = 30.0
_VELOCITY_VOLUME_WINDOW_SEC = 30.0
_RANK_WINDOW_SEC = 60.0


@dataclass(slots=True)
class _RunnerHistory:
    """Per-runner rolling state for one market.

    Stores ``(timestamp, value)`` pairs for the rolling-window features.
    Pruned on the fly inside ``extract`` so each window stays bounded
    by its lookback period.
    """

    ltp: deque[tuple[float, float]] = field(default_factory=deque)
    spread: deque[tuple[float, float]] = field(default_factory=deque)
    total_matched: deque[tuple[float, float]] = field(default_factory=deque)
    rank: deque[tuple[float, int]] = field(default_factory=deque)
    last_total_matched: float | None = None
    last_trade_ts: float | None = None  # epoch seconds of most recent
                                          # observed total_matched bump


@dataclass(slots=True)
class _MarketHistory:
    """Per-market rolling state.

    Tracks market-level traded volume velocity. Per-runner state is
    nested via ``runners[selection_id]``.
    """

    runners: dict[int, _RunnerHistory] = field(default_factory=dict)
    market_volume: deque[tuple[float, float]] = field(default_factory=deque)


class FeatureExtractor:
    """Compute scorer features from a (race, tick_idx, runner_idx, side)
    opportunity tuple.

    The extractor is **stateful** across ticks within a single market —
    velocity features need the rolling history of LTP / spread / volume.
    Iterate ticks in chronological order and call
    :meth:`update_history` on every tick before calling :meth:`extract`
    on opportunities at that tick. (``extract`` also lazily updates the
    history if the caller has already advanced past it, so the order
    just needs to be monotone.)

    State is keyed by ``market_id`` so ticks from interleaved markets
    don't pollute each other; call :meth:`forget_market` once a market
    is fully processed to release memory.
    """

    def __init__(self) -> None:
        self._markets: dict[str, _MarketHistory] = {}

    # ── Public API ──────────────────────────────────────────────────────

    def update_history(self, race: Race, tick: Tick) -> None:
        """Append this tick's per-runner LTP / spread / volume into the
        rolling deques. Idempotent: replaying the same timestamp is a
        no-op (we only keep monotone-time entries).
        """
        market_state = self._markets.setdefault(race.market_id, _MarketHistory())
        ts = _ts(tick.timestamp)

        if market_state.market_volume and market_state.market_volume[-1][0] >= ts:
            return  # already up-to-date through this tick

        market_state.market_volume.append((ts, float(tick.traded_volume)))
        _prune(market_state.market_volume, ts, _VELOCITY_VOLUME_WINDOW_SEC)

        # Per-runner update + ranking by current LTP for the rank-velocity
        # feature.
        active_lt_pairs: list[tuple[int, float]] = []
        for runner in tick.runners:
            sid = runner.selection_id
            rstate = market_state.runners.setdefault(sid, _RunnerHistory())

            ltp = runner.last_traded_price
            if ltp is not None and ltp > 1.0:
                rstate.ltp.append((ts, float(ltp)))
                _prune(rstate.ltp, ts, _VELOCITY_LTP_WINDOW_SEC)
                if runner.status == "ACTIVE":
                    active_lt_pairs.append((sid, float(ltp)))

            best_back = _best_back(runner)
            best_lay = _best_lay(runner)
            if best_back is not None and best_lay is not None:
                spread = best_lay - best_back
                rstate.spread.append((ts, spread))
                _prune(rstate.spread, ts, _VELOCITY_SPREAD_WINDOW_SEC)

            tm = float(runner.total_matched)
            if rstate.last_total_matched is None:
                rstate.last_total_matched = tm
            elif tm > rstate.last_total_matched + 1e-9:
                rstate.last_trade_ts = ts
                rstate.last_total_matched = tm
            rstate.total_matched.append((ts, tm))
            _prune(rstate.total_matched, ts, _VELOCITY_VOLUME_WINDOW_SEC)

        # Ranking: sort active runners by LTP ascending; rank 1 = favourite.
        active_lt_pairs.sort(key=lambda kv: kv[1])
        for rank, (sid, _) in enumerate(active_lt_pairs, start=1):
            rstate = market_state.runners.setdefault(sid, _RunnerHistory())
            rstate.rank.append((ts, rank))
            _prune(rstate.rank, ts, _RANK_WINDOW_SEC)

    def forget_market(self, market_id: str) -> None:
        self._markets.pop(market_id, None)

    def extract(
        self,
        race: Race,
        tick_idx: int,
        runner_idx: int,
        side: str,
    ) -> dict[str, float]:
        """Return a feature dict (str → float) keyed by ``FEATURE_NAMES``.

        Caller is expected to have called :meth:`update_history` for
        every tick from the start of the race up to and including
        ``race.ticks[tick_idx]``. Order in declaration order is preserved.
        """
        if side not in ("back", "lay"):
            raise ValueError(f"side must be 'back' or 'lay', got {side!r}")
        tick = race.ticks[tick_idx]
        runner = tick.runners[runner_idx]
        ts = _ts(tick.timestamp)

        feats: dict[str, float] = {}
        market_state = self._markets.setdefault(race.market_id, _MarketHistory())
        rstate = market_state.runners.setdefault(
            runner.selection_id, _RunnerHistory(),
        )

        # ── Price features ─────────────────────────────────────────────
        best_back = _best_back(runner)
        best_lay = _best_lay(runner)
        ltp = runner.last_traded_price if (
            runner.last_traded_price is not None and runner.last_traded_price > 1.0
        ) else None

        feats["best_back"] = float(best_back) if best_back is not None else math.nan
        feats["best_lay"] = float(best_lay) if best_lay is not None else math.nan
        feats["ltp"] = float(ltp) if ltp is not None else math.nan

        if best_back is not None and best_lay is not None:
            spread = best_lay - best_back
            feats["spread"] = float(spread)
            feats["spread_in_ticks"] = _spread_in_ticks(best_back, best_lay)
            feats["mid_price"] = 0.5 * (best_back + best_lay)
        else:
            feats["spread"] = math.nan
            feats["spread_in_ticks"] = math.nan
            feats["mid_price"] = math.nan

        # ── Book depth ─────────────────────────────────────────────────
        for i, key in enumerate(("back_size_l1", "back_size_l2", "back_size_l3")):
            feats[key] = (
                float(runner.available_to_back[i].size)
                if i < len(runner.available_to_back) else 0.0
            )
        for i, key in enumerate(("lay_size_l1", "lay_size_l2", "lay_size_l3")):
            feats[key] = (
                float(runner.available_to_lay[i].size)
                if i < len(runner.available_to_lay) else 0.0
            )
        feats["total_back_size"] = float(sum(lv.size for lv in runner.available_to_back))
        feats["total_lay_size"] = float(sum(lv.size for lv in runner.available_to_lay))

        # ── Time features ──────────────────────────────────────────────
        race_off_ts = _ts(race.market_start_time)
        feats["time_to_off_seconds"] = float(race_off_ts - ts)
        if rstate.last_trade_ts is not None:
            feats["time_since_last_trade_seconds"] = float(ts - rstate.last_trade_ts)
        else:
            feats["time_since_last_trade_seconds"] = math.nan

        # ── Velocity (rolling 30s windows) ─────────────────────────────
        feats["traded_volume_last_30s"] = _delta_window(
            rstate.total_matched, ts, _VELOCITY_VOLUME_WINDOW_SEC,
        )
        feats["ltp_change_last_30s"] = _value_delta(
            rstate.ltp, ts, _VELOCITY_LTP_WINDOW_SEC,
        )
        feats["spread_change_last_30s"] = _value_delta(
            rstate.spread, ts, _VELOCITY_SPREAD_WINDOW_SEC,
        )

        # ── Side one-hot ───────────────────────────────────────────────
        feats["side_back"] = 1.0 if side == "back" else 0.0
        feats["side_lay"] = 1.0 if side == "lay" else 0.0

        # ── Runner attributes ──────────────────────────────────────────
        feats["favourite_rank"] = _current_rank(rstate)
        meta = race.runner_metadata.get(runner.selection_id)
        if meta is not None and meta.sort_priority and meta.sort_priority.strip():
            try:
                feats["sort_priority"] = float(meta.sort_priority)
            except (ValueError, TypeError):
                feats["sort_priority"] = math.nan
        else:
            feats["sort_priority"] = math.nan
        feats["ltp_rank_change_last_60s"] = _rank_delta(
            rstate, ts, _RANK_WINDOW_SEC,
        )

        # ── Market attributes ──────────────────────────────────────────
        n_active = sum(1 for r in tick.runners if r.status == "ACTIVE")
        feats["n_active_runners"] = float(n_active)
        feats["total_market_volume"] = float(tick.traded_volume)
        feats["total_market_volume_velocity"] = _delta_window(
            market_state.market_volume, ts, _VELOCITY_VOLUME_WINDOW_SEC,
        )

        # ── Market type one-hot ────────────────────────────────────────
        mtype = (race.market_type or "").upper()
        feats["market_type_win"] = 1.0 if mtype == "WIN" else 0.0
        feats["market_type_each_way"] = 1.0 if mtype == "EACH_WAY" else 0.0
        feats["market_type_other"] = (
            1.0 if mtype not in ("WIN", "EACH_WAY") else 0.0
        )

        # Verify FEATURE_NAMES coverage in dev — the keyset must match
        # exactly. Cheap to assert at extraction time (one extra dict
        # keys() call); catches regressions where a new feature is
        # appended to FEATURE_NAMES but the extractor forgets to set
        # it (or vice versa).
        assert set(feats.keys()) == set(FEATURE_NAMES), (
            f"FEATURE_NAMES / extracted keys mismatch: "
            f"missing={set(FEATURE_NAMES) - set(feats.keys())}, "
            f"extra={set(feats.keys()) - set(FEATURE_NAMES)}"
        )
        return feats


# ── Helpers ────────────────────────────────────────────────────────────


def _ts(dt: datetime) -> float:
    return dt.timestamp()


def _best_back(runner: RunnerSnap) -> float | None:
    if not runner.available_to_back:
        return None
    return max(lv.price for lv in runner.available_to_back if lv.price > 0.0 and lv.size > 0.0) \
        if any(lv.price > 0.0 and lv.size > 0.0 for lv in runner.available_to_back) else None


def _best_lay(runner: RunnerSnap) -> float | None:
    if not runner.available_to_lay:
        return None
    valid = [lv for lv in runner.available_to_lay if lv.price > 0.0 and lv.size > 0.0]
    return min(lv.price for lv in valid) if valid else None


def _spread_in_ticks(best_back: float, best_lay: float) -> float:
    """Approximate ticks between the two prices using the env tick ladder.

    Uses ``env.tick_ladder.tick_offset`` indirectly: walking from
    best_back upward by one tick at a time until we reach or exceed
    best_lay. For typical horse-market spreads this is 1-5 ticks; the
    loop cap protects against pathological inputs.
    """
    from env.tick_ladder import tick_offset

    if best_lay <= best_back:
        return 0.0
    p = best_back
    for n in range(1, 50):
        p = tick_offset(best_back, n, +1)
        if p >= best_lay - 1e-9:
            return float(n)
    return math.nan


def _prune(buf: deque, now_ts: float, window_sec: float) -> None:
    """Drop entries older than ``now_ts - window_sec`` from the front."""
    threshold = now_ts - window_sec
    while buf and buf[0][0] < threshold:
        buf.popleft()


def _delta_window(buf: Iterable[tuple[float, float]], now_ts: float,
                  window_sec: float) -> float:
    """Total change in cumulative metric across the window.

    ``buf`` holds (ts, cumulative_value) pairs. Returns
    ``last_value - first_value_within_window``. Returns 0.0 when the
    window has fewer than 2 entries (caller treats this as "no data" but
    a literal 0 is the truthful answer for cumulatives).
    """
    items = list(buf)
    if len(items) < 2:
        return 0.0
    threshold = now_ts - window_sec
    first_in_window = next(
        (v for ts, v in items if ts >= threshold), items[-1][1],
    )
    last = items[-1][1]
    return float(last - first_in_window)


def _value_delta(buf: Iterable[tuple[float, float]], now_ts: float,
                 window_sec: float) -> float:
    """Change in a non-cumulative metric (LTP, spread) over the window.

    Semantic: ``last - earliest_kept``. The deque has already been
    pruned by :func:`_prune` to keep only entries with ``ts >= now -
    window_sec``, so the earliest kept entry is at most ``window_sec``
    old — but may be younger if the race only just started or the
    metric only just became valid (e.g. LTP first emitted mid-race).
    Returning the partial-window delta is more useful for the model
    than NaN — early-race noise is encoded in ``time_to_off_seconds``
    so the consumer can condition on window-coverage downstream.

    NaN is reserved for "no data at all" (deque empty or single entry).
    """
    items = list(buf)
    if len(items) < 2:
        return math.nan
    return float(items[-1][1] - items[0][1])


def _current_rank(rstate: _RunnerHistory) -> float:
    if not rstate.rank:
        return math.nan
    return float(rstate.rank[-1][1])


def _rank_delta(rstate: _RunnerHistory, now_ts: float, window_sec: float) -> float:
    """Same partial-window semantic as :func:`_value_delta`."""
    if len(rstate.rank) < 2:
        return math.nan
    return float(rstate.rank[-1][1] - rstate.rank[0][1])
