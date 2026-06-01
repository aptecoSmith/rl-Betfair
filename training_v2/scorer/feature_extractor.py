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

import numpy as np

from data.episode_builder import Race, RunnerSnap, Tick
from env.tick_ladder import _LADDER_BANDS, MAX_PRICE, MIN_PRICE

# Module-level precomputed band tables for the O(1) `_spread_in_ticks`
# closed form. Imported once at module load; the band tuple in
# `env/tick_ladder.py` is the single source of truth.
_BAND_LOWS: tuple[float, ...] = tuple(b[0] for b in _LADDER_BANDS)
_BAND_HIGHS: tuple[float, ...] = tuple(b[1] for b in _LADDER_BANDS)
_BAND_STEPS: tuple[float, ...] = tuple(b[2] for b in _LADDER_BANDS)
_N_BANDS: int = len(_LADDER_BANDS)

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


# Stable name→index map for the array-writing fast path (``extract_array``,
# training-speedup-v2 Step 2). Derived from FEATURE_NAMES so a
# declaration-order edit auto-reindexes — the position a feature occupies
# in the array is ALWAYS its position in FEATURE_NAMES, which is the same
# order the dict path's consumer re-keys by. This is the load-bearing
# ordering contract; the byte-equality test
# (tests/test_extract_array_parity.py) guards it.
_N_FEATURES: int = len(FEATURE_NAMES)
_FN_INDEX: dict[str, int] = {name: i for i, name in enumerate(FEATURE_NAMES)}
_I_BEST_BACK = _FN_INDEX["best_back"]
_I_BEST_LAY = _FN_INDEX["best_lay"]
_I_LTP = _FN_INDEX["ltp"]
_I_SPREAD = _FN_INDEX["spread"]
_I_SPREAD_IN_TICKS = _FN_INDEX["spread_in_ticks"]
_I_MID_PRICE = _FN_INDEX["mid_price"]
_I_BACK_SIZE = (
    _FN_INDEX["back_size_l1"], _FN_INDEX["back_size_l2"], _FN_INDEX["back_size_l3"],
)
_I_LAY_SIZE = (
    _FN_INDEX["lay_size_l1"], _FN_INDEX["lay_size_l2"], _FN_INDEX["lay_size_l3"],
)
_I_TOTAL_BACK_SIZE = _FN_INDEX["total_back_size"]
_I_TOTAL_LAY_SIZE = _FN_INDEX["total_lay_size"]
_I_TIME_TO_OFF = _FN_INDEX["time_to_off_seconds"]
_I_TIME_SINCE_LAST_TRADE = _FN_INDEX["time_since_last_trade_seconds"]
_I_TRADED_VOL_30 = _FN_INDEX["traded_volume_last_30s"]
_I_LTP_CHANGE_30 = _FN_INDEX["ltp_change_last_30s"]
_I_SPREAD_CHANGE_30 = _FN_INDEX["spread_change_last_30s"]
_I_SIDE_BACK = _FN_INDEX["side_back"]
_I_SIDE_LAY = _FN_INDEX["side_lay"]
_I_FAV_RANK = _FN_INDEX["favourite_rank"]
_I_SORT_PRIORITY = _FN_INDEX["sort_priority"]
_I_LTP_RANK_CHANGE_60 = _FN_INDEX["ltp_rank_change_last_60s"]
_I_N_ACTIVE = _FN_INDEX["n_active_runners"]
_I_TOTAL_MKT_VOL = _FN_INDEX["total_market_volume"]
_I_TOTAL_MKT_VOL_VEL = _FN_INDEX["total_market_volume_velocity"]
_I_MKT_WIN = _FN_INDEX["market_type_win"]
_I_MKT_EW = _FN_INDEX["market_type_each_way"]
_I_MKT_OTHER = _FN_INDEX["market_type_other"]


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

    # phase-3 Option B: lifted from per-call assert in ``extract`` to
    # a class-level set so the hot path doesn't pay ``set(feats.keys())
    # == set(FEATURE_NAMES)`` (~5-10 µs × 184k calls/day) on every
    # invocation. The contract is still checked — see ``extract`` —
    # but only the FIRST call per FeatureExtractor pays the cost.
    _FEATURE_NAME_SET: frozenset[str] = frozenset(FEATURE_NAMES)

    def __init__(self) -> None:
        self._markets: dict[str, _MarketHistory] = {}
        self._extract_contract_verified: bool = False
        # Reused float64 scratch for the dict-returning ``extract`` path
        # (Step 2). float64 so ``float(scratch[i])`` is exact → the
        # returned dict is byte-identical to the pre-refactor dict.
        self._scratch64: np.ndarray = np.empty(_N_FEATURES, dtype=np.float64)

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

        Step 2 (training-speedup-v2): delegates the computation to
        :meth:`_extract_into` (the single source of truth shared with
        :meth:`extract_array`) writing a float64 scratch buffer, then
        builds the dict from it. ``float(scratch[i])`` at float64 is
        exact, so the returned dict is byte-identical to the pre-refactor
        method. The dict keyset is now FEATURE_NAMES by construction, so
        the old once-per-instance keyset assert is redundant and dropped;
        the load-bearing guard is the byte-equality test
        (``tests/test_extract_array_parity.py``) plus the golden harness.
        """
        out = self._scratch64
        self._extract_into(race, tick_idx, runner_idx, side, out)
        return {name: float(out[i]) for i, name in enumerate(FEATURE_NAMES)}

    def extract_array(
        self,
        race: Race,
        tick_idx: int,
        runner_idx: int,
        side: str,
        out: np.ndarray,
    ) -> None:
        """Write the feature vector directly into ``out`` (no dict).

        Step 2 hot-path replacement for ``extract`` + the consumer's
        ``np.asarray([d[name] for name in FEATURE_NAMES])`` re-key.
        ``out`` must be a writable length-``_N_FEATURES`` array; its dtype
        controls the stored precision. When ``out`` is float32 the write
        casts each value float64→float32 with the SAME IEEE round-to-even
        as the dict path's ``np.asarray(..., dtype=np.float32)`` — hence
        byte-identical (test-guarded).
        """
        self._extract_into(race, tick_idx, runner_idx, side, out)

    def _extract_into(
        self,
        race: Race,
        tick_idx: int,
        runner_idx: int,
        side: str,
        out: np.ndarray,
    ) -> None:
        """Compute every feature and write it into ``out[_I_*]``.

        THE single source of feature computation (shared by ``extract``
        and ``extract_array``). Every assignment mirrors the original
        ``extract`` body's expression exactly; only the destination
        (``out[index]`` vs ``feats[name]``) changed. Writes all
        ``_N_FEATURES`` positions on every code path.
        """
        if side not in ("back", "lay"):
            raise ValueError(f"side must be 'back' or 'lay', got {side!r}")
        tick = race.ticks[tick_idx]
        runner = tick.runners[runner_idx]
        ts = _ts(tick.timestamp)

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

        out[_I_BEST_BACK] = float(best_back) if best_back is not None else math.nan
        out[_I_BEST_LAY] = float(best_lay) if best_lay is not None else math.nan
        out[_I_LTP] = float(ltp) if ltp is not None else math.nan

        if best_back is not None and best_lay is not None:
            spread = best_lay - best_back
            out[_I_SPREAD] = float(spread)
            out[_I_SPREAD_IN_TICKS] = _spread_in_ticks(best_back, best_lay)
            out[_I_MID_PRICE] = 0.5 * (best_back + best_lay)
        else:
            out[_I_SPREAD] = math.nan
            out[_I_SPREAD_IN_TICKS] = math.nan
            out[_I_MID_PRICE] = math.nan

        # ── Book depth ─────────────────────────────────────────────────
        for i, idx in enumerate(_I_BACK_SIZE):
            out[idx] = (
                float(runner.available_to_back[i].size)
                if i < len(runner.available_to_back) else 0.0
            )
        for i, idx in enumerate(_I_LAY_SIZE):
            out[idx] = (
                float(runner.available_to_lay[i].size)
                if i < len(runner.available_to_lay) else 0.0
            )
        out[_I_TOTAL_BACK_SIZE] = float(sum(lv.size for lv in runner.available_to_back))
        out[_I_TOTAL_LAY_SIZE] = float(sum(lv.size for lv in runner.available_to_lay))

        # ── Time features ──────────────────────────────────────────────
        race_off_ts = _ts(race.market_start_time)
        out[_I_TIME_TO_OFF] = float(race_off_ts - ts)
        if rstate.last_trade_ts is not None:
            out[_I_TIME_SINCE_LAST_TRADE] = float(ts - rstate.last_trade_ts)
        else:
            out[_I_TIME_SINCE_LAST_TRADE] = math.nan

        # ── Velocity (rolling 30s windows) ─────────────────────────────
        out[_I_TRADED_VOL_30] = _delta_window(
            rstate.total_matched, ts, _VELOCITY_VOLUME_WINDOW_SEC,
        )
        out[_I_LTP_CHANGE_30] = _value_delta(
            rstate.ltp, ts, _VELOCITY_LTP_WINDOW_SEC,
        )
        out[_I_SPREAD_CHANGE_30] = _value_delta(
            rstate.spread, ts, _VELOCITY_SPREAD_WINDOW_SEC,
        )

        # ── Side one-hot ───────────────────────────────────────────────
        out[_I_SIDE_BACK] = 1.0 if side == "back" else 0.0
        out[_I_SIDE_LAY] = 1.0 if side == "lay" else 0.0

        # ── Runner attributes ──────────────────────────────────────────
        out[_I_FAV_RANK] = _current_rank(rstate)
        meta = race.runner_metadata.get(runner.selection_id)
        if meta is not None and meta.sort_priority and meta.sort_priority.strip():
            try:
                out[_I_SORT_PRIORITY] = float(meta.sort_priority)
            except (ValueError, TypeError):
                out[_I_SORT_PRIORITY] = math.nan
        else:
            out[_I_SORT_PRIORITY] = math.nan
        out[_I_LTP_RANK_CHANGE_60] = _rank_delta(
            rstate, ts, _RANK_WINDOW_SEC,
        )

        # ── Market attributes ──────────────────────────────────────────
        n_active = sum(1 for r in tick.runners if r.status == "ACTIVE")
        out[_I_N_ACTIVE] = float(n_active)
        out[_I_TOTAL_MKT_VOL] = float(tick.traded_volume)
        out[_I_TOTAL_MKT_VOL_VEL] = _delta_window(
            market_state.market_volume, ts, _VELOCITY_VOLUME_WINDOW_SEC,
        )

        # ── Market type one-hot ────────────────────────────────────────
        mtype = (race.market_type or "").upper()
        out[_I_MKT_WIN] = 1.0 if mtype == "WIN" else 0.0
        out[_I_MKT_EW] = 1.0 if mtype == "EACH_WAY" else 0.0
        out[_I_MKT_OTHER] = (
            1.0 if mtype not in ("WIN", "EACH_WAY") else 0.0
        )


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


def _band_index_and_snap(price: float) -> tuple[int, float]:
    """Return ``(band_index, snapped_price)``. O(B), B=10 bands.

    Mirrors the snap semantics of ``env.tick_ladder.snap_to_tick``
    inline so the closed-form ``_spread_in_ticks`` does not call any
    function from ``env/tick_ladder.py`` (those are the slow walking
    paths we are bypassing).
    """
    if price <= MIN_PRICE:
        return 0, MIN_PRICE
    if price >= MAX_PRICE:
        return _N_BANDS - 1, MAX_PRICE
    for i in range(_N_BANDS):
        lo = _BAND_LOWS[i]
        hi = _BAND_HIGHS[i]
        if lo <= price < hi:
            step = _BAND_STEPS[i]
            n_steps = round((price - lo) / step)
            snapped = round(lo + n_steps * step, 2)
            return i, snapped
    return _N_BANDS - 1, MAX_PRICE


def _spread_in_ticks(best_back: float, best_lay: float) -> float:
    """Closed-form O(1) ladder-tick distance between two prices.

    Returns the integer count of Betfair ladder ticks between the two
    prices (smallest n in [1, 49] such that the price ``n`` ticks above
    ``snap(best_back)`` reaches or exceeds ``best_lay - 1e-9``). Returns
    ``0.0`` if ``best_lay <= best_back``. Returns ``nan`` if the spread
    exceeds 49 ticks (the cap inherited from the original walk's
    ``range(1, 50)`` loop).

    Bit-identical to the iterative walk it replaces (validated against
    a 10 000 random-pair sample and 752 hand-constructed edge cases
    spanning every band boundary).
    """
    if best_lay <= best_back:
        return 0.0
    target = best_lay - 1e-9
    bi, p = _band_index_and_snap(best_back)
    # If snapped best_back is at MAX_PRICE, the original walk clamps at
    # MAX for every n; one walk reaches MAX, so n=1 if MAX >= target.
    if p >= MAX_PRICE:
        return 1.0 if MAX_PRICE >= target else math.nan

    n_total = 0
    while bi < _N_BANDS:
        hi = _BAND_HIGHS[bi]
        step = _BAND_STEPS[bi]
        # Ticks remaining in this band before reaching hi (exclusive of p).
        ticks_in_band = round((hi - p) / step)
        # Smallest k >= 1 with p + k*step >= target. Original walks at
        # least 1 tick before checking, so always k >= 1.
        if target <= p:
            k = 1
        else:
            diff = target - p
            k_float = diff / step
            k = max(1, int(math.ceil(k_float)))
            # FP guards: ceiling can over- or under-shoot by 1 ULP.
            while k > 1 and (k - 1) * step + p >= target:
                k -= 1
            while k * step + p < target:
                k += 1

        if k <= ticks_in_band:
            n_total += k
            return float(n_total) if n_total <= 49 else math.nan

        # Walk through entire band; advance to next band.
        n_total += ticks_in_band
        if n_total > 49:
            return math.nan
        if bi + 1 >= _N_BANDS:
            # Reached MAX_PRICE without finding target.
            return math.nan
        p = hi  # equals lo of next band
        bi += 1
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
