"""env/features.py — Pure per-runner feature functions.

No dependencies beyond the standard library and duck-typed PriceLevel
inputs.  This file can be vendored verbatim into ai-betfair without
modification.

Design rules
------------
* Pure functions only — no side effects, no global state.
* No numpy, no env-internal imports.
* All inputs are duck-typed: any object with a ``.size`` attribute
  (e.g. ``PriceSize`` from ``data.episode_builder``) is accepted.
"""

from __future__ import annotations


def compute_microprice(back_levels, lay_levels, n: int, ltp_fallback) -> float:
    """Size-weighted midpoint of the top-N ladder levels per side.

    mp = (Σ back_size_i × best_back_price + Σ lay_size_i × best_lay_price)
         / (Σ back_size_i + Σ lay_size_i)

    The sizes from the top-N levels are summed and used as weights, but
    only the **best price on each side** (``levels[0].price``) is used in
    the numerator.  Using individual level prices (``price_i``) would pull
    the result outside ``[best_back, best_lay]`` whenever N > 1, because
    deeper back levels are priced *below* best_back and deeper lay levels
    are priced *above* best_lay.  Weighting on best prices preserves the
    N-level depth signal while guaranteeing the bounded constraint.

    Parameters
    ----------
    back_levels:
        Sequence of objects with ``.size`` and ``.price`` attributes, ordered
        best-first (highest back price first).  Only the first ``n`` entries
        are used.
    lay_levels:
        Same structure for the lay side (lowest lay price first).
    n:
        Number of levels to include per side.
    ltp_fallback:
        Last-traded price used when both sides sum to zero.  Must be a
        positive number; ``None`` or non-positive raises ``ValueError``.

    Returns
    -------
    float
        Size-weighted midpoint price.  Guaranteed to lie within
        ``[best_back_price, best_lay_price]`` for any non-degenerate book.

    Raises
    ------
    ValueError
        If both sides sum to zero *and* ``ltp_fallback`` is ``None`` or
        non-positive.  A runner with no liquidity and no LTP is
        unpriceable; do not silently return zero.
    """
    back_sliced = back_levels[:n]
    lay_sliced = lay_levels[:n]

    back_size_sum = sum(level.size for level in back_sliced)
    lay_size_sum = sum(level.size for level in lay_sliced)
    total_size = back_size_sum + lay_size_sum

    if total_size == 0.0:
        if ltp_fallback is None or ltp_fallback <= 0:
            raise ValueError(
                "compute_microprice: both sides sum to zero and ltp_fallback "
                f"is {ltp_fallback!r} (None or non-positive). "
                "A runner with no liquidity and no LTP is unpriceable."
            )
        return float(ltp_fallback)

    # Use the best price from each side (levels[0].price) weighted by the
    # total size across the top-N levels.  Using individual level prices
    # would allow levels beyond the best to pull the result outside the
    # [best_back, best_lay] range, violating the bounded constraint
    # (hard_constraints.md §12 / session-20 spec §2).
    best_back_price = back_sliced[0].price if back_sliced else None
    best_lay_price = lay_sliced[0].price if lay_sliced else None

    if best_back_price is None:
        # Only lay side has liquidity — result is best lay price
        return float(best_lay_price)  # type: ignore[arg-type]
    if best_lay_price is None:
        # Only back side has liquidity — result is best back price
        return float(best_back_price)

    return (back_size_sum * best_back_price + lay_size_sum * best_lay_price) / total_size


def betfair_tick_size(price: float) -> float:
    """Return the Betfair price tick size at the given price.

    Standard horse-racing ladder::

        1.01–2.00 → 0.01
        2.00–3.00 → 0.02
        3.00–4.00 → 0.05
        4.00–6.00 → 0.10
        6.00–10.0 → 0.20
        10.0–20.0 → 0.50
        20.0–30.0 → 1.00
        30.0–50.0 → 2.00
        50.0–100. → 5.00
        100–1000  → 10.0
    """
    if price < 2.0:
        return 0.01
    elif price < 3.0:
        return 0.02
    elif price < 4.0:
        return 0.05
    elif price < 6.0:
        return 0.10
    elif price < 10.0:
        return 0.20
    elif price < 20.0:
        return 0.50
    elif price < 30.0:
        return 1.00
    elif price < 50.0:
        return 2.00
    elif price < 100.0:
        return 5.00
    else:
        return 10.0


def compute_traded_delta(
    history,
    reference_microprice: float,
    window_seconds: float,
    now_ts: float,
) -> float:
    """Signed net traded volume relative to current microprice over the window.

    Each entry in *history* is a ``(timestamp_s, microprice, vol_delta)``
    tuple.  Volume that traded when microprice ≤ reference_microprice is
    counted positively (backing pressure); above reference_microprice is
    negative (laying pressure).

    Parameters
    ----------
    history:
        Iterable of ``(timestamp_s, microprice, vol_delta)`` tuples,
        ordered oldest-first.  ``timestamp_s`` must be in the same unit
        as ``now_ts`` (Unix epoch seconds).
    reference_microprice:
        Current microprice — the comparison baseline for sign assignment.
    window_seconds:
        Length of the look-back window in wall-clock seconds.
    now_ts:
        Current timestamp in the same unit as history timestamps.

    Returns
    -------
    float
        Positive → net backing pressure; negative → net laying pressure.
        Returns ``0.0`` when history is empty or no entries fall within
        the window.
    """
    cutoff = now_ts - window_seconds
    total = 0.0
    for ts, mp, delta in history:
        if ts < cutoff:
            continue
        total += delta if mp <= reference_microprice else -delta
    return total


def compute_mid_drift(
    history,
    window_seconds: float,
    now_ts: float,
    tick_size_fn,
) -> float:
    """Change in weighted_microprice over the window, in Betfair price ticks.

    Finds the entry in *history* whose timestamp is nearest to
    ``now_ts - window_seconds`` and still at or before that point (i.e.
    the latest entry with ``ts ≤ now_ts - window_seconds``).  If no such
    entry exists, returns ``0.0``.  The drift is the difference between
    the most recent microprice and that baseline microprice, divided by
    the Betfair tick size at the baseline price.

    Parameters
    ----------
    history:
        Iterable of ``(timestamp_s, microprice, vol_delta)`` tuples,
        ordered oldest-first.
    window_seconds:
        Length of the look-back window in wall-clock seconds.
    now_ts:
        Current timestamp in the same unit as history timestamps.
    tick_size_fn:
        Callable ``(price: float) -> float``.  Returns the Betfair tick
        size at *price*.  Injected by the caller so this function stays
        dependency-free.

    Returns
    -------
    float
        Drift in Betfair price ticks.  Positive → price rose (runner
        drifted out / became longer odds); negative → price fell (runner
        shortened).  Returns ``0.0`` if fewer than two entries are
        available or no baseline exists.
    """
    history_list = list(history)
    if len(history_list) < 2:
        return 0.0

    cutoff = now_ts - window_seconds
    baseline_mp: float | None = None
    for ts, mp, _ in history_list:
        if ts <= cutoff:
            baseline_mp = mp  # keep updating; final value = latest at-or-before cutoff

    if baseline_mp is None:
        return 0.0

    current_mp = history_list[-1][1]
    diff = current_mp - baseline_mp
    if diff == 0.0:
        return 0.0

    ts_size = tick_size_fn(baseline_mp)
    if ts_size <= 0.0:
        return 0.0
    return diff / ts_size


def compute_obi(back_levels, lay_levels, n: int) -> float:
    """Order Book Imbalance for the top-N ladder levels.

    obi = (sum(back_size_top_N) - sum(lay_size_top_N))
          / (sum(back_size_top_N) + sum(lay_size_top_N))

    Parameters
    ----------
    back_levels:
        Sequence of objects with a ``.size`` attribute, ordered best-first
        (i.e. highest back price first).  Only the first ``n`` entries are
        used.
    lay_levels:
        Same structure for the lay side (lowest lay price first).
    n:
        Number of levels to include.

    Returns
    -------
    float
        Value in ``[-1.0, 1.0]``.  Returns ``0.0`` when both sides sum to
        zero (empty book or all-NaN sizes).
    """
    back_sum = sum(level.size for level in back_levels[:n])
    lay_sum = sum(level.size for level in lay_levels[:n])
    total = back_sum + lay_sum
    if total == 0.0:
        return 0.0
    return (back_sum - lay_sum) / total
