"""
env/tick_ladder.py — Betfair Exchange price tick ladder utilities.

Betfair prices move on a non-linear tick ladder:

    1.01–2.00   →  0.01 increments  (100 ticks)
    2.00–3.00   →  0.02 increments  (50 ticks)
    3.00–4.00   →  0.05 increments  (20 ticks)
    4.00–6.00   →  0.10 increments  (20 ticks)
    6.00–10.0   →  0.20 increments  (20 ticks)
    10.0–20.0   →  0.50 increments  (20 ticks)
    20.0–30.0   →  1.00 increments  (10 ticks)
    30.0–50.0   →  2.00 increments  (10 ticks)
    50.0–100    →  5.00 increments  (10 ticks)
    100–1000    →  10.0 increments  (90 ticks)

Arbitrage counter-orders must rest at real ladder prices — a passive
order at a price that doesn't exist on the exchange would never fill.
The forced-arbitrage (scalping) feature uses ``tick_offset`` to
compute a paired passive price N ticks away from an aggressive fill.

Dependency-free (stdlib only) so the file can be vendored into the
live ``ai-betfair`` project without modification.
"""

from __future__ import annotations

# Ladder bands — (lower_bound_inclusive, upper_bound_exclusive, tick_size).
# Upper bound of the last band is 1000 inclusive; treated specially below.
_LADDER_BANDS: tuple[tuple[float, float, float], ...] = (
    (1.01, 2.00, 0.01),
    (2.00, 3.00, 0.02),
    (3.00, 4.00, 0.05),
    (4.00, 6.00, 0.10),
    (6.00, 10.0, 0.20),
    (10.0, 20.0, 0.50),
    (20.0, 30.0, 1.00),
    (30.0, 50.0, 2.00),
    (50.0, 100.0, 5.00),
    (100.0, 1000.0, 10.0),
)

MIN_PRICE: float = 1.01
MAX_PRICE: float = 1000.0


def _band_for(price: float) -> tuple[float, float, float]:
    """Return the ladder band containing *price* (clamped to [MIN, MAX])."""
    if price <= MIN_PRICE:
        return _LADDER_BANDS[0]
    if price >= MAX_PRICE:
        return _LADDER_BANDS[-1]
    for lo, hi, step in _LADDER_BANDS:
        if lo <= price < hi:
            return (lo, hi, step)
    # Exact upper endpoint (e.g. price == 1000.0) falls through.
    return _LADDER_BANDS[-1]


def snap_to_tick(price: float) -> float:
    """Snap an arbitrary price to the nearest valid Betfair tick."""
    if price <= MIN_PRICE:
        return MIN_PRICE
    if price >= MAX_PRICE:
        return MAX_PRICE
    lo, _hi, step = _band_for(price)
    # Steps are all whole-cent multiples except 0.01 band. Round to step grid
    # anchored at the band's lower bound.
    n_steps = round((price - lo) / step)
    snapped = lo + n_steps * step
    # Guard against float drift.
    return round(snapped, 2)


def tick_offset(price: float, n_ticks: int, direction: int) -> float:
    """Move *price* by ``n_ticks`` on the Betfair ladder.

    Parameters
    ----------
    price:
        Starting price. Snapped to the nearest valid tick before offsetting.
    n_ticks:
        Non-negative integer count of ticks to move. ``0`` returns the
        snapped starting price.
    direction:
        ``+1`` to move up (higher prices), ``-1`` to move down. Any other
        value raises ``ValueError``.

    Returns
    -------
    float
        The price ``n_ticks`` away on the Betfair ladder, clamped to
        ``[1.01, 1000]``. Band transitions use the destination band's
        tick size once the boundary is crossed.
    """
    if direction not in (-1, 1):
        raise ValueError(f"direction must be +1 or -1, got {direction}")
    if n_ticks < 0:
        raise ValueError(f"n_ticks must be >= 0, got {n_ticks}")

    current = snap_to_tick(price)
    if n_ticks == 0:
        return current

    for _ in range(n_ticks):
        if direction == 1 and current >= MAX_PRICE:
            return MAX_PRICE
        if direction == -1 and current <= MIN_PRICE:
            return MIN_PRICE
        lo, hi, step = _band_for(current)
        if direction == 1:
            nxt = current + step
            # If we've crossed into the next band, snap to its grid.
            if nxt >= hi:
                nxt = hi
            current = round(nxt, 2)
        else:
            # Stepping down: when exactly on a lower band boundary, the
            # next step uses the *previous* band's step size.
            if current == lo:
                # find previous band
                prev_band: tuple[float, float, float] | None = None
                for b in _LADDER_BANDS:
                    if b[1] == lo:
                        prev_band = b
                        break
                if prev_band is not None:
                    step = prev_band[2]
                    lo = prev_band[0]
            nxt = current - step
            if nxt < lo:
                nxt = lo
            if nxt < MIN_PRICE:
                nxt = MIN_PRICE
            current = round(nxt, 2)

    return current


def ticks_between(price_a: float, price_b: float) -> int:
    """Return the unsigned tick distance between two prices on the ladder.

    Both inputs are snapped first. Useful for computing passive fill
    proximity (how many ticks until the rest-price matches LTP).
    """
    a = snap_to_tick(price_a)
    b = snap_to_tick(price_b)
    if a == b:
        return 0
    direction = 1 if b > a else -1
    count = 0
    current = a
    # Upper bound prevents infinite loops if inputs are malformed.
    max_iter = 1000
    while current != b and count < max_iter:
        nxt = tick_offset(current, 1, direction)
        if nxt == current:
            break
        current = nxt
        count += 1
    return count
