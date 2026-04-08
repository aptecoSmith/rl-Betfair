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

    mp = (Σ back_size_i × back_price_i + Σ lay_size_i × lay_price_i)
         / (Σ back_size_i + Σ lay_size_i)

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
