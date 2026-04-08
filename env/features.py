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
