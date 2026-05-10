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
