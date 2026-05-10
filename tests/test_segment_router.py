"""Unit tests for predictors.segment_router.SegmentRouter.

Exercise rules from `plans/predictor-integration/predictor_contracts.md`
§1 + the Session 01 prompt's "Success bar":

    - test_loads_segment_performance — JSON loaded, axes indexed.
    - test_lookup_strong_segment       — known-strong combo returns STRONG.
    - test_lookup_weak_segment         — known-weak combo returns WEAK.
    - test_lookup_insufficient_data    — unseen bucket returns
                                         INSUFFICIENT_DATA.

Tests run against the real production sidecar
`betfair-predictors/production/race-outcome/segment_performance.json`.
The sibling repo's location is resolved relative to this rl-betfair
repo, mirroring the loader's `_BETFAIR_PREDICTORS_REPO` constant.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from predictors.segment_router import ConsumerHint, SegmentRouter


_RL_REPO_ROOT = Path(__file__).resolve().parents[1]
_BETFAIR_PREDICTORS_REPO = _RL_REPO_ROOT.parent / "betfair-predictors"
_CHAMPION_SEGMENTS = (
    _BETFAIR_PREDICTORS_REPO
    / "production"
    / "race-outcome"
    / "segment_performance.json"
)


def _require_sidecar() -> Path:
    if not _CHAMPION_SEGMENTS.exists():
        pytest.skip(
            f"sibling betfair-predictors repo not present at "
            f"{_BETFAIR_PREDICTORS_REPO}; predictor sidecar tests skipped"
        )
    return _CHAMPION_SEGMENTS


def test_loads_segment_performance():
    router = SegmentRouter.from_path(_require_sidecar())
    # Per predictor_contracts.md the champion's sidecar covers seven axes.
    expected_axes = {
        "field_size",
        "sp_band",
        "distance",
        "race_type",
        "surface",
        "agree_disagree_sp",
        "confidence_threshold",
    }
    assert expected_axes.issubset(set(router.axes))


def test_lookup_strong_segment():
    router = SegmentRouter.from_path(_require_sidecar())
    # field_size=12 is a documented strong segment per the manifest's
    # consumer_hints_summary.
    hint = router.lookup({"field_size": 12})
    assert hint is ConsumerHint.STRONG


def test_lookup_weak_segment():
    router = SegmentRouter.from_path(_require_sidecar())
    # field_size=5 is documented as a weak segment.
    hint = router.lookup({"field_size": 5})
    assert hint is ConsumerHint.WEAK


def test_lookup_insufficient_data():
    router = SegmentRouter.from_path(_require_sidecar())
    # Field size of 99 exists in no production market: unseen bucket.
    hint = router.lookup({"field_size": 99})
    assert hint is ConsumerHint.INSUFFICIENT_DATA


def test_weak_axis_dominates_strong_axis():
    """Reduce rule: WEAK on any axis trumps STRONG on another.

    Conservative — see SegmentRouter.lookup docstring. Exercised here
    so a future refactor that flips the precedence trips a known guard.
    """
    router = SegmentRouter.from_path(_require_sidecar())
    # field_size=12 is STRONG; field_size=5 is WEAK. Combined input
    # carries one weak axis -> overall WEAK.
    hint = router.lookup({"field_size": 5, "race_type": "Hcap"})
    assert hint is ConsumerHint.WEAK


def test_unknown_consumer_hint_raises():
    """A future segment_performance.json with an undocumented hint must
    NOT silently fall through (hard_constraints §10 — loader robustness)."""
    import json
    import tempfile

    payload = {
        "by_field_size": [
            {
                "bucket_label": "field_size=10",
                "bucket_definition": {"axis": "field_size", "value": 10},
                "consumer_hint": "completely_unknown_value",
                "n_markets_total": 99,
            }
        ]
    }
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as fh:
        json.dump(payload, fh)
        path = Path(fh.name)
    try:
        with pytest.raises(ValueError, match="unknown consumer_hint"):
            SegmentRouter.from_path(path)
    finally:
        path.unlink()
