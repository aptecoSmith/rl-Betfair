"""Regression guards for the persisted scorer v1 model.

These tests load the artefacts produced by
``training_v2.scorer.train_and_evaluate`` and verify the inference
contract that Phase 1's actor will rely on:

* The booster + calibrator load from disk and produce a working
  ``predict`` callable.
* A fixed feature vector produces a deterministic probability (within
  float tolerance).
* Calibrated predictions live in ``[0, 1]``.
* The persisted ``feature_spec.json`` matches the booster's expected
  feature names + count.

If ``models/scorer_v1/`` is missing (no training run yet), the tests
``skip``: training is the operator's responsibility, not CI's.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "models" / "scorer_v1"
MODEL_PATH = MODEL_DIR / "model.lgb"
CALIBRATOR_PATH = MODEL_DIR / "calibrator.joblib"
FEATURE_SPEC_PATH = MODEL_DIR / "feature_spec.json"


pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=(
        f"Scorer artefacts missing under {MODEL_DIR}; run "
        "`python -m training_v2.scorer.train_and_evaluate` first."
    ),
)


@pytest.fixture(scope="module")
def feature_spec() -> dict:
    with FEATURE_SPEC_PATH.open() as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def booster():
    import lightgbm as lgb

    return lgb.Booster(model_file=str(MODEL_PATH))


@pytest.fixture(scope="module")
def calibrator():
    import joblib

    return joblib.load(CALIBRATOR_PATH)


def _make_feature_vector(spec: dict) -> np.ndarray:
    """A plausible-looking 1-row feature vector matching the spec order.

    Values are intentionally simple constants (price midline, sane
    sizes) so the determinism test below has a concrete signature.
    Booster predictions on this vector are checked against itself
    across calls — the absolute value isn't asserted (it would drift
    every time training re-runs), only stability across two calls.
    """
    names = spec["feature_names"]
    values = {
        "best_back": 4.0, "best_lay": 4.5, "ltp": 4.2, "spread": 0.5,
        "spread_in_ticks": 5.0, "mid_price": 4.25,
        "back_size_l1": 50.0, "back_size_l2": 30.0, "back_size_l3": 20.0,
        "lay_size_l1": 40.0, "lay_size_l2": 25.0, "lay_size_l3": 15.0,
        "total_back_size": 100.0, "total_lay_size": 80.0,
        "time_to_off_seconds": 120.0, "time_since_last_trade_seconds": 5.0,
        "traded_volume_last_30s": 200.0, "ltp_change_last_30s": 0.05,
        "spread_change_last_30s": 0.01,
        "side_back": 1.0, "side_lay": 0.0,
        "favourite_rank": 3.0, "sort_priority": 4.0,
        "ltp_rank_change_last_60s": 0.0,
        "n_active_runners": 12.0, "total_market_volume": 5000.0,
        "total_market_volume_velocity": 100.0,
        "market_type_win": 1.0, "market_type_each_way": 0.0, "market_type_other": 0.0,
    }
    return np.array(
        [[values.get(n, 0.0) for n in names]], dtype=np.float32,
    )


def test_booster_loads_and_predicts(booster, feature_spec) -> None:
    x = _make_feature_vector(feature_spec)
    p = booster.predict(x)
    assert p.shape == (1,)
    assert np.isfinite(p).all()
    assert 0.0 <= float(p[0]) <= 1.0


def test_booster_prediction_is_deterministic(booster, feature_spec) -> None:
    x = _make_feature_vector(feature_spec)
    p1 = booster.predict(x)
    p2 = booster.predict(x)
    assert np.allclose(p1, p2, rtol=0, atol=1e-12)


def test_calibrator_outputs_in_unit_interval(booster, calibrator, feature_spec) -> None:
    rng = np.random.default_rng(42)
    n = 1000
    raw_grid = rng.uniform(0.0, 1.0, size=n)
    cal = calibrator.predict(raw_grid)
    assert cal.shape == (n,)
    assert np.all(cal >= 0.0)
    assert np.all(cal <= 1.0)


def test_calibrator_pipeline_round_trip_in_unit_interval(
    booster, calibrator, feature_spec,
) -> None:
    """End-to-end: feature vector → raw → calibrated, must land in [0, 1]."""
    x = _make_feature_vector(feature_spec)
    raw = booster.predict(x)
    cal = calibrator.predict(raw)
    assert 0.0 <= float(cal[0]) <= 1.0


def test_feature_spec_matches_booster(booster, feature_spec) -> None:
    """The persisted feature_spec.json must match the booster contract.

    Phase 1's actor loads ``feature_spec.json`` to know what to compute
    and in what order. If the spec drifts from the booster's feature
    names or input width, online inference will silently mis-feed the
    model and produce garbage predictions.
    """
    spec_names = feature_spec["feature_names"]
    booster_names = booster.feature_name()
    assert spec_names == booster_names, (
        f"feature_spec.json names diverge from booster.feature_name(); "
        f"spec[0:5]={spec_names[:5]} booster[0:5]={booster_names[:5]}"
    )
    assert int(feature_spec["feature_count"]) == len(spec_names)
    assert booster.num_feature() == len(spec_names)


def test_feature_spec_dtype_is_float32(feature_spec) -> None:
    """Phase 1 must build a float32 input array; lock the contract."""
    assert feature_spec["dtype"] == "float32"


def test_predict_with_correct_shape_only(booster, feature_spec) -> None:
    """Wrong-width input should raise — catches silent feature drift."""
    n_feat = booster.num_feature()
    bad_x = np.zeros((1, n_feat - 1), dtype=np.float32)
    with pytest.raises(Exception):
        booster.predict(bad_x)
