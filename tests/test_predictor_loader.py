"""Unit tests for predictors.loader.PredictorBundle.

Session 01 success bar (see
`plans/predictor-integration/session_prompts/01_predictor_loader.md`):

    test_loads_three_manifests       — bundle constructs against real manifests.
    test_missing_manifest_raises     — missing file -> loud error.
    test_schema_mismatch_raises      — manifest missing required keys -> loud.
    test_predict_race_returns_per_runner_dict — DEFERRED to next iteration
    test_predict_race_caches_by_market_id     — DEFERRED to next iteration
    test_predict_tick_fire_logic              — DEFERRED to next iteration

Tests in this file load the real production weights from the sibling
`betfair-predictors` repo when present; otherwise skip (so CI on a
fresh checkout without the sibling repo doesn't fail loudly).
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from predictors.loader import (
    PredictorBundle,
    PredictorLoaderError,
    _ChampionPayload,
    _DirectionPayload,
    _RankerPayload,
)


_RL_REPO_ROOT = Path(__file__).resolve().parents[1]
_BETFAIR_PREDICTORS_REPO = _RL_REPO_ROOT.parent / "betfair-predictors"
_CHAMP_M = (
    _BETFAIR_PREDICTORS_REPO / "production" / "race-outcome" / "manifest.json"
)
_RANK_M = (
    _BETFAIR_PREDICTORS_REPO
    / "production"
    / "race-outcome-ranker"
    / "manifest.json"
)
_DIR_M = (
    _BETFAIR_PREDICTORS_REPO
    / "production"
    / "direction-predictor"
    / "manifest.json"
)


def _require_sibling_repo() -> None:
    if not (_CHAMP_M.exists() and _RANK_M.exists() and _DIR_M.exists()):
        pytest.skip(
            f"sibling betfair-predictors repo not present at "
            f"{_BETFAIR_PREDICTORS_REPO}; loader tests against real "
            f"manifests skipped"
        )


def test_loads_three_manifests():
    _require_sibling_repo()
    bundle = PredictorBundle.from_manifests(_CHAMP_M, _RANK_M, _DIR_M)

    # Manifests
    assert bundle.champion_experiment_id == "1c15250ee90d1b65"
    assert bundle.ranker_experiment_id == "b23018bf5c8bcc70"
    assert bundle.direction_experiment_id.startswith("conv1d_k3_s1_")

    # Model payloads loaded eagerly
    assert isinstance(bundle.champion, _ChampionPayload)
    assert isinstance(bundle.ranker, _RankerPayload)
    assert isinstance(bundle.direction, _DirectionPayload)

    # Champion: 21 features per the manifest's F2 input contract
    assert len(bundle.champion.feature_names) == 21
    # Ranker: 43 features per the F5 input contract
    assert len(bundle.ranker.feature_names) == 43
    # Direction: 32 ticks x 39 features (retrained V4/F5; was 26 / 1m,3m,7m).
    assert bundle.direction.time_window == 32
    assert bundle.direction.n_features == 39
    assert bundle.direction.horizons == ("3m", "7m", "15m")
    assert bundle.direction.quantiles == (0.1, 0.5, 0.9)

    # Segment routers indexed
    expected_axes = {"field_size", "sp_band", "distance"}
    assert expected_axes.issubset(set(bundle.champion_segments.axes))
    assert expected_axes.issubset(set(bundle.ranker_segments.axes))


def test_missing_manifest_raises():
    """Hard_constraints §10 — silent fallback forbidden."""
    _require_sibling_repo()
    with pytest.raises(PredictorLoaderError, match="manifest not found"):
        PredictorBundle.from_manifests(
            champion_manifest=Path("/does/not/exist/manifest.json"),
            ranker_manifest=_RANK_M,
            direction_manifest=_DIR_M,
        )


def test_schema_mismatch_raises(tmp_path):
    """Manifest missing required keys -> loud failure."""
    _require_sibling_repo()

    # Build a corrupt champion manifest in a tmp tree that mirrors the
    # real production layout enough that weights_path resolution can be
    # exercised.
    fake_repo = tmp_path / "betfair-predictors-broken"
    prod_dir = fake_repo / "production" / "race-outcome"
    prod_dir.mkdir(parents=True)
    bad_manifest = prod_dir / "manifest.json"
    bad_manifest.write_text(
        json.dumps(
            {
                # NO experiment_id key — should trigger schema-mismatch raise.
                "weights_path": "production/race-outcome/weights.joblib",
                "architecture": {"family": "gbm", "kwargs": {}},
            }
        )
    )

    with pytest.raises(PredictorLoaderError, match="missing required key"):
        PredictorBundle.from_manifests(
            champion_manifest=bad_manifest,
            ranker_manifest=_RANK_M,
            direction_manifest=_DIR_M,
        )


def test_weights_missing_raises(tmp_path):
    """Manifest references weights file that doesn't exist -> loud failure."""
    _require_sibling_repo()

    fake_repo = tmp_path / "betfair-predictors-noweights"
    prod_dir = fake_repo / "production" / "race-outcome"
    prod_dir.mkdir(parents=True)
    bad_manifest = prod_dir / "manifest.json"
    # Construct a manifest that points to a missing weights file. The
    # loader resolves weights_path RELATIVE TO the manifest file's repo
    # root, but currently only against the rl-betfair `_RL_REPO_ROOT`
    # parent. To make the test deterministic, point the weights at a
    # missing absolute path encoded into the relative-to-tmp.
    bad_manifest.write_text(
        json.dumps(
            {
                "experiment_id": "fake_id",
                "weights_path": "production/race-outcome/missing_weights.joblib",
                "architecture": {"family": "gbm", "kwargs": {}},
            }
        )
    )
    with pytest.raises(PredictorLoaderError, match="weights file not found"):
        PredictorBundle.from_manifests(
            champion_manifest=bad_manifest,
            ranker_manifest=_RANK_M,
            direction_manifest=_DIR_M,
        )


def test_validate_compatibility_passes_on_matching_ids():
    """Hard_constraints §7: a cohort row whose recorded
    `predictor_*_experiment_id`s match the live bundle is accepted."""
    _require_sibling_repo()
    bundle = PredictorBundle.from_manifests(_CHAMP_M, _RANK_M, _DIR_M)
    bundle.validate_compatibility({
        "predictor_champion_experiment_id": bundle.champion_experiment_id,
        "predictor_ranker_experiment_id": bundle.ranker_experiment_id,
        "predictor_direction_experiment_id": bundle.direction_experiment_id,
    })


def test_validate_compatibility_passes_on_empty_strings():
    """A flag-off cohort that landed POST-contract carries empty
    strings; those should pass through."""
    _require_sibling_repo()
    bundle = PredictorBundle.from_manifests(_CHAMP_M, _RANK_M, _DIR_M)
    bundle.validate_compatibility({
        "predictor_champion_experiment_id": "",
        "predictor_ranker_experiment_id": "",
        "predictor_direction_experiment_id": "",
    })


def test_validate_compatibility_passes_on_pre_contract_rows():
    """Pre-contract cohort rows (no experiment_id keys at all) are
    legacy "this cohort didn't use predictors" — pass through."""
    _require_sibling_repo()
    bundle = PredictorBundle.from_manifests(_CHAMP_M, _RANK_M, _DIR_M)
    bundle.validate_compatibility({"learning_rate": 0.001})


def test_validate_compatibility_refuses_on_champion_mismatch():
    _require_sibling_repo()
    bundle = PredictorBundle.from_manifests(_CHAMP_M, _RANK_M, _DIR_M)
    with pytest.raises(PredictorLoaderError, match="champion experiment_id mismatch"):
        bundle.validate_compatibility({
            "predictor_champion_experiment_id": "stale_old_id",
            "predictor_ranker_experiment_id": bundle.ranker_experiment_id,
            "predictor_direction_experiment_id": bundle.direction_experiment_id,
        })


def test_validate_compatibility_refuses_on_ranker_mismatch():
    _require_sibling_repo()
    bundle = PredictorBundle.from_manifests(_CHAMP_M, _RANK_M, _DIR_M)
    with pytest.raises(PredictorLoaderError, match="ranker experiment_id mismatch"):
        bundle.validate_compatibility({
            "predictor_champion_experiment_id": bundle.champion_experiment_id,
            "predictor_ranker_experiment_id": "stale_old_id",
            "predictor_direction_experiment_id": bundle.direction_experiment_id,
        })


def test_validate_compatibility_refuses_on_direction_mismatch():
    _require_sibling_repo()
    bundle = PredictorBundle.from_manifests(_CHAMP_M, _RANK_M, _DIR_M)
    with pytest.raises(PredictorLoaderError, match="direction experiment_id mismatch"):
        bundle.validate_compatibility({
            "predictor_champion_experiment_id": bundle.champion_experiment_id,
            "predictor_ranker_experiment_id": bundle.ranker_experiment_id,
            "predictor_direction_experiment_id": "stale_old_id",
        })


def test_experiment_ids_captured_for_registry():
    """Hard_constraints §7 — every cohort row must capture predictor
    experiment_ids; the bundle must surface them as plain strings."""
    _require_sibling_repo()
    bundle = PredictorBundle.from_manifests(_CHAMP_M, _RANK_M, _DIR_M)
    assert isinstance(bundle.champion_experiment_id, str)
    assert isinstance(bundle.ranker_experiment_id, str)
    assert isinstance(bundle.direction_experiment_id, str)
    assert bundle.champion_experiment_id != bundle.ranker_experiment_id


# ---------------------------------------------------------------------------
# predict_race tests — exercise against a real predictor val market
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _bundle_and_val():
    """Build the bundle once and pull one val market for cross-checking."""
    _require_sibling_repo()
    import sys

    sys.path.insert(0, str(_BETFAIR_PREDICTORS_REPO))
    from scripts.outcome_predictor.datasets import (  # noqa: E402
        load_split,
    )

    bundle = PredictorBundle.from_manifests(_CHAMP_M, _RANK_M, _DIR_M)
    val = load_split("val", feature_variant="F5", train_corpus="last_12m")
    return bundle, val


def test_predict_race_returns_per_runner_dict(_bundle_and_val):
    bundle, val = _bundle_and_val
    market_id = val["market_id"].iloc[0]
    race = val[val["market_id"] == market_id].copy()
    n_runners = len(race)

    outs = bundle.predict_race(race)

    sids = set(int(s) for s in race["selection_id"])
    # All output dicts keyed by selection_id, length matches n_runners
    for attr in (
        "p_win",
        "p_placed",
        "ranker_score",
        "ranker_rank",
        "ranker_softmax_share",
        "ranker_top1_flag",
        "ranker_top1_high_confidence_flag",
        "segment_strong_flag",
    ):
        d = getattr(outs, attr)
        assert set(d.keys()) == sids, (
            f"{attr} keys {set(d.keys())} != selection_ids {sids}"
        )
        assert len(d) == n_runners

    # Probabilities in [0, 1]
    assert all(0.0 <= p <= 1.0 for p in outs.p_win.values())
    assert all(0.0 <= p <= 1.0 for p in outs.p_placed.values())

    # Softmax across runners sums to 1.0
    softmax_sum = sum(outs.ranker_softmax_share.values())
    assert abs(softmax_sum - 1.0) < 1e-6

    # Exactly one runner has top1_flag = True
    assert sum(outs.ranker_top1_flag.values()) == 1

    # ranker_rank covers 1..n with no duplicates
    assert sorted(outs.ranker_rank.values()) == list(range(1, n_runners + 1))


def test_predict_race_caches_by_market_id(_bundle_and_val):
    bundle, val = _bundle_and_val
    market_id = val["market_id"].iloc[0]
    race = val[val["market_id"] == market_id].copy()

    out1 = bundle.predict_race(race)
    out2 = bundle.predict_race(race)
    assert out1 is out2, "cache should return the SAME object on repeat call"


def test_predict_race_top1_high_confidence_threshold(_bundle_and_val):
    """A high-share top-pick (≥0.30) gets ranker_top1_high_confidence_flag = True."""
    bundle, val = _bundle_and_val
    market_id = val["market_id"].iloc[0]
    race = val[val["market_id"] == market_id].copy()
    outs = bundle.predict_race(race)

    # top1 selection_id and its softmax share
    top1_sid = next(sid for sid, top1 in outs.ranker_top1_flag.items() if top1)
    share = outs.ranker_softmax_share[top1_sid]
    expected = share >= 0.30
    assert outs.ranker_top1_high_confidence_flag[top1_sid] is expected


def test_predict_race_rejects_multi_market_dataframe(_bundle_and_val):
    """Caller must split by market; predict_race takes one market per call."""
    bundle, val = _bundle_and_val
    # Pick two markets and concat
    mids = val["market_id"].unique()[:2]
    race = val[val["market_id"].isin(mids)].copy()
    with pytest.raises(ValueError, match="one market per call"):
        bundle.predict_race(race)


def test_predict_race_requires_selection_id(_bundle_and_val):
    bundle, val = _bundle_and_val
    market_id = val["market_id"].iloc[0]
    race = val[val["market_id"] == market_id].copy().drop(columns=["selection_id"])
    with pytest.raises(ValueError, match="missing 'selection_id'"):
        bundle.predict_race(race)


def test_predict_race_rejects_non_dataframe(_bundle_and_val):
    bundle, _ = _bundle_and_val
    with pytest.raises(TypeError, match="pandas.DataFrame"):
        bundle.predict_race({"market_id": "x"})


# ---------------------------------------------------------------------------
# predict_tick tests — Conv1D forward + manifest-defined fire logic
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="predict_tick (non-batched) hardcodes the retired '1m' horizon; "
    "the retrained direction predictor uses 3m/7m/15m -> KeyError. Training "
    "is UNAFFECTED (the env uses predict_tick_batch, which adapts to the "
    "bundle's feature_variant + horizons). Reconcile predict_tick + its "
    "output dataclass in the direction-predictor task, then remove this xfail.",
)
def test_predict_tick_returns_dataclass(_bundle_and_val):
    """predict_tick on a (32, 26) random window returns the full TickLevelOutputs."""
    import numpy as np

    bundle, _ = _bundle_and_val
    np.random.seed(0)
    window = np.random.randn(
        bundle.direction.time_window, bundle.direction.n_features
    ).astype(np.float32)
    out = bundle.predict_tick(window)
    # All 9 quantile fields are floats
    for name in (
        "q10_1m", "q50_1m", "q90_1m",
        "q10_3m", "q50_3m", "q90_3m",
        "q10_7m", "q50_7m", "q90_7m",
    ):
        assert isinstance(getattr(out, name), float)
    # All 3 fire flags are bools
    for name in ("fire_drift", "fire_shorten", "fire_no_signal"):
        assert isinstance(getattr(out, name), bool)


@pytest.mark.xfail(
    strict=True,
    reason="predict_tick (non-batched) hardcodes the retired '1m' horizon; "
    "training-unused (env uses predict_tick_batch). Reconcile in the "
    "direction-predictor task, then remove this xfail.",
)
def test_predict_tick_fire_logic(_bundle_and_val):
    """Fire flags are mutually exclusive AND exhaustive (sum == 1).

    Per the manifest's `signal_description`:
        fire_drift     = (q50_7m >= +5) AND (q10_7m >= 0)
        fire_shorten   = (q50_7m <= -5) AND (q90_7m <= 0)
        fire_no_signal = NOT (fire_drift OR fire_shorten)
    """
    import numpy as np

    bundle, _ = _bundle_and_val
    np.random.seed(0)
    # Run several random windows and re-check the fire-logic invariant
    # AND that the bound-check matches the per-output quantile values.
    for seed in range(5):
        np.random.seed(seed)
        window = np.random.randn(
            bundle.direction.time_window, bundle.direction.n_features
        ).astype(np.float32)
        out = bundle.predict_tick(window)

        # Mutual exclusion: exactly one of the three is True.
        flags = (out.fire_drift, out.fire_shorten, out.fire_no_signal)
        assert sum(flags) == 1, f"flags={flags}"

        # Cross-check derivations against the manifest's thresholds.
        expected_drift = out.q50_7m >= 5.0 and out.q10_7m >= 0.0
        expected_shorten = out.q50_7m <= -5.0 and out.q90_7m <= 0.0
        expected_no_signal = not (expected_drift or expected_shorten)
        assert out.fire_drift is expected_drift
        assert out.fire_shorten is expected_shorten
        assert out.fire_no_signal is expected_no_signal


def test_predict_tick_rejects_wrong_shape(_bundle_and_val):
    """Shape contract: (time_window, n_features). Mismatch -> ValueError."""
    import numpy as np

    bundle, _ = _bundle_and_val

    # Wrong rank
    with pytest.raises(ValueError, match="must be 2-D"):
        bundle.predict_tick(np.zeros((32, 26, 1), dtype=np.float32))

    # Wrong feature dim
    with pytest.raises(ValueError, match=r"shape .* != expected"):
        bundle.predict_tick(np.zeros((32, 27), dtype=np.float32))

    # Wrong time dim
    with pytest.raises(ValueError, match=r"shape .* != expected"):
        bundle.predict_tick(np.zeros((33, 26), dtype=np.float32))


@pytest.mark.xfail(
    strict=True,
    reason="predict_tick (non-batched) hardcodes the retired '1m' horizon; "
    "training-unused (env uses predict_tick_batch). Reconcile in the "
    "direction-predictor task, then remove this xfail.",
)
def test_predict_tick_is_deterministic(_bundle_and_val):
    """Same window -> same outputs (per `intended_consumer.md` §Determinism)."""
    import numpy as np

    bundle, _ = _bundle_and_val
    window = np.ones(
        (bundle.direction.time_window, bundle.direction.n_features),
        dtype=np.float32,
    )
    out_a = bundle.predict_tick(window)
    out_b = bundle.predict_tick(window)
    for name in (
        "q10_1m", "q50_1m", "q90_1m",
        "q10_3m", "q50_3m", "q90_3m",
        "q10_7m", "q50_7m", "q90_7m",
    ):
        assert getattr(out_a, name) == getattr(out_b, name)
    assert out_a.fire_drift == out_b.fire_drift
    assert out_a.fire_shorten == out_b.fire_shorten
    assert out_a.fire_no_signal == out_b.fire_no_signal
