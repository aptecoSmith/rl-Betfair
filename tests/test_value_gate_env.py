"""Tests for the env-side value-bet gate (constructor validation).

Per plans/non-scalping-directional-probe/hard_constraints.md §2/§3:
fast unit-level tests for kwarg validation. End-to-end gate-fires
behaviour proof lives in ``tools/probe_directional.py`` (Phase 3
pre-flight smoke) — env construction takes ~30-60s per day so full
pytest integration tests at this layer don't fit the test budget.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def real_day():
    from data.episode_builder import load_day
    return load_day(
        "2026-05-20", data_dir=Path("data/processed"),
    )


@pytest.fixture(scope="module")
def predictor_bundle():
    from predictors import PredictorBundle
    root = Path(__file__).resolve().parents[1]
    sibling = root.parent / "betfair-predictors"
    return PredictorBundle.from_manifests(
        champion_manifest=str(
            sibling / "production" / "race-outcome" / "manifest.json",
        ),
        ranker_manifest=str(
            sibling / "production" / "race-outcome-ranker" / "manifest.json",
        ),
        direction_manifest=str(
            sibling / "production" / "direction-predictor" / "manifest.json",
        ),
    )


def _build_env(day, bundle, **kwargs):
    from env.betfair_env import BetfairEnv
    from training_v2.cohort.worker import scalping_train_config
    cfg = scalping_train_config()
    cfg["training"]["strategy_mode"] = kwargs.pop(
        "strategy_mode_cfg", "value_win",
    )
    cfg["training"]["scalping_mode"] = False
    defaults = dict(
        predictor_bundle=bundle,
        use_race_outcome_predictor=True,
        use_direction_predictor=True,
        predictor_lean_obs=True,
    )
    defaults.update(kwargs)
    return BetfairEnv(day, cfg, **defaults)


@pytest.mark.timeout(300)
class TestValueGateValidation:
    """Constructor-time validation per env's loud-fail convention."""

    def test_negative_threshold_raises(self, real_day, predictor_bundle):
        with pytest.raises(ValueError, match="value_edge_threshold"):
            _build_env(
                real_day, predictor_bundle,
                value_edge_threshold=-0.1,
            )

    def test_threshold_without_predictor_raises(self, real_day):
        with pytest.raises(ValueError, match="use_race_outcome_predictor"):
            _build_env(
                real_day, None,
                use_race_outcome_predictor=False,
                use_direction_predictor=False,
                value_edge_threshold=0.05,
            )

    def test_back_stake_zero_raises(self, real_day, predictor_bundle):
        with pytest.raises(ValueError, match="directional_back_stake"):
            _build_env(
                real_day, predictor_bundle,
                directional_back_stake=0.0,
            )

    def test_lay_liability_negative_raises(self, real_day, predictor_bundle):
        with pytest.raises(ValueError, match="directional_lay_liability"):
            _build_env(
                real_day, predictor_bundle,
                directional_lay_liability=-5.0,
            )

    def test_threshold_zero_constructs(self, real_day, predictor_bundle):
        """value_edge_threshold = 0 ⇒ no validation requirements."""
        env = _build_env(real_day, predictor_bundle, value_edge_threshold=0.0)
        assert env._value_edge_threshold == 0.0
        assert env._value_gate_refusals == 0

    def test_overrides_default_none(self, real_day, predictor_bundle):
        env = _build_env(real_day, predictor_bundle)
        assert env._directional_back_stake is None
        assert env._directional_lay_liability is None
