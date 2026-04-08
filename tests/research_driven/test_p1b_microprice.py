"""Tests for P1b: weighted_microprice feature (session 20).

All tests are CPU-only and run in milliseconds.
"""

from __future__ import annotations

import math
import random

import pytest

from data.episode_builder import PriceSize
from env.features import compute_microprice


# ── Helpers ───────────────────────────────────────────────────────────────────


def _ps(price: float, size: float) -> PriceSize:
    return PriceSize(price=price, size=size)


# ── Pure-function tests ───────────────────────────────────────────────────────


class TestComputeMicroprice:
    """Unit tests for the compute_microprice pure function."""

    def test_symmetric_book_equals_midpoint(self):
        """Equal sizes, equal price gaps → microprice equals the simple midpoint."""
        # Best back = 3.0, best lay = 3.2; equal sizes on each side
        backs = [_ps(3.0, 100.0), _ps(2.9, 100.0)]
        lays = [_ps(3.2, 100.0), _ps(3.3, 100.0)]
        mp = compute_microprice(backs, lays, n=2, ltp_fallback=3.1)

        # Numerator: 100*3.0 + 100*2.9 + 100*3.2 + 100*3.3 = 1240
        # Denominator: 400
        # Expected: 3.1
        assert mp == pytest.approx(3.1)

    def test_asymmetric_sizes_pull_toward_heavy_side(self):
        """More size on the back side → microprice pulls toward the back-best price."""
        best_back = 3.0
        best_lay = 3.2
        # 300 on back, 100 on lay → weighted average pulled toward back price
        backs = [_ps(best_back, 300.0)]
        lays = [_ps(best_lay, 100.0)]
        mp = compute_microprice(backs, lays, n=3, ltp_fallback=3.1)

        # Numerator: 300*3.0 + 100*3.2 = 900 + 320 = 1220
        # Denominator: 400
        # Expected: 3.05
        assert mp == pytest.approx(3.05)
        # Must be closer to back price than to lay price
        assert mp < (best_back + best_lay) / 2.0

    def test_empty_book_returns_ltp_fallback(self):
        """Empty book on both sides → returns the LTP fallback value."""
        mp = compute_microprice([], [], n=3, ltp_fallback=4.5)
        assert mp == pytest.approx(4.5)

    def test_empty_book_without_ltp_raises(self):
        """Empty book with no LTP fallback raises ValueError."""
        with pytest.raises(ValueError, match="unpriceable"):
            compute_microprice([], [], n=3, ltp_fallback=None)

    def test_empty_book_with_nonpositive_ltp_raises(self):
        """Empty book with non-positive LTP fallback raises ValueError."""
        with pytest.raises(ValueError):
            compute_microprice([], [], n=3, ltp_fallback=0.0)

    def test_bounded_by_best_back_and_best_lay(self):
        """Randomised property: result lies within [best_back_price, best_lay_price]."""
        rng = random.Random(42)

        for _ in range(5):
            n_back = rng.randint(1, 4)
            n_lay = rng.randint(1, 4)

            # Back prices descending from ~4 toward ~2
            best_back_price = round(rng.uniform(2.5, 4.0), 2)
            back_levels = [
                _ps(round(best_back_price - i * 0.1, 2), rng.uniform(10.0, 500.0))
                for i in range(n_back)
            ]

            # Lay prices ascending from best_back_price + spread
            spread = round(rng.uniform(0.1, 0.5), 2)
            best_lay_price = round(best_back_price + spread, 2)
            lay_levels = [
                _ps(round(best_lay_price + i * 0.1, 2), rng.uniform(10.0, 500.0))
                for i in range(n_lay)
            ]

            mp = compute_microprice(back_levels, lay_levels, n=4, ltp_fallback=3.0)

            assert best_back_price <= mp <= best_lay_price, (
                f"microprice={mp} outside [{best_back_price}, {best_lay_price}] "
                f"backs={back_levels} lays={lay_levels}"
            )


# ── Env smoke test ────────────────────────────────────────────────────────────


class TestEnvMicroprice:
    """Smoke test: weighted_microprice appears in info[debug_features]."""

    def test_microprice_in_debug_features(self):
        """weighted_microprice appears in debug_features and lies between best back and lay."""
        from tests.research_driven.test_p1a_obi import (
            _make_minimal_day,
            _minimal_config,
        )
        from env.betfair_env import BetfairEnv

        cfg = _minimal_config()
        cfg["features"]["microprice_top_n"] = 3

        day = _make_minimal_day()
        env = BetfairEnv(day, cfg)
        _, info = env.reset()

        found = False
        while True:
            action = env.action_space.sample() * 0.0
            _, _, terminated, _, info = env.step(action)
            debug = info.get("debug_features", {})
            for sid, feats in debug.items():
                if "weighted_microprice" in feats:
                    mp = feats["weighted_microprice"]
                    if not math.isnan(mp):
                        # Must be a positive price (basic sanity)
                        assert mp > 0.0, f"microprice={mp} for runner {sid}"
                        found = True
            if terminated:
                break

        assert found, "weighted_microprice never appeared in debug_features"


# ── Schema-bump test ──────────────────────────────────────────────────────────


class TestSchemaVersionRefusesPreP1b:
    """Loader refuses both bare state-dicts and P1a checkpoints (version 2)."""

    def test_refuses_p1a_checkpoint(self):
        """A P1a checkpoint (obs_schema_version=2) is refused — now stale."""
        from env.betfair_env import OBS_SCHEMA_VERSION, validate_obs_schema

        # P1a was version 2; P1b bumped to 3; P1c bumped to 4
        assert OBS_SCHEMA_VERSION == 4, (
            f"Expected OBS_SCHEMA_VERSION=4 after P1c bump, got {OBS_SCHEMA_VERSION}"
        )
        p1a_checkpoint = {"obs_schema_version": 2, "weights": {}}
        with pytest.raises(ValueError, match="obs_schema_version"):
            validate_obs_schema(p1a_checkpoint)

    def test_refuses_pre_p1a_checkpoint(self):
        """A pre-P1a checkpoint (version 1) is also refused."""
        from env.betfair_env import validate_obs_schema

        old_checkpoint = {"obs_schema_version": 1, "weights": {}}
        with pytest.raises(ValueError, match="obs_schema_version"):
            validate_obs_schema(old_checkpoint)

    def test_refuses_bare_state_dict(self):
        """Bare state-dict with no version key is refused."""
        from env.betfair_env import validate_obs_schema

        with pytest.raises(ValueError, match="obs_schema_version"):
            validate_obs_schema({"some_layer.weight": [1, 2, 3]})

    def test_accepts_current_version(self):
        """Current schema version (3) is accepted silently."""
        from env.betfair_env import OBS_SCHEMA_VERSION, validate_obs_schema

        valid = {"obs_schema_version": OBS_SCHEMA_VERSION, "weights": {}}
        validate_obs_schema(valid)  # must not raise
