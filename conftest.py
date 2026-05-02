"""Shared pytest fixtures for the rl-betfair test suite."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

# Phase 4 Session 05 (2026-05-02): default tests to STRICT mode for
# the per-tick attribution invariant assert. Must be set BEFORE any
# test imports ``training_v2.discrete_ppo.rollout`` so that the
# module-level ``_STRICT_ATTRIBUTION = os.environ.get(...)`` read
# picks up the strict default.
#
# Pytest imports this conftest before any test module — and any test
# module that imports ``rollout`` triggers ``rollout``'s import-time
# env-var read AFTER this line has run. Production code (training
# runs that don't go through pytest) imports ``rollout`` without this
# env var set, so the production default is sampled mode.
#
# Tests that want to exercise sampled-mode behaviour monkeypatch the
# rollout module's ``_STRICT_ATTRIBUTION`` attribute back to ``False``
# for the duration of the test. See
# ``tests/test_v2_rollout_invariant_assert.py``.
#
# We respect a pre-set env var (e.g. an operator deliberately
# disabling strict mode in CI to soak the sampled path) — only set
# the default if nothing was set before pytest started.
os.environ.setdefault("PHASE4_STRICT_ATTRIBUTION", "1")


@pytest.fixture(scope="session")
def config() -> dict:
    """Load and return the project config.yaml as a dict."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
