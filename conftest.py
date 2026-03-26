"""Shared pytest fixtures for the rl-betfair test suite."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.fixture(scope="session")
def config() -> dict:
    """Load and return the project config.yaml as a dict."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
