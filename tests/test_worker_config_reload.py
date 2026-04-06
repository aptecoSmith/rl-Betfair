"""Tests for TrainingWorker._reload_config_from_disk()."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from training.worker import TrainingWorker


def _make_worker(tmp: Path) -> TrainingWorker:
    """Create a TrainingWorker pointing at temp paths."""
    config = {
        "paths": {
            "registry_db": str(tmp / "models.db"),
            "model_weights": str(tmp / "weights"),
            "processed_data": str(tmp / "processed"),
        },
        "training": {
            "starting_budget": 100.0,
            "betting_constraints": {
                "max_back_price": None,
                "max_lay_price": None,
                "min_seconds_before_off": 0,
            },
        },
        "population": {"size": 10},
    }
    config_path = tmp / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return TrainingWorker(
        config=config,
        host="127.0.0.1",
        port=9999,
        config_path=str(config_path),
    )


class TestReloadConfigFromDisk:
    def test_reload_picks_up_disk_changes(self):
        """Changes to config.yaml on disk should be reflected after reload."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            worker = _make_worker(tmp_path)

            # Initial constraint is None (unlimited)
            assert worker.config["training"]["betting_constraints"]["max_back_price"] is None

            # Write new config to disk simulating an Admin UI save
            new_config = {
                "paths": worker.config["paths"],
                "training": {
                    "starting_budget": 100.0,
                    "betting_constraints": {
                        "max_back_price": 100.0,
                        "max_lay_price": None,
                        "min_seconds_before_off": 300,
                    },
                },
                "population": {"size": 10},
            }
            with open(worker.config_path, "w") as f:
                yaml.dump(new_config, f)

            # Reload
            result = worker._reload_config_from_disk()
            assert result is True

            # New values are picked up
            assert worker.config["training"]["betting_constraints"]["max_back_price"] == 100.0
            assert worker.config["training"]["betting_constraints"]["min_seconds_before_off"] == 300

    def test_reload_returns_false_on_missing_file(self):
        """Missing config file returns False and leaves config unchanged."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            worker = _make_worker(tmp_path)
            original_config = worker.config

            # Delete the file
            Path(worker.config_path).unlink()

            result = worker._reload_config_from_disk()
            assert result is False
            # Config unchanged
            assert worker.config is original_config

    def test_reload_returns_false_on_empty_file(self):
        """Empty config file returns False and leaves config unchanged."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            worker = _make_worker(tmp_path)
            original_config = worker.config

            # Truncate the file
            with open(worker.config_path, "w") as f:
                pass

            result = worker._reload_config_from_disk()
            assert result is False
            assert worker.config is original_config

    def test_reload_returns_false_on_invalid_yaml(self):
        """Invalid YAML returns False and leaves config unchanged."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            worker = _make_worker(tmp_path)
            original_config = worker.config

            with open(worker.config_path, "w") as f:
                f.write("invalid: yaml: content: [[[")

            result = worker._reload_config_from_disk()
            assert result is False
            assert worker.config is original_config

    def test_reload_preserves_all_config_sections(self):
        """Full config file sections should all be loaded."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            worker = _make_worker(tmp_path)

            full_config = {
                "paths": worker.config["paths"],
                "training": {
                    "starting_budget": 200.0,
                    "reevaluate_garaged_default": False,
                    "betting_constraints": {
                        "max_back_price": 50.0,
                        "max_lay_price": 25.0,
                        "min_seconds_before_off": 600,
                    },
                },
                "population": {
                    "size": 20,
                    "n_elite": 2,
                    "selection_top_pct": 0.4,
                    "mutation_rate": 0.25,
                },
            }
            with open(worker.config_path, "w") as f:
                yaml.dump(full_config, f)

            worker._reload_config_from_disk()

            assert worker.config["training"]["starting_budget"] == 200.0
            assert worker.config["training"]["reevaluate_garaged_default"] is False
            assert worker.config["population"]["size"] == 20
            assert worker.config["population"]["mutation_rate"] == 0.25
