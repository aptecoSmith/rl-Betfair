"""Tests for TrainingWorker._apply_run_overrides()."""

from __future__ import annotations

import pytest

from training.worker import TrainingWorker


def _base_config() -> dict:
    return {
        "paths": {"processed_data": "data/processed"},
        "training": {
            "architecture": "ppo_lstm_v1",
            "starting_budget": 100.0,
            "betting_constraints": {
                "max_back_price": None,
                "max_lay_price": None,
                "min_seconds_before_off": 0,
            },
        },
        "population": {
            "size": 50,
            "n_elite": 5,
        },
        "hyperparameters": {
            "search_ranges": {
                "architecture_name": {
                    "type": "str_choice",
                    "choices": ["ppo_lstm_v1", "ppo_time_lstm_v1"],
                },
            },
        },
    }


class TestApplyRunOverrides:
    def test_no_overrides_returns_deep_copy(self):
        """With no params, the result should equal the base but be a copy."""
        base = _base_config()
        result = TrainingWorker._apply_run_overrides(base, {})
        assert result == base
        assert result is not base
        # Modifying result shouldn't affect base
        result["population"]["size"] = 999
        assert base["population"]["size"] == 50

    def test_population_size_override(self):
        """population_size param overrides config and scales n_elite."""
        result = TrainingWorker._apply_run_overrides(_base_config(), {"population_size": 20})
        assert result["population"]["size"] == 20
        assert result["population"]["n_elite"] == 2  # max(1, 20 // 10)

    def test_population_size_small_scales_n_elite_to_one(self):
        """Population of 5 should give n_elite = 1."""
        result = TrainingWorker._apply_run_overrides(_base_config(), {"population_size": 5})
        assert result["population"]["size"] == 5
        assert result["population"]["n_elite"] == 1

    def test_population_size_none_keeps_config(self):
        """population_size=None leaves the config default."""
        result = TrainingWorker._apply_run_overrides(_base_config(), {"population_size": None})
        assert result["population"]["size"] == 50
        assert result["population"]["n_elite"] == 5

    def test_architectures_override_restricts_choices(self):
        """architectures param restricts the search range to the given list."""
        result = TrainingWorker._apply_run_overrides(
            _base_config(),
            {"architectures": ["ppo_lstm_v1"]},
        )
        arch_spec = result["hyperparameters"]["search_ranges"]["architecture_name"]
        assert arch_spec["choices"] == ["ppo_lstm_v1"]
        assert arch_spec["type"] == "str_choice"

    def test_single_architecture_sets_training_default(self):
        """When only one architecture is selected, training.architecture is set."""
        result = TrainingWorker._apply_run_overrides(
            _base_config(),
            {"architectures": ["ppo_time_lstm_v1"]},
        )
        assert result["training"]["architecture"] == "ppo_time_lstm_v1"

    def test_multiple_architectures_preserves_training_default(self):
        """With multiple architectures, training.architecture is not overridden."""
        result = TrainingWorker._apply_run_overrides(
            _base_config(),
            {"architectures": ["ppo_lstm_v1", "ppo_time_lstm_v1"]},
        )
        # Unchanged from base
        assert result["training"]["architecture"] == "ppo_lstm_v1"
        # But choices is the full list
        assert set(result["hyperparameters"]["search_ranges"]["architecture_name"]["choices"]) == {
            "ppo_lstm_v1", "ppo_time_lstm_v1",
        }

    def test_empty_architectures_list_ignored(self):
        """Empty architectures list should not override (treated as 'no preference')."""
        base = _base_config()
        result = TrainingWorker._apply_run_overrides(base, {"architectures": []})
        assert result["hyperparameters"]["search_ranges"]["architecture_name"]["choices"] == [
            "ppo_lstm_v1", "ppo_time_lstm_v1",
        ]

    def test_architectures_none_keeps_config(self):
        """architectures=None keeps the config choices."""
        result = TrainingWorker._apply_run_overrides(_base_config(), {"architectures": None})
        assert result["hyperparameters"]["search_ranges"]["architecture_name"]["choices"] == [
            "ppo_lstm_v1", "ppo_time_lstm_v1",
        ]

    def test_max_back_price_override(self):
        """max_back_price param overrides the betting constraint."""
        result = TrainingWorker._apply_run_overrides(
            _base_config(),
            {"max_back_price": 50.0},
        )
        assert result["training"]["betting_constraints"]["max_back_price"] == 50.0
        # Others unchanged
        assert result["training"]["betting_constraints"]["max_lay_price"] is None
        assert result["training"]["betting_constraints"]["min_seconds_before_off"] == 0

    def test_max_lay_price_override(self):
        result = TrainingWorker._apply_run_overrides(
            _base_config(),
            {"max_lay_price": 25.0},
        )
        assert result["training"]["betting_constraints"]["max_lay_price"] == 25.0

    def test_min_seconds_before_off_override(self):
        result = TrainingWorker._apply_run_overrides(
            _base_config(),
            {"min_seconds_before_off": 300},
        )
        assert result["training"]["betting_constraints"]["min_seconds_before_off"] == 300

    def test_all_constraint_overrides_together(self):
        result = TrainingWorker._apply_run_overrides(
            _base_config(),
            {
                "max_back_price": 100.0,
                "max_lay_price": 50.0,
                "min_seconds_before_off": 600,
            },
        )
        bc = result["training"]["betting_constraints"]
        assert bc["max_back_price"] == 100.0
        assert bc["max_lay_price"] == 50.0
        assert bc["min_seconds_before_off"] == 600

    def test_constraint_override_preserves_existing_admin_defaults(self):
        """If the base config has admin-set constraints, unspecified overrides keep them."""
        base = _base_config()
        base["training"]["betting_constraints"]["max_back_price"] = 200.0
        base["training"]["betting_constraints"]["min_seconds_before_off"] = 60

        # Override only max_lay_price
        result = TrainingWorker._apply_run_overrides(base, {"max_lay_price": 30.0})
        bc = result["training"]["betting_constraints"]
        assert bc["max_back_price"] == 200.0  # Admin default preserved
        assert bc["max_lay_price"] == 30.0  # Per-run override applied
        assert bc["min_seconds_before_off"] == 60  # Admin default preserved

    def test_all_overrides_applied_together(self):
        """Multiple override types can coexist."""
        result = TrainingWorker._apply_run_overrides(
            _base_config(),
            {
                "population_size": 10,
                "architectures": ["ppo_lstm_v1"],
                "max_back_price": 100.0,
                "min_seconds_before_off": 300,
            },
        )
        assert result["population"]["size"] == 10
        assert result["hyperparameters"]["search_ranges"]["architecture_name"]["choices"] == ["ppo_lstm_v1"]
        assert result["training"]["betting_constraints"]["max_back_price"] == 100.0
        assert result["training"]["betting_constraints"]["min_seconds_before_off"] == 300

    def test_does_not_mutate_base_config(self):
        """Base config must not be modified — future runs should see original values."""
        base = _base_config()
        TrainingWorker._apply_run_overrides(
            base,
            {
                "population_size": 10,
                "architectures": ["ppo_lstm_v1"],
                "max_back_price": 100.0,
            },
        )
        # Base is untouched
        assert base["population"]["size"] == 50
        assert base["hyperparameters"]["search_ranges"]["architecture_name"]["choices"] == [
            "ppo_lstm_v1", "ppo_time_lstm_v1",
        ]
        assert base["training"]["betting_constraints"]["max_back_price"] is None
