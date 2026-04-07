"""Tests: config.yaml loads correctly and contains all required keys."""

from __future__ import annotations


REQUIRED_TOP_LEVEL_KEYS = [
    "database",
    "population",
    "reward",
    "paths",
    "training",
    "discard_policy",
    "hyperparameters",
]

REQUIRED_DB_KEYS = ["host", "port", "cold_data_db", "hot_data_db"]
REQUIRED_POPULATION_KEYS = ["size", "n_elite", "selection_top_pct"]
REQUIRED_REWARD_COEFFICIENT_KEYS = ["win_rate", "sharpe", "mean_daily_pnl", "efficiency"]
REQUIRED_PATH_KEYS = ["processed_data", "model_weights", "logs", "registry_db"]
REQUIRED_DISCARD_KEYS = ["min_win_rate", "min_mean_pnl", "min_sharpe"]


def test_config_loads(config):
    assert config is not None
    assert isinstance(config, dict)


def test_required_top_level_keys(config):
    for key in REQUIRED_TOP_LEVEL_KEYS:
        assert key in config, f"Missing required top-level config key: '{key}'"


def test_database_keys(config):
    db = config["database"]
    for key in REQUIRED_DB_KEYS:
        assert key in db, f"Missing database config key: '{key}'"


def test_database_port_is_3306(config):
    assert config["database"]["port"] == 3306


def test_population_keys(config):
    pop = config["population"]
    for key in REQUIRED_POPULATION_KEYS:
        assert key in pop, f"Missing population config key: '{key}'"


def test_population_values_sane(config):
    pop = config["population"]
    assert pop["size"] > 0
    assert pop["n_elite"] > 0
    assert pop["n_elite"] < pop["size"]
    assert 0 < pop["selection_top_pct"] <= 1.0


def test_reward_coefficient_keys(config):
    coeffs = config["reward"]["coefficients"]
    for key in REQUIRED_REWARD_COEFFICIENT_KEYS:
        assert key in coeffs, f"Missing reward coefficient: '{key}'"


def test_reward_coefficients_sum_to_one(config):
    coeffs = config["reward"]["coefficients"]
    total = sum(coeffs.values())
    assert abs(total - 1.0) < 1e-9, (
        f"Reward coefficients must sum to 1.0, got {total:.6f}"
    )


def test_path_keys(config):
    paths = config["paths"]
    for key in REQUIRED_PATH_KEYS:
        assert key in paths, f"Missing paths config key: '{key}'"


def test_discard_policy_keys(config):
    discard = config["discard_policy"]
    for key in REQUIRED_DISCARD_KEYS:
        assert key in discard, f"Missing discard_policy key: '{key}'"


def test_hyperparameter_search_ranges_present(config):
    hp = config["hyperparameters"]
    assert "search_ranges" in hp
    ranges = hp["search_ranges"]
    expected_params = [
        "learning_rate",
        "ppo_clip_epsilon",
        "entropy_coefficient",
        "lstm_hidden_size",
        "mlp_hidden_size",
        "mlp_layers",
        "early_pick_bonus_min",
        "early_pick_bonus_max",
        "early_pick_min_seconds",
        "terminal_bonus_weight",
        "reward_efficiency_penalty",
        "reward_precision_bonus",
        "gamma",
        "gae_lambda",
        "value_loss_coeff",
        "lstm_num_layers",
        "lstm_dropout",
        "lstm_layer_norm",
    ]
    for param in expected_params:
        assert param in ranges, f"Missing hyperparameter search range: '{param}'"


def test_training_architecture_key(config):
    assert "architecture" in config["training"]


def test_starting_budget_positive(config):
    assert config["training"]["starting_budget"] > 0
