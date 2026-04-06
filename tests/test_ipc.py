"""Unit tests for training/ipc.py — IPC protocol messages."""

from __future__ import annotations

import json

from training.ipc import (
    CMD_FINISH,
    CMD_START,
    make_finish_cmd,
    make_start_cmd,
)


class TestMakeStartCmd:
    def test_default_fields(self):
        raw = make_start_cmd()
        msg = json.loads(raw)
        assert msg["type"] == CMD_START
        assert msg["n_generations"] == 3
        assert msg["n_epochs"] == 3
        assert msg["train_dates"] is None
        assert msg["test_dates"] is None

    def test_explicit_dates_included(self):
        raw = make_start_cmd(
            train_dates=["2026-01-01", "2026-01-02"],
            test_dates=["2026-01-03"],
        )
        msg = json.loads(raw)
        assert msg["train_dates"] == ["2026-01-01", "2026-01-02"]
        assert msg["test_dates"] == ["2026-01-03"]

    def test_all_params_round_trip(self):
        raw = make_start_cmd(
            n_generations=5,
            n_epochs=2,
            population_size=30,
            seed=42,
            reevaluate_garaged=True,
            reevaluate_min_score=0.5,
            train_dates=["2026-03-01"],
            test_dates=["2026-03-02"],
        )
        msg = json.loads(raw)
        assert msg["n_generations"] == 5
        assert msg["n_epochs"] == 2
        assert msg["population_size"] == 30
        assert msg["seed"] == 42
        assert msg["reevaluate_garaged"] is True
        assert msg["reevaluate_min_score"] == 0.5
        assert msg["train_dates"] == ["2026-03-01"]
        assert msg["test_dates"] == ["2026-03-02"]

    def test_architectures_and_constraints_included(self):
        raw = make_start_cmd(
            architectures=["ppo_lstm_v1", "ppo_time_lstm_v1"],
            max_back_price=100.0,
            max_lay_price=50.0,
            min_seconds_before_off=300,
        )
        msg = json.loads(raw)
        assert msg["architectures"] == ["ppo_lstm_v1", "ppo_time_lstm_v1"]
        assert msg["max_back_price"] == 100.0
        assert msg["max_lay_price"] == 50.0
        assert msg["min_seconds_before_off"] == 300

    def test_architectures_default_none(self):
        raw = make_start_cmd()
        msg = json.loads(raw)
        assert msg["architectures"] is None
        assert msg["max_back_price"] is None
        assert msg["max_lay_price"] is None
        assert msg["min_seconds_before_off"] is None


class TestMakeFinishCmd:
    def test_finish_cmd_type(self):
        raw = make_finish_cmd()
        msg = json.loads(raw)
        assert msg["type"] == CMD_FINISH
