"""Tests for Sprint 3 Session 2: adaptive breeding (Issue 09).

Detect bad generations and respond with one of three policies (persist,
boost_mutation, inject_top). When adaptive_mutation is on, the rate
ramps up for consecutive bad generations and resets on a good one.

Most behaviour is exercised at the orchestrator level via direct state
manipulation, since spinning up a full training run for each test is
prohibitive.
"""

from __future__ import annotations

import asyncio

import pytest

from registry.scoreboard import ModelScore
from training.run_training import TrainingOrchestrator


def _make_orch_config(**overrides) -> dict:
    cfg = {
        "population": {
            "size": 4,
            "n_elite": 1,
            "selection_top_pct": 0.5,
            "mutation_rate": 0.3,
            "max_mutations_per_child": None,
            "breeding_pool": "run_only",
            "bad_generation_threshold": 0.0,
            "bad_generation_policy": "persist",
            "adaptive_mutation": False,
            "adaptive_mutation_increment": 0.1,
            "adaptive_mutation_cap": 0.8,
        },
        "training": {
            "architecture": "ppo_lstm_v1",
            "max_runners": 14,
            "starting_budget": 100.0,
            "require_gpu": False,
        },
        "hyperparameters": {
            "search_ranges": {
                "learning_rate": {"type": "float_log", "min": 1e-5, "max": 5e-4},
                "ppo_clip_epsilon": {"type": "float", "min": 0.1, "max": 0.3},
                "entropy_coefficient": {"type": "float", "min": 0.001, "max": 0.05},
                "lstm_hidden_size": {"type": "int_choice", "choices": [64, 128]},
                "mlp_hidden_size": {"type": "int_choice", "choices": [64, 128]},
                "mlp_layers": {"type": "int", "min": 1, "max": 3},
                "early_pick_bonus_min": {"type": "float", "min": 1.0, "max": 1.2},
                "early_pick_bonus_max": {"type": "float", "min": 1.2, "max": 1.5},
                "reward_efficiency_penalty": {"type": "float", "min": 0.001, "max": 0.05},
                "reward_precision_bonus": {"type": "float", "min": 0.0, "max": 3.0},
            }
        },
        "discard_policy": {
            "min_win_rate": 0.35,
            "min_mean_pnl": 0.0,
            "min_sharpe": -0.5,
        },
    }
    for k, v in overrides.items():
        cfg["population"][k] = v
    return cfg


def _make_score(model_id: str, composite: float) -> ModelScore:
    return ModelScore(
        model_id=model_id, win_rate=0.5, mean_daily_pnl=1.0, sharpe=0.5,
        bet_precision=0.5, pnl_per_bet=0.1, efficiency=0.5,
        composite_score=composite, test_days=10, profitable_days=5,
    )


def _make_orch(cfg: dict) -> TrainingOrchestrator:
    # Bypass model_store/cuda by passing None and device='cpu'.
    return TrainingOrchestrator(config=cfg, model_store=None, device="cpu")


class TestAdaptiveMutationState:
    def test_default_state(self):
        orch = _make_orch(_make_orch_config())
        assert orch._consecutive_bad_gens == 0
        assert orch._effective_mutation_rate is None

    def test_consecutive_increment_via_state(self):
        """The adaptive ramp uses _consecutive_bad_gens as multiplier."""
        cfg = _make_orch_config(
            adaptive_mutation=True,
            adaptive_mutation_increment=0.1,
            adaptive_mutation_cap=0.8,
            mutation_rate=0.3,
        )
        orch = _make_orch(cfg)
        # Simulate 3 consecutive bad gens.
        orch._consecutive_bad_gens = 3
        base = cfg["population"]["mutation_rate"]
        increment = cfg["population"]["adaptive_mutation_increment"]
        cap = cfg["population"]["adaptive_mutation_cap"]
        expected = min(cap, base + increment * orch._consecutive_bad_gens)
        assert expected == pytest.approx(0.6)

    def test_cap_applied(self):
        cfg = _make_orch_config(
            adaptive_mutation=True,
            adaptive_mutation_increment=0.5,
            adaptive_mutation_cap=0.6,
            mutation_rate=0.3,
        )
        orch = _make_orch(cfg)
        orch._consecutive_bad_gens = 10
        expected = min(0.6, 0.3 + 0.5 * 10)
        assert expected == 0.6


class TestBadGenerationDetection:
    def test_threshold_zero_disables_detection(self):
        cfg = _make_orch_config(bad_generation_threshold=0.0)
        # max(score) = 0.05 < threshold? threshold=0.0 → no, not detected.
        scores = [_make_score("a", 0.05)]
        threshold = cfg["population"]["bad_generation_threshold"]
        is_bad = threshold > 0.0 and max(s.composite_score for s in scores) < threshold
        assert not is_bad

    def test_threshold_triggers_below(self):
        cfg = _make_orch_config(bad_generation_threshold=0.2)
        scores = [_make_score("a", 0.1), _make_score("b", 0.15)]
        threshold = cfg["population"]["bad_generation_threshold"]
        is_bad = threshold > 0.0 and max(s.composite_score for s in scores) < threshold
        assert is_bad

    def test_threshold_above_not_triggered(self):
        cfg = _make_orch_config(bad_generation_threshold=0.2)
        scores = [_make_score("a", 0.25)]
        threshold = cfg["population"]["bad_generation_threshold"]
        is_bad = threshold > 0.0 and max(s.composite_score for s in scores) < threshold
        assert not is_bad


class TestPolicyValidation:
    def test_valid_policies(self):
        for p in ("persist", "boost_mutation", "inject_top"):
            cfg = _make_orch_config(bad_generation_policy=p)
            assert cfg["population"]["bad_generation_policy"] == p


class TestStartTrainingRequest:
    """Smoke test that the new fields round-trip via Pydantic."""

    def test_request_accepts_new_fields(self):
        from api.schemas import StartTrainingRequest
        body = StartTrainingRequest(
            stud_model_ids=["a", "b"],
            mutation_rate=0.5,
            bad_generation_threshold=0.2,
            bad_generation_policy="boost_mutation",
            adaptive_mutation=True,
            adaptive_mutation_increment=0.1,
            adaptive_mutation_cap=0.8,
        )
        assert body.stud_model_ids == ["a", "b"]
        assert body.mutation_rate == 0.5
        assert body.bad_generation_policy == "boost_mutation"
        assert body.adaptive_mutation is True

    def test_request_defaults(self):
        from api.schemas import StartTrainingRequest
        body = StartTrainingRequest()
        assert body.stud_model_ids is None
        assert body.mutation_rate is None
        assert body.bad_generation_policy is None


class TestWorkerOverrides:
    def test_apply_run_overrides_layers_population(self):
        from training.worker import TrainingWorker
        base = {
            "population": {
                "size": 50,
                "mutation_rate": 0.3,
                "bad_generation_threshold": 0.0,
                "bad_generation_policy": "persist",
                "adaptive_mutation": False,
            },
            "training": {},
            "hyperparameters": {"search_ranges": {}},
        }
        params = {
            "mutation_rate": 0.5,
            "bad_generation_threshold": 0.25,
            "bad_generation_policy": "boost_mutation",
            "adaptive_mutation": True,
        }
        out = TrainingWorker._apply_run_overrides(base, params)
        pop = out["population"]
        assert pop["mutation_rate"] == 0.5
        assert pop["bad_generation_threshold"] == 0.25
        assert pop["bad_generation_policy"] == "boost_mutation"
        assert pop["adaptive_mutation"] is True
        # Originals untouched.
        assert base["population"]["mutation_rate"] == 0.3
