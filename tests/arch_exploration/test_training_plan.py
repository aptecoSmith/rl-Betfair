"""Tests for ``training/training_plan.py`` and the matching API router.

Session 4 — all CPU, all fast.  No PPO loops, no GPU, no real days.
"""

from __future__ import annotations

import random

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agents.population_manager import HyperparamSpec, parse_search_ranges
from training.training_plan import (
    CoverageReport,
    GenerationOutcome,
    HistoricalAgent,
    PlanRegistry,
    TrainingPlan,
    bias_sampler,
    compute_coverage,
    generate_coverage_seed,
    generate_sobol_points,
    is_launchable,
    validate_plan,
)
from api.routers import training_plans as training_plans_router


# -- Fixtures -----------------------------------------------------------------


@pytest.fixture
def hp_ranges() -> dict[str, dict]:
    """Minimal but realistic search-range block matching ``config.yaml``."""
    return {
        "learning_rate": {"type": "float_log", "min": 1.0e-5, "max": 5.0e-4},
        "gamma": {"type": "float", "min": 0.95, "max": 0.999},
        "gae_lambda": {"type": "float", "min": 0.9, "max": 0.98},
        "early_pick_min_seconds": {"type": "int", "min": 120, "max": 900},
        "architecture_name": {
            "type": "str_choice",
            "choices": ["ppo_lstm_v1", "ppo_time_lstm_v1"],
        },
    }


@pytest.fixture
def hp_specs(hp_ranges) -> list[HyperparamSpec]:
    return parse_search_ranges(hp_ranges)


@pytest.fixture
def basic_plan(hp_ranges) -> TrainingPlan:
    return TrainingPlan.new(
        name="even-split-21",
        population_size=21,
        architectures=["ppo_lstm_v1", "ppo_time_lstm_v1"],
        hp_ranges=hp_ranges,
        seed=42,
        notes="Three slots per arch, generous range",
    )


@pytest.fixture
def registry(tmp_path) -> PlanRegistry:
    return PlanRegistry(tmp_path / "training_plans")


# -- 1. Plan round-trip -------------------------------------------------------


def test_plan_round_trip_via_registry(registry, basic_plan):
    """Save -> reload -> equality on every persisted field."""
    registry.save(basic_plan)
    reloaded = registry.load(basic_plan.plan_id)

    assert reloaded.plan_id == basic_plan.plan_id
    assert reloaded.name == basic_plan.name
    assert reloaded.created_at == basic_plan.created_at
    assert reloaded.population_size == basic_plan.population_size
    assert reloaded.architectures == basic_plan.architectures
    assert reloaded.hp_ranges == basic_plan.hp_ranges
    assert reloaded.seed == basic_plan.seed
    assert reloaded.min_arch_samples == basic_plan.min_arch_samples
    assert reloaded.notes == basic_plan.notes
    assert reloaded.outcomes == basic_plan.outcomes

    listed = registry.list()
    assert any(p.plan_id == basic_plan.plan_id for p in listed)


# -- 2. Validate rejects undersized populations -------------------------------


def test_validate_rejects_undersized_population(hp_ranges):
    """3 arches × 5 minimum should refuse pop=6."""
    plan = TrainingPlan.new(
        name="too-small",
        population_size=6,
        architectures=["ppo_lstm_v1", "ppo_time_lstm_v1", "ppo_transformer_v1"],
        hp_ranges=hp_ranges,
        min_arch_samples=5,
    )
    issues = validate_plan(plan)
    error_codes = {i.code for i in issues if i.severity == "error"}
    assert "population_too_small" in error_codes
    assert not is_launchable(issues)


def test_validate_passes_correctly_sized_population(hp_ranges):
    plan = TrainingPlan.new(
        name="just-right",
        population_size=15,
        architectures=["ppo_lstm_v1", "ppo_time_lstm_v1", "ppo_transformer_v1"],
        hp_ranges=hp_ranges,
        min_arch_samples=5,
    )
    issues = validate_plan(plan)
    assert is_launchable(issues), [i.message for i in issues]


# -- 3. Coverage with empty history -------------------------------------------


def test_coverage_empty_history_flags_everything(hp_specs):
    report = compute_coverage([], hp_specs)
    assert report.total_agents == 0
    # Every numeric gene should be poorly-covered (0% buckets non-empty).
    numeric_names = {
        "learning_rate", "gamma", "gae_lambda", "early_pick_min_seconds",
    }
    assert numeric_names.issubset(set(report.gene_coverage))
    for name in numeric_names:
        gc = report.gene_coverage[name]
        assert gc.nonempty_buckets == 0
        assert gc.coverage_fraction == 0.0
        assert gc.well_covered is False
        assert name in report.poorly_covered_genes
    # Both architectures should be flagged as under-covered.
    assert set(report.arch_undercovered) == {
        "ppo_lstm_v1", "ppo_time_lstm_v1",
    }


# -- 4. Coverage with synthetic history ---------------------------------------


def test_coverage_synthetic_history_matches_hand_count(hp_specs):
    """Build a tightly-controlled history and assert exact bucket counts."""
    rng = random.Random(0)
    history: list[HistoricalAgent] = []

    # 25 ppo_lstm_v1 agents with gamma uniformly across the *whole* range,
    # 25 ppo_time_lstm_v1 agents with gamma clustered ONLY in the bottom
    # decile (so the bias test below has something to nudge).
    for i in range(25):
        history.append(HistoricalAgent(
            architecture_name="ppo_lstm_v1",
            hyperparameters={
                "gamma": 0.95 + (0.999 - 0.95) * (i / 24),
                "gae_lambda": 0.9 + (0.98 - 0.9) * (i / 24),
                "learning_rate": 1e-5 + (5e-4 - 1e-5) * (i / 24),
                "early_pick_min_seconds": 120 + (900 - 120) * i // 24,
            },
        ))
    for _ in range(25):
        history.append(HistoricalAgent(
            architecture_name="ppo_time_lstm_v1",
            hyperparameters={
                "gamma": 0.951,        # bottom bucket
                "gae_lambda": 0.901,   # bottom bucket
                "learning_rate": 1.1e-5,  # bottom bucket
                "early_pick_min_seconds": 125,
            },
        ))

    report = compute_coverage(history, hp_specs, min_arch_samples=15)

    assert report.total_agents == 50
    assert report.arch_counts["ppo_lstm_v1"] == 25
    assert report.arch_counts["ppo_time_lstm_v1"] == 25
    # 15-floor is satisfied by both.
    assert report.arch_undercovered == []

    # gamma: the 25 evenly-spaced lstm agents alone should hit every
    # bucket; the 25 clustered time-lstm agents only add to bucket 0.
    gamma_cov = report.gene_coverage["gamma"]
    assert gamma_cov.nonempty_buckets == 10
    assert gamma_cov.well_covered is True
    # Bucket 0 should be the heaviest.
    assert gamma_cov.bucket_counts[0] == max(gamma_cov.bucket_counts)
    # Sum of all bucket counts equals number of agents that supplied gamma.
    assert sum(gamma_cov.bucket_counts) == 50

    # learning_rate is float_log -- the spaced lstm history is uniform in
    # *linear* space, so log buckets won't be evenly populated, but the
    # bottom bucket will still be the heaviest because the log scale puts
    # more linear-uniform mass at the low end.
    lr_cov = report.gene_coverage["learning_rate"]
    assert sum(lr_cov.bucket_counts) == 50
    assert lr_cov.bucket_counts[0] >= 25  # all 25 clustered + most lstm


# -- 5. Bias sampler nudges empty buckets -------------------------------------


def test_bias_sampler_nudges_empty_gamma_buckets(hp_specs):
    """History only in [0.95, 0.96] => empty upper buckets get heavier weight."""
    history = [
        HistoricalAgent(
            architecture_name="ppo_lstm_v1",
            hyperparameters={"gamma": 0.951 + 0.001 * (i % 9)},
        )
        for i in range(20)
    ]

    biased_specs = bias_sampler(hp_specs, history)
    by_name = {b.spec.name: b for b in biased_specs}

    gamma_biased = by_name["gamma"]
    assert gamma_biased.is_biased, "gamma should be flagged for biasing"
    assert gamma_biased.bucket_weights is not None
    assert gamma_biased.bucket_edges is not None
    assert len(gamma_biased.bucket_weights) == 10

    # Identify empty buckets directly: every gamma value is in [0.951, 0.96],
    # so the *bottom-most* bucket should be the only populated one and
    # its weight should be lower than the empty upper buckets.
    weights = gamma_biased.bucket_weights
    populated_weight = weights[0]
    upper_weights = weights[1:]
    assert populated_weight < max(upper_weights), (
        f"empty buckets should outweight populated ones; got {weights}"
    )
    # Every upper bucket should be strictly above the populated bucket.
    for w in upper_weights:
        assert w >= populated_weight
    assert any(w > populated_weight for w in upper_weights)


# -- 6. Outcome update round-trip ---------------------------------------------


def test_outcome_round_trip(registry, basic_plan):
    registry.save(basic_plan)
    outcome = GenerationOutcome(
        generation=0,
        recorded_at="2026-04-06T12:00:00+00:00",
        best_fitness=0.42,
        mean_fitness=0.13,
        architectures_alive=["ppo_lstm_v1", "ppo_time_lstm_v1"],
        architectures_died=[],
        n_agents=21,
        notes="Synthetic outcome",
    )
    registry.record_outcome(basic_plan.plan_id, outcome)

    reloaded = registry.load(basic_plan.plan_id)
    assert len(reloaded.outcomes) == 1
    got = reloaded.outcomes[0]
    assert got == outcome


# -- 7. API endpoints ---------------------------------------------------------


def _make_test_app(registry: PlanRegistry, hp_ranges: dict, history=None) -> TestClient:
    """Mount the planner router on a fresh FastAPI with seeded state."""
    app = FastAPI()
    app.include_router(training_plans_router.router)
    app.state.plan_registry = registry
    app.state.config = {"hyperparameters": {"search_ranges": hp_ranges}}
    app.state.coverage_history = history or []
    return TestClient(app)


def test_api_post_validate_list_get_coverage(registry, hp_ranges):
    client = _make_test_app(registry, hp_ranges)

    # Create a valid plan
    valid_payload = {
        "name": "api-test",
        "population_size": 14,
        "architectures": ["ppo_lstm_v1", "ppo_time_lstm_v1"],
        "hp_ranges": hp_ranges,
        "min_arch_samples": 5,
    }
    resp = client.post("/api/training-plans", json=valid_payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    plan_id = body["plan"]["plan_id"]
    assert body["plan"]["population_size"] == 14
    assert body["validation"] == []

    # List should now contain it
    resp = client.get("/api/training-plans")
    assert resp.status_code == 200
    listed_ids = [p["plan_id"] for p in resp.json()["plans"]]
    assert plan_id in listed_ids

    # Get by id
    resp = client.get(f"/api/training-plans/{plan_id}")
    assert resp.status_code == 200
    assert resp.json()["plan"]["plan_id"] == plan_id

    # Coverage endpoint with empty history is still 200
    resp = client.get("/api/training-plans/coverage")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["report"]["total_agents"] == 0
    # gamma should be biased because it's poorly covered.
    assert "gamma" in payload["biased_genes"]


def test_api_post_invalid_plan_returns_422(registry, hp_ranges):
    client = _make_test_app(registry, hp_ranges)
    invalid_payload = {
        "name": "too-small",
        "population_size": 4,                      # too small
        "architectures": ["ppo_lstm_v1", "ppo_time_lstm_v1"],
        "hp_ranges": hp_ranges,
        "min_arch_samples": 5,                     # need >= 10
    }
    resp = client.post("/api/training-plans", json=invalid_payload)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    issue_codes = {i["code"] for i in detail["issues"]}
    assert "population_too_small" in issue_codes


def test_api_get_unknown_plan_returns_404(registry, hp_ranges):
    client = _make_test_app(registry, hp_ranges)
    resp = client.get("/api/training-plans/does-not-exist")
    assert resp.status_code == 404


# -- 8. Per-plan starting_budget (Session 01) ---------------------------------


def test_plan_with_budget_round_trips(registry, hp_ranges):
    """Plan with starting_budget persists and reloads correctly."""
    plan = TrainingPlan.new(
        name="budget-test",
        population_size=10,
        architectures=["ppo_lstm_v1"],
        hp_ranges=hp_ranges,
        min_arch_samples=5,
        starting_budget=10.0,
    )
    assert plan.starting_budget == 10.0

    registry.save(plan)
    reloaded = registry.load(plan.plan_id)
    assert reloaded.starting_budget == 10.0


def test_plan_without_budget_defaults_to_none(registry, hp_ranges):
    """Plans that don't specify a budget have starting_budget=None."""
    plan = TrainingPlan.new(
        name="no-budget",
        population_size=10,
        architectures=["ppo_lstm_v1"],
        hp_ranges=hp_ranges,
        min_arch_samples=5,
    )
    assert plan.starting_budget is None

    registry.save(plan)
    reloaded = registry.load(plan.plan_id)
    assert reloaded.starting_budget is None


def test_plan_budget_in_to_dict(hp_ranges):
    """to_dict() includes starting_budget."""
    plan = TrainingPlan.new(
        name="dict-test",
        population_size=10,
        architectures=["ppo_lstm_v1"],
        hp_ranges=hp_ranges,
        starting_budget=25.0,
    )
    d = plan.to_dict()
    assert d["starting_budget"] == 25.0


def test_plan_from_dict_without_budget_field(hp_ranges):
    """Old JSON without starting_budget → defaults to None (backward compat)."""
    plan = TrainingPlan.new(
        name="old-plan",
        population_size=10,
        architectures=["ppo_lstm_v1"],
        hp_ranges=hp_ranges,
    )
    d = plan.to_dict()
    del d["starting_budget"]  # simulate old JSON
    reloaded = TrainingPlan.from_dict(d)
    assert reloaded.starting_budget is None


def test_api_create_plan_with_budget(registry, hp_ranges):
    """API accepts starting_budget and persists it."""
    client = _make_test_app(registry, hp_ranges)
    payload = {
        "name": "budget-api",
        "population_size": 10,
        "architectures": ["ppo_lstm_v1", "ppo_time_lstm_v1"],
        "hp_ranges": hp_ranges,
        "min_arch_samples": 5,
        "starting_budget": 10.0,
    }
    resp = client.post("/api/training-plans", json=payload)
    assert resp.status_code == 200, resp.text
    assert resp.json()["plan"]["starting_budget"] == 10.0

    # Verify persisted
    plan_id = resp.json()["plan"]["plan_id"]
    reloaded = registry.load(plan_id)
    assert reloaded.starting_budget == 10.0


def test_api_rejects_negative_budget(registry, hp_ranges):
    """API returns 422 for budget <= 0."""
    client = _make_test_app(registry, hp_ranges)
    payload = {
        "name": "bad-budget",
        "population_size": 10,
        "architectures": ["ppo_lstm_v1"],
        "hp_ranges": hp_ranges,
        "starting_budget": -5.0,
    }
    resp = client.post("/api/training-plans", json=payload)
    assert resp.status_code == 422


def test_api_rejects_zero_budget(registry, hp_ranges):
    """API returns 422 for budget == 0."""
    client = _make_test_app(registry, hp_ranges)
    payload = {
        "name": "zero-budget",
        "population_size": 10,
        "architectures": ["ppo_lstm_v1"],
        "hp_ranges": hp_ranges,
        "starting_budget": 0,
    }
    resp = client.post("/api/training-plans", json=payload)
    assert resp.status_code == 422


def test_orchestrator_uses_plan_budget(hp_ranges):
    """TrainingOrchestrator patches config when plan has starting_budget."""
    import copy
    config = {
        "training": {
            "starting_budget": 100.0,
            "max_runners": 14,
            "max_bets_per_race": 20,
            "architecture": "ppo_lstm_v1",
        },
        "population": {"size": 5, "n_elite": 1, "selection_top_pct": 0.5, "mutation_rate": 0.1},
        "hyperparameters": {"search_ranges": hp_ranges},
        "paths": {"processed_data": "/tmp/fake"},
    }
    plan = TrainingPlan.new(
        name="budget-orch",
        population_size=5,
        architectures=["ppo_lstm_v1"],
        hp_ranges=hp_ranges,
        starting_budget=10.0,
    )
    test_config = copy.deepcopy(config)
    from training.run_training import TrainingOrchestrator
    orch = TrainingOrchestrator(test_config, model_store=None, training_plan=plan)
    assert orch.config["training"]["starting_budget"] == 10.0


def test_orchestrator_uses_global_budget_when_plan_has_none(hp_ranges):
    """TrainingOrchestrator keeps global budget when plan doesn't override."""
    import copy
    config = {
        "training": {
            "starting_budget": 100.0,
            "max_runners": 14,
            "max_bets_per_race": 20,
            "architecture": "ppo_lstm_v1",
        },
        "population": {"size": 5, "n_elite": 1, "selection_top_pct": 0.5, "mutation_rate": 0.1},
        "hyperparameters": {"search_ranges": hp_ranges},
        "paths": {"processed_data": "/tmp/fake"},
    }
    plan = TrainingPlan.new(
        name="no-budget-orch",
        population_size=5,
        architectures=["ppo_lstm_v1"],
        hp_ranges=hp_ranges,
    )
    test_config = copy.deepcopy(config)
    from training.run_training import TrainingOrchestrator
    orch = TrainingOrchestrator(test_config, model_store=None, training_plan=plan)
    assert orch.config["training"]["starting_budget"] == 100.0


# -- 9. Sobol seed point generator (Sprint 4, Session 02) ---------------------


class TestSobolSeedGenerator:
    """Tests for ``generate_sobol_points``."""

    def test_all_points_within_bounds(self, hp_specs):
        points = generate_sobol_points(hp_specs, n_points=10)
        assert len(points) == 10
        spec_by_name = {s.name: s for s in hp_specs}
        for pt in points:
            lr = pt["learning_rate"]
            assert spec_by_name["learning_rate"].min <= lr <= spec_by_name["learning_rate"].max
            gamma = pt["gamma"]
            assert spec_by_name["gamma"].min <= gamma <= spec_by_name["gamma"].max
            lam = pt["gae_lambda"]
            assert spec_by_name["gae_lambda"].min <= lam <= spec_by_name["gae_lambda"].max
            secs = pt["early_pick_min_seconds"]
            assert 120 <= secs <= 900
            assert isinstance(secs, int)
            assert pt["architecture_name"] in ["ppo_lstm_v1", "ppo_time_lstm_v1"]

    def test_points_are_well_spaced(self, hp_specs):
        """Sobol points should be better-distributed than random."""
        points = generate_sobol_points(hp_specs, n_points=10)
        spec_by_name = {s.name: s for s in hp_specs}
        # Normalise gamma values to [0,1] and check pairwise distances.
        gmin = spec_by_name["gamma"].min
        gmax = spec_by_name["gamma"].max
        normed = [(pt["gamma"] - gmin) / (gmax - gmin) for pt in points]
        normed.sort()
        # With 10 quasi-random points in [0,1], the max gap should be
        # much less than 0.5 (a random sample could have large gaps).
        gaps = [normed[i + 1] - normed[i] for i in range(len(normed) - 1)]
        assert max(gaps) < 0.3, f"Max gap {max(gaps)} too large for Sobol"

    def test_skip_produces_different_points(self, hp_specs):
        a = generate_sobol_points(hp_specs, n_points=5, skip=0)
        b = generate_sobol_points(hp_specs, n_points=5, skip=5)
        # At least one numeric gene should differ across all 5 points.
        a_gammas = [pt["gamma"] for pt in a]
        b_gammas = [pt["gamma"] for pt in b]
        assert a_gammas != b_gammas

    def test_round_trip_valid_hyperparameters(self, hp_specs):
        """Each Sobol point should be usable as hyperparameters."""
        points = generate_sobol_points(hp_specs, n_points=5)
        for pt in points:
            # Every spec should have a corresponding key.
            for spec in hp_specs:
                assert spec.name in pt, f"Missing gene {spec.name}"
                val = pt[spec.name]
                if spec.type in ("float", "float_log"):
                    assert isinstance(val, float)
                elif spec.type == "int":
                    assert isinstance(val, int)
                elif spec.type in ("int_choice", "str_choice"):
                    assert val in spec.choices


# -- 9b. Exploration strategy on TrainingPlan (Sprint 4, Session 05) -----------


class TestExplorationStrategy:
    """Tests for exploration_strategy field on TrainingPlan."""

    def test_default_strategy_is_random(self, hp_ranges):
        plan = TrainingPlan.new(
            name="default", population_size=10,
            architectures=["ppo_lstm_v1"], hp_ranges=hp_ranges,
        )
        assert plan.exploration_strategy == "random"
        assert plan.manual_seed_point is None

    def test_strategy_round_trips(self, registry, hp_ranges):
        plan = TrainingPlan.new(
            name="sobol-plan", population_size=10,
            architectures=["ppo_lstm_v1"], hp_ranges=hp_ranges,
            exploration_strategy="sobol",
        )
        registry.save(plan)
        reloaded = registry.load(plan.plan_id)
        assert reloaded.exploration_strategy == "sobol"

    def test_manual_seed_point_round_trips(self, registry, hp_ranges):
        seed_pt = {"learning_rate": 0.0001, "gamma": 0.99}
        plan = TrainingPlan.new(
            name="manual-plan", population_size=10,
            architectures=["ppo_lstm_v1"], hp_ranges=hp_ranges,
            exploration_strategy="manual",
            manual_seed_point=seed_pt,
        )
        registry.save(plan)
        reloaded = registry.load(plan.plan_id)
        assert reloaded.exploration_strategy == "manual"
        assert reloaded.manual_seed_point == seed_pt

    def test_old_plan_without_strategy_defaults(self, hp_ranges):
        """Old JSON without exploration_strategy → defaults to 'random'."""
        plan = TrainingPlan.new(
            name="old", population_size=10,
            architectures=["ppo_lstm_v1"], hp_ranges=hp_ranges,
        )
        d = plan.to_dict()
        del d["exploration_strategy"]
        del d["manual_seed_point"]
        reloaded = TrainingPlan.from_dict(d)
        assert reloaded.exploration_strategy == "random"
        assert reloaded.manual_seed_point is None

    def test_api_accepts_strategy(self, registry, hp_ranges):
        client = _make_test_app(registry, hp_ranges)
        payload = {
            "name": "api-sobol",
            "population_size": 10,
            "architectures": ["ppo_lstm_v1", "ppo_time_lstm_v1"],
            "hp_ranges": hp_ranges,
            "exploration_strategy": "coverage",
        }
        resp = client.post("/api/training-plans", json=payload)
        assert resp.status_code == 200, resp.text
        assert resp.json()["plan"]["exploration_strategy"] == "coverage"

    def test_api_default_strategy(self, registry, hp_ranges):
        """API without strategy field → defaults to 'random'."""
        client = _make_test_app(registry, hp_ranges)
        payload = {
            "name": "api-default",
            "population_size": 10,
            "architectures": ["ppo_lstm_v1"],
            "hp_ranges": hp_ranges,
        }
        resp = client.post("/api/training-plans", json=payload)
        assert resp.status_code == 200
        assert resp.json()["plan"]["exploration_strategy"] == "random"


# -- 10. Coverage-biased seed generation (Sprint 4, Session 03) ----------------


class TestCoverageSeed:
    """Tests for ``generate_coverage_seed``."""

    def test_empty_history_produces_uniform_seed(self, hp_specs):
        """With no models, the seed is effectively random (all genes present)."""
        seed, report = generate_coverage_seed(hp_specs, [], seed=42)
        assert report.total_agents == 0
        # Every spec should have a value.
        for spec in hp_specs:
            assert spec.name in seed

    def test_clustered_history_biases_away(self, hp_specs):
        """50 models all in one gamma region → seed biased toward other regions."""
        history = [
            HistoricalAgent(
                architecture_name="ppo_lstm_v1",
                hyperparameters={"gamma": 0.951},
            )
            for _ in range(50)
        ]
        # Run many seeds and check the average gamma is higher than the
        # clustered region (0.951 is bottom of range 0.95–0.999).
        gammas = []
        for i in range(50):
            seed, _ = generate_coverage_seed(hp_specs, history, seed=i)
            gammas.append(seed["gamma"])
        avg = sum(gammas) / len(gammas)
        # Midpoint of range is ~0.975. Biased sampling should push avg
        # well above the clustered 0.951 region.
        assert avg > 0.96, f"Expected avg gamma > 0.96, got {avg}"

    def test_coverage_report_is_serialisable(self, hp_specs):
        """CoverageReport.to_dict() should produce JSON-serialisable output."""
        import json
        history = [
            HistoricalAgent("ppo_lstm_v1", {"gamma": 0.97, "learning_rate": 1e-4})
            for _ in range(5)
        ]
        _, report = generate_coverage_seed(hp_specs, history, seed=0)
        d = report.to_dict()
        # Should round-trip through JSON without error.
        serialised = json.dumps(d)
        assert isinstance(serialised, str)
        parsed = json.loads(serialised)
        assert parsed["total_agents"] == 5
