"""Unit tests for genetic selection in agents/population_manager.py."""

from __future__ import annotations

import pytest

from agents.population_manager import (
    PopulationManager,
    SelectionResult,
)
from registry.model_store import ModelStore
from registry.scoreboard import ModelScore


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_score(
    model_id: str,
    composite: float,
    win_rate: float = 0.5,
    mean_daily_pnl: float = 5.0,
    sharpe: float = 1.0,
) -> ModelScore:
    """Create a ModelScore with sensible defaults."""
    return ModelScore(
        model_id=model_id,
        win_rate=win_rate,
        mean_daily_pnl=mean_daily_pnl,
        sharpe=sharpe,
        bet_precision=0.5,
        pnl_per_bet=1.0,
        efficiency=0.5,
        composite_score=composite,
        test_days=10,
        profitable_days=int(win_rate * 10),
    )


def _make_config(
    pop_size: int = 10,
    n_elite: int = 2,
    top_pct: float = 0.5,
) -> dict:
    """Minimal config for selection tests."""
    return {
        "population": {
            "size": pop_size,
            "n_elite": n_elite,
            "selection_top_pct": top_pct,
        },
        "training": {
            "architecture": "ppo_lstm_v1",
            "max_runners": 14,
            "starting_budget": 100.0,
        },
        "hyperparameters": {
            "search_ranges": {
                "learning_rate": {"type": "float_log", "min": 1e-5, "max": 5e-4},
                "ppo_clip_epsilon": {"type": "float", "min": 0.1, "max": 0.3},
                "entropy_coefficient": {"type": "float", "min": 0.001, "max": 0.05},
                "lstm_hidden_size": {"type": "int_choice", "choices": [64, 128, 256, 512, 1024, 2048]},
                "mlp_hidden_size": {"type": "int_choice", "choices": [64, 128, 256]},
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


# ── Tournament selection ─────────────────────────────────────────────────────


class TestSelect:
    def test_returns_selection_result(self):
        config = _make_config(pop_size=10, n_elite=2, top_pct=0.5)
        pm = PopulationManager(config, model_store=None)
        scores = [_make_score(f"m{i}", composite=i * 0.1) for i in range(10)]
        result = pm.select(scores)
        assert isinstance(result, SelectionResult)

    def test_survivor_count_is_top_50pct(self):
        config = _make_config(pop_size=10, n_elite=2, top_pct=0.5)
        pm = PopulationManager(config, model_store=None)
        scores = [_make_score(f"m{i}", composite=i * 0.1) for i in range(10)]
        result = pm.select(scores)
        assert len(result.survivors) == 5

    def test_eliminated_count_is_bottom_50pct(self):
        config = _make_config(pop_size=10, n_elite=2, top_pct=0.5)
        pm = PopulationManager(config, model_store=None)
        scores = [_make_score(f"m{i}", composite=i * 0.1) for i in range(10)]
        result = pm.select(scores)
        assert len(result.eliminated) == 5

    def test_survivors_plus_eliminated_equals_total(self):
        config = _make_config(pop_size=10, n_elite=2, top_pct=0.5)
        pm = PopulationManager(config, model_store=None)
        scores = [_make_score(f"m{i}", composite=i * 0.1) for i in range(10)]
        result = pm.select(scores)
        assert len(result.survivors) + len(result.eliminated) == len(scores)

    def test_elites_are_top_n(self):
        config = _make_config(pop_size=10, n_elite=3, top_pct=0.5)
        pm = PopulationManager(config, model_store=None)
        scores = [_make_score(f"m{i}", composite=i * 0.1) for i in range(10)]
        result = pm.select(scores)
        # Highest scores are m9, m8, m7
        assert result.elites == ["m9", "m8", "m7"]

    def test_elites_are_subset_of_survivors(self):
        config = _make_config(pop_size=10, n_elite=3, top_pct=0.5)
        pm = PopulationManager(config, model_store=None)
        scores = [_make_score(f"m{i}", composite=i * 0.1) for i in range(10)]
        result = pm.select(scores)
        for elite_id in result.elites:
            assert elite_id in result.survivors

    def test_survivors_sorted_by_score_desc(self):
        config = _make_config(pop_size=10, n_elite=2, top_pct=0.5)
        pm = PopulationManager(config, model_store=None)
        scores = [_make_score(f"m{i}", composite=i * 0.1) for i in range(10)]
        result = pm.select(scores)
        survivor_scores = [
            next(s for s in result.ranked_scores if s.model_id == mid).composite_score
            for mid in result.survivors
        ]
        assert survivor_scores == sorted(survivor_scores, reverse=True)

    def test_eliminated_have_lower_scores_than_survivors(self):
        config = _make_config(pop_size=10, n_elite=2, top_pct=0.5)
        pm = PopulationManager(config, model_store=None)
        scores = [_make_score(f"m{i}", composite=i * 0.1) for i in range(10)]
        result = pm.select(scores)
        worst_survivor = min(
            s.composite_score
            for s in result.ranked_scores
            if s.model_id in result.survivors
        )
        for s in result.ranked_scores:
            if s.model_id in result.eliminated:
                assert s.composite_score <= worst_survivor

    def test_no_overlap_between_survivors_and_eliminated(self):
        config = _make_config(pop_size=10, n_elite=2, top_pct=0.5)
        pm = PopulationManager(config, model_store=None)
        scores = [_make_score(f"m{i}", composite=i * 0.1) for i in range(10)]
        result = pm.select(scores)
        assert set(result.survivors).isdisjoint(set(result.eliminated))

    def test_ranked_scores_preserved(self):
        config = _make_config(pop_size=10, n_elite=2, top_pct=0.5)
        pm = PopulationManager(config, model_store=None)
        scores = [_make_score(f"m{i}", composite=i * 0.1) for i in range(10)]
        result = pm.select(scores)
        assert len(result.ranked_scores) == 10
        # Sorted descending
        cs = [s.composite_score for s in result.ranked_scores]
        assert cs == sorted(cs, reverse=True)


# ── Elitism edge cases ───────────────────────────────────────────────────────


class TestElitism:
    def test_n_elite_equals_population(self):
        """If n_elite >= population, all survive."""
        config = _make_config(pop_size=5, n_elite=5, top_pct=0.5)
        pm = PopulationManager(config, model_store=None)
        scores = [_make_score(f"m{i}", composite=i * 0.1) for i in range(5)]
        result = pm.select(scores)
        assert len(result.survivors) == 5
        assert len(result.eliminated) == 0

    def test_n_elite_exceeds_population(self):
        """If n_elite > population size, all survive (no crash)."""
        config = _make_config(pop_size=3, n_elite=10, top_pct=0.5)
        pm = PopulationManager(config, model_store=None)
        scores = [_make_score(f"m{i}", composite=i * 0.1) for i in range(3)]
        result = pm.select(scores)
        assert len(result.survivors) == 3
        assert len(result.eliminated) == 0

    def test_n_elite_one(self):
        config = _make_config(pop_size=6, n_elite=1, top_pct=0.5)
        pm = PopulationManager(config, model_store=None)
        scores = [_make_score(f"m{i}", composite=i * 0.1) for i in range(6)]
        result = pm.select(scores)
        assert len(result.elites) == 1
        assert result.elites[0] == "m5"  # highest score

    def test_n_elite_zero(self):
        config = _make_config(pop_size=6, n_elite=0, top_pct=0.5)
        pm = PopulationManager(config, model_store=None)
        scores = [_make_score(f"m{i}", composite=i * 0.1) for i in range(6)]
        result = pm.select(scores)
        assert len(result.elites) == 0
        assert len(result.survivors) == 3  # 50% of 6


# ── Selection percentages ────────────────────────────────────────────────────


class TestSelectionPercentage:
    def test_top_30pct(self):
        config = _make_config(pop_size=10, n_elite=2, top_pct=0.3)
        pm = PopulationManager(config, model_store=None)
        scores = [_make_score(f"m{i}", composite=i * 0.1) for i in range(10)]
        result = pm.select(scores)
        assert len(result.survivors) == 3  # 30% of 10

    def test_top_70pct(self):
        config = _make_config(pop_size=10, n_elite=2, top_pct=0.7)
        pm = PopulationManager(config, model_store=None)
        scores = [_make_score(f"m{i}", composite=i * 0.1) for i in range(10)]
        result = pm.select(scores)
        assert len(result.survivors) == 7

    def test_top_100pct(self):
        config = _make_config(pop_size=10, n_elite=2, top_pct=1.0)
        pm = PopulationManager(config, model_store=None)
        scores = [_make_score(f"m{i}", composite=i * 0.1) for i in range(10)]
        result = pm.select(scores)
        assert len(result.survivors) == 10
        assert len(result.eliminated) == 0

    def test_elite_guarantees_minimum_survivors(self):
        """Even if top_pct rounds to fewer than n_elite, n_elite survive."""
        config = _make_config(pop_size=10, n_elite=4, top_pct=0.1)
        pm = PopulationManager(config, model_store=None)
        scores = [_make_score(f"m{i}", composite=i * 0.1) for i in range(10)]
        result = pm.select(scores)
        # 10% of 10 = 1, but n_elite = 4, so 4 survive
        assert len(result.survivors) == 4

    def test_odd_population_rounding(self):
        config = _make_config(pop_size=7, n_elite=1, top_pct=0.5)
        pm = PopulationManager(config, model_store=None)
        scores = [_make_score(f"m{i}", composite=i * 0.1) for i in range(7)]
        result = pm.select(scores)
        # round(7 * 0.5) = round(3.5) = 4
        assert len(result.survivors) == 4
        assert len(result.eliminated) == 3


# ── Empty and single-element populations ─────────────────────────────────────


class TestEdgeCasePopulations:
    def test_empty_population(self):
        config = _make_config(pop_size=10, n_elite=2, top_pct=0.5)
        pm = PopulationManager(config, model_store=None)
        result = pm.select([])
        assert result.survivors == []
        assert result.eliminated == []
        assert result.elites == []

    def test_single_agent(self):
        config = _make_config(pop_size=1, n_elite=1, top_pct=0.5)
        pm = PopulationManager(config, model_store=None)
        scores = [_make_score("m0", composite=0.5)]
        result = pm.select(scores)
        assert result.survivors == ["m0"]
        assert result.eliminated == []
        assert result.elites == ["m0"]

    def test_two_agents(self):
        config = _make_config(pop_size=2, n_elite=1, top_pct=0.5)
        pm = PopulationManager(config, model_store=None)
        scores = [
            _make_score("m0", composite=0.3),
            _make_score("m1", composite=0.7),
        ]
        result = pm.select(scores)
        assert result.survivors == ["m1"]
        assert result.eliminated == ["m0"]
        assert result.elites == ["m1"]

    def test_tied_scores(self):
        """Agents with identical scores should not cause errors."""
        config = _make_config(pop_size=4, n_elite=1, top_pct=0.5)
        pm = PopulationManager(config, model_store=None)
        scores = [_make_score(f"m{i}", composite=0.5) for i in range(4)]
        result = pm.select(scores)
        assert len(result.survivors) == 2
        assert len(result.eliminated) == 2


# ── Discard policy ───────────────────────────────────────────────────────────


class TestDiscardPolicy:
    def test_bad_model_discarded(self):
        """Model meeting ALL discard criteria is discarded."""
        config = _make_config()
        pm = PopulationManager(config, model_store=None)
        scores = [
            _make_score("bad", composite=0.1, win_rate=0.2, mean_daily_pnl=-5.0, sharpe=-1.0),
        ]
        discarded = pm.apply_discard_policy(scores)
        assert "bad" in discarded

    def test_good_model_not_discarded(self):
        config = _make_config()
        pm = PopulationManager(config, model_store=None)
        scores = [
            _make_score("good", composite=0.8, win_rate=0.7, mean_daily_pnl=10.0, sharpe=2.0),
        ]
        discarded = pm.apply_discard_policy(scores)
        assert discarded == []

    def test_only_win_rate_bad_not_discarded(self):
        """Failing only win_rate is NOT enough for discard."""
        config = _make_config()
        pm = PopulationManager(config, model_store=None)
        scores = [
            _make_score("partial", composite=0.3, win_rate=0.2, mean_daily_pnl=5.0, sharpe=1.0),
        ]
        discarded = pm.apply_discard_policy(scores)
        assert discarded == []

    def test_only_pnl_bad_not_discarded(self):
        config = _make_config()
        pm = PopulationManager(config, model_store=None)
        scores = [
            _make_score("partial", composite=0.3, win_rate=0.5, mean_daily_pnl=-5.0, sharpe=1.0),
        ]
        discarded = pm.apply_discard_policy(scores)
        assert discarded == []

    def test_only_sharpe_bad_not_discarded(self):
        config = _make_config()
        pm = PopulationManager(config, model_store=None)
        scores = [
            _make_score("partial", composite=0.3, win_rate=0.5, mean_daily_pnl=5.0, sharpe=-1.0),
        ]
        discarded = pm.apply_discard_policy(scores)
        assert discarded == []

    def test_two_of_three_bad_not_discarded(self):
        """Failing two out of three criteria is NOT enough."""
        config = _make_config()
        pm = PopulationManager(config, model_store=None)
        scores = [
            _make_score("two_bad", composite=0.2, win_rate=0.2, mean_daily_pnl=-5.0, sharpe=1.0),
        ]
        discarded = pm.apply_discard_policy(scores)
        assert discarded == []

    def test_boundary_values_not_discarded(self):
        """Values exactly at thresholds should NOT be discarded (strict <)."""
        config = _make_config()
        pm = PopulationManager(config, model_store=None)
        scores = [
            _make_score("boundary", composite=0.2, win_rate=0.35, mean_daily_pnl=0.0, sharpe=-0.5),
        ]
        discarded = pm.apply_discard_policy(scores)
        assert discarded == []

    def test_multiple_models_mixed(self):
        """Only the truly bad models are discarded from a mixed population."""
        config = _make_config()
        pm = PopulationManager(config, model_store=None)
        scores = [
            _make_score("good1", composite=0.8, win_rate=0.7, mean_daily_pnl=10.0, sharpe=2.0),
            _make_score("bad1", composite=0.1, win_rate=0.2, mean_daily_pnl=-5.0, sharpe=-1.0),
            _make_score("ok", composite=0.4, win_rate=0.4, mean_daily_pnl=2.0, sharpe=0.5),
            _make_score("bad2", composite=0.05, win_rate=0.1, mean_daily_pnl=-10.0, sharpe=-2.0),
            _make_score("partial_bad", composite=0.2, win_rate=0.2, mean_daily_pnl=-3.0, sharpe=0.1),
        ]
        discarded = pm.apply_discard_policy(scores)
        assert set(discarded) == {"bad1", "bad2"}

    def test_discard_updates_model_store(self, tmp_path):
        """Discard policy marks models as 'discarded' in the model store."""
        store = ModelStore(str(tmp_path / "test.db"), str(tmp_path / "weights"))
        config = _make_config()
        pm = PopulationManager(config, model_store=store)

        # Create a model in the store
        model_id = store.create_model(
            generation=0,
            architecture_name="ppo_lstm_v1",
            architecture_description="test",
            hyperparameters={},
        )
        assert store.get_model(model_id).status == "active"

        scores = [
            _make_score(model_id, composite=0.1, win_rate=0.2, mean_daily_pnl=-5.0, sharpe=-1.0),
        ]
        pm.apply_discard_policy(scores)
        assert store.get_model(model_id).status == "discarded"

    def test_discard_without_store_still_returns_ids(self):
        """Discard returns IDs even without a model store (no DB update)."""
        config = _make_config()
        pm = PopulationManager(config, model_store=None)
        scores = [
            _make_score("bad", composite=0.1, win_rate=0.2, mean_daily_pnl=-5.0, sharpe=-1.0),
        ]
        discarded = pm.apply_discard_policy(scores)
        assert "bad" in discarded

    def test_empty_scores_no_discards(self):
        config = _make_config()
        pm = PopulationManager(config, model_store=None)
        discarded = pm.apply_discard_policy([])
        assert discarded == []

    def test_custom_thresholds(self):
        """Custom discard thresholds from config are respected."""
        config = _make_config()
        config["discard_policy"] = {
            "min_win_rate": 0.5,
            "min_mean_pnl": 5.0,
            "min_sharpe": 0.0,
        }
        pm = PopulationManager(config, model_store=None)
        # This model is below the stricter thresholds
        scores = [
            _make_score("marginal", composite=0.3, win_rate=0.4, mean_daily_pnl=2.0, sharpe=-0.1),
        ]
        discarded = pm.apply_discard_policy(scores)
        assert "marginal" in discarded


# ── Select + Discard combined ────────────────────────────────────────────────


class TestSelectAndDiscard:
    def test_eliminated_models_can_be_discarded(self):
        """Eliminated models that also meet discard criteria get discarded."""
        config = _make_config(pop_size=6, n_elite=1, top_pct=0.5)
        pm = PopulationManager(config, model_store=None)
        scores = [
            _make_score("m0", composite=0.05, win_rate=0.1, mean_daily_pnl=-10.0, sharpe=-2.0),
            _make_score("m1", composite=0.1, win_rate=0.2, mean_daily_pnl=-5.0, sharpe=-1.0),
            _make_score("m2", composite=0.2, win_rate=0.3, mean_daily_pnl=-2.0, sharpe=-0.3),
            _make_score("m3", composite=0.4, win_rate=0.5, mean_daily_pnl=3.0, sharpe=0.5),
            _make_score("m4", composite=0.6, win_rate=0.6, mean_daily_pnl=7.0, sharpe=1.5),
            _make_score("m5", composite=0.8, win_rate=0.8, mean_daily_pnl=12.0, sharpe=2.5),
        ]
        result = pm.select(scores)
        assert set(result.eliminated) == {"m0", "m1", "m2"}

        # Only m0 and m1 meet ALL discard criteria
        discarded = pm.apply_discard_policy(
            [s for s in scores if s.model_id in result.eliminated]
        )
        assert set(discarded) == {"m0", "m1"}

    def test_survivors_not_discarded_even_if_borderline(self):
        """Survivors are not subject to discard (discard only applies to eliminated)."""
        config = _make_config(pop_size=4, n_elite=1, top_pct=0.5)
        pm = PopulationManager(config, model_store=None)
        scores = [
            _make_score("m0", composite=0.1, win_rate=0.2, mean_daily_pnl=-5.0, sharpe=-1.0),
            _make_score("m1", composite=0.2, win_rate=0.3, mean_daily_pnl=-2.0, sharpe=-0.3),
            _make_score("m2", composite=0.5, win_rate=0.5, mean_daily_pnl=5.0, sharpe=1.0),
            _make_score("m3", composite=0.8, win_rate=0.8, mean_daily_pnl=12.0, sharpe=2.5),
        ]
        result = pm.select(scores)
        # Only check eliminated for discard
        discarded = pm.apply_discard_policy(
            [s for s in scores if s.model_id in result.eliminated]
        )
        # m0 meets all criteria, m1 does not (sharpe > -0.5)
        assert set(discarded) == {"m0"}
