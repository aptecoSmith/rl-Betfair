"""Session 3 — verify the expanded reward hyperparameter schema.

The four new genes (``early_pick_bonus_min``, ``early_pick_bonus_max``,
``early_pick_min_seconds``, ``terminal_bonus_weight``) must:

1. Be sampled by the genetic search and stay within their declared ranges.
2. Reach ``BetfairEnv`` via the Session 1 ``reward_overrides`` plumbing.
3. Repair (swap) an inverted ``early_pick_bonus`` interval rather than
   crash — the plan deliberately favours repair over rejection so the
   genetic search never throws away a candidate.
4. Preserve the ``early_pick_bonus`` zero-mean symmetry around random
   policies (winners and losers cancel for identical placement times).
5. Apply ``terminal_bonus_weight`` to the **raw** terminal contribution
   (real cash P&L), not the shaped bucket — and the
   ``raw + shaped ≈ total_reward`` invariant continues to hold.
6. Survive 200 mutation rounds with both range and pair-order invariants
   intact.

All tests are CPU-only and run in well under a second each.
"""

from __future__ import annotations

import random
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from agents.population_manager import (
    PopulationManager,
    parse_search_ranges,
    sample_hyperparams,
)
from env.betfair_env import BetfairEnv
from env.bet_manager import Bet, BetManager, BetOutcome, BetSide

# Re-use the synthetic Day/Race/Tick builders from the existing env tests.
from tests.test_betfair_env import _make_day


REWARD_GENES = (
    "early_pick_bonus_min",
    "early_pick_bonus_max",
    "early_pick_min_seconds",
    "terminal_bonus_weight",
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def reward_config() -> dict:
    """Minimal config block satisfied by ``BetfairEnv.__init__``."""
    return {
        "training": {
            "max_runners": 14,
            "starting_budget": 100.0,
            "max_bets_per_race": 20,
        },
        "reward": {
            "early_pick_bonus_min": 1.2,
            "early_pick_bonus_max": 1.5,
            "early_pick_min_seconds": 300,
            "terminal_bonus_weight": 1.0,
            "efficiency_penalty": 0.01,
            "precision_bonus": 0.0,
            "commission": 0.05,
        },
    }


@pytest.fixture
def real_search_ranges() -> dict:
    """The hyperparameter search_ranges block from the live config.yaml."""
    config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["hyperparameters"]["search_ranges"]


# ── Test 1 — sampling produces all four genes in range ─────────────────────


def test_sampler_emits_new_reward_genes_in_range(real_search_ranges):
    specs = parse_search_ranges(real_search_ranges)
    rng = random.Random(0)

    for _ in range(50):
        hp = sample_hyperparams(specs, rng)
        for gene in REWARD_GENES:
            assert gene in hp, f"Gene {gene!r} was not sampled"
            spec = real_search_ranges[gene]
            assert spec["min"] <= hp[gene] <= spec["max"], (
                f"{gene}={hp[gene]} outside [{spec['min']}, {spec['max']}]"
            )

        # The sampler also runs the paired-gene repair, so the post-sample
        # invariant ``max >= min`` must already hold.
        assert hp["early_pick_bonus_max"] >= hp["early_pick_bonus_min"]
        # int gene must remain an int (sampler emits ``rng.randint``).
        assert isinstance(hp["early_pick_min_seconds"], int)


# ── Test 2 — env picks up the new overrides ────────────────────────────────


def test_env_picks_up_new_reward_overrides(reward_config):
    """Extreme overrides must land on the corresponding env attributes."""
    day = _make_day(n_races=1, n_pre_ticks=3, n_inplay_ticks=1)
    env = BetfairEnv(
        day,
        reward_config,
        reward_overrides={
            "early_pick_bonus_min": 1.0,
            "early_pick_bonus_max": 1.8,
            "early_pick_min_seconds": 120,
            "terminal_bonus_weight": 3.0,
        },
    )
    assert env._early_pick_min == pytest.approx(1.0)
    assert env._early_pick_max == pytest.approx(1.8)
    assert env._early_pick_seconds == 120
    assert env._terminal_bonus_weight == pytest.approx(3.0)


# ── Test 3 — inverted early-pick interval is repaired, not crashed ─────────


def test_env_repairs_inverted_early_pick_interval(reward_config):
    """``min > max`` must be repaired by swapping (documented choice).

    Independent sampling/mutation can produce an inverted interval; the
    Session 3 plan opts for *repair* over rejection so the genome is
    preserved. ``BetfairEnv`` performs the swap defensively even if an
    upstream consumer forgot to.
    """
    day = _make_day(n_races=1, n_pre_ticks=3, n_inplay_ticks=1)
    env = BetfairEnv(
        day,
        reward_config,
        reward_overrides={
            "early_pick_bonus_min": 1.5,  # deliberately the larger value
            "early_pick_bonus_max": 1.1,  # deliberately the smaller value
        },
    )
    # Repair is a swap (not a clamp): the ordering is now well-formed
    # and both end values are preserved.
    assert env._early_pick_min == pytest.approx(1.1)
    assert env._early_pick_max == pytest.approx(1.5)
    assert env._early_pick_min <= env._early_pick_max


# ── Test 4 — early_pick_bonus is symmetric around random policies ─────────


def test_early_pick_bonus_is_symmetric_for_winner_and_loser(reward_config):
    """Equal-magnitude winning and losing back bets, placed at the same
    time before the off, must contribute zero net shaped reward.

    This is the zero-mean invariant from CLAUDE.md. Promoting min/max to
    genes only widens the multiplier — it must not break symmetry.
    """
    day = _make_day(n_races=1, n_pre_ticks=5, n_inplay_ticks=1)
    env = BetfairEnv(
        day,
        reward_config,
        reward_overrides={
            "early_pick_bonus_min": 1.0,
            "early_pick_bonus_max": 1.8,
            "early_pick_min_seconds": 120,
        },
    )
    env.reset()
    race = day.races[0]

    winner_bet = Bet(
        selection_id=101,
        side=BetSide.BACK,
        requested_stake=10.0,
        matched_stake=10.0,
        average_price=2.0,
        market_id=race.market_id,
        outcome=BetOutcome.WON,
        pnl=+10.0,
    )
    loser_bet = Bet(
        selection_id=102,
        side=BetSide.BACK,
        requested_stake=10.0,
        matched_stake=10.0,
        average_price=2.0,
        market_id=race.market_id,
        outcome=BetOutcome.LOST,
        pnl=-10.0,
    )
    env.bet_manager.bets = [winner_bet, loser_bet]
    # Both placed 10 minutes out → identical multiplier for each.
    env._bet_times = {0: 600.0, 1: 600.0}

    bonus, count = env._compute_early_pick_bonus(
        race, [winner_bet, loser_bet],
    )
    assert count == 2
    assert bonus == pytest.approx(0.0, abs=1e-9)


# ── Test 5 — terminal_bonus_weight scales the raw bucket, not shaped ──────


def test_terminal_bonus_weight_lands_in_raw_not_shaped(reward_config):
    """``terminal_bonus_weight`` is a multiplier on real cash P&L.

    The terminal contribution must accumulate into ``raw_pnl_reward``
    (not ``shaped_bonus``), and the ``raw + shaped ≈ total`` invariant
    must continue to hold. To verify the *weight* is actually applied
    (and not silently dropped), we patch ``settle_race`` to inject a
    known race P&L so the terminal bonus has a non-zero base to scale.
    """
    day = _make_day(n_races=1, n_pre_ticks=2, n_inplay_ticks=1)
    weight = 2.0
    env = BetfairEnv(
        day,
        reward_config,
        reward_overrides={"terminal_bonus_weight": weight},
    )
    env.reset()

    fake_race_pnl = 10.0
    with patch.object(BetManager, "settle_race", return_value=fake_race_pnl):
        zero = np.zeros(env.action_space.shape, dtype=np.float32)
        total_reward = 0.0
        done = False
        info: dict = {}
        while not done:
            _, reward, terminated, truncated, info = env.step(zero)
            total_reward += reward
            done = terminated or truncated

    starting_budget = reward_config["training"]["starting_budget"]
    expected_terminal = weight * fake_race_pnl / starting_budget
    expected_raw = fake_race_pnl + expected_terminal

    assert env._terminal_bonus_weight == pytest.approx(weight)
    assert info["raw_pnl_reward"] == pytest.approx(expected_raw)
    # No bets were placed and precision_bonus is 0 → shaped contribution
    # is exactly zero.
    assert info["shaped_bonus"] == pytest.approx(0.0)
    # Invariant: raw + shaped == total reward.
    assert info["raw_pnl_reward"] + info["shaped_bonus"] == pytest.approx(
        total_reward, abs=1e-6,
    )


# ── Test 6 — mutation respects ranges and the pair-order invariant ────────


def test_mutation_respects_ranges_and_repair_invariant(real_search_ranges):
    """Seeded mutation rounds must keep every gene in range and the
    early-pick interval well-formed (``max >= min``)."""
    specs = parse_search_ranges(real_search_ranges)
    rng = random.Random(42)

    pm_config = {
        "population": {
            "size": 4,
            "n_elite": 1,
            "selection_top_pct": 0.5,
            "mutation_rate": 1.0,
        },
        "training": {
            "architecture": "ppo_lstm_v1",
            "max_runners": 14,
            "starting_budget": 100.0,
            "max_bets_per_race": 20,
        },
        "hyperparameters": {"search_ranges": real_search_ranges},
    }
    pm = PopulationManager(pm_config, model_store=None)

    spec_map = {s.name: s for s in specs}
    hp = sample_hyperparams(specs, rng)

    for _ in range(200):
        hp, _ = pm.mutate(hp, mutation_rate=1.0, rng=rng)
        for gene in REWARD_GENES:
            spec = spec_map[gene]
            value = hp[gene]
            assert spec.min <= value <= spec.max, (
                f"After mutation, {gene}={value} outside "
                f"[{spec.min}, {spec.max}]"
            )
        assert hp["early_pick_bonus_max"] >= hp["early_pick_bonus_min"], (
            "Mutation produced an inverted early-pick interval that "
            "the repair step failed to swap"
        )
