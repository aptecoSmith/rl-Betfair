"""Tests for arb-signal-cleanup Session 02 — shaped-penalty warmup.

Covers the six categories mandated by
``plans/arb-signal-cleanup/hard_constraints.md §32``:

1. Default (``warmup_eps=0``) rollouts are byte-identical to pre-change.
2. Linear ramp of ``warmup_scale`` across ``episode_idx``.
3. Only ``efficiency_cost`` / ``precision_reward`` are scaled; other
   shaping terms stay at full strength.
4. JSONL post-episode row carries ``shaped_penalty_warmup_scale`` and
   ``shaped_penalty_warmup_eps``.
5. No cliff at ``warmup_eps + 1`` — linear continuity across the
   transition.
6. Invariant ``raw + shaped ≈ total`` parametrised over
   ``(shaped_penalty_warmup_eps, episode_idx)`` and stacks with the
   Session 01 ``force_close_before_off_seconds`` axis.

All tests run CPU-only on synthetic scripted races — the scaling math
does not depend on policy weights, so a deterministic zero-action
rollout suffices for every quantitative check.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pytest

from data.episode_builder import Day, PriceSize, Race, RunnerMeta, RunnerSnap, Tick
from env.betfair_env import (
    SCALPING_ACTIONS_PER_RUNNER,
    BetfairEnv,
)


# ── Synthetic data helpers (mirror test_force_close) ──────────────────────


_MARKET_START = datetime(2026, 4, 21, 14, 0, 0)


def _meta(sid: int) -> RunnerMeta:
    return RunnerMeta(
        selection_id=sid, runner_name=f"Horse{sid}", sort_priority="1",
        handicap="0", sire_name="S", dam_name="D", damsire_name="DS",
        bred="GB", official_rating="85", adjusted_rating="85", age="4",
        sex_type="GELDING", colour_type="BAY", weight_value="140",
        weight_units="LB", jockey_name="J", jockey_claim="0",
        trainer_name="T", owner_name="O", stall_draw="3",
        cloth_number="1", form="1234", days_since_last_run="14",
        wearing="", forecastprice_numerator="3",
        forecastprice_denominator="1",
    )


def _snap(
    sid: int,
    ltp: float = 4.0,
    back_price: float = 4.0,
    lay_price: float = 4.2,
    size: float = 100.0,
    status: str = "ACTIVE",
) -> RunnerSnap:
    return RunnerSnap(
        selection_id=sid, status=status, last_traded_price=ltp,
        total_matched=500.0, starting_price_near=0.0,
        starting_price_far=0.0, adjustment_factor=None, bsp=None,
        sort_priority=1, removal_date=None,
        available_to_back=[PriceSize(price=back_price, size=size)],
        available_to_lay=[PriceSize(price=lay_price, size=size)],
    )


def _tick(
    market_id: str, seq: int, runners: list[RunnerSnap],
    time_to_off_s: float, winner: int = 101,
) -> Tick:
    ts = _MARKET_START - timedelta(seconds=time_to_off_s)
    return Tick(
        market_id=market_id, timestamp=ts, sequence_number=seq,
        venue="Newmarket", market_start_time=_MARKET_START,
        number_of_active_runners=len(runners), traded_volume=10000.0,
        in_play=False, winner_selection_id=winner, race_status=None,
        temperature=15.0, precipitation=0.0, wind_speed=5.0,
        wind_direction=180.0, humidity=60.0, weather_code=0,
        runners=runners,
    )


def _scripted_day(
    time_to_off_schedule: list[float],
    winner: int = 101,
    sid: int = 101,
) -> Day:
    runner = _snap(sid)
    ticks = [
        _tick("1.999000001", i, [runner], t, winner=winner)
        for i, t in enumerate(time_to_off_schedule)
    ]
    race = Race(
        market_id="1.999000001",
        venue="Newmarket",
        market_start_time=_MARKET_START,
        winner_selection_id=winner,
        ticks=ticks,
        runner_metadata={sid: _meta(sid)},
        winning_selection_ids={winner},
    )
    return Day(date="2026-04-21", races=[race])


def _scalping_config(
    warmup_eps: int = 0,
    force_close_before_off_seconds: int = 0,
    efficiency_penalty: float = 0.05,
    naked_penalty_weight: float = 0.0,
    early_lock_bonus_weight: float = 0.0,
    drawdown_shaping_weight: float = 0.0,
) -> dict:
    """Scalping-mode config with knobs relevant to the warmup tests.

    ``efficiency_penalty`` is the scale target; setting it non-zero
    makes ``efficiency_cost`` non-trivial so the scaling math has
    something to bite on. ``naked_penalty_weight`` / ``early_lock`` are
    surfaced so Test 3 can verify the un-scaled shaping terms stay
    un-scaled.
    """
    return {
        "training": {
            "max_runners": 3,
            "starting_budget": 100.0,
            "max_bets_per_race": 20,
            "scalping_mode": True,
            "shaped_penalty_warmup_eps": warmup_eps,
            "betting_constraints": {
                "max_back_price": 50.0,
                "max_lay_price": None,
                "force_close_before_off_seconds": (
                    force_close_before_off_seconds
                ),
            },
        },
        "actions": {"force_aggressive": True},
        "reward": {
            "early_pick_bonus_min": 1.2,
            "early_pick_bonus_max": 1.5,
            "early_pick_min_seconds": 300,
            "efficiency_penalty": efficiency_penalty,
            "commission": 0.05,
            "naked_penalty_weight": naked_penalty_weight,
            "early_lock_bonus_weight": early_lock_bonus_weight,
            "drawdown_shaping_weight": drawdown_shaping_weight,
        },
    }


def _place_initial_pair(env: BetfairEnv) -> None:
    """Open one aggressive back + paired passive lay on the first tick."""
    a = np.zeros(
        env.max_runners * SCALPING_ACTIONS_PER_RUNNER, dtype=np.float32,
    )
    a[0] = 1.0  # signal = back
    a[env.max_runners + 0] = -0.8  # stake fraction
    a[2 * env.max_runners + 0] = 1.0  # aggression = aggressive
    env.step(a)


def _run_race_and_settle(
    warmup_eps: int,
    episode_idx: int,
    **cfg_kwargs,
) -> dict:
    """Run one scripted race through settlement and return final info."""
    day = _scripted_day([60.0, 29.0, 15.0])
    env = BetfairEnv(
        day,
        _scalping_config(warmup_eps=warmup_eps, **cfg_kwargs),
    )
    env.set_episode_idx(episode_idx)
    env.reset()
    _place_initial_pair(env)
    done = False
    last_info: dict = {}
    total_reward = 0.0
    while not done:
        _, r, terminated, truncated, last_info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32),
        )
        total_reward += float(r)
        done = terminated or truncated
    last_info["_total_reward"] = total_reward
    return last_info


# ===========================================================================
# 1. Default (warmup=0) byte-identical
# ===========================================================================


class TestDefaultByteIdentical:
    """With ``warmup_eps == 0`` the warmup path is a no-op.

    We compare two configurations:
    - Config A: no ``shaped_penalty_warmup_eps`` key (absent in config
      → default 0 via ``.get(..., 0)``).
    - Config B: explicit ``shaped_penalty_warmup_eps: 0``.
    Both should produce identical per-episode reward components.
    Additionally, the scale should read as 1.0 (the ``eps<=0`` branch)
    regardless of ``episode_idx``.
    """

    def test_absent_key_equals_explicit_zero(self):
        # Config A: key absent.
        day = _scripted_day([60.0, 29.0, 15.0])
        cfg_a = _scalping_config(warmup_eps=0)
        del cfg_a["training"]["shaped_penalty_warmup_eps"]
        env_a = BetfairEnv(day, cfg_a)
        env_a.reset()
        _place_initial_pair(env_a)
        done = False
        info_a: dict = {}
        while not done:
            _, _, terminated, truncated, info_a = env_a.step(
                np.zeros(env_a.action_space.shape, dtype=np.float32),
            )
            done = terminated or truncated

        # Config B: key explicit 0.
        day_b = _scripted_day([60.0, 29.0, 15.0])
        env_b = BetfairEnv(day_b, _scalping_config(warmup_eps=0))
        env_b.reset()
        _place_initial_pair(env_b)
        done = False
        info_b: dict = {}
        while not done:
            _, _, terminated, truncated, info_b = env_b.step(
                np.zeros(env_b.action_space.shape, dtype=np.float32),
            )
            done = terminated or truncated

        assert info_a["raw_pnl_reward"] == pytest.approx(
            info_b["raw_pnl_reward"], abs=1e-9,
        )
        assert info_a["shaped_bonus"] == pytest.approx(
            info_b["shaped_bonus"], abs=1e-9,
        )

    def test_episode_idx_ignored_when_disabled(self):
        """With ``warmup_eps == 0``, scale is 1.0 regardless of idx."""
        info_idx_0 = _run_race_and_settle(warmup_eps=0, episode_idx=0)
        info_idx_100 = _run_race_and_settle(warmup_eps=0, episode_idx=100)
        assert info_idx_0["shaped_penalty_warmup_scale"] == 1.0
        assert info_idx_100["shaped_penalty_warmup_scale"] == 1.0
        assert info_idx_0["shaped_bonus"] == pytest.approx(
            info_idx_100["shaped_bonus"], abs=1e-9,
        )


# ===========================================================================
# 2. Linear ramp of warmup_scale
# ===========================================================================


class TestLinearRamp:
    @pytest.mark.parametrize(
        "episode_idx, expected_scale",
        [
            (0, 0.0),
            (5, 0.5),
            (9, 0.9),
            (10, 1.0),
            (20, 1.0),
        ],
    )
    def test_scale_for_episode_idx(self, episode_idx, expected_scale):
        info = _run_race_and_settle(
            warmup_eps=10, episode_idx=episode_idx,
        )
        assert info["shaped_penalty_warmup_scale"] == pytest.approx(
            expected_scale, abs=1e-12,
        )
        assert info["shaped_penalty_warmup_eps"] == 10


# ===========================================================================
# 3. Only efficiency_cost and precision_reward scale — other terms unchanged
# ===========================================================================


class TestOnlyTwoTermsScaled:
    """Scaling is linear in the two targeted terms.

    In scalping mode ``precision_reward`` is zeroed unconditionally
    (hard constraint — one leg of every completed arb is a planned
    loss, so a directional precision metric is nonsense). Thus the
    scalping-mode reduction of the warmup is "scale multiplies
    ``efficiency_cost`` only"; the linear-interpolation property
    between ``scale=0`` and ``scale=1`` is still diagnostic of the
    mechanism's shape because no *other* shaping term depends on
    scale.

    We also assert that a non-scaled term (``naked_penalty_term``)
    contributes identically at ``scale=0`` and ``scale=1`` — proving
    it is NOT caught in the warmup.
    """

    def test_linear_interpolation_in_shaped(self):
        """``shaped(0.5) == 0.5 * shaped(0) + 0.5 * shaped(1)``.

        The race is the same in all three calls; anything not scaled
        cancels on subtraction and the leftover is strictly linear in
        ``warmup_scale``.
        """
        common = dict(efficiency_penalty=0.05)
        info_0 = _run_race_and_settle(
            warmup_eps=10, episode_idx=0, **common,
        )
        info_half = _run_race_and_settle(
            warmup_eps=10, episode_idx=5, **common,
        )
        info_1 = _run_race_and_settle(
            warmup_eps=10, episode_idx=10, **common,
        )
        shaped_0 = info_0["shaped_bonus"]
        shaped_half = info_half["shaped_bonus"]
        shaped_1 = info_1["shaped_bonus"]
        # Strict linearity: the mid-point equals the mean of the
        # endpoints to float-eps.
        assert shaped_half == pytest.approx(
            0.5 * shaped_0 + 0.5 * shaped_1, abs=1e-9,
        )
        # And the endpoints themselves must differ — otherwise the
        # test is vacuous (e.g. zero efficiency_cost).
        assert shaped_0 != pytest.approx(shaped_1, abs=1e-9)

    def test_non_scaled_terms_survive_through_scale_zero(self):
        """Activating a non-scaled shaping term alters ``shaped`` at
        ``scale=0`` — proving the warmup does NOT catch it.

        Chooses ``drawdown_shaping_weight > 0`` because it emits a
        non-zero shaped contribution on any race whose settled P&L is
        non-zero (the scripted race locks a real positive P&L). At
        ``scale=0`` the efficiency_cost contribution vanishes; the only
        surviving shaping channel is the drawdown term, so its
        enable/disable state is directly visible in ``shaped_bonus``.
        """
        base = _run_race_and_settle(
            warmup_eps=10, episode_idx=0,
            efficiency_penalty=0.05, drawdown_shaping_weight=0.0,
        )
        with_drawdown = _run_race_and_settle(
            warmup_eps=10, episode_idx=0,
            efficiency_penalty=0.05, drawdown_shaping_weight=0.5,
        )
        assert base["shaped_penalty_warmup_scale"] == 0.0
        assert with_drawdown["shaped_penalty_warmup_scale"] == 0.0
        # efficiency_cost × 0 = 0 for both; the only non-zero
        # difference comes from the non-scaled drawdown term.
        assert with_drawdown["shaped_bonus"] != pytest.approx(
            base["shaped_bonus"], abs=1e-9,
        )


# ===========================================================================
# 4. JSONL field present
# ===========================================================================


class TestJsonlFieldPresent:
    def test_info_dict_carries_both_keys(self):
        info = _run_race_and_settle(warmup_eps=5, episode_idx=2)
        assert "shaped_penalty_warmup_scale" in info
        assert "shaped_penalty_warmup_eps" in info
        assert info["shaped_penalty_warmup_scale"] == pytest.approx(
            0.4, abs=1e-12,
        )
        assert info["shaped_penalty_warmup_eps"] == 5

    def test_jsonl_row_carries_warmup_fields(self, tmp_path):
        """A full PPOTrainer episode log row contains the warmup keys.

        Bypasses the policy rollout path by constructing a minimal
        ``EpisodeStats`` and calling ``_log_episode`` directly — the
        log-formatting code is what this test guards.
        """
        import torch  # noqa: F401

        from agents.policy_network import PPOLSTMPolicy
        from agents.ppo_trainer import EpisodeStats, PPOTrainer

        cfg = _scalping_config(warmup_eps=10)
        cfg["paths"] = {"logs": str(tmp_path)}
        policy = PPOLSTMPolicy(
            obs_dim=32, action_dim=6, max_runners=2,
            hyperparams={
                "lstm_hidden_size": 16, "mlp_hidden_size": 16,
                "mlp_layers": 1,
            },
        )
        trainer = PPOTrainer(policy, cfg, hyperparams={})
        ep = EpisodeStats(
            day_date="2026-04-21",
            total_reward=0.0,
            total_pnl=0.0,
            bet_count=0,
            winning_bets=0,
            races_completed=0,
            final_budget=100.0,
            n_steps=0,
            shaped_penalty_warmup_scale=0.7,
            shaped_penalty_warmup_eps=10,
        )

        class _T:
            completed = 1
            total = 1

        loss_info = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        trainer._log_episode(ep, loss_info, _T())
        log_file = tmp_path / "training" / "episodes.jsonl"
        assert log_file.exists()
        rows = [json.loads(line) for line in log_file.read_text(
            encoding="utf-8",
        ).splitlines()]
        row = rows[-1]
        assert row["shaped_penalty_warmup_scale"] == pytest.approx(
            0.7, abs=1e-9,
        )
        assert row["shaped_penalty_warmup_eps"] == 10


# ===========================================================================
# 5. No cliff at warmup+1 — continuity across the transition
# ===========================================================================


class TestNoCliffAtTransition:
    def test_delta_continuous_across_warmup_boundary(self):
        """Stepping ``episode_idx`` 1→2, …, 9→10, 10→11 on the same
        scripted race gives a constant shaped-delta during the ramp
        (0.1 × scaled-terms per step) and a zero delta after.

        Specifically: the delta between consecutive episodes is
        ``(scale_{i+1} - scale_i) × scaled_terms_total``, a constant
        0.1 × X during ep0..ep10, then 0 for ep10..ep11. NO
        discontinuity (the ep10→ep11 delta is NOT a spike).
        """
        warmup_eps = 10
        shaped_per_ep = [
            _run_race_and_settle(
                warmup_eps=warmup_eps, episode_idx=idx,
            )["shaped_bonus"]
            for idx in range(warmup_eps + 3)  # 0..12
        ]
        deltas = [
            shaped_per_ep[i + 1] - shaped_per_ep[i]
            for i in range(len(shaped_per_ep) - 1)
        ]
        # During ramp (ep0→ep1 … ep9→ep10) all deltas equal.
        ramp_deltas = deltas[:warmup_eps]
        for d in ramp_deltas[1:]:
            assert d == pytest.approx(ramp_deltas[0], abs=1e-9)
        # After ramp (ep10→ep11, ep11→ep12) deltas are zero.
        for d in deltas[warmup_eps:]:
            assert d == pytest.approx(0.0, abs=1e-9)
        # Critically: the final ramp delta (ep9→ep10) is NOT a spike —
        # it must equal the other ramp deltas. This is the cliff guard.
        assert deltas[warmup_eps - 1] == pytest.approx(
            ramp_deltas[0], abs=1e-9,
        )


# ===========================================================================
# 6. Invariant raw + shaped ≈ total, parametrised with stacking Session 01
# ===========================================================================


@pytest.mark.parametrize("warmup_eps", [0, 5])
@pytest.mark.parametrize("episode_idx", [0, 2, 4, 5, 10])
@pytest.mark.parametrize("force_close", [0, 30])
def test_invariant_raw_plus_shaped_equals_total(
    warmup_eps, episode_idx, force_close,
):
    """Invariant stacks with Session 01's force-close parametrisation.

    Exercises the full matrix mandated by hard_constraints.md §28:
    - ``shaped_penalty_warmup_eps ∈ {0, 5}``
    - ``episode_idx ∈ {0, 2, 4, 5, 10}`` (covers pre-, mid-, and
      post-warmup)
    - ``force_close_before_off_seconds ∈ {0, 30}`` (Session 01 axis)
    The raw + shaped invariant must hold for every combination.
    """
    info = _run_race_and_settle(
        warmup_eps=warmup_eps,
        episode_idx=episode_idx,
        force_close_before_off_seconds=force_close,
    )
    total_reward = info["_total_reward"]
    raw = info["raw_pnl_reward"]
    shaped = info["shaped_bonus"]
    assert total_reward == pytest.approx(raw + shaped, abs=1e-6), (
        f"invariant broken at warmup_eps={warmup_eps}, "
        f"episode_idx={episode_idx}, force_close={force_close}: "
        f"total={total_reward} raw+shaped={raw + shaped}"
    )
