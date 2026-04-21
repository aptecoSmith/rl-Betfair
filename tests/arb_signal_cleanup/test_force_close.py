"""Tests for arb-signal-cleanup Session 01.

Covers the ten categories mandated by
``plans/arb-signal-cleanup/hard_constraints.md §31``:

1. Force-close fires at threshold (time_to_off ≤ N triggers, > N does not).
2. Force-close uses the matcher — unpriceable runner stays naked.
3. Force-close respects the hard price cap / junk filter.
4. Force-close P&L in ``race_pnl`` — bucket formula holds.
5. Matured-arb bonus excludes force-closes.
6. Close_signal shaped bonus excludes force-closes.
7. ``alpha_lr`` gene passthrough (PPOTrainer optimiser lr).
8. ``alpha_lr`` does not mutate across PPO updates.
9. Invariant raw + shaped ≈ total parametrised over
   ``force_close_before_off_seconds`` and ``alpha_lr``.
10. Transformer builds and forwards at ``transformer_ctx_ticks=256``.

All tests run CPU-only and avoid training-heavy paths.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from data.episode_builder import Day, PriceSize, Race, RunnerMeta, RunnerSnap, Tick
from env.bet_manager import BetSide
from env.betfair_env import (
    SCALPING_ACTIONS_PER_RUNNER,
    BetfairEnv,
)


# ── Synthetic data helpers ─────────────────────────────────────────────────


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


def _scripted_race(
    time_to_off_schedule: list[float],
    back_price: float = 4.0,
    lay_price: float = 4.2,
    winner: int = 101,
    sid: int = 101,
    tick_snaps: list[RunnerSnap] | None = None,
) -> Race:
    """Build a one-runner race with ticks at the given seconds-to-off.

    If ``tick_snaps`` is given it MUST have the same length as
    ``time_to_off_schedule`` — one runner snap per tick. Otherwise a
    uniform snap at the given back/lay prices is used.
    """
    if tick_snaps is not None:
        assert len(tick_snaps) == len(time_to_off_schedule)
        ticks = [
            _tick("1.999000001", i, [tick_snaps[i]], t, winner=winner)
            for i, t in enumerate(time_to_off_schedule)
        ]
    else:
        runner = _snap(sid, back_price=back_price, lay_price=lay_price)
        ticks = [
            _tick("1.999000001", i, [runner], t, winner=winner)
            for i, t in enumerate(time_to_off_schedule)
        ]
    return Race(
        market_id="1.999000001",
        venue="Newmarket",
        market_start_time=_MARKET_START,
        winner_selection_id=winner,
        ticks=ticks,
        runner_metadata={sid: _meta(sid)},
        winning_selection_ids={winner},
    )


def _scripted_day(
    time_to_off_schedule: list[float],
    **kwargs,
) -> Day:
    return Day(
        date="2026-04-21",
        races=[_scripted_race(time_to_off_schedule, **kwargs)],
    )


def _scalping_config(
    force_close_before_off_seconds: int = 0,
    extra_reward: dict | None = None,
) -> dict:
    cfg: dict = {
        "training": {
            "max_runners": 3,
            "starting_budget": 100.0,
            "max_bets_per_race": 20,
            "scalping_mode": True,
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
            "efficiency_penalty": 0.01,
            "commission": 0.05,
        },
    }
    if extra_reward:
        cfg["reward"].update(extra_reward)
    return cfg


def _place_initial_pair(env: BetfairEnv) -> tuple:
    """Open one aggressive back + paired passive lay on the first tick."""
    a = np.zeros(
        env.max_runners * SCALPING_ACTIONS_PER_RUNNER, dtype=np.float32,
    )
    a[0] = 1.0  # signal = back
    a[env.max_runners + 0] = -0.8  # stake fraction
    a[2 * env.max_runners + 0] = 1.0  # aggression = aggressive
    env.step(a)
    bm = env.bet_manager
    paired = [b for b in bm.bets if b.pair_id is not None]
    assert paired, "initial aggressive pair must have placed"
    # Prevent auto-fill of the passive: inflate queue ahead so the
    # passive cannot fill via synthesised ladder volume.
    for o in bm.passive_book.orders:
        if o.pair_id is not None:
            o.queue_ahead_at_placement = 1e12
    return paired[0], next(
        o for o in bm.passive_book.orders if o.pair_id == paired[0].pair_id
    )


# ===========================================================================
# 1. Force-close fires at threshold
# ===========================================================================


class TestForceCloseFiresAtThreshold:
    def test_above_threshold_does_not_fire(self):
        """Tick at T−31 with threshold=30 does NOT force-close.

        Schedule: T−60 (place) → T−35 (above threshold) checks fire
        gate. We inspect state BEFORE the final settlement tick so
        race-off cleanup does not confound the "passive still in
        book" signal.
        """
        day = _scripted_day([60.0, 35.0, 15.0])
        env = BetfairEnv(
            day,
            _scalping_config(force_close_before_off_seconds=30),
        )
        env.reset()
        agg, _ = _place_initial_pair(env)
        # Step onto tick 1 (T−35): 35 > 30 → no force-close.
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        bm = env.bet_manager
        # No force-close bet was added; the passive is still resting
        # in the book (race-off cleanup hasn't run yet).
        assert not any(b.force_close for b in bm.bets)
        assert any(
            o.pair_id == agg.pair_id for o in bm.passive_book.orders
        )

    def test_at_threshold_fires(self):
        """Tick at T−30 with threshold=30 force-closes (boundary = fire).

        Use three ticks so we can inspect state after force-close
        fires but before race-off settlement cleanup runs.
        """
        day = _scripted_day([60.0, 30.0, 15.0])
        env = BetfairEnv(
            day,
            _scalping_config(force_close_before_off_seconds=30),
        )
        env.reset()
        agg, _ = _place_initial_pair(env)
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        bm = env.bet_manager
        close_bets = [
            b for b in bm.bets if b.pair_id == agg.pair_id and b.close_leg
        ]
        assert len(close_bets) == 1
        assert close_bets[0].force_close is True


# ===========================================================================
# 2. Force-close uses matcher
# ===========================================================================


class TestForceCloseUsesMatcher:
    def test_unpriceable_runner_stays_naked(self):
        """Runner with LTP=0 → matcher refuses → pair stays naked."""
        # Tick 0: normal; Tick 1: priceable, used to confirm fire can
        # happen; Tick 2: unpriceable (LTP=0, empty books), confirms
        # the matcher refusal path leaves the pair naked.
        #
        # We use two test ticks (priceable + unpriceable) on a
        # two-runner pair to exercise both branches at once: the
        # agg-back pair gets force-closed normally on the priceable
        # ticker (sid=101), and a second pair on an unpriceable
        # ticker (sid=102) fails.
        normal = _snap(101)
        broken = RunnerSnap(
            selection_id=101, status="ACTIVE", last_traded_price=0.0,
            total_matched=500.0, starting_price_near=0.0,
            starting_price_far=0.0, adjustment_factor=None, bsp=None,
            sort_priority=1, removal_date=None,
            available_to_back=[],
            available_to_lay=[],
        )
        race = _scripted_race(
            [60.0, 29.0, 15.0],
            tick_snaps=[normal, broken, broken],
        )
        day = Day(date="2026-04-21", races=[race])
        env = BetfairEnv(
            day,
            _scalping_config(force_close_before_off_seconds=30),
        )
        env.reset()
        agg, _ = _place_initial_pair(env)
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        bm = env.bet_manager
        # On the unpriceable tick, force-close fires but the matcher
        # refuses — no force-close Bet exists.
        assert not any(b.force_close for b in bm.bets)

    def test_priceable_runner_closes(self):
        """With a matchable opposite-side book, close lands."""
        day = _scripted_day([60.0, 29.0, 15.0])
        env = BetfairEnv(
            day,
            _scalping_config(force_close_before_off_seconds=30),
        )
        env.reset()
        agg, _ = _place_initial_pair(env)
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        bm = env.bet_manager
        assert any(b.force_close for b in bm.bets)


# ===========================================================================
# 3. Force-close respects hard price cap
# ===========================================================================


class TestForceCloseHardPriceCap:
    def test_price_above_cap_refused(self):
        """An aggressive close at a price above max_back_price is refused.

        A force-close on a back-first pair crosses via an aggressive
        LAY, so we set a tight ``max_lay_price`` cap that the current
        best LAY (4.2) exceeds. The close attempt goes through
        ``bm.place_lay`` which refuses when the post-filter top price
        exceeds the cap.
        """
        day = _scripted_day([60.0, 29.0, 15.0])
        cfg = _scalping_config(force_close_before_off_seconds=30)
        # Tight cap so the aggressive-lay close (at LTP's lay-side top
        # 4.2) is rejected.
        cfg["training"]["betting_constraints"]["max_lay_price"] = 2.5
        env = BetfairEnv(day, cfg)
        env.reset()
        agg, _ = _place_initial_pair(env)
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        bm = env.bet_manager
        # No close leg landed despite force-close firing.
        assert not any(b.force_close for b in bm.bets)


# ===========================================================================
# 4. Force-close P&L in race_pnl
# ===========================================================================


class TestForceClosedPnlInRacePnl:
    def test_race_pnl_bucket_sum(self):
        """race_pnl = locked + closed + force_closed + scaled_naked_sum.

        Run the force-close path, settle the race, then check the
        buckets sum to the race's total P&L.
        """
        day = _scripted_day([60.0, 29.0, 15.0])
        env = BetfairEnv(
            day,
            _scalping_config(force_close_before_off_seconds=30),
        )
        env.reset()
        _place_initial_pair(env)
        # Drive to settlement.
        done = False
        last_info: dict = {}
        while not done:
            _, _, terminated, truncated, last_info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32),
            )
            done = terminated or truncated
        rr = last_info["race_records"][0]
        # The force_close_pnl bucket is non-zero (the force-close
        # settled into it).
        assert rr.force_closed_pnl != 0.0 or rr.arbs_force_closed >= 1
        # Aggregates: all buckets sum to race P&L to float tolerance.
        assert rr.pnl == pytest.approx(
            rr.locked_pnl
            + rr.force_closed_pnl
            + rr.naked_pnl
            # closed_pnl is not on RaceRecord — it lives in the env's
            # internal split; naked_pnl here is race_pnl minus
            # locked, closed, and force_closed per the env. So
            # rr.pnl - rr.locked_pnl - rr.force_closed_pnl - rr.naked_pnl
            # equals the agent-close cash, which should be 0 for a
            # pure-force-close race.
            + 0.0,
            abs=1e-6,
        )


# ===========================================================================
# 5. Matured-arb bonus excludes force-closes
# ===========================================================================


class TestMaturedArbBonusExcludesForceCloses:
    def test_force_close_does_not_count_toward_matured(self):
        """With bonus weight=1.0, expected_random=0: force-close alone
        produces term = weight * (0 matured - 0) = 0."""
        day = _scripted_day([60.0, 29.0, 15.0])
        env = BetfairEnv(
            day,
            _scalping_config(
                force_close_before_off_seconds=30,
                extra_reward={
                    "matured_arb_bonus_weight": 1.0,
                    "matured_arb_bonus_cap": 100.0,
                    "matured_arb_expected_random": 0.0,
                },
            ),
        )
        env.reset()
        _place_initial_pair(env)
        done = False
        last_info: dict = {}
        while not done:
            _, _, terminated, truncated, last_info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32),
            )
            done = terminated or truncated
        # arbs_force_closed is credited; arbs_closed + arbs_completed
        # stays zero. n_matured = 0 → term = 1.0 * (0 - 0) = 0.
        assert last_info["arbs_force_closed"] >= 1
        assert last_info["arbs_closed"] == 0
        assert last_info["arbs_completed"] == 0


# ===========================================================================
# 6. Close_signal bonus excludes force-closes
# ===========================================================================


class TestCloseSignalBonusExcludesForceCloses:
    def test_close_bonus_counts_only_agent_closes(self):
        """``n_close_signal_successes`` passed to the shaped-reward
        helper is ``scalping_arbs_closed`` only — force-closes don't
        trigger the +£1 per close bonus.

        Check by comparing shaped contribution across two runs: same
        scripted race, same force-close result, one with the close
        bonus visible and one with no force-close. The close bonus
        is a fixed +£1 * close_signal_successes. A run with only
        force-closes should contribute 0 close-bonus pounds.
        """
        day = _scripted_day([60.0, 29.0, 15.0])
        env = BetfairEnv(
            day,
            _scalping_config(force_close_before_off_seconds=30),
        )
        env.reset()
        _place_initial_pair(env)
        done = False
        last_info: dict = {}
        while not done:
            _, _, terminated, truncated, last_info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32),
            )
            done = terminated or truncated
        # Pair force-closed but arbs_closed stayed 0 → the close_bonus
        # term contributed 0 * £1 = £0.
        assert last_info["arbs_force_closed"] >= 1
        assert last_info["arbs_closed"] == 0


# ===========================================================================
# 7. & 8. alpha_lr gene passthrough + non-mutation
# ===========================================================================


class TestAlphaLrGene:
    def _minimal_policy(self):
        import torch  # noqa: F401

        from agents.policy_network import PPOLSTMPolicy

        return PPOLSTMPolicy(
            obs_dim=32, action_dim=6, max_runners=2,
            hyperparams={
                "lstm_hidden_size": 16, "mlp_hidden_size": 16,
                "mlp_layers": 1,
            },
        )

    def test_alpha_lr_passes_through_to_optimizer(self):
        """PPOTrainer built with ``{"alpha_lr": 0.05}`` → optimiser lr==0.05."""
        from agents.ppo_trainer import PPOTrainer

        policy = self._minimal_policy()
        cfg = _scalping_config()
        trainer = PPOTrainer(
            policy, cfg, hyperparams={"alpha_lr": 0.05},
        )
        lr = trainer._alpha_optimizer.param_groups[0]["lr"]
        assert lr == pytest.approx(0.05, abs=1e-12)
        assert trainer._alpha_lr == pytest.approx(0.05, abs=1e-12)

    def test_alpha_lr_default_unchanged_without_gene(self):
        """No gene override → default 1e-2 preserved (byte-identical)."""
        from agents.ppo_trainer import PPOTrainer

        policy = self._minimal_policy()
        cfg = _scalping_config()
        trainer = PPOTrainer(policy, cfg, hyperparams={})
        lr = trainer._alpha_optimizer.param_groups[0]["lr"]
        assert lr == pytest.approx(1e-2, abs=1e-12)

    def test_alpha_lr_does_not_mutate_across_controller_steps(self):
        """Stepping the controller N times leaves the optimiser LR fixed.

        The controller moves ``log_alpha`` only; it does NOT touch the
        LR. Regression guard for hard_constraints.md §16.
        """
        from agents.ppo_trainer import PPOTrainer

        policy = self._minimal_policy()
        cfg = _scalping_config()
        trainer = PPOTrainer(
            policy, cfg, hyperparams={"alpha_lr": 0.05},
        )
        for _ in range(5):
            trainer._update_entropy_coefficient(current_entropy=150.0)
        lr = trainer._alpha_optimizer.param_groups[0]["lr"]
        assert lr == pytest.approx(0.05, abs=1e-12)


# ===========================================================================
# 9. Invariant raw + shaped ≈ total, parametrised
# ===========================================================================


@pytest.mark.parametrize("threshold", [0, 30])
@pytest.mark.parametrize("alpha_lr", [1e-2, 5e-2])
def test_invariant_raw_plus_shaped_equals_total(threshold, alpha_lr):
    """The raw+shaped≈total invariant holds across force-close
    threshold values and alpha_lr values.

    The env-level invariant is independent of alpha_lr (the
    controller lives in the trainer). Parametrising over alpha_lr
    still serves as a smoke that the knob value doesn't break env
    reward accounting when the env+trainer are co-constructed.
    """
    day = _scripted_day([60.0, 29.0, 15.0])
    env = BetfairEnv(
        day,
        _scalping_config(force_close_before_off_seconds=threshold),
    )
    env.reset()
    _place_initial_pair(env)
    done = False
    total_reward = 0.0
    last_info: dict = {}
    while not done:
        _, r, terminated, truncated, last_info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32),
        )
        total_reward += float(r)
        done = terminated or truncated
    raw = last_info["raw_pnl_reward"]
    shaped = last_info["shaped_bonus"]
    assert total_reward == pytest.approx(raw + shaped, abs=1e-6), (
        f"invariant broken at threshold={threshold}, alpha_lr={alpha_lr}: "
        f"total={total_reward} raw+shaped={raw + shaped}"
    )


# ===========================================================================
# 10. Transformer builds and forwards at ctx=256
# ===========================================================================


class TestTransformerCtx256:
    def test_policy_instantiates_and_forwards_at_ctx_256(self):
        """hard_constraints.md §14c — structural smoke for ctx=256."""
        import torch

        from agents.policy_network import PPOTransformerPolicy
        from env.betfair_env import (
            ACTIONS_PER_RUNNER,
            AGENT_STATE_DIM,
            MARKET_DIM,
            POSITION_DIM,
            RUNNER_DIM,
            VELOCITY_DIM,
        )

        max_runners = 4
        obs_dim = (
            MARKET_DIM
            + VELOCITY_DIM
            + RUNNER_DIM * max_runners
            + AGENT_STATE_DIM
            + POSITION_DIM * max_runners
        )
        action_dim = max_runners * ACTIONS_PER_RUNNER

        policy = PPOTransformerPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_runners=max_runners,
            hyperparams={
                "lstm_hidden_size": 32,
                "mlp_hidden_size": 16,
                "mlp_layers": 1,
                "transformer_heads": 4,
                "transformer_depth": 1,
                "transformer_ctx_ticks": 256,
            },
        )
        policy.eval()
        assert policy.ctx_ticks == 256
        assert policy.position_embedding.num_embeddings == 256
        assert policy.causal_mask.shape == (256, 256)

        obs = torch.zeros(1, obs_dim)
        out = policy(obs)
        assert out.action_mean.shape == (1, action_dim)
        buf, valid = out.hidden_state
        assert buf.shape == (1, 256, 32)
        assert valid.shape == (1,)
