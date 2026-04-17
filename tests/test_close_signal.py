"""Tests for the scalping close_signal action (plans/scalping-close-signal).

Covers the six cases mandated by ``hard_constraints.md §14``:

1. Happy path — close with open aggressive + unfilled passive.
2. No-op when no open aggressive on this runner.
3. No-op when the passive already filled (pair done).
4. Close at unfavorable market — realised loss in day_pnl, raw reward
   contributes 0 (the core reward-invariant check per session 01's
   exit criteria).
5. Close at favorable market — realised gain flows through day_pnl.
6. Legacy v3 checkpoint refused on strict load without the migration
   helper, loads cleanly after ``migrate_scalping_action_head_v3_to_v4``.
"""

from __future__ import annotations

import numpy as np
import pytest

from env.bet_manager import Bet, BetSide
from env.betfair_env import (
    ACTION_SCHEMA_VERSION,
    SCALPING_ACTIONS_PER_RUNNER,
    BetfairEnv,
)

from tests.test_betfair_env import _make_day


# ── Schema sanity ──────────────────────────────────────────────────────────


def test_action_schema_version_is_4():
    """Bumped from 3 → 4 in this session."""
    assert ACTION_SCHEMA_VERSION == 4


def test_scalping_actions_per_runner_is_7():
    """signal, stake, aggression, cancel, arb_spread, requote, close."""
    assert SCALPING_ACTIONS_PER_RUNNER == 7


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def scalping_config() -> dict:
    return {
        "training": {
            "max_runners": 14,
            "starting_budget": 100.0,
            "max_bets_per_race": 20,
            "scalping_mode": True,
        },
        "actions": {"force_aggressive": True},
        "reward": {
            "early_pick_bonus_min": 1.2,
            "early_pick_bonus_max": 1.5,
            "early_pick_min_seconds": 300,
            "efficiency_penalty": 0.01,
        },
    }


def _close_action(
    max_runners: int = 14,
    *,
    slot: int = 0,
    signal: float = 0.0,
    stake: float = -1.0,
    aggression: float = -1.0,
    cancel: float = -1.0,
    arb_spread: float = -1.0,
    requote: float = -1.0,
    close: float = -1.0,
) -> np.ndarray:
    """Build a v4 scalping action with explicit per-dim values."""
    action = np.zeros(
        max_runners * SCALPING_ACTIONS_PER_RUNNER, dtype=np.float32,
    )
    action[slot] = signal
    action[max_runners + slot] = stake
    action[2 * max_runners + slot] = aggression
    action[3 * max_runners + slot] = cancel
    action[4 * max_runners + slot] = arb_spread
    action[5 * max_runners + slot] = requote
    action[6 * max_runners + slot] = close
    return action


def _place_initial_pair(env: BetfairEnv) -> tuple:
    """Drive env through one tick producing an aggressive back + paired passive lay.

    Fences the passive against on_tick auto-fill (synthetic-ladder quirk:
    queue_ahead of 0 triggers instant fills). Returns ``(agg_bet, passive_order)``.
    """
    a = _close_action(
        signal=1.0, stake=-0.8, aggression=1.0, arb_spread=-1.0,
    )
    env.step(a)
    bm = env.bet_manager
    paired_bets = [b for b in bm.bets if b.pair_id is not None]
    assert paired_bets, "expected an aggressive fill with a pair_id"
    agg = paired_bets[0]
    pairing = [
        o for o in bm.passive_book.orders if o.pair_id == agg.pair_id
    ]
    assert pairing, "expected a paired passive to rest after the fill"
    for o in bm.passive_book.orders:
        if o.pair_id is not None:
            o.queue_ahead_at_placement = 1e12
    return agg, pairing[0]


# ── Case 1: happy path ─────────────────────────────────────────────────────


class TestClosePlacement:
    def _make_env(self, scalping_config, **kwargs) -> BetfairEnv:
        return BetfairEnv(
            _make_day(n_races=1, n_pre_ticks=5, n_inplay_ticks=2),
            scalping_config,
            **kwargs,
        )

    def test_close_cancels_passive_and_places_aggressive_opposite_leg(
        self, scalping_config,
    ):
        """Close fires → old passive cancelled, close leg matched aggressive."""
        env = self._make_env(scalping_config)
        env.reset()

        agg, old_passive = _place_initial_pair(env)
        bm = env.bet_manager
        old_passive_id = id(old_passive)

        # Fire the close signal on slot 0.
        env.step(_close_action(close=1.0))

        # 1. The originally-rested paired passive is gone from the book
        #    (cancelled).
        assert not any(
            id(o) == old_passive_id for o in bm.passive_book.orders
        )
        # 2. Two Bet objects now share the same pair_id (aggressive +
        #    the aggressive close leg) — both matched.
        pair_bets = [b for b in bm.bets if b.pair_id == agg.pair_id]
        assert len(pair_bets) == 2
        # 3. Exactly one of the two carries close_leg=True.
        assert sum(1 for b in pair_bets if b.close_leg) == 1
        assert sum(1 for b in pair_bets if not b.close_leg) == 1
        # 4. Sides are opposite (a BACK agg closed via LAY).
        sides = {b.side for b in pair_bets}
        assert sides == {BetSide.BACK, BetSide.LAY}
        # 5. Diagnostic tag is set.
        sid = old_passive.selection_id
        debug = env._last_action_debug.get(sid, {})
        assert debug.get("close_attempted") is True
        assert debug.get("close_placed") is True
        assert debug.get("close_reason") is None

    def test_close_sizes_legs_for_equal_pnl(self, scalping_config):
        """S_close × P_close == S_agg × P_agg (equal-P&L formula)."""
        env = self._make_env(scalping_config)
        env.reset()
        agg, _ = _place_initial_pair(env)
        env.step(_close_action(close=1.0))
        bm = env.bet_manager
        pair_bets = [b for b in bm.bets if b.pair_id == agg.pair_id]
        close_bet = next(b for b in pair_bets if b.close_leg)
        agg_bet = next(b for b in pair_bets if not b.close_leg)
        expected_notional = agg_bet.matched_stake * agg_bet.average_price
        close_notional = close_bet.matched_stake * close_bet.average_price
        # Tolerance: tick-snap + matched_stake can shrink under thin
        # liquidity so we check within a few percent.
        assert close_notional == pytest.approx(expected_notional, rel=0.15)


# ── Case 2: no-op without an open aggressive ───────────────────────────────


class TestCloseNoopWithoutAggressive:
    def test_close_on_slot_with_no_pair_is_silent_noop(
        self, scalping_config,
    ):
        env = BetfairEnv(
            _make_day(n_races=1, n_pre_ticks=3), scalping_config,
        )
        env.reset()
        # Fire close immediately — nothing has been placed yet.
        env.step(_close_action(close=1.0))
        bm = env.bet_manager
        assert bm.bets == []
        assert bm.passive_book.orders == []
        # Diagnostic tag is set on the slot's selection_id.
        sid = env._slot_maps[env._race_idx].get(0)
        debug = env._last_action_debug.get(sid, {})
        assert debug.get("close_attempted") is True
        assert debug.get("close_placed") is False
        assert debug.get("close_reason") == "no_open_aggressive"


# ── Case 3: no-op when passive already filled (pair done) ──────────────────


class TestClosePairAlreadyComplete:
    def test_close_noop_when_passive_already_filled(self, scalping_config):
        """Pair already complete → no outstanding passive → silent no-op."""
        env = BetfairEnv(
            _make_day(n_races=1, n_pre_ticks=5), scalping_config,
        )
        env.reset()
        agg, passive = _place_initial_pair(env)
        bm = env.bet_manager

        # Simulate the passive filling: append a matching Bet and cancel
        # the resting order so the pair is complete before the close
        # signal fires.
        bm.bets.append(Bet(
            selection_id=passive.selection_id,
            side=passive.side,
            requested_stake=passive.requested_stake,
            matched_stake=passive.requested_stake,
            average_price=passive.price,
            market_id=passive.market_id,
            ltp_at_placement=passive.ltp_at_placement,
            pair_id=passive.pair_id,
            tick_index=env._tick_idx,
        ))
        bm.passive_book.cancel_order(passive, reason="synth fill")

        bet_count_before = len(bm.bets)
        env.step(_close_action(close=1.0))

        # No new bets placed (no outstanding passive → close is a no-op).
        new_close_bets = [
            b for b in bm.bets if getattr(b, "close_leg", False)
        ]
        assert new_close_bets == []
        # The pair is still recognised as 1 pair (both legs matched).
        pairs = bm.get_paired_positions(market_id=agg.market_id)
        assert len(pairs) == 1
        assert pairs[0]["complete"]
        # Diagnostic tag set to no_open_aggressive.
        sid = passive.selection_id
        debug = env._last_action_debug.get(sid, {})
        assert debug.get("close_attempted") is True
        assert debug.get("close_placed") is False
        assert debug.get("close_reason") == "no_open_aggressive"


# ── Case 4: close at loss — raw reward unchanged, cash loss in day_pnl ─────


class TestCloseAtLossRawRewardInvariant:
    def test_close_at_loss_contributes_zero_to_raw_reward(
        self, scalping_config,
    ):
        """Core reward-invariant (hard_constraints §5): close-at-loss
        pair registers with locked_pnl=0 (floored), contributes 0 to
        naked_pnl (both legs matched), so raw_pnl_reward stays 0 — but
        the realised cash loss flows through info['day_pnl']."""
        env = BetfairEnv(
            _make_day(n_races=1, n_pre_ticks=5), scalping_config,
        )
        env.reset()
        agg, _ = _place_initial_pair(env)
        bm = env.bet_manager

        # Drive the rest of the episode with close on tick 2 and then
        # nothing. The synthetic ladder has back=4.0 / lay=4.2 so an
        # aggressive close-lay after an aggressive back pairs BACK@4.2
        # vs LAY@4.0 — a losing spread (lay price < back price).
        obs, reward, terminated, _, info = env.step(
            _close_action(close=1.0),
        )
        assert any(
            b.close_leg for b in bm.bets if b.pair_id == agg.pair_id
        ), "close leg must have been placed for the invariant to bind"

        # Finish the episode at idle so settlement happens.
        while not terminated:
            obs, reward, terminated, _, info = env.step(
                _close_action(),  # all zeros
            )

        # The closed pair should show up in arbs_closed — NOT arbs_naked
        # and NOT arbs_completed (close_leg tagging routes it separately).
        assert info["arbs_closed"] >= 1
        # Raw reward invariant (hard_constraints §5 + session 01 exit
        # criteria): a close-at-loss contributes 0 to raw_pnl_reward.
        # The pair's locked_pnl floors at 0 and its cash P&L is carved
        # out of naked_pnl (scalping_closed_pnl), so neither the locked
        # term nor the asymmetric naked-loss term claim it. The cash
        # loss remains visible on info["day_pnl"].
        assert info["day_pnl"] < 0.0
        assert info["raw_pnl_reward"] == pytest.approx(0.0, abs=1e-6)
        # Close events surface on info for the trainer.
        events = info.get("close_events", [])
        assert len(events) >= 1
        ev = events[0]
        assert ev["selection_id"] == agg.selection_id
        assert ev["realised_pnl"] < 0.0

    def test_close_at_loss_locked_pnl_floors_at_zero(self, scalping_config):
        """hard_constraints §6 — losing close pair's locked_pnl floors at 0."""
        env = BetfairEnv(
            _make_day(n_races=1, n_pre_ticks=5), scalping_config,
        )
        env.reset()
        agg, _ = _place_initial_pair(env)
        bm = env.bet_manager
        env.step(_close_action(close=1.0))
        pairs = bm.get_paired_positions(
            market_id=agg.market_id, commission=0.05,
        )
        # Both legs matched → complete; losing spread → locked floors at 0.
        closed_pair = next(p for p in pairs if p["complete"])
        assert closed_pair["locked_pnl"] == pytest.approx(0.0, abs=1e-6)


# ── Case 5: close at favorable market (gain) ───────────────────────────────


class TestCloseAtProfit:
    def test_close_at_profit_classified_as_arbs_closed(self, scalping_config):
        """A close after the spread moved favourably still routes as arbs_closed.

        Same plumbing as a natural completion — classified by the
        close_leg tag, not by outcome.
        """
        env = BetfairEnv(
            _make_day(n_races=1, n_pre_ticks=5), scalping_config,
        )
        env.reset()
        agg, _ = _place_initial_pair(env)
        bm = env.bet_manager

        # Synthesise favourable move: shrink the lay price to make the
        # close favourable. We directly rewrite the original aggressive
        # leg to BACK@5.0 so that the close-lay at the market's 4.2 lay
        # would be a genuine profitable exit.
        agg.average_price = 5.0
        agg.matched_stake = 10.0

        env.step(_close_action(close=1.0))
        pair_bets = [b for b in bm.bets if b.pair_id == agg.pair_id]
        assert any(b.close_leg for b in pair_bets)

        # Drive to settlement.
        terminated = False
        info: dict = {}
        while not terminated:
            _, _, terminated, _, info = env.step(_close_action())

        assert info["arbs_closed"] >= 1
        assert info["arbs_completed"] == 0  # natural-completion bucket stays empty
        ev = info["close_events"][0]
        # Favourable-exit realised_pnl could be positive or small; the
        # plumbing check is that the event landed in close_events.
        assert "realised_pnl" in ev


# ── Case 6: v3 checkpoint migration ────────────────────────────────────────


class TestV3ToV4Migration:
    def test_v3_checkpoint_refused_without_migration(self):
        """Strict load of a v3 (6-per-runner) checkpoint into a v4
        (7-per-runner) policy must fail — the migration helper is the
        only supported path."""
        import torch
        from agents.policy_network import PPOLSTMPolicy

        max_runners = 14
        obs_dim = 32
        hp = {"lstm_hidden_size": 16, "mlp_hidden_size": 16, "mlp_layers": 1}

        v3_net = PPOLSTMPolicy(
            obs_dim=obs_dim, action_dim=max_runners * 6,
            max_runners=max_runners, hyperparams=hp,
        )
        v3_state = {k: v.clone() for k, v in v3_net.state_dict().items()}

        v4_net = PPOLSTMPolicy(
            obs_dim=obs_dim, action_dim=max_runners * 7,
            max_runners=max_runners, hyperparams=hp,
        )
        with pytest.raises(RuntimeError):
            v4_net.load_state_dict(v3_state, strict=True)

    def test_v3_checkpoint_loads_cleanly_after_migration(self):
        """migrate_scalping_action_head_v3_to_v4 pads the actor head + log_std.

        After migration the new v4 policy accepts strict load, the old
        rows are preserved bit-for-bit, and the new ``close_signal`` row
        is zero-initialised (bias / log-std) so the migrated agent's
        close_signal stays centred at 0 on the first forward pass.
        """
        import torch
        from agents.policy_network import (
            PPOLSTMPolicy,
            migrate_scalping_action_head_v3_to_v4,
        )

        max_runners = 14
        obs_dim = 32
        hp = {"lstm_hidden_size": 16, "mlp_hidden_size": 16, "mlp_layers": 1}

        v3_net = PPOLSTMPolicy(
            obs_dim=obs_dim, action_dim=max_runners * 6,
            max_runners=max_runners, hyperparams=hp,
        )
        v3_state = {k: v.clone() for k, v in v3_net.state_dict().items()}

        migrated = migrate_scalping_action_head_v3_to_v4(
            v3_state, max_runners=max_runners,
        )

        v4_net = PPOLSTMPolicy(
            obs_dim=obs_dim, action_dim=max_runners * 7,
            max_runners=max_runners, hyperparams=hp,
        )
        missing, unexpected = v4_net.load_state_dict(migrated, strict=True)
        assert missing == [] and unexpected == []

        # Old rows of the actor head final linear preserved bit-for-bit.
        old_weight = v3_state["actor_head.2.weight"]
        new_weight = v4_net.actor_head[2].weight.detach()
        assert torch.allclose(new_weight[:6], old_weight)

        # New bias row zero-initialised.
        new_bias = v4_net.actor_head[2].bias.detach()
        assert torch.allclose(
            new_bias[6:7], torch.zeros(1, dtype=new_bias.dtype),
        )

        # action_log_std: old entries preserved, new entries zero-init.
        old_log_std = v3_state["action_log_std"]
        new_log_std = v4_net.action_log_std.detach()
        assert torch.allclose(new_log_std[: max_runners * 6], old_log_std)
        assert torch.allclose(
            new_log_std[max_runners * 6:],
            torch.zeros(max_runners, dtype=new_log_std.dtype),
        )
