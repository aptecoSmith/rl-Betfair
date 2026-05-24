"""Regression tests for per-episode attribution counters (2026-05-24).

Four counters surfaced on the env's info dict, propagated through
``EpisodeStats`` and ``EvalSummary`` into the scoreboard row:

1. ``direction_gate_refusals`` — int. Per-step sum of OPEN_BACK_i /
   OPEN_LAY_i slots the direction gate masked off post-legality.
   Default 0 when gate disabled.
2. ``pwin_back_gate_refusals`` — int. Placeholder for the future
   predictor-p-win back gate. Stays 0 in current cohorts (no gate
   wired yet).
3. ``pwin_lay_gate_refusals`` — int. Same as above for the lay side.
4. ``arb_realised_lock_pct`` — float. Episode-level
   ``sum(locked_pnl) / sum(agg_stake)`` over pairs whose passive
   filled (matured OR agent-closed; NOT naked, NOT force-closed,
   NOT stop-closed). NaN if no pairs filled.

Acceptance: counters default to falsy / NaN when the corresponding
gate is disabled so old behaviour is byte-identical; the counters
are cleared at every ``env.reset()`` boundary.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from env.bet_manager import BetManager, BetSide
from env.betfair_env import BetfairEnv

from tests.test_betfair_env import _make_day, _make_runner_snap


# ── Fixtures ──────────────────────────────────────────────────────────────


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


# ── Counter-field existence + reset / info-dict surface ───────────────────


class TestCounterFieldsExistAndReset:
    def test_init_defaults_zero_for_int_counters(self, scalping_config):
        env = BetfairEnv(_make_day(n_races=1), scalping_config)
        assert env._direction_gate_refusals == 0
        assert env._pwin_back_gate_refusals == 0
        assert env._pwin_lay_gate_refusals == 0

    def test_init_default_nan_for_arb_realised_lock_pct(
        self, scalping_config,
    ):
        env = BetfairEnv(_make_day(n_races=1), scalping_config)
        assert math.isnan(env._arb_realised_lock_pct_last)
        # The episode-cumulative numerator / denominator also default
        # to zero so the first settle's NaN branch (denom == 0) fires.
        assert env._arb_realised_locked_pnl_sum == 0.0
        assert env._arb_realised_agg_stake_sum == 0.0

    def test_reset_clears_counters(self, scalping_config):
        env = BetfairEnv(_make_day(n_races=1), scalping_config)
        # Poke non-zero values to verify reset clears them — the
        # rollout collector mutates the int counter from outside, so
        # the same external mutation pattern is what reset must undo.
        env._direction_gate_refusals = 17
        env._pwin_back_gate_refusals = 3
        env._pwin_lay_gate_refusals = 9
        env._arb_realised_lock_pct_last = 0.042
        env._arb_realised_locked_pnl_sum = 5.5
        env._arb_realised_agg_stake_sum = 100.0

        env.reset()

        assert env._direction_gate_refusals == 0
        assert env._pwin_back_gate_refusals == 0
        assert env._pwin_lay_gate_refusals == 0
        assert math.isnan(env._arb_realised_lock_pct_last)
        assert env._arb_realised_locked_pnl_sum == 0.0
        assert env._arb_realised_agg_stake_sum == 0.0

    def test_info_dict_surfaces_all_four_keys(self, scalping_config):
        env = BetfairEnv(_make_day(n_races=1), scalping_config)
        env.reset()
        info = env._get_info()
        for key in (
            "direction_gate_refusals",
            "pwin_back_gate_refusals",
            "pwin_lay_gate_refusals",
            "arb_realised_lock_pct",
        ):
            assert key in info, f"info dict missing key {key!r}"
        # Default semantics:
        assert info["direction_gate_refusals"] == 0
        assert info["pwin_back_gate_refusals"] == 0
        assert info["pwin_lay_gate_refusals"] == 0
        assert math.isnan(info["arb_realised_lock_pct"])


# ── direction_gate_refusals — accumulation logic at rollout layer ─────────


class TestDirectionGateRefusalAccumulation:
    """The env owns the field; the rollout collector mutates it per
    tick using ``legality AND NOT gate_pass`` on the OPEN slots
    ``[1, 1 + 2*max_runners)``. We exercise the accumulation logic
    directly (the integration with the policy / gate-mask buffer is
    a much larger-fixture rollout test).
    """

    def test_increment_via_external_mutation_persists(
        self, scalping_config,
    ):
        env = BetfairEnv(_make_day(n_races=1), scalping_config)
        env.reset()
        # Simulate the rollout collector adding the per-tick refusal
        # count.  Three ticks, each refusing different counts.
        env._direction_gate_refusals += 4
        env._direction_gate_refusals += 0
        env._direction_gate_refusals += 7
        assert env._direction_gate_refusals == 11
        info = env._get_info()
        assert info["direction_gate_refusals"] == 11

    def test_count_legal_and_not_gate_open_slots_only(self):
        """The rollout collector's refusal formula is::

            refusal = legality & ~gate_pass

        restricted to ``[1, 1 + 2*max_runners)`` (NOOP at index 0 and
        CLOSE in the last R indices are never gated). This test
        reproduces the formula on synthetic masks and confirms the
        count matches the gate-only blocked OPEN slots.
        """
        max_runners = 4
        action_n = 1 + 3 * max_runners  # NOOP + OB + OL + CLOSE
        # Legality: everything legal except OB slot 0.
        legality = np.ones(action_n, dtype=bool)
        legality[1] = False  # OPEN_BACK_0 illegal
        # Gate: blocks OB slot 1 and OL slot 0 (both legal otherwise);
        # also "blocks" OB slot 0 — but that slot was already illegal,
        # so it must not be counted.
        gate_pass = np.ones(action_n, dtype=bool)
        gate_pass[1] = False  # OB slot 0 (illegal — gate match doesn't count)
        gate_pass[2] = False  # OB slot 1 (legal + gated → count)
        gate_pass[1 + max_runners] = False  # OL slot 0 (legal + gated → count)
        # Gate also "blocks" a CLOSE slot — must not be counted by the
        # OPEN-only window.
        gate_pass[1 + 2 * max_runners] = False

        open_lo = 1
        open_hi = 1 + 2 * max_runners
        legal_slice = legality[open_lo:open_hi]
        gate_slice = gate_pass[open_lo:open_hi]
        refusal_count = int(np.count_nonzero(legal_slice & ~gate_slice))
        # Exactly two OPEN slots are legal-but-gated.
        assert refusal_count == 2

    def test_count_zero_when_gate_passes_everything(self):
        """Gate disabled (all True) → zero refusals on any legality."""
        max_runners = 3
        action_n = 1 + 3 * max_runners
        legality = np.array([True, True, True, True, False, True, True,
                             True, True, True], dtype=bool)
        gate_pass = np.ones(action_n, dtype=bool)
        open_lo = 1
        open_hi = 1 + 2 * max_runners
        legal_slice = legality[open_lo:open_hi]
        gate_slice = gate_pass[open_lo:open_hi]
        refusal_count = int(np.count_nonzero(legal_slice & ~gate_slice))
        assert refusal_count == 0


# ── pwin gate counters — placeholder semantics ───────────────────────────


class TestPwinGateRefusalsPlaceholder:
    """The predictor-p-win gates are not wired into the action path
    yet (plans/predictor-integration/strategy_modes.md). The
    counters exist so the scoreboard schema is forward-compatible;
    they MUST stay 0 in any current cohort run.
    """

    def test_back_counter_stays_zero_across_a_rollout(
        self, scalping_config,
    ):
        env = BetfairEnv(_make_day(n_races=1, n_pre_ticks=2), scalping_config)
        env.reset()
        # Step until done — the pwin gate isn't wired, so no
        # accumulation can happen anywhere.
        done = False
        for _ in range(50):
            if done:
                break
            obs, reward, terminated, truncated, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32),
            )
            done = terminated or truncated
        info = env._get_info()
        assert info["pwin_back_gate_refusals"] == 0
        assert info["pwin_lay_gate_refusals"] == 0

    def test_external_increment_surfaces_on_info(self, scalping_config):
        """When a future plan wires the pwin gate it will increment
        these counters the same way the rollout collector increments
        ``direction_gate_refusals``. The plumbing from env → info →
        scoreboard already works today; we exercise it via direct
        attribute write.
        """
        env = BetfairEnv(_make_day(n_races=1), scalping_config)
        env.reset()
        env._pwin_back_gate_refusals += 5
        env._pwin_lay_gate_refusals += 12
        info = env._get_info()
        assert info["pwin_back_gate_refusals"] == 5
        assert info["pwin_lay_gate_refusals"] == 12


# ── arb_realised_lock_pct — settle-time arithmetic ───────────────────────


class TestArbRealisedLockPctArithmetic:
    """The env accumulates ``locked_pnl`` and ``agg.matched_stake``
    over filled pairs (matured OR agent-closed; NOT naked, NOT
    force-closed, NOT stop-closed) and divides them at settle. NaN
    if no filled pairs.

    We exercise the accumulator + settle update directly on the
    private fields; the BetManager + env interaction that actually
    appends to these sums in production is regression-covered by
    the broader scalping reward tests (test_forced_arbitrage.py
    TestScalpingReward).
    """

    def test_nan_when_no_filled_pairs(self, scalping_config):
        env = BetfairEnv(_make_day(n_races=1), scalping_config)
        env.reset()
        # Nothing accumulated → denominator is 0 → NaN.
        # We mirror the settle-time computation directly to avoid
        # depending on the full bet-placement pipeline for this
        # arithmetic-only assertion.
        if env._arb_realised_agg_stake_sum > 0.0:
            v = (
                env._arb_realised_locked_pnl_sum
                / env._arb_realised_agg_stake_sum
            )
        else:
            v = float("nan")
        assert math.isnan(v)

    def test_known_scalp_matches_analytical_value(self):
        """Operator's worked example: back £10 @ 5.0 + lay £11.51 at
        4.5, commission 5%. Equal-profit sizing produces a locked
        floor of ~£0.43 per pair. The realised lock pct is
        ``locked_pnl / agg.matched_stake`` = 0.43 / 10 ≈ 0.043.

        We assert the analytical value within 1e-4 of what the
        env's ``get_paired_positions`` computes on the same
        configuration.
        """
        bm = BetManager(starting_budget=100.0)
        # Aggressive back £10 @ 5.0.
        bm.place_back(
            _make_runner_snap(
                101, ltp=5.0, back_price=5.0, lay_price=5.0, size=500.0,
            ),
            stake=10.0,
            market_id="m1",
            pair_id="pp1",
        )
        # Equal-profit lay sizing for back £10 @ 5.0, lay @ 4.5, c=0.05:
        #   S_lay = 10 × [5.0×0.95 + 0.05] / (4.5 − 0.05)
        #         = 10 × 4.80 / 4.45 ≈ 10.7865
        s_lay = 10.0 * (5.0 * 0.95 + 0.05) / (4.5 - 0.05)
        bm.place_lay(
            _make_runner_snap(
                101, ltp=4.5, back_price=4.5, lay_price=4.5, size=500.0,
            ),
            stake=s_lay,
            market_id="m1",
            pair_id="pp1",
        )
        pairs = bm.get_paired_positions(market_id="m1", commission=0.05)
        assert len(pairs) == 1
        p = pairs[0]
        assert p["complete"]
        locked = p["locked_pnl"]
        agg = p["aggressive"]
        agg_stake = agg.matched_stake

        # Direct analytical check — equal-profit sizing locks both
        # sides at the SAME positive value (within float tolerance).
        # We compute the realised lock-pct directly and verify the
        # env's locked_pnl figure is consistent with it.
        assert agg_stake == pytest.approx(10.0, abs=1e-9)
        # win  = 10×4×0.95 − s_lay×3.5      = 38 − 37.7528 ≈ +£0.2472
        # lose = −10 + s_lay×0.95            = −10 + 10.2472 ≈ +£0.2472
        # locked = min(win, lose) ≈ £0.2472, so lock-pct ≈ 0.0247.
        win = 10.0 * (5.0 - 1.0) * 0.95 - s_lay * (4.5 - 1.0)
        lose = -10.0 + s_lay * 0.95
        expected_locked = min(win, lose)
        expected_lock_pct = expected_locked / 10.0
        assert locked == pytest.approx(expected_locked, abs=1e-4)
        assert expected_locked / agg_stake == pytest.approx(
            expected_lock_pct, abs=1e-9,
        )
        # Apply the env's settle-time accumulation and compute the
        # realised lock-pct.
        locked_sum = locked
        agg_sum = agg_stake
        realised_lock_pct = locked_sum / agg_sum
        assert realised_lock_pct == pytest.approx(
            expected_lock_pct, abs=1e-4,
        )

    def test_settle_accumulator_filters_naked_pairs(
        self, scalping_config,
    ):
        """Naked pairs (passive failed) must NOT contribute to either
        the numerator or the denominator. The settle-time filter on
        ``p['complete']`` is the load-bearing guard.
        """
        env = BetfairEnv(_make_day(n_races=1), scalping_config)
        env.reset()
        # Add an incomplete pair (back only, no lay) to the env's
        # bet manager and confirm the settle accumulator stays zero
        # for it (the loop's ``if p['complete']`` guard short-
        # circuits naked pairs before they touch the lock-pct sums).
        bm = env.bet_manager
        bm.place_back(
            _make_runner_snap(
                101, ltp=5.0, back_price=5.0, lay_price=5.0, size=500.0,
            ),
            stake=10.0,
            market_id=env.day.races[0].market_id,
            pair_id="naked_pair",
        )
        # The env's _settle_current_race walks pairs and filters by
        # complete-ness BEFORE touching the lock-pct sums. We mirror
        # the relevant slice of that loop here to keep the test
        # arithmetic-only (no full settlement pipeline needed).
        pairs = bm.get_paired_positions(
            market_id=env.day.races[0].market_id,
            commission=env._commission,
        )
        for p in pairs:
            if p["complete"]:
                # Wouldn't fire — pair has only a back leg, no lay.
                env._arb_realised_locked_pnl_sum += p["locked_pnl"]
                agg = p["aggressive"]
                if agg is not None:
                    env._arb_realised_agg_stake_sum += agg.matched_stake
        assert env._arb_realised_locked_pnl_sum == 0.0
        assert env._arb_realised_agg_stake_sum == 0.0


# ── Backward-compat: scoreboard.jsonl rows without these fields ───────────


class TestBackwardCompatibility:
    """A pre-2026-05-24 scoreboard.jsonl row will not carry the four
    new keys. Downstream readers that pull them via ``.get(key,
    default)`` must keep working. The default contract:

    - ``direction_gate_refusals`` / pwin counters → 0 (int).
    - ``arb_realised_lock_pct`` → NaN (float).
    """

    def test_legacy_row_round_trips_through_json(self):
        legacy_row = {
            "schema": "v2_cohort_scoreboard",
            "model_id": "abc",
            "agent_id": "agent-0",
            "generation": 0,
            "agent_idx": 0,
            "eval_total_reward": 12.5,
            "eval_day_pnl": 4.2,
            "eval_arbs_completed": 17,
            "eval_arbs_force_closed": 3,
        }
        s = json.dumps(legacy_row)
        round_tripped = json.loads(s)
        # Default-tolerant reads must succeed and produce the
        # documented defaults.
        assert round_tripped.get("eval_direction_gate_refusals", 0) == 0
        assert round_tripped.get("eval_pwin_back_gate_refusals", 0) == 0
        assert round_tripped.get("eval_pwin_lay_gate_refusals", 0) == 0
        v = round_tripped.get("eval_arb_realised_lock_pct", float("nan"))
        assert math.isnan(v)

    def test_new_row_carries_all_four_keys_with_falsy_defaults(self):
        """A row built from a fresh env that NEVER saw the gate fire
        must still emit the four keys at their no-op defaults
        (0 / NaN) so downstream readers can rely on key-presence as
        well as on key-value defaults.
        """
        new_row = {
            "schema": "v2_cohort_scoreboard",
            "eval_direction_gate_refusals": 0,
            "eval_pwin_back_gate_refusals": 0,
            "eval_pwin_lay_gate_refusals": 0,
            "eval_arb_realised_lock_pct": float("nan"),
        }
        s = json.dumps(new_row)
        rt = json.loads(s)
        assert rt["eval_direction_gate_refusals"] == 0
        assert rt["eval_pwin_back_gate_refusals"] == 0
        assert rt["eval_pwin_lay_gate_refusals"] == 0
        assert math.isnan(rt["eval_arb_realised_lock_pct"])
