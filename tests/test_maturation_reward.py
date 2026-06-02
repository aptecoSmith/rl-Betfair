"""Operator's maturation-only reward shape (ga-recipe-search §R).

Pure-logic guards for ``maturation_only_reward``: the positive raw
channel = naturally-matured locked P&L + agent-closes-at-a-profit;
agent-close-at-loss, force-close, stop-close, naked all contribute 0.
"""

from __future__ import annotations

import pytest

from env.betfair_env import maturation_only_reward


def test_naturally_matured_pays_its_locked_spread():
    out = [{"kind": "matured", "locked": 0.93, "covered_cash": 0.0}]
    assert maturation_only_reward(out) == pytest.approx(0.93)


def test_agent_close_at_profit_pays_the_profit():
    out = [{"kind": "agent_closed", "locked": 0.0, "covered_cash": 1.50}]
    assert maturation_only_reward(out) == pytest.approx(1.50)


def test_agent_close_at_loss_pays_zero():
    out = [{"kind": "agent_closed", "locked": 0.0, "covered_cash": -27.0}]
    assert maturation_only_reward(out) == 0.0


def test_force_stop_naked_pay_zero():
    out = [
        {"kind": "force", "locked": -5.0, "covered_cash": -5.0},
        {"kind": "stop", "locked": -3.0, "covered_cash": -3.0},
        {"kind": "naked", "locked": 0.0, "covered_cash": 0.0},
        {"kind": "naked", "locked": 0.0, "covered_cash": -80.0},
    ]
    assert maturation_only_reward(out) == 0.0


def test_mixed_race_sums_only_the_positive_matured_channel():
    # 2 matured (+0.5, +1.2), 1 agent-close profit (+0.8), 1 agent-close
    # loss (0), 1 force (0), 1 naked loss (0).
    out = [
        {"kind": "matured", "locked": 0.5, "covered_cash": 0.0},
        {"kind": "matured", "locked": 1.2, "covered_cash": 0.0},
        {"kind": "agent_closed", "locked": 0.0, "covered_cash": 0.8},
        {"kind": "agent_closed", "locked": 0.0, "covered_cash": -4.0},
        {"kind": "force", "locked": 0.0, "covered_cash": -30.0},
        {"kind": "naked", "locked": 0.0, "covered_cash": -80.0},
    ]
    assert maturation_only_reward(out) == pytest.approx(0.5 + 1.2 + 0.8)


def test_empty_is_zero():
    assert maturation_only_reward([]) == 0.0


def test_spam_of_non_maturing_opens_earns_zero_raw():
    # The degeneracy guard's premise: 100 opens that all force-close/naked
    # earn ZERO from this channel — so the open_cost toll (shaped) is what
    # must make them net-negative. This reward alone is spam-neutral, NOT
    # spam-penalising.
    out = [{"kind": "force", "locked": 0.0, "covered_cash": -1.0}] * 100
    assert maturation_only_reward(out) == 0.0
