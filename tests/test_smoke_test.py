"""Unit tests for the Session 04 smoke-test gate.

Covers the pure assertion evaluator ``evaluate_probe_episodes`` against
fabricated episode rows. The full ``run_smoke_test`` orchestrator is
exercised at the worker-integration level; this suite keeps the
gate's logic provable without standing up a trainer.

hard_constraints.md §15 defines the three assertions:

1. ``ep1.policy_loss < 100`` on BOTH probe agents
2. ``ep3.entropy <= ep1.entropy`` on BOTH probe agents
3. ``max(ep1..ep3.arbs_closed) >= 1`` on AT LEAST ONE probe agent

Every test below has exactly one of these assertions as its primary
concern; other fields are stubbed to clearly-passing values so a
regression points at the right assertion.
"""

from __future__ import annotations

from agents.smoke_test import (
    ARBS_CLOSED_MIN,
    EP1_POLICY_LOSS_MAX,
    PROBE_EPISODE_COUNT,
    SmokeAssertionResult,
    SmokeResult,
    evaluate_probe_episodes,
)


def _row(
    model_id: str,
    episode: int,
    *,
    policy_loss: float = 1.0,
    entropy: float = 5.0,
    arbs_closed: int = 0,
) -> dict:
    """Fabricate one episodes.jsonl-shaped row. Defaults all land in
    the clearly-passing zone so each test only needs to override the
    field under test."""
    return {
        "model_id": model_id,
        "episode": episode,
        "policy_loss": policy_loss,
        "entropy": entropy,
        "arbs_closed": arbs_closed,
    }


def _three_eps(model_id: str, **kwargs_by_ep) -> list[dict]:
    """Fabricate three episodes for a probe agent.

    ``kwargs_by_ep`` keys are ``ep1`` / ``ep2`` / ``ep3``; each maps
    to a dict overriding that episode's fields.
    """
    rows = []
    for ep in (1, 2, 3):
        overrides = kwargs_by_ep.get(f"ep{ep}", {}) or {}
        rows.append(_row(model_id, ep, **overrides))
    return rows


# ── Assertion 1 — ep1.policy_loss < 100 ───────────────────────────────


class TestEp1PolicyLossAssertion:

    def test_both_agents_below_threshold_passes(self):
        rows = (
            _three_eps("a", ep1={"policy_loss": 50.0})
            + _three_eps("b", ep1={"policy_loss": 75.0})
        )
        rows[-1]["arbs_closed"] = 2  # keep assertion 3 green
        result = evaluate_probe_episodes(rows)
        pl = next(a for a in result.assertions if a.name == "ep1_policy_loss")
        assert pl.passed
        assert pl.observed == 75.0
        assert pl.threshold == EP1_POLICY_LOSS_MAX

    def test_one_agent_at_threshold_fails(self):
        # Strict <, so 100.0 exactly fails.
        rows = (
            _three_eps("a", ep1={"policy_loss": 50.0})
            + _three_eps("b", ep1={"policy_loss": 100.0})
        )
        result = evaluate_probe_episodes(rows)
        pl = next(a for a in result.assertions if a.name == "ep1_policy_loss")
        assert not pl.passed

    def test_both_agents_above_threshold_fails(self):
        rows = (
            _three_eps("a", ep1={"policy_loss": 1.0e4})
            + _three_eps("b", ep1={"policy_loss": 1.0e17})
        )
        result = evaluate_probe_episodes(rows)
        pl = next(a for a in result.assertions if a.name == "ep1_policy_loss")
        assert not pl.passed
        # Worst is reported — reproducing the transformer 0a8cacd3 blow-up.
        assert pl.observed == 1.0e17


# ── Assertion 2 — ep3.entropy <= ep1.entropy ─────────────────────────


class TestEntropyMonotoneAssertion:

    def test_strictly_decreasing_passes(self):
        rows = (
            _three_eps("a", ep1={"entropy": 150.0}, ep3={"entropy": 140.0})
            + _three_eps("b", ep1={"entropy": 200.0}, ep3={"entropy": 195.0})
        )
        rows[0]["arbs_closed"] = 3  # keep assertion 3 green
        result = evaluate_probe_episodes(rows)
        ent = next(a for a in result.assertions if a.name == "entropy_non_increasing")
        assert ent.passed
        # Least-monotone delta is reported (closest-to-zero negative / positive).
        assert ent.observed == -5.0  # ep3 − ep1 for agent b

    def test_flat_passes(self):
        rows = (
            _three_eps("a", ep1={"entropy": 100.0}, ep3={"entropy": 100.0})
            + _three_eps("b", ep1={"entropy": 100.0}, ep3={"entropy": 100.0})
        )
        rows[0]["arbs_closed"] = 1
        result = evaluate_probe_episodes(rows)
        ent = next(a for a in result.assertions if a.name == "entropy_non_increasing")
        assert ent.passed

    def test_one_agent_rising_fails(self):
        # Reproduces the transformer 0a8cacd3 rising-entropy pathology.
        rows = (
            _three_eps("a", ep1={"entropy": 140.0}, ep3={"entropy": 180.0})
            + _three_eps("b", ep1={"entropy": 100.0}, ep3={"entropy": 95.0})
        )
        result = evaluate_probe_episodes(rows)
        ent = next(a for a in result.assertions if a.name == "entropy_non_increasing")
        assert not ent.passed
        assert ent.observed == 40.0  # worst Δ surfaces


# ── Assertion 3 — max arbs_closed >= 1 on at-least-one agent ─────────


class TestArbsClosedAssertion:

    def test_one_agent_nonzero_passes(self):
        rows = (
            _three_eps("a", ep3={"arbs_closed": 3})
            + _three_eps("b")  # all zero
        )
        result = evaluate_probe_episodes(rows)
        arb = next(a for a in result.assertions if a.name == "arbs_closed_any_agent")
        assert arb.passed
        assert arb.observed == 3.0

    def test_both_agents_zero_fails(self):
        # The gen-2 0a8cacd3 arbs_closed-collapse pathology.
        rows = _three_eps("a") + _three_eps("b")
        result = evaluate_probe_episodes(rows)
        arb = next(a for a in result.assertions if a.name == "arbs_closed_any_agent")
        assert not arb.passed
        assert arb.observed == 0.0
        assert arb.threshold == float(ARBS_CLOSED_MIN)

    def test_at_least_one_pair_any_episode_passes(self):
        # Assertion 3 looks at ``max across eps 1..3`` — a single
        # non-zero episode on one agent is sufficient.
        rows = (
            _three_eps("a", ep2={"arbs_closed": 1})
            + _three_eps("b")
        )
        result = evaluate_probe_episodes(rows)
        arb = next(a for a in result.assertions if a.name == "arbs_closed_any_agent")
        assert arb.passed


# ── Aggregate SmokeResult ────────────────────────────────────────────


class TestSmokeResultAggregate:

    def test_all_three_pass_means_overall_pass(self):
        rows = _three_eps("a", ep1={"policy_loss": 10.0, "arbs_closed": 2})
        rows += _three_eps("b", ep1={"policy_loss": 20.0})
        result = evaluate_probe_episodes(rows)
        assert result.passed is True
        assert all(a.passed for a in result.assertions)
        assert len(result.assertions) == 3

    def test_any_failed_means_overall_fail(self):
        rows = _three_eps("a", ep1={"policy_loss": 10.0, "arbs_closed": 2})
        # Agent b's ep3 entropy rises — assertion 2 fails
        rows += _three_eps("b", ep1={"entropy": 100.0}, ep3={"entropy": 180.0})
        result = evaluate_probe_episodes(rows)
        assert result.passed is False

    def test_empty_rows_fail_every_assertion(self):
        result = evaluate_probe_episodes([])
        assert result.passed is False
        # Three assertion rows even with no data — the UI needs a full
        # table to render, so the evaluator always produces one per
        # assertion.
        assert len(result.assertions) == 3
        assert all(not a.passed for a in result.assertions)

    def test_probe_model_ids_sorted(self):
        rows = _three_eps("zebra") + _three_eps("apple")
        result = evaluate_probe_episodes(rows)
        # Lexical sort — keeps the assertion-failure detail strings
        # deterministic across runs.
        assert result.probe_model_ids == ["apple", "zebra"]

    def test_to_dict_roundtrip(self):
        rows = _three_eps("a") + _three_eps("b")
        result = evaluate_probe_episodes(rows)
        d = result.to_dict()
        assert set(d.keys()) == {"passed", "assertions", "probe_model_ids"}
        assert len(d["assertions"]) == 3
        first = d["assertions"][0]
        assert set(first.keys()) == {
            "name", "passed", "observed", "threshold", "detail",
        }


# ── End-to-end purpose-table style vignettes ────────────────────────


class TestPurposeTableScenarios:
    """Recreate the three-pathology table from
    ``plans/naked-clip-and-stability/purpose.md`` and assert the gate
    catches it. Pre-fix transformer 0a8cacd3 would have failed all
    three assertions on its ep-1..ep-3 window — a belt-and-braces
    regression guard for any future reward-shape / PPO change."""

    def test_gen2_transformer_0a8cacd3_would_fail_gate(self):
        # Fabricate ep 1..3 matching the purpose.md table's first
        # three rows: policy_loss 1.04e17, entropy climbing 139→145,
        # arbs_closed 5→0→0.
        rows = _three_eps(
            "0a8cacd3",
            ep1={"policy_loss": 1.04e17, "entropy": 139.0, "arbs_closed": 5},
            ep2={"policy_loss": 1.72e4, "entropy": 141.0, "arbs_closed": 0},
            ep3={"policy_loss": 0.24, "entropy": 145.0, "arbs_closed": 0},
        )
        # Second probe agent — a parallel-universe LSTM that was
        # calm. Per assertion 2, its monotone-decreasing entropy
        # isn't enough to save the run; the transformer's rising
        # window blocks the gate. Per assertion 1, the transformer's
        # 1e17 blows the ceiling by itself. Per assertion 3, the
        # transformer DID close 5 arbs on ep 1, so assertion 3 would
        # actually pass on this exact vignette — but assertions 1 and
        # 2 still fail, so the gate fails overall.
        rows += _three_eps(
            "calmlstm",
            ep1={"entropy": 90.0}, ep3={"entropy": 80.0},
        )
        result = evaluate_probe_episodes(rows)
        assert result.passed is False
        by_name = {a.name: a for a in result.assertions}
        assert not by_name["ep1_policy_loss"].passed
        assert not by_name["entropy_non_increasing"].passed
        # assertion 3 is interesting — the transformer closed 5
        # arbs on ep 1, which on its own satisfies the at-least-one
        # rule. So the gate fails on assertions 1+2, not 3. This is
        # the conservative "any assertion fails → gate fails"
        # invariant in action.
        assert by_name["arbs_closed_any_agent"].passed
