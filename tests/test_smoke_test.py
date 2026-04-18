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

import json

import pytest

from agents.smoke_test import (
    ARBS_CLOSED_MIN,
    ENTROPY_RISE_TOLERANCE,
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
    """Entropy assertion allows a small positive drift — the
    ``0a8cacd3`` diffusion pathology climbs +50 over 7 episodes;
    normal early exploration can add +3-7 over 3. Tolerance lives
    in ``ENTROPY_RISE_TOLERANCE`` (= 10.0 at time of writing).
    """

    def test_strictly_decreasing_passes(self):
        rows = (
            _three_eps("a", ep1={"entropy": 150.0}, ep3={"entropy": 140.0})
            + _three_eps("b", ep1={"entropy": 200.0}, ep3={"entropy": 195.0})
        )
        rows[0]["arbs_closed"] = 3  # keep assertion 3 green
        result = evaluate_probe_episodes(rows)
        ent = next(a for a in result.assertions if a.name == "entropy_non_increasing")
        assert ent.passed
        # Worst (most-positive / least-negative) delta is reported.
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

    def test_mild_rise_under_tolerance_passes(self):
        """The 2026-04-18 post-clamp-fix probe showed +3.62
        (transformer) and +7.14 (LSTM). Both under ±10 tolerance —
        gate should pass."""
        rows = (
            _three_eps("a", ep1={"entropy": 139.52}, ep3={"entropy": 143.15})
            + _three_eps("b", ep1={"entropy": 139.59}, ep3={"entropy": 146.72})
        )
        rows[0]["arbs_closed"] = 3  # keep assertion 3 green
        result = evaluate_probe_episodes(rows)
        ent = next(a for a in result.assertions if a.name == "entropy_non_increasing")
        assert ent.passed
        # Worst Δ = agent b = 7.13 (reported regardless of pass/fail).
        assert ent.observed == pytest.approx(7.13, abs=1e-2)
        assert ent.threshold == ENTROPY_RISE_TOLERANCE

    def test_rise_at_tolerance_exactly_passes(self):
        """Boundary: Δ == tolerance passes (``<=`` semantics)."""
        rows = (
            _three_eps("a", ep1={"entropy": 100.0},
                       ep3={"entropy": 100.0 + ENTROPY_RISE_TOLERANCE})
            + _three_eps("b", ep1={"entropy": 100.0}, ep3={"entropy": 99.0})
        )
        rows[0]["arbs_closed"] = 3
        result = evaluate_probe_episodes(rows)
        ent = next(a for a in result.assertions if a.name == "entropy_non_increasing")
        assert ent.passed

    def test_rise_just_past_tolerance_fails(self):
        """Boundary: Δ strictly greater than tolerance fails."""
        rows = (
            _three_eps("a", ep1={"entropy": 100.0},
                       ep3={"entropy": 100.0 + ENTROPY_RISE_TOLERANCE + 0.01})
            + _three_eps("b", ep1={"entropy": 100.0}, ep3={"entropy": 99.0})
        )
        result = evaluate_probe_episodes(rows)
        ent = next(a for a in result.assertions if a.name == "entropy_non_increasing")
        assert not ent.passed

    def test_one_agent_rising_hard_fails(self):
        """Reproduces the full ``0a8cacd3`` entropy pathology
        (+40 rise over 3 episodes). Far above any tolerance — flags
        regardless of how we tune ``ENTROPY_RISE_TOLERANCE``."""
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
        # calm.
        rows += _three_eps(
            "calmlstm",
            ep1={"entropy": 90.0}, ep3={"entropy": 80.0},
        )
        result = evaluate_probe_episodes(rows)
        assert result.passed is False
        by_name = {a.name: a for a in result.assertions}
        # Assertion 1 (ep1_policy_loss) hard-fails on its own —
        # 1.04e17 blows the ceiling.
        assert not by_name["ep1_policy_loss"].passed
        # Assertion 2 after 2026-04-18 tolerance relaxation: this
        # vignette's entropy Δ is only +6 over 3 episodes. That
        # PASSES the new ``ENTROPY_RISE_TOLERANCE = 10`` threshold.
        # The full ``0a8cacd3`` pathology climbed +50 over 7 episodes;
        # over the 3-episode probe window it's only +6, too small to
        # flag on entropy alone. Assertions 1 and 3 still catch it.
        assert by_name["entropy_non_increasing"].passed
        # Assertion 3 — the transformer closed 5 arbs on ep 1, which
        # on its own satisfies the at-least-one rule.
        assert by_name["arbs_closed_any_agent"].passed
        # Conservative "any assertion fails → gate fails" invariant:
        # assertion 1's blow-up is sufficient even when 2 and 3 pass.


# ── Orchestrator smoke (regression guard for signature drift) ────────
#
# ``evaluate_probe_episodes`` is the pure gate logic and has tight
# coverage above. ``run_smoke_test`` is the glue that builds env +
# policy + trainer; when it called ``BetfairEnv(config=..., days=...)``
# (plural kwarg) and ``policy_cls(env=env)`` in the Session 04 landing,
# neither the unit tests above nor the DTO-level integration in
# ``test_api_training.py`` caught it — the bug only surfaced the first
# time an operator clicked Launch after the Session 05 reset.
#
# This class exercises the whole ``run_smoke_test`` path end-to-end
# with a fabricated ``PPOTrainer`` so the training loop itself stays
# out of the hot path (seconds → milliseconds) but every real
# construction call in the orchestrator — ``BetfairEnv(...)``,
# ``create_policy(...)``, ``PPOTrainer(...)``, ``trainer.train(...)``
# — fires with real arguments. A future kwarg drift on any of those
# signatures fails the test before a launch.


class TestRunSmokeTestOrchestrator:
    """Full ``run_smoke_test`` path with a fake PPOTrainer.

    The fake trainer records its constructor kwargs, appends two
    fabricated clearly-passing episode rows to the shared
    ``episodes.jsonl`` file on ``train()``, and otherwise mirrors the
    real trainer's surface that ``run_smoke_test`` touches
    (``log_dir``, ``smoke_test_tag``).
    """

    @pytest.fixture
    def scalping_config(self, tmp_path) -> dict:
        """Minimal scalping config pointing logs into ``tmp_path``."""
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
            "paths": {"logs": str(tmp_path)},
        }

    @pytest.fixture
    def probe_days(self):
        """Three tiny synthetic days — real ``Day`` objects so the
        env constructor can't be faked around."""
        from tests.test_betfair_env import _make_day

        day = _make_day(n_races=1, n_pre_ticks=3, n_inplay_ticks=1)
        return [day, day, day]

    def test_run_smoke_test_builds_real_env_and_policy(
        self, monkeypatch, scalping_config, probe_days, tmp_path,
    ):
        """Regression for the Session 04→05 bug: ``run_smoke_test``
        passed ``days=`` (plural) to ``BetfairEnv`` and ``env=`` to the
        policy class. Either failure raises ``TypeError`` from real
        constructors long before training starts — this test asserts
        the construction path is wired to the real signatures by
        letting both real constructors run."""
        import agents.ppo_trainer as ppo_trainer_mod
        from agents import smoke_test as st

        # Fake trainer: records instantiation, fabricates episodes
        # rows on train(), mirrors log_dir so _tail_probe_rows can
        # read them back.
        log_training_dir = tmp_path / "training"
        log_training_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_training_dir / "episodes.jsonl"

        init_kwargs: list[dict] = []
        train_kwargs: list[dict] = []

        class FakeTrainer:
            def __init__(self, **kw):
                init_kwargs.append(dict(kw))
                self.policy = kw["policy"]
                self.model_id = kw["model_id"]
                self.architecture_name = kw["architecture_name"]
                self.log_dir = log_training_dir
                self.smoke_test_tag = False

            def train(self, *, days, n_epochs):
                train_kwargs.append({"n_days": len(days), "n_epochs": n_epochs})
                # Fabricate PROBE_EPISODE_COUNT clearly-passing rows
                # for this model_id — enough to exercise
                # _tail_probe_rows + evaluate_probe_episodes on real
                # output rather than stubbed returns.
                with open(log_path, "a") as f:
                    for ep in range(1, PROBE_EPISODE_COUNT + 1):
                        f.write(json.dumps({
                            "model_id": self.model_id,
                            "episode": ep,
                            "policy_loss": 1.0,
                            "entropy": 50.0 - ep,  # monotone decreasing
                            "arbs_closed": 2 if ep == 1 else 0,
                            "smoke_test": self.smoke_test_tag,
                        }) + "\n")

        monkeypatch.setattr(ppo_trainer_mod, "PPOTrainer", FakeTrainer)

        result = st.run_smoke_test(
            config=scalping_config,
            train_days=probe_days,
        )

        # Both probe architectures were instantiated — real create_policy
        # ran without TypeError; real BetfairEnv(day, config) ran without
        # TypeError.
        assert len(init_kwargs) == 2
        assert {kw["architecture_name"] for kw in init_kwargs} == {
            "ppo_transformer_v1", "ppo_lstm_v1",
        }
        # train() fired with the probe day list and n_epochs=1.
        assert train_kwargs == [
            {"n_days": PROBE_EPISODE_COUNT, "n_epochs": 1},
            {"n_days": PROBE_EPISODE_COUNT, "n_epochs": 1},
        ]
        # Probe rows came back through _tail_probe_rows and were
        # evaluated — not short-circuited to an empty fallback.
        assert isinstance(result, SmokeResult)
        assert result.passed is True
        assert set(result.probe_model_ids) == {
            "smoke-ppo_transformer_v1", "smoke-ppo_lstm_v1",
        }

    def test_run_smoke_test_short_day_list_rejected_before_construction(
        self, scalping_config, probe_days,
    ):
        """Defensive — fewer days than PROBE_EPISODE_COUNT raises
        ValueError before any env / policy / trainer construction."""
        from agents import smoke_test as st

        with pytest.raises(ValueError, match="at least"):
            st.run_smoke_test(
                config=scalping_config,
                train_days=probe_days[:PROBE_EPISODE_COUNT - 1],
            )

    def test_run_smoke_test_unknown_architecture_rejected(
        self, monkeypatch, scalping_config, probe_days,
    ):
        """Defensive — an architecture name not in REGISTRY raises
        ValueError. Asserts the check fires after env construction (so
        a bad architecture name doesn't mask a BetfairEnv TypeError)."""
        from agents import smoke_test as st

        with pytest.raises(ValueError, match="not in registry"):
            st.run_smoke_test(
                config=scalping_config,
                train_days=probe_days,
                probe_architectures=("ppo_transformer_v1", "no_such_arch"),
            )

    # ── Device + progress-queue observability ────────────────────────
    #
    # The user-visible symptom after the Session 05 launch was a warm
    # CPU, idle GPU, and empty activity log. PPOTrainer defaulted to
    # ``device="cpu"`` when run_smoke_test didn't forward a device, and
    # the probe's trainer.train() call didn't emit the phase_start /
    # phase_complete events the /training page's activity log reads.
    # Both are fixed by threading device through from config /
    # torch.cuda.is_available() and emitting explicit phase events
    # around each probe agent + the aggregate pass/fail.

    def _install_fake_trainer(
        self, monkeypatch, tmp_path,
    ) -> tuple[list[dict], list[dict]]:
        """Install a FakeTrainer that records its constructor kwargs
        and train() calls. Returns (init_kwargs_list, train_kwargs_list)
        that tests can assert against."""
        import json as _json
        import agents.ppo_trainer as ppo_trainer_mod

        log_training_dir = tmp_path / "training"
        log_training_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_training_dir / "episodes.jsonl"

        init_kwargs: list[dict] = []
        train_kwargs: list[dict] = []

        class FakeTrainer:
            def __init__(self, **kw):
                init_kwargs.append(dict(kw))
                self.policy = kw["policy"]
                self.model_id = kw["model_id"]
                self.architecture_name = kw["architecture_name"]
                self.log_dir = log_training_dir
                self.smoke_test_tag = False

            def train(self, *, days, n_epochs):
                train_kwargs.append({
                    "n_days": len(days), "n_epochs": n_epochs,
                })
                with open(log_path, "a") as f:
                    for ep in range(1, PROBE_EPISODE_COUNT + 1):
                        f.write(_json.dumps({
                            "model_id": self.model_id,
                            "episode": ep,
                            "policy_loss": 1.0,
                            "entropy": 50.0 - ep,
                            "arbs_closed": 2 if ep == 1 else 0,
                            "smoke_test": self.smoke_test_tag,
                        }) + "\n")

        monkeypatch.setattr(ppo_trainer_mod, "PPOTrainer", FakeTrainer)
        return init_kwargs, train_kwargs

    def test_run_smoke_test_forwards_cuda_device_when_available(
        self, monkeypatch, scalping_config, probe_days, tmp_path,
    ):
        """With CUDA available and no ``config.training.device``
        override, the probe must pass ``device="cuda"`` to
        PPOTrainer. Regression for the Session 05 launch where the
        probe ran on CPU beside an idle RTX 3090."""
        import agents.smoke_test as st

        init_kwargs, _ = self._install_fake_trainer(monkeypatch, tmp_path)

        # Force the auto-detect branch to see a GPU.
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        result = st.run_smoke_test(
            config=scalping_config, train_days=probe_days,
        )
        assert result.passed is True
        # Both probe agents received ``device="cuda"``.
        assert [kw["device"] for kw in init_kwargs] == ["cuda", "cuda"]

    def test_run_smoke_test_respects_explicit_cpu_config(
        self, monkeypatch, scalping_config, probe_days, tmp_path,
    ):
        """``config.training.device="cpu"`` wins over auto-detect —
        same precedence as run_training.py. Keeps the CPU-only
        CI / developer path reproducible."""
        import agents.smoke_test as st

        init_kwargs, _ = self._install_fake_trainer(monkeypatch, tmp_path)

        import torch
        # CUDA visible but config forces CPU.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        scalping_config["training"]["device"] = "cpu"

        st.run_smoke_test(config=scalping_config, train_days=probe_days)
        assert [kw["device"] for kw in init_kwargs] == ["cpu", "cpu"]

    def test_run_smoke_test_falls_back_to_cpu_without_cuda(
        self, monkeypatch, scalping_config, probe_days, tmp_path,
    ):
        """No CUDA, no config override → ``device="cpu"``. Covers
        CI + any developer without a GPU."""
        import agents.smoke_test as st

        init_kwargs, _ = self._install_fake_trainer(monkeypatch, tmp_path)

        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        st.run_smoke_test(config=scalping_config, train_days=probe_days)
        assert [kw["device"] for kw in init_kwargs] == ["cpu", "cpu"]

    def test_run_smoke_test_emits_phase_events_for_activity_log(
        self, monkeypatch, scalping_config, probe_days, tmp_path,
    ):
        """The activity log on /training renders ``phase_start`` /
        ``phase_complete`` events from the worker progress queue. The
        probe must emit them or the operator sees charts updating with
        no text log — confusing."""
        import queue
        import agents.smoke_test as st

        self._install_fake_trainer(monkeypatch, tmp_path)
        pq: queue.Queue = queue.Queue()

        # Suppress CUDA auto-detect — the test doesn't care about
        # device selection, it cares about events.
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        result = st.run_smoke_test(
            config=scalping_config,
            train_days=probe_days,
            progress_queue=pq,
        )
        assert result.passed is True

        events = []
        while not pq.empty():
            events.append(pq.get_nowait())

        # One top-level phase_start + per-agent phase_start and
        # phase_complete + one top-level phase_complete. Exact content
        # matters for the UI so operators see meaningful text.
        phases = [(e["event"], e["phase"]) for e in events]
        assert ("phase_start", "smoke_test") in phases
        assert ("phase_start", "smoke_test_agent") in phases
        assert ("phase_complete", "smoke_test_agent") in phases
        assert ("phase_complete", "smoke_test") in phases

        top_complete = next(
            e for e in events
            if e["event"] == "phase_complete" and e["phase"] == "smoke_test"
        )
        assert "PASSED" in top_complete["detail"]

    def test_run_smoke_test_emits_failing_assertions_in_final_detail(
        self, monkeypatch, scalping_config, probe_days, tmp_path,
    ):
        """On fail, the final phase_complete detail names the failing
        assertions so the activity log is self-explanatory."""
        import json as _json
        import queue
        import agents.ppo_trainer as ppo_trainer_mod
        import agents.smoke_test as st

        log_dir = tmp_path / "training"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "episodes.jsonl"

        class BlowUpTrainer:
            """Writes rows that deliberately fail ep1_policy_loss."""
            def __init__(self, **kw):
                self.model_id = kw["model_id"]
                self.log_dir = log_dir
                self.smoke_test_tag = False

            def train(self, *, days, n_epochs):
                with open(log_path, "a") as f:
                    for ep in range(1, PROBE_EPISODE_COUNT + 1):
                        f.write(_json.dumps({
                            "model_id": self.model_id,
                            "episode": ep,
                            # 1e17 > EP1_POLICY_LOSS_MAX → assertion 1 fails.
                            "policy_loss": 1.0e17 if ep == 1 else 1.0,
                            "entropy": 50.0 - ep,
                            "arbs_closed": 2 if ep == 1 else 0,
                        }) + "\n")

        monkeypatch.setattr(ppo_trainer_mod, "PPOTrainer", BlowUpTrainer)

        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        pq: queue.Queue = queue.Queue()
        result = st.run_smoke_test(
            config=scalping_config,
            train_days=probe_days,
            progress_queue=pq,
        )
        assert result.passed is False

        events = []
        while not pq.empty():
            events.append(pq.get_nowait())

        top_complete = next(
            e for e in events
            if e["event"] == "phase_complete" and e["phase"] == "smoke_test"
        )
        assert "FAILED" in top_complete["detail"]
        assert "ep1_policy_loss" in top_complete["detail"]

    def test_run_smoke_test_survives_full_progress_queue(
        self, monkeypatch, scalping_config, probe_days, tmp_path,
    ):
        """A full progress_queue must NOT crash the probe. The gate
        is the critical path; UI observability is best-effort."""
        import agents.smoke_test as st

        self._install_fake_trainer(monkeypatch, tmp_path)
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        class FullQueue:
            def put_nowait(self, item):
                raise Exception("queue full")  # generic — smoke_test
                                               # must swallow any error

        # No assertion beyond "doesn't raise" — that's the contract.
        result = st.run_smoke_test(
            config=scalping_config,
            train_days=probe_days,
            progress_queue=FullQueue(),
        )
        assert result.passed is True
