"""Tests for the v2 cohort worker (Session 03).

The end-to-end worker test runs the full multi-day train + eval
pipeline on a synthetic 1-day dataset (using the in-test
``_make_day`` fixture) with ``hidden_size=64`` to keep instantiation
cheap. Scorer artefacts must be present (see ``_runtime_ok`` in
``test_v2_multi_day_train.py``).

Pure-helper tests cover:
- ``arch_name_for_genes`` — discriminator format.
- ``scalping_train_config`` — fresh-per-call so workers can't leak.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from training_v2.cohort.genes import sample_genes
from training_v2.cohort.worker import (
    arch_name_for_genes,
    scalping_train_config,
)


# ── Pure helpers ─────────────────────────────────────────────────────────


def test_arch_name_encodes_hidden_size():
    rng = random.Random(0)
    g = sample_genes(rng)
    name = arch_name_for_genes(g)
    assert name.startswith("v2_discrete_ppo_lstm_h")
    assert name.endswith(str(g.hidden_size))


def test_arch_name_differs_per_hidden_size():
    from training_v2.cohort.genes import CohortGenes
    base = dict(
        learning_rate=1e-4,
        entropy_coeff=1e-3,
        clip_range=0.2,
        gae_lambda=0.95,
        value_coeff=0.5,
        mini_batch_size=64,
    )
    a = CohortGenes(**base, hidden_size=64)
    b = CohortGenes(**base, hidden_size=128)
    c = CohortGenes(**base, hidden_size=256)
    assert len({arch_name_for_genes(a), arch_name_for_genes(b),
                arch_name_for_genes(c)}) == 3


def test_scalping_train_config_returns_fresh_dict():
    """Mutating one config must not leak into the next caller's view."""
    cfg_a = scalping_train_config()
    cfg_a["training"]["max_runners"] = 999
    cfg_b = scalping_train_config()
    assert cfg_b["training"]["max_runners"] != 999


# ── Phase 5 (restore-genes, 2026-05-03) ──────────────────────────────────


class TestPerAgentOverrideHelpers:
    """``_build_per_agent_reward_overrides`` /
    ``_build_per_agent_scalping_overrides`` translate a Phase 5
    enabled_set + an agent's gene draws into the dicts the env
    consumes."""

    def _genes(self, **overrides):
        from training_v2.cohort.genes import CohortGenes
        base = dict(
            learning_rate=1e-4, entropy_coeff=1e-3, clip_range=0.2,
            gae_lambda=0.95, value_coeff=0.5, mini_batch_size=64,
            hidden_size=128,
        )
        base.update(overrides)
        return CohortGenes(**base)

    def test_legacy_no_enabled_genes_returns_none(self):
        from training_v2.cohort.worker import (
            _build_per_agent_reward_overrides,
            _build_per_agent_scalping_overrides,
        )
        g = self._genes(open_cost=1.5, arb_spread_target_lock_pct=0.03)
        # No enabled_set, no cohort overrides → None for both (legacy
        # byte-identity invariant).
        assert _build_per_agent_reward_overrides(
            cohort_overrides=None, genes=g, enabled_set=frozenset(),
        ) is None
        assert _build_per_agent_scalping_overrides(
            genes=g, enabled_set=frozenset(),
        ) is None

    def test_disabled_gene_does_not_appear_even_with_non_default_value(
        self,
    ):
        """A gene with a non-default field value (e.g. set manually
        on a CohortGenes for a test) must NOT leak into the overrides
        dict unless the gene is in ``enabled_set``."""
        from training_v2.cohort.worker import (
            _build_per_agent_reward_overrides,
        )
        g = self._genes(open_cost=1.5, mark_to_market_weight=0.07)
        out = _build_per_agent_reward_overrides(
            cohort_overrides=None, genes=g, enabled_set=frozenset(),
        )
        assert out is None

    def test_enabled_gene_value_lands_in_reward_overrides(self):
        from training_v2.cohort.worker import (
            _build_per_agent_reward_overrides,
        )
        g = self._genes(open_cost=1.5, mark_to_market_weight=0.07)
        out = _build_per_agent_reward_overrides(
            cohort_overrides=None, genes=g,
            enabled_set=frozenset({"open_cost"}),
        )
        assert out == {"open_cost": 1.5}

    def test_combines_cohort_overrides_with_enabled_gene_value(self):
        from training_v2.cohort.worker import (
            _build_per_agent_reward_overrides,
        )
        g = self._genes(open_cost=1.5)
        out = _build_per_agent_reward_overrides(
            cohort_overrides={"force_close_before_off_seconds": 60},
            genes=g, enabled_set=frozenset({"open_cost"}),
        )
        assert out == {
            "force_close_before_off_seconds": 60,
            "open_cost": 1.5,
        }

    def test_arb_spread_target_lock_pct_routes_to_scalping_overrides(self):
        """``arb_spread_target_lock_pct`` lives in scalping_overrides, not
        reward_overrides — the env reads it from a different dict. (Renamed
        from the retired arb_spread_scale; the routing is unchanged.)"""
        from training_v2.cohort.worker import (
            _build_per_agent_reward_overrides,
            _build_per_agent_scalping_overrides,
        )
        g = self._genes(arb_spread_target_lock_pct=0.03)
        ro = _build_per_agent_reward_overrides(
            cohort_overrides=None, genes=g,
            enabled_set=frozenset({"arb_spread_target_lock_pct"}),
        )
        so = _build_per_agent_scalping_overrides(
            genes=g, enabled_set=frozenset({"arb_spread_target_lock_pct"}),
        )
        # Goes to scalping_overrides ONLY.
        assert ro is None
        assert so == {"arb_spread_target_lock_pct": 0.03}

    def test_enables_all_eight_env_consumed_genes(self):
        """All Phase 5 genes routed through reward_overrides land in
        the dict when enabled."""
        from training_v2.cohort.worker import (
            _PHASE5_GENES_VIA_REWARD_OVERRIDES,
            _build_per_agent_reward_overrides,
        )
        g = self._genes(
            open_cost=1.5,
            matured_arb_bonus_weight=2.0,
            mark_to_market_weight=0.08,
            naked_loss_scale=0.7,
            stop_loss_pnl_threshold=0.1,
            fill_prob_loss_weight=0.2,
            mature_prob_loss_weight=0.1,
            risk_loss_weight=0.15,
            reward_clip=5.0,
        )
        out = _build_per_agent_reward_overrides(
            cohort_overrides=None, genes=g,
            enabled_set=_PHASE5_GENES_VIA_REWARD_OVERRIDES,
        )
        assert out is not None
        assert set(out) == set(_PHASE5_GENES_VIA_REWARD_OVERRIDES)
        # Every value matches the gene field.
        for name in _PHASE5_GENES_VIA_REWARD_OVERRIDES:
            assert out[name] == float(getattr(g, name))


# ── Promoted-to-Phase-5 gate-gene resolution (spray-and-bail, 2026-06-06) ─


class TestGateGeneResolution:
    """``_resolve_gate_genes`` resolves the four selectivity gates (five
    values) from per-agent genes (when enabled) vs cohort-wide flag args,
    with the env-side gates pinned disabled when the predictor is absent."""

    def _genes(self, **overrides):
        from training_v2.cohort.genes import CohortGenes
        base = dict(
            learning_rate=1e-4, entropy_coeff=1e-3, clip_range=0.2,
            gae_lambda=0.95, value_coeff=0.5, mini_batch_size=64,
            hidden_size=128,
        )
        base.update(overrides)
        return CohortGenes(**base)

    _ALL_GATES = frozenset({
        "mature_prob_open_threshold", "race_confidence_threshold",
        "lay_price_max", "predictor_p_win_back_threshold",
        "predictor_p_win_lay_threshold",
    })

    def _resolve(self, *, genes, enabled_set, predictor_active,
                 cohort=(0.0, 0.0, 0.0, 0.0, 1.0)):
        from training_v2.cohort.worker import _resolve_gate_genes
        return _resolve_gate_genes(
            genes=genes, enabled_set=enabled_set,
            predictor_active=predictor_active,
            cohort_mature_prob_open_threshold=cohort[0],
            cohort_race_confidence_threshold=cohort[1],
            cohort_lay_price_max=cohort[2],
            cohort_predictor_p_win_back_threshold=cohort[3],
            cohort_predictor_p_win_lay_threshold=cohort[4],
        )

    def test_disabled_genes_pass_cohort_flag_through(self):
        """No gene enabled ⇒ the cohort-wide flag args pass through verbatim
        (the legacy cohort-flag path is unchanged)."""
        g = self._genes(
            mature_prob_open_threshold=0.3, race_confidence_threshold=0.25,
            lay_price_max=20.0, predictor_p_win_back_threshold=0.3,
            predictor_p_win_lay_threshold=0.2,
        )
        out = self._resolve(
            genes=g, enabled_set=frozenset(), predictor_active=True,
            cohort=(0.15, 0.4, 30.0, 0.25, 0.3),
        )
        assert out == (0.15, 0.4, 30.0, 0.25, 0.3)

    def test_enabled_genes_win_with_predictor(self):
        """Genes in enabled_set + predictor ON ⇒ per-agent gene values win
        over the (disabled-default) cohort flags."""
        g = self._genes(
            mature_prob_open_threshold=0.3, race_confidence_threshold=0.25,
            lay_price_max=20.0, predictor_p_win_back_threshold=0.3,
            predictor_p_win_lay_threshold=0.2,
        )
        out = self._resolve(
            genes=g, enabled_set=self._ALL_GATES, predictor_active=True,
        )
        assert out == (0.3, 0.25, 20.0, 0.3, 0.2)

    def test_predictor_off_pins_env_gates_but_keeps_policy_gate(self):
        """Predictor OFF ⇒ the four ENV-side gates pin to their disabled
        default (the env would raise otherwise); the POLICY-side
        mature_prob_open_threshold still honours its gene."""
        g = self._genes(
            mature_prob_open_threshold=0.3, race_confidence_threshold=0.25,
            lay_price_max=20.0, predictor_p_win_back_threshold=0.3,
            predictor_p_win_lay_threshold=0.2,
        )
        out = self._resolve(
            genes=g, enabled_set=self._ALL_GATES, predictor_active=False,
        )
        # mature kept; race_conf/lay/pwin pinned to disabled (0/0/0/1).
        assert out == (0.3, 0.0, 0.0, 0.0, 1.0)

    def test_per_gate_independence(self):
        """Only the enabled gate is taken from the gene; the rest stay on
        their cohort flag."""
        g = self._genes(mature_prob_open_threshold=0.4)
        out = self._resolve(
            genes=g, enabled_set=frozenset({"mature_prob_open_threshold"}),
            predictor_active=True, cohort=(0.0, 0.1, 15.0, 0.2, 0.3),
        )
        # mature from gene (0.4); others from cohort flags.
        assert out == (0.4, 0.1, 15.0, 0.2, 0.3)


# ── End-to-end worker (real env + scorer) ───────────────────────────────


REPO_ROOT = Path(__file__).resolve().parents[1]
SCORER_DIR = REPO_ROOT / "models" / "scorer_v1"


def _runtime_ok() -> tuple[bool, str]:
    if not (SCORER_DIR / "model.lgb").exists():
        return False, f"scorer artefacts missing under {SCORER_DIR}"
    try:
        import lightgbm  # noqa: F401
        import joblib  # noqa: F401
    except Exception as exc:  # pragma: no cover
        return False, f"scorer runtime missing: {exc!r}"
    return True, ""


_runtime, _runtime_reason = _runtime_ok()


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.skipif(not _runtime, reason=_runtime_reason)
def test_train_one_agent_runs_end_to_end_on_synthetic_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One agent through 1 train day + 1 eval day on a synthetic dataset.

    Validates: the worker produces a non-empty :class:`AgentResult`,
    train summary has 1 day with steps > 0, eval summary populated,
    and the registry round-tripped (model_id, weights, eval_run row).
    """
    from registry.model_store import ModelStore
    from tests.test_betfair_env import _make_day

    from training_v2.cohort.genes import CohortGenes
    from training_v2.cohort import worker as worker_mod

    # Synthetic days — replace load_day so the worker doesn't read from
    # data/processed. Two distinct day objects so the eval rollout sees
    # a fresh env and we exercise the full path.
    days = {
        "2026-04-23": _make_day(
            n_races=1, n_pre_ticks=4, n_inplay_ticks=2,
        ),
        "2026-04-24": _make_day(
            n_races=1, n_pre_ticks=4, n_inplay_ticks=2,
        ),
    }

    def _fake_load_day(date_str, data_dir):
        return days[date_str]

    monkeypatch.setattr(worker_mod, "load_day", _fake_load_day)

    genes = CohortGenes(
        learning_rate=3e-4,
        entropy_coeff=0.01,
        clip_range=0.2,
        gae_lambda=0.95,
        value_coeff=0.5,
        mini_batch_size=32,
        hidden_size=64,  # cheap to instantiate
    )

    store = ModelStore(
        db_path=tmp_path / "models.db",
        weights_dir=tmp_path / "weights",
        bet_logs_dir=tmp_path / "bet_logs",
    )

    result = worker_mod.train_one_agent(
        agent_id="test-agent-0001",
        genes=genes,
        days_to_train=["2026-04-23"],
        eval_day="2026-04-24",
        data_dir=tmp_path,
        device="cpu",
        seed=42,
        model_store=store,
        scorer_dir=SCORER_DIR,
        generation=0,
    )

    # ── Result shape ─────────────────────────────────────────────
    assert result.agent_id == "test-agent-0001"
    assert result.architecture_name == "v2_discrete_ppo_lstm_h64"
    assert result.train.n_days == 1
    assert result.train.total_steps > 0
    assert isinstance(result.eval.total_reward, float)
    assert result.eval.eval_day == "2026-04-24"
    assert result.eval.n_steps > 0

    # ── Registry round-trip ──────────────────────────────────────
    record = store.get_model(result.model_id)
    assert record is not None
    assert record.architecture_name == "v2_discrete_ppo_lstm_h64"
    assert record.generation == 0
    assert record.hyperparameters["hidden_size"] == 64
    # Weights file exists.
    assert Path(result.weights_path).exists()
    # Evaluation run + day row written.
    eval_run = store.get_latest_evaluation_run(result.model_id)
    assert eval_run is not None
    assert eval_run.test_days == ["2026-04-24"]
    days_rows = store.get_evaluation_days(eval_run.run_id)
    assert len(days_rows) == 1
    assert days_rows[0].date == "2026-04-24"


# ── Phase 7 Session 02 — load-bearing reward-overrides integration ────────


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.skipif(not _runtime, reason=_runtime_reason)
@pytest.mark.parametrize("weight_key", [
    "fill_prob_loss_weight",
    "mature_prob_loss_weight",
    "risk_loss_weight",
])
def test_reward_overrides_for_aux_weights_reaches_constructed_trainer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    weight_key: str,
) -> None:
    """``--reward-overrides <weight_key>=0.5`` flows into the trainer.

    Phase 7 Session 02 — the **load-bearing regression guard for the
    bug this whole plan exists to fix**. The 2026-05-04 cohort
    ``v2_phase5_oc1_mpw05_clean5day_1777849498`` was byte-identical
    to its predecessor because the trainer never read the override:
    v2's ``CohortGenes.to_dict`` always populates the three weight
    keys with their default 0.0, so a v1-style ``hp.get(name,
    config_fallback)`` would return 0.0 and silently swallow the
    override. The S02 fix routes overrides through the worker's
    ``_build_trainer_hp`` so the hp dict carries the override value
    BEFORE the trainer is constructed (Path A).

    This integration test exercises the full ``train_one_agent`` flow
    with a synthetic day, captures the hp dict the worker passed to
    ``DiscretePPOTrainer.__init__``, and asserts the override survived
    end-to-end. Spy-style: monkeypatches ``DiscretePPOTrainer`` to
    record the constructor's ``hp`` kwarg, then asserts on the
    captured value AND the constructed trainer's stored attribute.
    """
    from tests.test_betfair_env import _make_day

    from training_v2.cohort.genes import CohortGenes
    from training_v2.cohort import worker as worker_mod
    from training_v2.discrete_ppo.trainer import DiscretePPOTrainer

    days = {
        "2026-04-23": _make_day(
            n_races=1, n_pre_ticks=4, n_inplay_ticks=2,
        ),
        "2026-04-24": _make_day(
            n_races=1, n_pre_ticks=4, n_inplay_ticks=2,
        ),
    }

    def _fake_load_day(date_str, data_dir):
        return days[date_str]

    monkeypatch.setattr(worker_mod, "load_day", _fake_load_day)

    captured_hp: list[dict] = []
    captured_trainer: list[DiscretePPOTrainer] = []
    real_init = DiscretePPOTrainer.__init__

    def _spy_init(self, *args, **kwargs):
        captured_hp.append(dict(kwargs.get("hp") or {}))
        real_init(self, *args, **kwargs)
        captured_trainer.append(self)

    monkeypatch.setattr(DiscretePPOTrainer, "__init__", _spy_init)

    genes = CohortGenes(
        learning_rate=3e-4,
        entropy_coeff=0.01,
        clip_range=0.2,
        gae_lambda=0.95,
        value_coeff=0.5,
        mini_batch_size=32,
        hidden_size=64,
    )
    # Sanity — the gene's default is 0.0 in to_dict(), the precondition
    # for the v2-specific failure mode this test guards against.
    assert genes.to_dict()[weight_key] == 0.0

    worker_mod.train_one_agent(
        agent_id=f"smoke-{weight_key}",
        genes=genes,
        days_to_train=["2026-04-23"],
        eval_day="2026-04-24",
        data_dir=tmp_path,
        device="cpu",
        seed=42,
        model_store=None,  # skip registry IO; we only care about hp.
        scorer_dir=SCORER_DIR,
        generation=0,
        reward_overrides={weight_key: 0.5},
        enabled_set=frozenset(),  # gene NOT enabled — pure override path.
    )

    # Spy invariants — exactly one trainer constructed, hp carries the
    # override, and the constructed trainer's stored attribute reads
    # 0.5 (not the gene default 0.0).
    assert len(captured_hp) == 1, (
        f"expected one trainer construction, got {len(captured_hp)}"
    )
    assert captured_hp[0][weight_key] == 0.5, (
        f"hp dict swallowed the override: hp[{weight_key!r}]="
        f"{captured_hp[0].get(weight_key)!r} (expected 0.5). "
        "v1↔v2 precedence trap regression — see "
        "plans/rewrite/phase-7-port-aux-heads/lessons_learnt.md"
    )
    trainer = captured_trainer[0]
    assert getattr(trainer, weight_key) == 0.5


# ── scalping-lay-quality-gate Phase 2a (2026-05-13) — bet log capture ────


class TestPerBetLogCapture:
    """``_build_eval_bet_records`` converts ``env.all_settled_bets``
    into ``EvaluationBetRecord`` rows with predictor context and a
    derived per-pair lifecycle classification. The writer side
    (``ModelStore.write_bet_logs_parquet``) is verified end-to-end
    in the synthetic-data integration test below; these unit tests
    pin the categorization logic and predictor-field capture that
    forensic analysis (per ``memory/feedback_per_bet_logging.md``)
    depends on.
    """

    def _make_env_day_stubs(self, *, bets, race_pwins):
        """Build minimal stubs sufficient for ``_build_eval_bet_records``.

        ``race_pwins`` is a list of ``{sid: pwin}`` dicts indexed by
        race position; one race per dict. ``bets`` are
        ``env.bet_manager.Bet`` instances whose ``market_id`` references
        the synthesized races.
        """
        from datetime import datetime, timedelta
        from types import SimpleNamespace

        races = []
        for i, _ in enumerate(race_pwins):
            mid = f"1.{i + 100}"
            ticks = [SimpleNamespace(
                timestamp=datetime(2026, 4, 28, 12, 0, 0)
                + timedelta(seconds=10 * t),
            ) for t in range(5)]
            race = SimpleNamespace(
                market_id=mid,
                market_start_time=datetime(2026, 4, 28, 12, 1, 0),
                ticks=ticks,
                runner_metadata={
                    sid: SimpleNamespace(runner_name=f"runner_{sid}")
                    for sid in race_pwins[i]
                },
            )
            races.append(race)

        day = SimpleNamespace(date="2026-04-28", races=races)
        env = SimpleNamespace(
            all_settled_bets=bets,
            _race_p_win_by_race=list(race_pwins),
            starting_budget=100.0,
        )
        return env, day

    def _bet(
        self, *, market_id, sid, side, price=5.0, stake=10.0,
        outcome="won", pnl=10.0, tick_index=2, pair_id=None,
        close_leg=False, force_close=False, stop_close=False,
    ):
        from env.bet_manager import Bet, BetOutcome, BetSide
        return Bet(
            selection_id=sid,
            side=BetSide(side),
            requested_stake=stake,
            matched_stake=stake,
            average_price=price,
            market_id=market_id,
            outcome=BetOutcome(outcome),
            pnl=pnl,
            tick_index=tick_index,
            pair_id=pair_id,
            close_leg=close_leg,
            force_close=force_close,
            stop_close=stop_close,
        )

    def test_empty_bets_returns_empty_list(self):
        from training_v2.cohort.worker import _build_eval_bet_records
        env, day = self._make_env_day_stubs(bets=[], race_pwins=[{1: 0.4}])
        out = _build_eval_bet_records(
            env=env, day=day, starting_budget=100.0,
        )
        assert out == []

    def test_predictor_context_captured(self):
        """``runner_champion_p_win`` and ``race_max_pwin`` come from
        the env's ``_race_p_win_by_race`` cache and ``max(...)`` across
        runners in the same race."""
        from training_v2.cohort.worker import _build_eval_bet_records
        bets = [self._bet(market_id="1.100", sid=2, side="lay")]
        env, day = self._make_env_day_stubs(
            bets=bets,
            race_pwins=[{1: 0.6, 2: 0.15, 3: 0.05}],
        )
        out = _build_eval_bet_records(
            env=env, day=day, starting_budget=100.0,
        )
        assert len(out) == 1
        assert out[0].runner_champion_p_win == pytest.approx(0.15)
        assert out[0].race_max_pwin == pytest.approx(0.6)

    def test_predictor_disabled_yields_none(self):
        """When ``_race_p_win_by_race`` is empty (predictor off), the
        predictor fields stay ``None`` rather than zero."""
        from training_v2.cohort.worker import _build_eval_bet_records
        bets = [self._bet(market_id="1.100", sid=2, side="lay")]
        env, day = self._make_env_day_stubs(
            bets=bets, race_pwins=[{1: 0.4}],
        )
        env._race_p_win_by_race = []
        out = _build_eval_bet_records(
            env=env, day=day, starting_budget=100.0,
        )
        assert out[0].runner_champion_p_win is None
        assert out[0].race_max_pwin is None

    def test_final_outcome_matured(self):
        """Two legs sharing a pair_id, no close flags → ``matured``."""
        from training_v2.cohort.worker import _build_eval_bet_records
        bets = [
            self._bet(
                market_id="1.100", sid=1, side="back", pair_id="P1",
            ),
            self._bet(
                market_id="1.100", sid=1, side="lay", pair_id="P1",
            ),
        ]
        env, day = self._make_env_day_stubs(
            bets=bets, race_pwins=[{1: 0.5}],
        )
        out = _build_eval_bet_records(
            env=env, day=day, starting_budget=100.0,
        )
        assert all(r.final_outcome == "matured" for r in out)

    def test_final_outcome_naked(self):
        """Single leg with pair_id (passive never filled) → ``naked``."""
        from training_v2.cohort.worker import _build_eval_bet_records
        bets = [self._bet(
            market_id="1.100", sid=1, side="back", pair_id="P1",
        )]
        env, day = self._make_env_day_stubs(
            bets=bets, race_pwins=[{1: 0.5}],
        )
        out = _build_eval_bet_records(
            env=env, day=day, starting_budget=100.0,
        )
        assert out[0].final_outcome == "naked"

    def test_final_outcome_agent_closed(self):
        """A pair where one leg carries ``close_leg=True`` (not
        force/stop) → ``agent_closed``."""
        from training_v2.cohort.worker import _build_eval_bet_records
        bets = [
            self._bet(
                market_id="1.100", sid=1, side="back", pair_id="P1",
            ),
            self._bet(
                market_id="1.100", sid=1, side="lay", pair_id="P1",
                close_leg=True,
            ),
        ]
        env, day = self._make_env_day_stubs(
            bets=bets, race_pwins=[{1: 0.5}],
        )
        out = _build_eval_bet_records(
            env=env, day=day, starting_budget=100.0,
        )
        assert all(r.final_outcome == "agent_closed" for r in out)

    def test_final_outcome_force_closed(self):
        """A pair where one leg carries ``force_close=True`` →
        ``force_closed`` (precedes ``agent_closed``)."""
        from training_v2.cohort.worker import _build_eval_bet_records
        bets = [
            self._bet(
                market_id="1.100", sid=1, side="back", pair_id="P1",
            ),
            self._bet(
                market_id="1.100", sid=1, side="lay", pair_id="P1",
                close_leg=True, force_close=True,
            ),
        ]
        env, day = self._make_env_day_stubs(
            bets=bets, race_pwins=[{1: 0.5}],
        )
        out = _build_eval_bet_records(
            env=env, day=day, starting_budget=100.0,
        )
        assert all(r.final_outcome == "force_closed" for r in out)

    def test_final_outcome_stop_closed(self):
        """``stop_close=True`` wins over both ``force_close`` and
        ``close_leg`` in the precedence."""
        from training_v2.cohort.worker import _build_eval_bet_records
        bets = [
            self._bet(
                market_id="1.100", sid=1, side="back", pair_id="P1",
            ),
            self._bet(
                market_id="1.100", sid=1, side="lay", pair_id="P1",
                close_leg=True, stop_close=True,
            ),
        ]
        env, day = self._make_env_day_stubs(
            bets=bets, race_pwins=[{1: 0.5}],
        )
        out = _build_eval_bet_records(
            env=env, day=day, starting_budget=100.0,
        )
        assert all(r.final_outcome == "stop_closed" for r in out)

    def test_final_outcome_directional(self):
        """Bet with ``pair_id=None`` is a directional bet, not a
        scalping leg → ``directional``."""
        from training_v2.cohort.worker import _build_eval_bet_records
        bets = [self._bet(
            market_id="1.100", sid=1, side="back", pair_id=None,
        )]
        env, day = self._make_env_day_stubs(
            bets=bets, race_pwins=[{1: 0.5}],
        )
        out = _build_eval_bet_records(
            env=env, day=day, starting_budget=100.0,
        )
        assert out[0].final_outcome == "directional"

    def test_parquet_round_trip_carries_new_fields(self, tmp_path):
        """``write_bet_logs_parquet`` round-trips the new columns —
        the operator must be able to read predictor context + lifecycle
        class back out for joining to scoreboard.jsonl."""
        import pandas as pd
        from registry.model_store import EvaluationBetRecord, ModelStore

        store = ModelStore(
            db_path=tmp_path / "models.db",
            weights_dir=tmp_path / "weights",
            bet_logs_dir=tmp_path / "bet_logs",
        )
        records = [EvaluationBetRecord(
            run_id="run-A",
            date="2026-04-28",
            market_id="1.100",
            tick_timestamp="2026-04-28T12:00:20",
            seconds_to_off=40.0,
            runner_id=2,
            runner_name="runner_2",
            action="lay",
            price=8.0,
            stake=10.0,
            matched_size=10.0,
            outcome="lost",
            pnl=10.0,
            pair_id="P1",
            close_leg=True,
            force_close=False,
            stop_close=False,
            runner_champion_p_win=0.15,
            race_max_pwin=0.6,
            final_outcome="agent_closed",
        )]
        path = store.write_bet_logs_parquet(
            run_id="run-A", date="2026-04-28", records=records,
        )
        assert path is not None and path.exists()
        df = pd.read_parquet(path)
        assert df.loc[0, "stop_close"] is False or df.loc[0, "stop_close"] == 0
        assert df.loc[0, "runner_champion_p_win"] == pytest.approx(0.15)
        assert df.loc[0, "race_max_pwin"] == pytest.approx(0.6)
        assert df.loc[0, "final_outcome"] == "agent_closed"
