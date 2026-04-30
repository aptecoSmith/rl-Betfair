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
