"""Tests for the v2 cohort runner (Session 03).

Lightweight integration: the runner is wired against a stub
``train_one_agent_fn`` so the test exercises the cohort orchestration
loop (gene sampling, breeding, scoreboard write, registry write)
without touching the env / policy / PPO machinery.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path

from registry.model_store import ModelStore
from training_v2.cohort import runner as runner_mod
from training_v2.cohort.genes import CohortGenes, sample_genes
from training_v2.cohort.worker import (
    AgentResult,
    EvalSummary,
    TrainSummary,
    arch_name_for_genes,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _stub_train_one_agent(
    *,
    agent_id: str,
    genes: CohortGenes,
    days_to_train: list[str],
    eval_day: str,
    data_dir: Path,
    device: str,
    seed: int,
    model_store: ModelStore | None,
    generation: int = 0,
    parent_a_id: str | None = None,
    parent_b_id: str | None = None,
    **kwargs,
) -> AgentResult:
    """Pretend to train an agent. Reward = ``-learning_rate × 1000``.

    The negative-LR scoring makes the ranking deterministic and easy
    to assert on: lower LR ⇒ higher reward. Used by the runner's
    breeding loop to elect a clear set of elites.
    """
    arch = arch_name_for_genes(genes)
    fake_reward = -float(genes.learning_rate) * 1000.0

    # Always write the model + an eval row into the registry so the
    # runner's persistence path is covered too.
    weights_path = ""
    run_id = ""
    if model_store is not None:
        from registry.model_store import EvaluationDayRecord
        model_id = model_store.create_model(
            generation=int(generation),
            architecture_name=arch,
            architecture_description="stub",
            hyperparameters=genes.to_dict(),
            parent_a_id=parent_a_id,
            parent_b_id=parent_b_id,
            model_id=str(agent_id),
        )
        # Save a tiny dummy state dict so save_weights writes a real
        # file path the test can later assert on.
        import torch
        weights_path = model_store.save_weights(
            model_id=model_id,
            state_dict={"dummy": torch.zeros(1)},
        )
        run_id = model_store.create_evaluation_run(
            model_id=model_id,
            train_cutoff_date=days_to_train[-1],
            test_days=[eval_day],
        )
        model_store.record_evaluation_day(EvaluationDayRecord(
            run_id=run_id,
            date=eval_day,
            day_pnl=fake_reward,
            bet_count=10,
            winning_bets=5,
            bet_precision=0.5,
            pnl_per_bet=fake_reward / 10.0,
            early_picks=0,
            profitable=fake_reward > 0,
        ))
        model_store.update_composite_score(
            model_id=model_id, score=fake_reward,
        )
    else:
        model_id = str(agent_id)

    train_summary = TrainSummary(
        n_days=len(days_to_train),
        total_steps=10,
        total_reward=fake_reward,
        mean_reward=fake_reward,
        mean_pnl=fake_reward,
        mean_value_loss=1.0,
        mean_policy_loss=0.0,
        mean_approx_kl=0.01,
        wall_time_sec=0.01,
        per_day_rows=[
            {"day_idx": i, "day_str": d, "n_steps": 10,
             "total_reward": fake_reward, "day_pnl": fake_reward,
             "value_loss_mean": 1.0, "policy_loss_mean": 0.0,
             "approx_kl_mean": 0.01, "entropy_mean": 0.0,
             "wall_time_sec": 0.0}
            for i, d in enumerate(days_to_train)
        ],
    )
    eval_summary = EvalSummary(
        eval_day=eval_day,
        total_reward=fake_reward,
        day_pnl=fake_reward,
        n_steps=10,
        bet_count=10,
        winning_bets=5,
        bet_precision=0.5,
        pnl_per_bet=fake_reward / 10.0,
        early_picks=0,
        profitable=fake_reward > 0,
        action_histogram={"NOOP": 8, "OPEN_BACK": 2},
    )
    return AgentResult(
        agent_id=str(agent_id),
        model_id=str(model_id),
        architecture_name=arch,
        genes=genes,
        train=train_summary,
        eval=eval_summary,
        weights_path=str(weights_path),
        run_id=str(run_id),
    )


def _populate_data_dir(tmp_path: Path, dates: list[str]) -> None:
    for d in dates:
        (tmp_path / f"{d}.parquet").write_bytes(b"")


# ── Tests ────────────────────────────────────────────────────────────────


def test_run_cohort_writes_scoreboard_and_registry(tmp_path: Path) -> None:
    """Smoke: 2 agents × 1 generation runs end-to-end."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _populate_data_dir(data_dir, [
        "2026-04-21", "2026-04-22", "2026-04-23",
    ])
    out_dir = tmp_path / "cohort_out"

    last_results = runner_mod.run_cohort(
        n_agents=2,
        n_generations=1,
        days=3,  # 2 training + 1 eval
        data_dir=data_dir,
        device="cpu",
        seed=42,
        output_dir=out_dir,
        train_one_agent_fn=_stub_train_one_agent,
    )

    # 2 results returned (last generation = generation 0 here).
    assert len(last_results) == 2

    # Scoreboard JSONL has one row per agent in the (single) generation.
    scoreboard = (out_dir / "scoreboard.jsonl").read_text().splitlines()
    assert len(scoreboard) == 2
    rows = [json.loads(line) for line in scoreboard]
    for row in rows:
        assert row["schema"] == "v2_cohort_scoreboard"
        assert "hyperparameters" in row
        assert set(row["hyperparameters"].keys()) == {
            "learning_rate", "entropy_coeff", "clip_range",
            "gae_lambda", "value_coeff", "mini_batch_size",
            "hidden_size",
        }
        assert row["eval_day"] == "2026-04-23"
        assert len(row["training_days"]) == 2
        assert row["generation"] == 0

    # Registry has 2 active models.
    store = ModelStore(
        db_path=out_dir / "models.db",
        weights_dir=out_dir / "weights",
        bet_logs_dir=out_dir / "bet_logs",
    )
    assert len(store.list_models(status="active")) == 2


def test_run_cohort_breeds_second_generation(tmp_path: Path) -> None:
    """4 agents × 2 generations writes 8 scoreboard rows (4 per gen)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _populate_data_dir(data_dir, [
        "2026-04-21", "2026-04-22", "2026-04-23",
    ])
    out_dir = tmp_path / "cohort_out"

    runner_mod.run_cohort(
        n_agents=4,
        n_generations=2,
        days=3,
        data_dir=data_dir,
        device="cpu",
        seed=7,
        output_dir=out_dir,
        train_one_agent_fn=_stub_train_one_agent,
    )

    rows = [
        json.loads(line)
        for line in (out_dir / "scoreboard.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 8, "expect 4 agents × 2 generations of rows"
    assert sorted({r["generation"] for r in rows}) == [0, 1]

    # Ranking inside each generation is independent — assert each gen has
    # 4 rows.
    by_gen: dict[int, int] = {}
    for r in rows:
        by_gen[r["generation"]] = by_gen.get(r["generation"], 0) + 1
    assert by_gen == {0: 4, 1: 4}

    # Registry: 8 active models + at least one genetic_event for the
    # breeding pass between gen 0 → gen 1.
    store = ModelStore(
        db_path=out_dir / "models.db",
        weights_dir=out_dir / "weights",
        bet_logs_dir=out_dir / "bet_logs",
    )
    models = store.list_models(status="active")
    assert len(models) == 8
    events = store.get_genetic_events(generation=1)
    # n_agents=4 → 2 elites carried, 2 children bred, so 2 events.
    assert len(events) == 2
    for ev in events:
        assert ev.event_type == "crossover"
        assert ev.parent_a_id is not None
        assert ev.parent_b_id is not None


def test_breed_next_generation_keeps_top_half_elites_verbatim() -> None:
    """Direct unit on the breeding helper.

    Top half (sorted descending by eval reward) carries over verbatim.
    Bottom half is replaced with bred + mutated children.
    """
    rng = random.Random(0)
    # 4 mock agents, eval_total_reward 1.0 / 0.5 / -0.5 / -1.0.
    parents: list[AgentResult] = []
    for i, reward in enumerate([1.0, 0.5, -0.5, -1.0]):
        g = sample_genes(rng)
        parents.append(AgentResult(
            agent_id=f"a{i}",
            model_id=f"m{i}",
            architecture_name=arch_name_for_genes(g),
            genes=g,
            train=TrainSummary(
                n_days=1, total_steps=1, total_reward=0.0,
                mean_reward=0.0, mean_pnl=0.0, mean_value_loss=0.0,
                mean_policy_loss=0.0, mean_approx_kl=0.0,
                wall_time_sec=0.0, per_day_rows=[],
            ),
            eval=EvalSummary(
                eval_day="x", total_reward=reward, day_pnl=reward,
                n_steps=1, bet_count=0, winning_bets=0,
                bet_precision=0.0, pnl_per_bet=0.0, early_picks=0,
                profitable=reward > 0, action_histogram={},
            ),
            weights_path="", run_id="",
        ))

    next_genes, parent_ids = runner_mod._breed_next_generation(
        parents_ranked=parents,
        rng=random.Random(99),
        n_agents=4,
        mutation_rate=0.1,
        model_store=None,
        next_generation=1,
    )

    assert len(next_genes) == 4
    # Top 2 (rewards 1.0 and 0.5 — already sorted by caller) carry
    # over verbatim with no parents recorded.
    assert next_genes[0] == parents[0].genes
    assert next_genes[1] == parents[1].genes
    assert parent_ids[0] == (None, None)
    assert parent_ids[1] == (None, None)
    # Bottom 2 are bred from elites — both parents must be elite IDs.
    elite_ids = {parents[0].model_id, parents[1].model_id}
    for child_idx in (2, 3):
        pa, pb = parent_ids[child_idx]
        assert pa in elite_ids
        assert pb in elite_ids


def test_scoreboard_writes_per_agent_in_sequential_mode(
    tmp_path: Path,
) -> None:
    """Cohort-visibility S01a: in sequential (non-batched) mode, the
    scoreboard.jsonl row for agent N must be on disk before agent N+1
    starts. Pre-plan code wrote all rows after the per-generation
    loop, so mid-cohort visibility was zero. See
    plans/rewrite/phase-3-followups/cohort-visibility/.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _populate_data_dir(data_dir, [
        "2026-04-21", "2026-04-22", "2026-04-23",
    ])
    out_dir = tmp_path / "cohort_out"
    scoreboard_path = out_dir / "scoreboard.jsonl"

    seen_row_counts: list[int] = []

    def _spying_stub(**kwargs):
        # Read the scoreboard at the START of every agent's run. By
        # the per-agent-write contract, row count here == number of
        # agents that have already completed.
        if scoreboard_path.exists():
            seen_row_counts.append(
                len(scoreboard_path.read_text().splitlines())
            )
        else:
            seen_row_counts.append(0)
        return _stub_train_one_agent(**kwargs)

    runner_mod.run_cohort(
        n_agents=3,
        n_generations=1,
        days=3,
        data_dir=data_dir,
        device="cpu",
        seed=42,
        output_dir=out_dir,
        train_one_agent_fn=_spying_stub,
    )

    # Agent 1 starts: 0 rows on disk. Agent 2 starts: 1 row. Agent
    # 3 starts: 2 rows. After all 3 finish: 3 rows on disk.
    assert seen_row_counts == [0, 1, 2], (
        f"expected per-agent live writes [0, 1, 2], got {seen_row_counts}"
    )
    final = scoreboard_path.read_text().splitlines()
    assert len(final) == 3

    # Each row is well-formed JSON with the expected schema.
    for line in final:
        row = json.loads(line)
        assert row["schema"] == "v2_cohort_scoreboard"
        assert row["generation"] == 0


def test_run_cohort_rejects_invalid_args(tmp_path: Path) -> None:
    """Boundary checks: n_agents >= 2, n_generations >= 1, days >= 2."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "cohort_out"
    import pytest
    with pytest.raises(ValueError):
        runner_mod.run_cohort(
            n_agents=1, n_generations=1, days=2,
            data_dir=data_dir, device="cpu", seed=0,
            output_dir=out_dir,
            train_one_agent_fn=_stub_train_one_agent,
        )
    with pytest.raises(ValueError):
        runner_mod.run_cohort(
            n_agents=2, n_generations=0, days=2,
            data_dir=data_dir, device="cpu", seed=0,
            output_dir=out_dir,
            train_one_agent_fn=_stub_train_one_agent,
        )
    with pytest.raises(ValueError):
        runner_mod.run_cohort(
            n_agents=2, n_generations=1, days=1,
            data_dir=data_dir, device="cpu", seed=0,
            output_dir=out_dir,
            train_one_agent_fn=_stub_train_one_agent,
        )
