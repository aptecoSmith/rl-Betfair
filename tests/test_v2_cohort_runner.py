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

import pytest

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
    eval_days: list[str] | None = None,
    eval_day: str | None = None,  # legacy single-string kwarg
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
    # Backward-compat: stub originally took eval_day=str; runner now
    # passes eval_days=list. Accept either; record eval against the
    # first (and typically only, in unit tests) date.
    if eval_days is None and eval_day is not None:
        eval_days = [eval_day]
    eval_day = eval_days[0] if eval_days else ""

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
            # Phase 3 (legacy 7) — always evolved.
            "learning_rate", "entropy_coeff", "clip_range",
            "gae_lambda", "value_coeff", "mini_batch_size",
            "hidden_size",
            # Phase 5 (promoted 11, 2026-05-03) — at default unless
            # the cohort enables them via --enable-gene NAME.
            "open_cost", "matured_arb_bonus_weight",
            "mark_to_market_weight", "naked_loss_scale",
            "stop_loss_pnl_threshold", "arb_spread_target_lock_pct",
            "fill_prob_loss_weight", "mature_prob_loss_weight",
            "risk_loss_weight", "alpha_lr", "reward_clip",
            # Phase 8 (added 2026-05-05). Operator-controlled via the
            # runner's ``--bc-pretrain-steps`` flag, not GA-evolved.
            # Always present in the persisted gene dict at their pinned
            # defaults (0 / 3e-4 / 5) so registry rows have a stable
            # schema and downstream tooling can read them uniformly.
            "bc_pretrain_steps", "bc_learning_rate",
            "bc_target_entropy_warmup_eps",
            # Phase-13 (added 2026-05-06). Direction-prob aux head —
            # operator-controlled via ``--reward-overrides``; pinned
            # at defaults (0.0 / 60 / 5 / 60.0) on sample.
            "direction_prob_loss_weight", "direction_horizon_ticks",
            "direction_threshold_ticks",
            "direction_force_close_seconds",
            # Phase-13 S05 (added 2026-05-06). Direction-targeted BC.
            "bc_direction_target_weight",
            # Phase-14 S03 (added 2026-05-07). Direction-confidence
            # gate — flag is cohort-wide; threshold is GA-evolved
            # when operator passes ``--enable-gene
            # direction_gate_threshold``.
            "direction_gate_enabled", "direction_gate_threshold",
            # Phase-14 S06 (added 2026-05-07). Threshold-warmup
            # window — operator-controlled, default 5.
            "direction_gate_warmup_eps",
            # Predictor-integration Session 03 (added 2026-05-10).
            "predictor_feature_gain",
            "value_edge_threshold",
            "value_kelly_fraction",
            "each_way_edge_threshold",
            "each_way_kelly_fraction",
            # scalping-tight-naked-variance Phase 2A (added 2026-05-15).
            "naked_variance_penalty_beta",
            # pbt-breeding Step 1b (added 2026-06-03). Structural
            # architecture genes — pinned to the LSTM default under the
            # base sampler (only sample_fresh_blood_genes draws them).
            "architecture", "transformer_depth", "transformer_heads",
            "transformer_ctx_ticks",
            # pbt-breeding (2026-06-04). Obs-representation structural gene.
            "predictor_lean_obs",
            # pbt-gpu-forward (2026-06-04). Transformer-config genes.
            "transformer_ffn_mult", "transformer_pos_encoding",
            # 2026-06-05. Direction mechanism + safety-exit genes.
            "use_direction_predictor", "force_close_before_off_seconds",
            "close_walk_ticks",
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


# ── Phase 5 (restore-genes, 2026-05-03): --enable-gene plumbing ─────────


class TestPhase5EnableGeneCli:
    """CLI parsing + enabled_set plumbing through run_cohort."""

    def test_parse_enabled_genes_dedupes_and_returns_frozenset(self):
        result = runner_mod._parse_enabled_genes([
            "open_cost", "alpha_lr", "open_cost",
        ])
        assert isinstance(result, frozenset)
        assert result == frozenset({"open_cost", "alpha_lr"})

    def test_parse_enabled_genes_empty(self):
        assert runner_mod._parse_enabled_genes([]) == frozenset()
        assert runner_mod._parse_enabled_genes(None) == frozenset()  # type: ignore[arg-type]

    def test_parse_enabled_genes_unknown_name_errors(self):
        import pytest
        with pytest.raises(ValueError, match="unknown gene name"):
            runner_mod._parse_enabled_genes(["fake_gene"])
        # Legacy gene names are NOT valid for --enable-gene either —
        # they're unconditionally evolved already.
        with pytest.raises(ValueError, match="unknown gene name"):
            runner_mod._parse_enabled_genes(["learning_rate"])

    def test_main_errors_on_reward_overrides_enable_gene_collision(
        self, tmp_path: Path,
    ) -> None:
        """``--reward-overrides open_cost=1.0 --enable-gene open_cost``
        errors at startup."""
        import pytest
        with pytest.raises(ValueError, match="Cannot combine"):
            runner_mod.main([
                "--n-agents", "2", "--generations", "1", "--days", "2",
                "--data-dir", str(tmp_path),
                "--output-dir", str(tmp_path / "out"),
                "--reward-overrides", "open_cost=1.0",
                "--enable-gene", "open_cost",
            ])

    def test_legacy_launch_no_enable_gene_passes_no_phase5_keys(
        self, tmp_path: Path,
    ) -> None:
        """A cohort launched WITHOUT any ``--enable-gene`` flags MUST
        produce a per-agent ``reward_overrides`` dict that contains no
        Phase 5 gene keys — preserving byte-identity for legacy
        launches at the same seed."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _populate_data_dir(data_dir, [
            "2026-04-21", "2026-04-22", "2026-04-23",
        ])
        out_dir = tmp_path / "cohort_out"

        captured: list[dict | None] = []

        def _spying_stub(**kwargs):
            captured.append(kwargs.get("reward_overrides"))
            return _stub_train_one_agent(**kwargs)

        runner_mod.run_cohort(
            n_agents=2, n_generations=1, days=3,
            data_dir=data_dir, device="cpu", seed=42,
            output_dir=out_dir,
            train_one_agent_fn=_spying_stub,
        )

        # Stub receives the cohort-level ``reward_overrides`` (None
        # here). The per-agent reward_overrides dict is built INSIDE
        # the real worker; the stub bypasses that path. So the
        # invariant we assert at the runner level is: no cohort
        # reward_overrides were passed (the runner never injects
        # gene values at the cohort level — those happen per-agent
        # inside the worker).
        for ro in captured:
            assert ro is None or not (
                set(ro) & {
                    "open_cost", "matured_arb_bonus_weight",
                    "mark_to_market_weight", "naked_loss_scale",
                    "stop_loss_pnl_threshold", "fill_prob_loss_weight",
                    "mature_prob_loss_weight", "risk_loss_weight",
                    "reward_clip",
                }
            )

    def test_run_cohort_passes_enabled_set_to_worker(
        self, tmp_path: Path,
    ) -> None:
        """``run_cohort(enabled_set=...)`` reaches the worker function."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _populate_data_dir(data_dir, [
            "2026-04-21", "2026-04-22", "2026-04-23",
        ])
        out_dir = tmp_path / "cohort_out"

        captured: list[frozenset[str]] = []

        def _spying_stub(**kwargs):
            captured.append(kwargs.get("enabled_set", frozenset()))
            return _stub_train_one_agent(**kwargs)

        enabled = frozenset({"open_cost", "alpha_lr"})
        runner_mod.run_cohort(
            n_agents=2, n_generations=1, days=3,
            data_dir=data_dir, device="cpu", seed=42,
            output_dir=out_dir,
            train_one_agent_fn=_spying_stub,
            enabled_set=enabled,
        )
        # Both agents see the same enabled_set.
        assert all(es == enabled for es in captured)

    def test_run_cohort_with_enabled_set_samples_varied_gene_values(
        self, tmp_path: Path,
    ) -> None:
        """With ``open_cost`` enabled, each agent's gene draw differs.
        With it disabled, every agent has the cohort-wide default."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _populate_data_dir(data_dir, [
            "2026-04-21", "2026-04-22", "2026-04-23",
        ])
        from training_v2.cohort.genes import PHASE5_GENE_DEFAULTS

        # ── enabled ──
        out_dir = tmp_path / "enabled"
        results = runner_mod.run_cohort(
            n_agents=4, n_generations=1, days=3,
            data_dir=data_dir, device="cpu", seed=7,
            output_dir=out_dir,
            train_one_agent_fn=_stub_train_one_agent,
            enabled_set=frozenset({"open_cost"}),
        )
        open_costs_enabled = {r.genes.open_cost for r in results}
        # 4 fresh continuous samples — overwhelmingly distinct.
        assert len(open_costs_enabled) >= 3

        # ── disabled ──
        out_dir2 = tmp_path / "disabled"
        results2 = runner_mod.run_cohort(
            n_agents=4, n_generations=1, days=3,
            data_dir=data_dir, device="cpu", seed=7,
            output_dir=out_dir2,
            train_one_agent_fn=_stub_train_one_agent,
        )
        for r in results2:
            assert r.genes.open_cost == PHASE5_GENE_DEFAULTS["open_cost"]


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


# ── composite_score (2026-05-04) ───────────────────────────────────────────


def _make_eval_summary(
    *, total_reward: float, arbs_completed: int, arbs_closed: int,
    day_pnl: float = 0.0,
) -> EvalSummary:
    """Tiny EvalSummary builder for the composite-score tests."""
    return EvalSummary(
        eval_day="2026-05-03",
        total_reward=float(total_reward),
        day_pnl=float(day_pnl),
        n_steps=1,
        bet_count=0,
        winning_bets=0,
        bet_precision=0.0,
        pnl_per_bet=0.0,
        early_picks=0,
        profitable=False,
        action_histogram={},
        arbs_completed=int(arbs_completed),
        arbs_closed=int(arbs_closed),
    )


def test_composite_score_default_weight_equals_total_reward() -> None:
    """``maturation_bonus_weight = 0.0`` ⇒ composite == total_reward.

    Byte-identical regression guard: the default-disabled knob must
    not change the GA selection signal compared to pre-2026-05-04.
    """
    e = _make_eval_summary(
        total_reward=-500.0, arbs_completed=40, arbs_closed=10,
    )
    assert runner_mod._composite_score(e, 0.0) == -500.0


def test_composite_score_adds_weight_times_completed_pairs() -> None:
    """Per-completed-pair bonus lands in the score at the right scale."""
    e = _make_eval_summary(
        total_reward=-500.0, arbs_completed=40, arbs_closed=10,
    )
    # 40 + 10 = 50 completed; weight 5 ⇒ +250 bonus.
    assert runner_mod._composite_score(e, 5.0) == -250.0


def test_composite_score_excludes_force_closed_pairs() -> None:
    """``arbs_force_closed`` must NOT contribute (env bail-out, not skill)."""
    e = EvalSummary(
        eval_day="2026-05-03", total_reward=-500.0, day_pnl=0.0,
        n_steps=1, bet_count=0, winning_bets=0, bet_precision=0.0,
        pnl_per_bet=0.0, early_picks=0, profitable=False,
        action_histogram={},
        arbs_completed=10, arbs_closed=0,
        arbs_force_closed=200,  # large — would dominate if mistakenly counted
    )
    # Only 10 completed should count; weight 5 ⇒ +50.
    assert runner_mod._composite_score(e, 5.0) == -450.0


def test_run_cohort_sort_uses_composite_score(tmp_path: Path) -> None:
    """High-maturation low-reward agent ranks above low-maturation high-
    reward agent when the bonus is large enough.

    Stub workforce: agent A has total_reward=-500 with 50 completed
    pairs; agent B has total_reward=-100 with 0 completed pairs. With
    a £5/pair bonus, A's composite is -250 and B's is -100 — B wins.
    With a £20/pair bonus, A's composite is +500 and B's is -100 — A
    wins. The test confirms the knob's direction.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _populate_data_dir(data_dir, [
        "2026-04-21", "2026-04-22", "2026-04-23",
    ])

    def _two_agent_stub(*, agent_id, genes, **kwargs) -> AgentResult:
        # Agent A (low LR ⇒ high-arbs profile): -500 reward, 50 completed
        # Agent B (high LR ⇒ low-arbs profile):  -100 reward,  0 completed
        is_a = float(genes.learning_rate) < 5e-4
        if is_a:
            total_reward, completed, closed = -500.0, 40, 10
        else:
            total_reward, completed, closed = -100.0, 0, 0
        eval_summary = _make_eval_summary(
            total_reward=total_reward, arbs_completed=completed,
            arbs_closed=closed, day_pnl=total_reward,
        )
        train_summary = TrainSummary(
            n_days=1, total_steps=1, total_reward=total_reward,
            mean_reward=total_reward, mean_pnl=total_reward,
            mean_value_loss=0.0, mean_policy_loss=0.0,
            mean_approx_kl=0.0, wall_time_sec=0.0,
            per_day_rows=[],
        )
        return AgentResult(
            agent_id=str(agent_id), model_id=str(agent_id),
            architecture_name="lstm-stub", genes=genes,
            train=train_summary, eval=eval_summary,
            weights_path="", run_id="",
        )

    # Force one agent each side of the LR threshold by overriding
    # gene draws via a controlled seed search.
    rng_a = random.Random(0)
    while True:
        ga = sample_genes(rng_a)
        if ga.learning_rate < 5e-4:
            break
    rng_b = random.Random(0)
    while True:
        gb = sample_genes(rng_b)
        if gb.learning_rate >= 5e-4:
            break

    # Patch sample_genes for one cohort run so the two seeded genes
    # land on slot 0 and slot 1.
    seeded = [ga, gb]
    real_sample_genes = runner_mod.sample_genes
    runner_mod.sample_genes = lambda *a, **k: seeded.pop(0)
    try:
        # Bonus = £5/pair: A composite = -250, B composite = -100 ⇒ B wins.
        out_dir_low = tmp_path / "low_bonus"
        seeded[:] = [ga, gb]
        results_low = runner_mod.run_cohort(
            n_agents=2, n_generations=1, days=2,
            data_dir=data_dir, device="cpu", seed=0,
            output_dir=out_dir_low,
            train_one_agent_fn=_two_agent_stub,
            maturation_bonus_weight=5.0,
        )
        # Bonus = £20/pair: A composite = +500, B composite = -100 ⇒ A wins.
        out_dir_high = tmp_path / "high_bonus"
        seeded[:] = [ga, gb]
        results_high = runner_mod.run_cohort(
            n_agents=2, n_generations=1, days=2,
            data_dir=data_dir, device="cpu", seed=0,
            output_dir=out_dir_high,
            train_one_agent_fn=_two_agent_stub,
            maturation_bonus_weight=20.0,
        )
    finally:
        runner_mod.sample_genes = real_sample_genes

    assert results_low[0].eval.total_reward == -100.0, (
        "Low bonus should rank low-arbs/high-reward agent first"
    )
    assert results_high[0].eval.total_reward == -500.0, (
        "High bonus should rank high-arbs/low-reward agent first"
    )


# ── multi-eval-day aggregator (2026-05-05) ────────────────────────────────


def test_aggregate_eval_summaries_means_numeric_fields() -> None:
    """All numeric fields are MEANED across the per-day inputs."""
    from training_v2.cohort.worker import aggregate_eval_summaries

    summaries = [
        EvalSummary(
            eval_day="2026-05-01", total_reward=-300.0, day_pnl=-100.0,
            n_steps=10, bet_count=400, winning_bets=200,
            bet_precision=0.5, pnl_per_bet=-0.25,
            early_picks=10, profitable=False,
            action_histogram={"NOOP": 100, "OPEN_BACK": 300},
            arbs_completed=40, arbs_naked=10, locked_pnl=200.0,
            naked_pnl=-50.0,
        ),
        EvalSummary(
            eval_day="2026-05-02", total_reward=-100.0, day_pnl=+100.0,
            n_steps=20, bet_count=600, winning_bets=200,
            bet_precision=1/3, pnl_per_bet=100/600,
            early_picks=20, profitable=True,
            action_histogram={"NOOP": 200, "OPEN_BACK": 400},
            arbs_completed=60, arbs_naked=20, locked_pnl=300.0,
            naked_pnl=+150.0,
        ),
    ]
    agg = aggregate_eval_summaries(summaries)

    # Means
    assert agg.total_reward == -200.0
    assert agg.day_pnl == 0.0
    assert agg.locked_pnl == 250.0
    assert agg.naked_pnl == 50.0
    assert agg.arbs_completed == 50
    assert agg.bet_count == 500
    # bet_precision recomputed from the means, not a mean-of-ratios.
    assert agg.bet_precision == 200.0 / 500.0
    # profitable iff mean day_pnl > 0; here mean is exactly 0 ⇒ False
    assert agg.profitable is False
    # eval_day = first summary's day
    assert agg.eval_day == "2026-05-01"
    # per_day populated with both inputs
    assert len(agg.per_day) == 2
    # action_histogram is mean per key (here both keys present in both
    # so means are halves of summed values; rounded back to int)
    assert agg.action_histogram == {"NOOP": 150, "OPEN_BACK": 350}


def test_aggregate_eval_summaries_single_input_is_passthrough() -> None:
    """Single-element list: returns the input wrapped with per_day=[input]."""
    from training_v2.cohort.worker import aggregate_eval_summaries

    s = EvalSummary(
        eval_day="2026-05-03", total_reward=-55.0, day_pnl=-55.0,
        n_steps=5, bet_count=10, winning_bets=5, bet_precision=0.5,
        pnl_per_bet=-5.5, early_picks=0, profitable=False,
        action_histogram={},
        arbs_completed=20, arbs_closed=5,
    )
    agg = aggregate_eval_summaries([s])
    assert agg.total_reward == s.total_reward
    assert agg.day_pnl == s.day_pnl
    assert agg.arbs_completed == 20
    assert len(agg.per_day) == 1
    assert agg.per_day[0].eval_day == "2026-05-03"


def test_aggregate_eval_summaries_naked_pnl_averages_to_zero() -> None:
    """The motivating use case: averaging across days drives naked_pnl
    variance toward zero. Three days with naked outcomes +£200 / -£150 /
    +£100 mean to +£50 — the agent's expected naked tail."""
    from training_v2.cohort.worker import aggregate_eval_summaries

    base = dict(
        total_reward=-100.0, n_steps=1, bet_count=10,
        winning_bets=5, bet_precision=0.5, pnl_per_bet=-10.0,
        early_picks=0, profitable=False, action_histogram={},
    )
    summaries = [
        EvalSummary(eval_day=f"d{i}", day_pnl=v, naked_pnl=v, **base)
        for i, v in enumerate([200.0, -150.0, 100.0])
    ]
    agg = aggregate_eval_summaries(summaries)
    assert abs(agg.naked_pnl - 50.0) < 1e-9
    assert abs(agg.day_pnl - 50.0) < 1e-9


def test_run_cohort_default_n_eval_days_is_50_50_split(
    tmp_path: Path,
) -> None:
    """Default ``n_eval_days = n_days // 2`` — restores v1 50/50 split."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _populate_data_dir(data_dir, [
        "2026-04-25", "2026-04-26", "2026-04-27", "2026-04-28",
        "2026-04-29", "2026-04-30", "2026-05-01",
    ])
    out_dir = tmp_path / "cohort_out"

    runner_mod.run_cohort(
        n_agents=2, n_generations=1, days=7,
        data_dir=data_dir, device="cpu", seed=42,
        output_dir=out_dir,
        train_one_agent_fn=_stub_train_one_agent,
    )

    rows = [
        json.loads(line)
        for line in (out_dir / "scoreboard.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    for row in rows:
        # 7 days → 3 eval (last 3) + 4 training. Training gets the
        # bigger half on odd day counts.
        assert len(row["eval_days"]) == 3, row["eval_days"]
        assert row["eval_days"] == [
            "2026-04-29", "2026-04-30", "2026-05-01",
        ]
        assert len(row["training_days"]) == 4
        # eval_day kept for backward compat = first eval day
        assert row["eval_day"] == "2026-04-29"


def test_scoreboard_row_persists_composite_score_field(tmp_path: Path) -> None:
    """``composite_score`` and ``maturation_bonus_weight`` land in the
    scoreboard row at the values the GA selected on."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _populate_data_dir(data_dir, [
        "2026-04-21", "2026-04-22", "2026-04-23",
    ])
    out_dir = tmp_path / "cohort_out"

    runner_mod.run_cohort(
        n_agents=2, n_generations=1, days=2,
        data_dir=data_dir, device="cpu", seed=0,
        output_dir=out_dir,
        train_one_agent_fn=_stub_train_one_agent,
        maturation_bonus_weight=3.0,
    )

    rows = [
        json.loads(line)
        for line in (out_dir / "scoreboard.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    assert rows, "scoreboard should not be empty"
    for row in rows:
        assert "composite_score" in row
        assert "maturation_bonus_weight" in row
        assert row["maturation_bonus_weight"] == 3.0
        # Stub gives arbs_completed = arbs_closed = 0 by default ⇒
        # composite == total_reward.
        assert row["composite_score"] == row["eval_total_reward"]


# ── Pre-flight cache schema check (2026-05-24) ────────────────────────────


class TestPreflightCacheSchemaCheck:
    """Unit tests for ``runner_mod._preflight_cache_schema_check``.

    Today's bug (2026-05-24): a 12-agent × 3-gen cohort crashed 30s into
    agent 1 because a direction-label cache was at
    ``obs_schema_version=7`` while the env expected 9. The operator had
    committed to ~28h of compute; ~5min of debug + relaunch was an
    unnecessary cost when the failure was knowable at launch.

    These tests cover the four contract points in the spec:

    1. Stale oracle cache → ValueError naming the date + the oracle_cli
       scan command.
    2. Stale direction cache → ValueError naming the date + the
       direction_label_cli scan command.
    3. Both stale → ONE ValueError listing both groups.
    4. Neither cache needed → no-op (no crash, no read).

    The all-up-to-date case is covered by the byte-identity test below
    (preflight passes silently when every header matches).
    """

    # Mirror direction_label_scan._cache_stem so the test stays
    # self-contained and breaks LOUDLY if that naming convention drifts.
    @staticmethod
    def _direction_stem(
        horizon: int = 60, threshold: int = 5, fc_seconds: float = 60.0,
    ) -> str:
        fc_token = f"{fc_seconds:g}".replace(".", "_")
        return f"horizon{horizon}_thresh{threshold}_fc{fc_token}"

    @staticmethod
    def _write_oracle_header(
        data_dir: Path, date: str, obs_dim: int,
    ) -> None:
        cache_dir = data_dir.parent / "oracle_cache_v2" / date
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "header.json").write_text(
            json.dumps({"obs_dim": int(obs_dim)}),
            encoding="utf-8",
        )

    @classmethod
    def _write_direction_header(
        cls, data_dir: Path, date: str, obs_schema_version: int,
        horizon: int = 60, threshold: int = 5, fc_seconds: float = 60.0,
    ) -> None:
        cache_dir = data_dir.parent / "direction_labels" / date
        cache_dir.mkdir(parents=True, exist_ok=True)
        stem = cls._direction_stem(horizon, threshold, fc_seconds)
        (cache_dir / f"{stem}_header.json").write_text(
            json.dumps({
                "obs_schema_version": int(obs_schema_version),
                "direction_horizon_ticks": int(horizon),
                "direction_threshold_ticks": int(threshold),
                "force_close_before_off_seconds": float(fc_seconds),
            }),
            encoding="utf-8",
        )

    # ── Contract point 1 ──────────────────────────────────────────

    def test_stale_oracle_cache_raises_with_date_and_command(
        self, tmp_path: Path,
    ) -> None:
        """One stale oracle header → ValueError mentions the date AND
        the ``oracle_cli scan`` re-scan command."""
        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        stale = "2026-04-21"
        good = "2026-04-22"
        # Operator-running env expects obs_dim=99; cache has 77.
        self._write_oracle_header(data_dir, stale, obs_dim=77)
        self._write_oracle_header(data_dir, good, obs_dim=99)

        with pytest.raises(ValueError) as exc:
            runner_mod._preflight_cache_schema_check(
                training_days=[stale, good],
                data_dir=data_dir,
                needs_oracle=True,
                expected_oracle_obs_dim=99,
                needs_direction=False,
                direction_horizon_ticks=60,
                direction_threshold_ticks=5,
                direction_force_close_seconds=60.0,
            )

        msg = str(exc.value)
        assert "Pre-flight cache schema check FAILED" in msg
        assert stale in msg
        assert "obs_dim=77" in msg
        assert "expects 99" in msg
        assert "python -m training_v2.oracle_cli scan" in msg
        assert f"--dates {stale}" in msg
        # No direction command since needs_direction=False.
        assert "direction_label_cli" not in msg

    def test_missing_oracle_header_raises_with_rescan_command(
        self, tmp_path: Path,
    ) -> None:
        """Missing header file is also a failure (treated as stale)."""
        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        with pytest.raises(ValueError) as exc:
            runner_mod._preflight_cache_schema_check(
                training_days=["2026-04-21"],
                data_dir=data_dir,
                needs_oracle=True,
                expected_oracle_obs_dim=99,
                needs_direction=False,
                direction_horizon_ticks=60,
                direction_threshold_ticks=5,
                direction_force_close_seconds=60.0,
            )
        msg = str(exc.value)
        assert "header.json missing" in msg
        assert "2026-04-21" in msg
        assert "python -m training_v2.oracle_cli scan" in msg

    # ── Contract point 2 ──────────────────────────────────────────

    def test_stale_direction_cache_raises_with_date_and_command(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """Stale direction header (obs_schema_version 7 when env expects
        9) → ValueError mentions the date AND the ``direction_label_cli
        scan`` re-scan command."""
        # Pin the env's schema version inside the runner module to 9
        # so the test is stable regardless of the live env constant.
        monkeypatch.setattr(
            runner_mod, "_DIRECTION_OBS_SCHEMA_VERSION", 9, raising=True,
        )
        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        stale = "2026-04-21"
        self._write_direction_header(
            data_dir, stale, obs_schema_version=7,
        )

        with pytest.raises(ValueError) as exc:
            runner_mod._preflight_cache_schema_check(
                training_days=[stale],
                data_dir=data_dir,
                needs_oracle=False,
                expected_oracle_obs_dim=None,
                needs_direction=True,
                direction_horizon_ticks=60,
                direction_threshold_ticks=5,
                direction_force_close_seconds=60.0,
            )

        msg = str(exc.value)
        assert "Pre-flight cache schema check FAILED" in msg
        assert stale in msg
        assert "obs_schema_version=7" in msg
        assert "expects 9" in msg
        assert "python -m training_v2.direction_label_cli scan" in msg
        assert f"--dates {stale}" in msg
        # The horizon/threshold/fc triple appears in the command.
        assert "--horizon-ticks 60" in msg
        assert "--threshold-ticks 5" in msg
        assert "--force-close-before-off-seconds 60" in msg
        # No oracle command since needs_oracle=False.
        assert "oracle_cli" not in msg

    # ── Contract point 3 ──────────────────────────────────────────

    def test_both_stale_raises_one_error_listing_both(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """Stale oracle AND stale direction → ONE ValueError grouping
        both, with both re-scan commands."""
        monkeypatch.setattr(
            runner_mod, "_DIRECTION_OBS_SCHEMA_VERSION", 9, raising=True,
        )
        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        oracle_stale = "2026-04-21"
        direction_stale = "2026-04-22"
        self._write_oracle_header(data_dir, oracle_stale, obs_dim=77)
        self._write_direction_header(
            data_dir, direction_stale, obs_schema_version=7,
        )

        with pytest.raises(ValueError) as exc:
            runner_mod._preflight_cache_schema_check(
                training_days=[oracle_stale, direction_stale],
                data_dir=data_dir,
                needs_oracle=True,
                expected_oracle_obs_dim=99,
                needs_direction=True,
                direction_horizon_ticks=60,
                direction_threshold_ticks=5,
                direction_force_close_seconds=60.0,
            )

        msg = str(exc.value)
        # Single error covers both classes.
        assert msg.count("Pre-flight cache schema check FAILED") == 1
        # Both failure dates surface.
        assert oracle_stale in msg
        assert direction_stale in msg
        # Both re-scan commands surface.
        assert "python -m training_v2.oracle_cli scan" in msg
        assert "python -m training_v2.direction_label_cli scan" in msg
        # Each cache type has its own section header.
        assert "Oracle cache" in msg
        assert "Direction-label cache" in msg

    # ── Contract point 4 ──────────────────────────────────────────

    def test_neither_needed_is_noop_even_with_no_caches(
        self, tmp_path: Path,
    ) -> None:
        """When neither oracle nor direction caches are needed, the
        check is a no-op — no crash even with zero caches on disk."""
        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        # No caches written. Should silently pass.
        runner_mod._preflight_cache_schema_check(
            training_days=["2026-04-21", "2026-04-22"],
            data_dir=data_dir,
            needs_oracle=False,
            expected_oracle_obs_dim=None,
            needs_direction=False,
            direction_horizon_ticks=60,
            direction_threshold_ticks=5,
            direction_force_close_seconds=60.0,
        )

    # ── All-good passes silently ──────────────────────────────────

    def test_all_caches_uptodate_passes_silently(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """Every header matches → no raise, no log noise."""
        monkeypatch.setattr(
            runner_mod, "_DIRECTION_OBS_SCHEMA_VERSION", 9, raising=True,
        )
        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        for d in ("2026-04-21", "2026-04-22"):
            self._write_oracle_header(data_dir, d, obs_dim=99)
            self._write_direction_header(
                data_dir, d, obs_schema_version=9,
            )

        # Must not raise.
        runner_mod._preflight_cache_schema_check(
            training_days=["2026-04-21", "2026-04-22"],
            data_dir=data_dir,
            needs_oracle=True,
            expected_oracle_obs_dim=99,
            needs_direction=True,
            direction_horizon_ticks=60,
            direction_threshold_ticks=5,
            direction_force_close_seconds=60.0,
        )

    # ── Wiring guard (matches spec contract 4 end-to-end) ─────────

    def test_run_cohort_with_no_bc_or_direction_does_not_crash_on_missing_caches(
        self, tmp_path: Path,
    ) -> None:
        """End-to-end: a launch with ``bc_pretrain_steps_override=None``
        (the default) and no direction-related reward_overrides MUST
        proceed past the preflight even when no cache files exist on
        disk.

        This guards against the preflight ever being called when it
        shouldn't be — the byte-identical pre-patch contract.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _populate_data_dir(data_dir, [
            "2026-04-21", "2026-04-22", "2026-04-23",
        ])
        out_dir = tmp_path / "cohort_out"

        last_results = runner_mod.run_cohort(
            n_agents=2,
            n_generations=1,
            days=3,
            data_dir=data_dir,
            device="cpu",
            seed=42,
            output_dir=out_dir,
            train_one_agent_fn=_stub_train_one_agent,
            # No bc_pretrain_steps_override, no reward_overrides
            # mentioning direction_* knobs.
        )
        # If preflight had run with needs_oracle/needs_direction True
        # it would have crashed (no caches on disk); since both flags
        # default to inert, it must short-circuit and the cohort
        # completes.
        assert len(last_results) == 2
