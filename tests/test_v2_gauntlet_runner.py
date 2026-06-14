"""Integration test for the runner's --breeding gauntlet dispatch (Phase 5).

Exercises `_run_gauntlet_breeding` end-to-end through the REAL
executor + ledger + breeder + orchestrator, stubbing only the actual training
(`train_one_agent_fn`). No data dir / model store needed — the stub ignores days
and writes its own checkpoint files so the recipe-purity sidecar chain is real.
"""
from __future__ import annotations

import json
from pathlib import Path

from training_v2.cohort.ledger import GauntletLedger
from training_v2.cohort.pbt import PbtConfig
from training_v2.cohort.runner import _run_gauntlet_breeding
from training_v2.cohort.worker import AgentResult, EvalSummary, TrainSummary


def _make_stub(output_dir: Path, calls: list):
    wdir = output_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)

    def stub(**spec):
        calls.append(spec)
        aid = spec["agent_id"]
        wp = wdir / f"{aid}.pt"
        wp.write_bytes(b"weights")
        locked = float(spec["genes"].open_cost)  # deterministic stub score
        ev = EvalSummary(
            eval_day=spec["eval_days"][0], total_reward=locked, day_pnl=0.0,
            n_steps=1, bet_count=2, winning_bets=1, bet_precision=0.5,
            pnl_per_bet=0.0, early_picks=0, profitable=False,
            action_histogram={"NOOP": 1}, locked_pnl=locked, naked_pnl=-1.0)
        tr = TrainSummary(
            n_days=1, total_steps=1, total_reward=0.0, mean_reward=0.0,
            mean_pnl=0.0, mean_value_loss=0.0, mean_policy_loss=0.0,
            mean_approx_kl=0.0, wall_time_sec=1.0)
        return AgentResult(
            agent_id=aid, model_id=aid, architecture_name="v2_lstm_h64",
            genes=spec["genes"], train=tr, eval=ev, weights_path=str(wp),
            run_id="run")

    return stub


def test_gauntlet_dispatch_runs_pipeline(tmp_path):
    output_dir = tmp_path / "era"
    output_dir.mkdir()
    calls: list = []
    # 16 chronological training days + 2 validation; tranche size 8 -> 2 tranches.
    training_days = [f"2026-01-{d:02d}" for d in range(1, 17)]
    validation_days = ["2026-02-01", "2026-02-02"]
    pbt = PbtConfig(n_agents=4, train_per_rotation=6, eval_per_rotation=2,
                    survivor_fraction=0.5, perturb_frac=0.2)

    out = _run_gauntlet_breeding(
        n_agents=4, n_generations=2,  # 1 climb + 1 breed round
        training_days=training_days, eval_pool=[], validation_days=validation_days,
        data_dir=tmp_path / "data", output_dir=output_dir, model_store=None,
        seed=0, device="cpu", pbt_config=pbt,
        parallel_agents=0,  # sequential -> uses train_one_agent_fn stub
        enabled_set=frozenset(), seed_bands=None,
        reward_overrides=None, maturation_bonus_weight=0.0,
        era_id="gauntlet_smoke_unit", era_type="gauntlet", hypothesis_id=None,
        train_one_agent_fn=_make_stub(output_dir, calls),
        predictor_bundle=None, predictor_manifests=None,
        use_race_outcome_predictor=False, use_direction_predictor=False,
        strategy_mode=None, composite_score_mode="locked_weighted",
        argmax_eval=False, per_transition_credit=False,
        bc_pretrain_steps_override=None, bc_learning_rate_override=None,
        bc_target_entropy_warmup_eps_override=None,
        bc_include_negative_samples=False, bc_positive_weight=1.0,
        bc_include_close_hold_samples=False,
        arb_spread_target_lock_pct_override=None,
        predictor_p_win_back_threshold=0.0,
        predictor_p_win_back_max_threshold=1.0,
        predictor_p_win_lay_threshold=1.0, direction_gate_enabled=False,
        mature_prob_open_threshold=0.0, race_confidence_threshold=0.0,
        lay_price_max=0.0, frozen_direction_head_path=None,
        big_model_threads=1, gpu_policy_lane=False, gpu_lane_max_concurrent=2)

    # Ledger written + frontier at full depth (2 tranches).
    ledger = GauntletLedger.load(output_dir / "gauntlet_ledger.jsonl")
    assert ledger.split is not None
    assert ledger.split.validation_days == validation_days
    assert ledger.frontier_depth() == 2
    # Returned the frontier recipes' AgentResults, best-locked first.
    assert len(out) >= 1
    locked = [r.eval.locked_pnl for r in out]
    assert locked == sorted(locked, reverse=True)

    # Recipe purity in action: at least one tranche-2 run warm-started a
    # checkpoint that carries a matching .genehash sidecar (else it'd have
    # raised RecipePurityError inside run_tranche).
    t2_calls = [c for c in calls if c["init_weights_path"]]
    assert t2_calls, "expected warm-started (tranche>1) runs"
    for c in t2_calls:
        side = Path(c["init_weights_path"] + ".genehash")
        assert side.exists()

    # Validation set reached every eval (fixed fc=0 selection set).
    for c in calls:
        assert sorted(c["eval_days"]) == sorted(validation_days)

    # Scoreboard rows written (the Phase 6 judge + gene_register read these).
    sb = output_dir / "scoreboard.jsonl"
    assert sb.exists()
    rows = [json.loads(ln) for ln in sb.read_text().splitlines() if ln.strip()]
    assert rows, "expected scoreboard rows on the gauntlet path"
    r0 = rows[0]
    assert r0["schema"] == "v2_cohort_scoreboard"
    assert "hyperparameters" in r0 and "lineage_id" in r0 and "tranche_K" in r0
    assert r0["era_id"] == "gauntlet_smoke_unit"


def test_gauntlet_requires_validation_days(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        _run_gauntlet_breeding(
            n_agents=4, n_generations=1, training_days=["2026-01-01"],
            eval_pool=[], validation_days=None, data_dir=tmp_path,
            output_dir=tmp_path, model_store=None, seed=0, device="cpu",
            pbt_config=None, parallel_agents=0,
            enabled_set=frozenset(), seed_bands=None, reward_overrides=None,
            train_one_agent_fn=lambda **k: None, predictor_bundle=None,
            predictor_manifests=None, use_race_outcome_predictor=False,
            use_direction_predictor=False, strategy_mode=None,
            composite_score_mode="locked_weighted", argmax_eval=False,
            per_transition_credit=False, bc_pretrain_steps_override=None,
            bc_learning_rate_override=None,
            bc_target_entropy_warmup_eps_override=None,
            bc_include_negative_samples=False, bc_positive_weight=1.0,
            bc_include_close_hold_samples=False,
            arb_spread_target_lock_pct_override=None,
            predictor_p_win_back_threshold=0.0,
            predictor_p_win_back_max_threshold=1.0,
            predictor_p_win_lay_threshold=1.0, direction_gate_enabled=False,
            mature_prob_open_threshold=0.0, race_confidence_threshold=0.0,
            lay_price_max=0.0, frozen_direction_head_path=None,
            big_model_threads=1, gpu_policy_lane=False,
            gpu_lane_max_concurrent=2)
