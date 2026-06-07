"""Fast runner-level integration of the PBT path (pbt-breeding Step 2-3).

Drives ``run_cohort(breeding="pbt")`` end-to-end through a STUB
``train_one_agent_fn`` (no env / policy / PPO) so it is deterministic and
quick, yet exercises the real runner wiring: per-tier rotation day
assignment, the warm-start ``init_weights_path`` thread-through, the
promotion-ladder breed between generations, and the ``pbt_lineage.jsonl``
instrumentation. Reuses the gene-only GA test's stub + empty-parquet
data fixture.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from training_v2.cohort import runner as runner_mod
from training_v2.cohort.pbt import PbtConfig

sys.path.insert(0, os.path.dirname(__file__))
from test_v2_cohort_runner import (  # noqa: E402
    _populate_data_dir,
    _stub_train_one_agent,
)


def test_pbt_runner_warm_starts_and_rotates(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _populate_data_dir(data_dir, [f"2026-04-{d:02d}" for d in range(6, 18)])
    out_dir = tmp_path / "pbt_out"

    calls: list[dict] = []

    def capturing_stub(**kw):
        calls.append({
            "generation": kw["generation"],
            "agent_id": kw["agent_id"],
            "init_weights_path": kw.get("init_weights_path"),
            "days_to_train": tuple(kw["days_to_train"]),
            "eval_days": tuple(kw["eval_days"]),
        })
        return _stub_train_one_agent(**kw)

    cfg = PbtConfig(
        n_agents=4, n_rotations=2, train_per_rotation=1, eval_per_rotation=1,
        r2_size=2, r3_size=1, promote_from_r1=1, promote_from_r2=1,
        freeze_top_r3=1,
    )
    runner_mod.run_cohort(
        n_agents=4, n_generations=2, days=12, data_dir=data_dir,
        device="cpu", seed=7, output_dir=out_dir, breeding="pbt",
        pbt_config=cfg, parallel_agents=0,
        train_one_agent_fn=capturing_stub,
    )

    gen0 = [c for c in calls if c["generation"] == 0]
    gen1 = [c for c in calls if c["generation"] == 1]
    assert len(gen0) == 4 and len(gen1) == 4

    # Gen 0: every agent is fresh blood — cold-start, all on rotation 1.
    assert all(c["init_weights_path"] is None for c in gen0)
    assert len({c["days_to_train"] for c in gen0}) == 1  # one rotation

    # Gen 1: the promoted R2 agents (1 elite + 1 offspring) WARM-START from
    # real gen-0 weight files; the refilled R1 fresh blood stays cold.
    warm = [c for c in gen1 if c["init_weights_path"]]
    cold = [c for c in gen1 if not c["init_weights_path"]]
    assert len(warm) == cfg.r2_size      # 2 promoted (elite + offspring)
    assert len(cold) == cfg.n_agents - cfg.r2_size
    for c in warm:
        assert Path(c["init_weights_path"]).exists(), c["init_weights_path"]

    # The warm-started R2 agents train on rotation 2 — DIFFERENT days from
    # the R1 fresh blood (the gauntlet: winning earned the next rotation).
    r1_days = {c["days_to_train"] for c in cold}
    r2_days = {c["days_to_train"] for c in warm}
    assert r1_days and r2_days and not (r1_days & r2_days)

    # Lineage instrumentation: one row per agent per gen, with tiers/roles
    # and the warm-start pointer recorded for Step 4.
    rows = [
        json.loads(line)
        for line in (out_dir / "pbt_lineage.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 8  # 4 agents × 2 generations
    assert any(r["tier"] == 2 and r["role"] in ("elite", "offspring")
               for r in rows)
    assert any(r["init_weights_path"] for r in rows)
    # Roles present: fresh (both gens) + elite/offspring (gen 1).
    assert {r["role"] for r in rows} >= {"fresh", "elite", "offspring"}
    # The live leaderboard + register are regenerated each gen.
    assert (out_dir / "leaderboard.txt").exists()
    assert (out_dir / "model_register.csv").exists()

    # Per-agent train time flows TrainSummary.wall_time_sec -> lineage row ->
    # leaderboard + register (2026-06-04). The stub measures a positive time.
    assert all(r.get("train_seconds", 0) > 0 for r in rows)
    assert "train" in (out_dir / "leaderboard.txt").read_text()
    reg_header = (out_dir / "model_register.csv").read_text().splitlines()[0]
    assert "train_seconds" in reg_header
    # Every lineage row carries a wall-clock trained_at datetime, and it lands
    # in the register + the R1/R2 boards (recency across the multi-day campaign).
    assert all(r.get("trained_at") for r in rows)
    assert "trained_at" in reg_header
    assert "trained_at" in (out_dir / "leaderboard_r1.txt").read_text()


def test_pbt_runner_freezes_r3_to_hall_of_fame_and_leaderboard(
    tmp_path: Path,
) -> None:
    """Run enough gens for the gauntlet to reach R3, then assert an R3
    champion freezes with a frozen_at datetime, lands in
    pbt_hall_of_fame.jsonl, and appears in leaderboard.txt."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _populate_data_dir(data_dir, [f"2026-04-{d:02d}" for d in range(6, 20)])
    out_dir = tmp_path / "pbt_hof"

    cfg = PbtConfig(
        n_agents=6, n_rotations=3, train_per_rotation=1, eval_per_rotation=1,
        r2_size=2, r3_size=2, promote_from_r1=1, promote_from_r2=1,
        freeze_top_r3=1,
    )
    runner_mod.run_cohort(
        n_agents=6, n_generations=4, days=12, data_dir=data_dir,
        device="cpu", seed=3, output_dir=out_dir, breeding="pbt",
        pbt_config=cfg, parallel_agents=0,
        train_one_agent_fn=_stub_train_one_agent,
    )

    hof = out_dir / "pbt_hall_of_fame.jsonl"
    assert hof.exists(), "no hall-of-fame written"
    champs = [json.loads(line) for line in hof.read_text().splitlines() if line]
    assert champs, "no R3 champion frozen"
    for c in champs:
        assert c["frozen_at"], "champion missing frozen_at datetime"
        assert c["tier"] == 3
        assert set(c["rotations_seen"]) == {1, 2, 3}  # climbed all 3 rotations
        assert "genes" in c and "learning_rate" in c["genes"]
        assert "locked_pnl" in c and "bet_count" in c

    lb = (out_dir / "leaderboard.txt").read_text()
    assert "R3 HALL-OF-FAME" in lb
    assert "frozen_at(R3)" in lb and "locked" in lb
    assert "train" in lb  # train-time column (2026-06-04)
    assert "train_seconds" in champs[0]  # plumbed into the hall-of-fame row
    assert "trained_at" not in lb  # R3 uses frozen_at, not trained_at
    # The frozen champion's short model id appears in the table.
    assert champs[0]["model_id"][:8] in lb

    # R1 + R2 live-tier leaderboards are produced too (tier filter on the
    # same per-model rows; no frozen_at column).
    r1 = (out_dir / "leaderboard_r1.txt").read_text()
    r2 = (out_dir / "leaderboard_r2.txt").read_text()
    assert "R1 TIER" in r1 and "R2 TIER" in r2
    assert "frozen_at(R3)" not in r1  # tiers aren't frozen
    assert "trained_at" in r1 and "trained_at" in r2  # when-trained datetime
    assert "locked" in r1 and "lineage" in r2


def test_pbt_runner_chronological_four_tier(tmp_path: Path) -> None:
    """Rotation-rework end-to-end: run_cohort threads rotation_mode through to
    make_rotations AND drives a 4-tier (R4) ladder. Asserts the folds are
    chronological (R1 = oldest days, ordered old→new) and the gauntlet reaches
    R4 with a champion that climbed all four rotations."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # 16 consecutive days → 4 rotations × (2 train + 2 eval).
    dates = [f"2026-04-{d:02d}" for d in range(6, 22)]
    _populate_data_dir(data_dir, dates)
    out_dir = tmp_path / "pbt_chrono"

    calls: list[dict] = []

    def capturing_stub(**kw):
        calls.append({
            "generation": kw["generation"],
            "days_to_train": tuple(kw["days_to_train"]),
            "eval_days": tuple(kw["eval_days"]),
        })
        return _stub_train_one_agent(**kw)

    cfg = PbtConfig(
        n_agents=8, n_rotations=4, train_per_rotation=2, eval_per_rotation=2,
        tier_sizes=(2, 1, 1), promote_counts=(1, 1, 1), freeze_top=1,
    )
    runner_mod.run_cohort(
        n_agents=8, n_generations=4, days=16, data_dir=data_dir,
        device="cpu", seed=5, output_dir=out_dir, breeding="pbt",
        pbt_config=cfg, parallel_agents=0, rotation_mode="chronological",
        train_one_agent_fn=capturing_stub,
    )

    # Gen 0: all agents on R1 = the OLDEST block's train days (chronological).
    gen0_days = {c["days_to_train"] for c in calls if c["generation"] == 0}
    assert gen0_days == {("2026-04-06", "2026-04-07")}, gen0_days

    # Four distinct rotations, chronologically ordered (R1 oldest → R4 newest).
    rot_train = sorted({c["days_to_train"] for c in calls}, key=lambda t: t[0])
    assert len(rot_train) == 4, rot_train
    mins = [t[0] for t in rot_train]
    assert mins == sorted(mins)  # strictly old→new
    assert rot_train[0] == ("2026-04-06", "2026-04-07")   # R1 oldest
    assert rot_train[-1] == ("2026-04-18", "2026-04-19")  # R4 newest

    # The gauntlet reached R4 and froze a champion that climbed all 4 tiers.
    champs = [
        json.loads(line)
        for line in (out_dir / "pbt_hall_of_fame.jsonl").read_text().splitlines()
        if line
    ]
    assert champs, "no champion frozen"
    assert any(set(c["rotations_seen"]) == {1, 2, 3, 4} for c in champs)
    assert all(c["tier"] == 4 for c in champs)  # only the top tier (R4) freezes


def test_resolve_holdout_days_requests_non_excluded_count() -> None:
    """Regression guard for the launch crash: days_arg must be the
    NON-excluded racing-day count (select_days RAISES if n_days > post-exclude
    count), NOT the full count."""
    days = [f"2026-04-{d:02d}" for d in range(1, 29)]  # 28 ascending
    ex, days_arg, hold = runner_mod._resolve_holdout_days(days, 7, [])
    assert hold == days[-7:]
    assert set(ex) == set(days[-7:])
    assert days_arg == 21              # 28 - 7 (the bug set this to 28 → crash)
    assert days_arg == len([d for d in days if d not in set(ex)])
    # base exclude merges and is counted out too.
    ex2, n2, _ = runner_mod._resolve_holdout_days(days, 7, [days[0]])
    assert n2 == 20 and days[0] in ex2  # 28 - 7 - 1
    # off (holdout_recent <= 0): no holdout, full count.
    assert runner_mod._resolve_holdout_days(days, 0, ["x"]) == (["x"], 28, [])
    # too few days → loud exit, never a silent under-request.
    import pytest
    with pytest.raises(SystemExit, match="holdout-recent"):
        runner_mod._resolve_holdout_days(days[:5], 7, [])


def test_resolve_generations_self_heals_with_tiers() -> None:
    """G = n_tiers + K when --maturation-gens set (constant top-tier window as
    the ladder deepens); explicit --generations when not."""
    r = runner_mod._resolve_generations
    # None ⇒ explicit (byte-identical).
    assert r(maturation_gens=None, n_tiers=3, explicit_generations=5) == 5
    # K=2 reproduces the 3-tier/5-gen baseline AND auto-scales with depth.
    assert r(maturation_gens=2, n_tiers=3, explicit_generations=99) == 5
    assert r(maturation_gens=2, n_tiers=4, explicit_generations=99) == 6
    assert r(maturation_gens=2, n_tiers=5, explicit_generations=99) == 7
    # Top-tier maturation window (G - N + 1) is constant = K+1 for every depth.
    for n in (3, 4, 5, 6):
        g = r(maturation_gens=2, n_tiers=n, explicit_generations=0)
        assert g - n + 1 == 3


def test_select_final_freeze_picks_top_r3_nonfrozen_deduped() -> None:
    """Unit guard for the end-of-run freeze selector: tier-3 only, drops
    already-frozen specs + already-frozen model_ids, ranks by score, caps
    at freeze_top_r3."""
    from dataclasses import dataclass

    @dataclass
    class _Spec:
        frozen: bool
        tier: int

    @dataclass
    class _Res:
        model_id: str
        s: float

    pairs = [
        (_Spec(False, 3), _Res("a", 5.0)),    # eligible
        (_Spec(False, 3), _Res("b", 9.0)),    # eligible by tier but already frozen
        (_Spec(False, 3), _Res("c", 3.0)),    # eligible
        (_Spec(False, 2), _Res("d", 100.0)),  # tier 2 -> excluded
        (_Spec(True, 3), _Res("e", 100.0)),   # spec.frozen -> excluded
    ]
    out = runner_mod._select_final_freeze(
        pairs, already_frozen_ids={"b"},
        score_fn=lambda r: r.s, freeze_top_r3=2,
    )
    assert [r.model_id for _s, r in out] == ["a", "c"]  # top-2, b/d/e dropped

    # Cap honoured + the cap value is respected.
    one = runner_mod._select_final_freeze(
        pairs, already_frozen_ids=set(),
        score_fn=lambda r: r.s, freeze_top_r3=1,
    )
    assert [r.model_id for _s, r in one] == ["b"]  # highest eligible R3

    # No tier-3 -> empty (short runs never form R3).
    assert runner_mod._select_final_freeze(
        [(_Spec(False, 1), _Res("x", 1.0))], already_frozen_ids=set(),
        score_fn=lambda r: r.s, freeze_top_r3=3,
    ) == []


def test_pbt_runner_end_of_run_freezes_final_generation(tmp_path: Path) -> None:
    """The FINAL generation's R3 winners must reach the hall-of-fame.

    The in-loop breed-step freeze is gated ``generation < n_generations - 1``,
    so without the end-of-run pass the last gen's R3 (the most-trained agents)
    is silently dropped. R3 first forms at gen 2; in-loop freezes gen 2; the
    end-of-run pass must additionally freeze the final gen (n_generations-1).
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _populate_data_dir(data_dir, [f"2026-04-{d:02d}" for d in range(6, 20)])
    out_dir = tmp_path / "pbt_eor"

    n_gens = 4
    cfg = PbtConfig(
        n_agents=6, n_rotations=3, train_per_rotation=1, eval_per_rotation=1,
        r2_size=2, r3_size=2, promote_from_r1=1, promote_from_r2=1,
        freeze_top_r3=1,
    )
    runner_mod.run_cohort(
        n_agents=6, n_generations=n_gens, days=12, data_dir=data_dir,
        device="cpu", seed=3, output_dir=out_dir, breeding="pbt",
        pbt_config=cfg, parallel_agents=0,
        train_one_agent_fn=_stub_train_one_agent,
    )

    champs = [
        json.loads(line)
        for line in (out_dir / "pbt_hall_of_fame.jsonl").read_text().splitlines()
        if line
    ]
    gens_frozen = {c["generation"] for c in champs}
    assert (n_gens - 1) in gens_frozen, (
        f"final gen {n_gens - 1} not frozen (end-of-run pass missing); "
        f"got gens {sorted(gens_frozen)}"
    )
    # In-loop gen-2 freeze is still present (no regression).
    assert 2 in gens_frozen
    # Idempotent: no duplicate (model_id, generation) rows.
    keys = [(c["model_id"], c["generation"]) for c in champs]
    assert len(keys) == len(set(keys)), "duplicate champion rows"
    # The final-gen champion climbed all three rotations.
    final_champ = next(c for c in champs if c["generation"] == n_gens - 1)
    assert set(final_champ["rotations_seen"]) == {1, 2, 3}
