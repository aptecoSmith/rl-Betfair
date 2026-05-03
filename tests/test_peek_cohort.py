"""Tests for tools/peek_cohort.py.

Validates the verdict-bar metrics (fc_rate with true denominator,
policy-close fraction, stop-close fraction, positive-pnl count) on
synthetic models.db + scoreboard.jsonl pairs.

See plans/rewrite/phase-3-followups/cohort-visibility/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from registry.model_store import EvaluationDayRecord, ModelStore
from tools.peek_cohort import (
    AgentSnapshot,
    collect_snapshots,
    render_json,
    render_table,
)


def _seed_db(
    run_dir: Path,
    *,
    agents: list[dict],
) -> None:
    """Insert one row per agent into the cohort run dir's models.db.

    Each ``agent`` dict supplies the per-pair lifecycle counters.
    Mimics what the cohort worker writes after each agent's eval.
    """
    store = ModelStore(
        db_path=run_dir / "models.db",
        weights_dir=run_dir / "weights",
        bet_logs_dir=run_dir / "bet_logs",
    )
    for a in agents:
        mid = store.create_model(
            generation=0, architecture_name="test", architecture_description="",
            hyperparameters={},
            model_id=a["agent_id"],
        )
        rid = store.create_evaluation_run(
            model_id=mid, train_cutoff_date="2026-04-25",
            test_days=["2026-04-28"],
        )
        store.record_evaluation_day(EvaluationDayRecord(
            run_id=rid, date="2026-04-28",
            day_pnl=a.get("day_pnl", 0.0),
            bet_count=a.get("bet_count", 0),
            winning_bets=0, bet_precision=0.0, pnl_per_bet=0.0,
            early_picks=0, profitable=a.get("day_pnl", 0) > 0,
            arbs_completed=a.get("matured", 0),
            arbs_closed=a.get("closed", 0),
            arbs_stop_closed=a.get("stop", 0),
            arbs_force_closed=a.get("forced", 0),
            arbs_naked=a.get("naked", 0),
            locked_pnl=a.get("locked", 0.0),
            naked_pnl=a.get("naked_pnl", 0.0),
            closed_pnl=a.get("closed_pnl", 0.0),
            stop_closed_pnl=a.get("stop_pnl", 0.0),
            force_closed_pnl=a.get("force_pnl", 0.0),
        ))


def test_collects_snapshots_from_db_with_full_breakdown(tmp_path: Path):
    """A post-S01b cohort dir produces snapshots whose lifecycle
    counters match what was written. Verdict-bar metrics compute
    correctly off the TRUE denominator."""
    _seed_db(tmp_path, agents=[
        # Agent 1: lots of stop-closed pairs → low fc_rate, high scf.
        dict(agent_id="aaaa", day_pnl=+12.50, bet_count=100,
             matured=20, closed=10, stop=30, forced=0, naked=10),
        # Agent 2: AMBER v2 shape (no closes/stops) → high fc_rate.
        dict(agent_id="bbbb", day_pnl=-50.0, bet_count=80,
             matured=15, closed=0, stop=0, forced=0, naked=85),
    ])

    snaps = collect_snapshots(tmp_path)
    assert len(snaps) == 2

    a1 = snaps[0]
    # Total outcomes = 20+10+30+0+10 = 70.
    assert a1.total_outcomes == 70
    # fc_rate = naked / total = 10/70 = 0.143.
    assert a1.fc_rate == pytest.approx(10 / 70)
    # pcf = 10/70 = 0.143.
    assert a1.policy_close_fraction == pytest.approx(10 / 70)
    # scf = 30/70 = 0.429.
    assert a1.stop_close_fraction == pytest.approx(30 / 70)

    a2 = snaps[1]
    # total = 15+85 = 100. fc_rate = 85/100 = 0.85 (AMBER v2-ish).
    assert a2.fc_rate == pytest.approx(0.85)
    assert a2.policy_close_fraction == pytest.approx(0.0)
    assert a2.stop_close_fraction == pytest.approx(0.0)


def test_render_json_produces_aggregate_verdict_metrics(tmp_path: Path):
    """The JSON output exposes mean fc_rate, median pcf / scf, and
    positive_pnl_count for downstream tooling / asserts."""
    _seed_db(tmp_path, agents=[
        dict(agent_id="aaaa", day_pnl=+10.0, bet_count=50,
             matured=10, closed=5, stop=15, forced=0, naked=5),
        dict(agent_id="bbbb", day_pnl=+20.0, bet_count=60,
             matured=12, closed=6, stop=18, forced=0, naked=6),
        dict(agent_id="cccc", day_pnl=-5.0, bet_count=40,
             matured=8, closed=4, stop=12, forced=0, naked=4),
    ])
    snaps = collect_snapshots(tmp_path)
    payload = json.loads(render_json(snaps))
    assert payload["n_agents_complete"] == 3
    assert payload["positive_day_pnl_count"] == 2
    # fc_rate per agent: 5/35, 6/42, 4/28 — all ≈ 0.143. Mean ≈ 0.143.
    assert payload["mean_fc_rate"] == pytest.approx(0.142857, abs=0.001)
    # All agents have pcf = 5/35 ≈ 0.143; median = 0.143.
    assert payload["median_policy_close_fraction"] == pytest.approx(
        5 / 35, abs=0.001,
    )
    # scf = 15/35, 18/42, 12/28 → all 0.429. Median = 0.429.
    assert payload["median_stop_close_fraction"] == pytest.approx(
        15 / 35, abs=0.001,
    )
    # Per-agent payload is preserved.
    assert len(payload["agents"]) == 3
    assert all("fc_rate" in a for a in payload["agents"])


def test_skill_metrics_compute_correctly(tmp_path: Path):
    """Scalping-skill metrics (maturation_rate, naked_open_rate,
    locked_per_matured) compute from the right fields and handle
    div-by-zero cleanly."""
    _seed_db(tmp_path, agents=[
        # Agent: 100 pairs opened, 30 matured, 10 closed, 20 stop, 30 forced, 10 naked.
        # locked_pnl=£120, naked_pnl=-£20.
        # maturation_rate = (30+10)/100 = 0.40
        # naked_open_rate = 10/100 = 0.10
        # locked_per_matured = 120/(30+10) = £3.00
        dict(agent_id="aaaa", day_pnl=+10.0, bet_count=200,
             matured=30, closed=10, stop=20, forced=30, naked=10,
             locked=120.0, naked_pnl=-20.0),
    ])
    # Patch the test's _seed_db to also write pairs_opened, since the
    # default helper sets it to 0 (we need 100 to validate the rates).
    store = ModelStore(
        db_path=tmp_path / "models.db",
        weights_dir=tmp_path / "weights",
        bet_logs_dir=tmp_path / "bet_logs",
    )
    # Re-write the row with pairs_opened = 100.
    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "models.db"))
    conn.execute("UPDATE evaluation_days SET pairs_opened = 100")
    conn.commit()
    conn.close()

    snaps = collect_snapshots(tmp_path)
    assert len(snaps) == 1
    s = snaps[0]
    assert s.maturation_rate == pytest.approx(0.40)
    assert s.naked_open_rate == pytest.approx(0.10)
    assert s.locked_per_matured == pytest.approx(3.0)


def test_skill_metrics_handle_zero_pairs_opened(tmp_path: Path):
    """If pairs_opened == 0 (no opens, e.g. collapsed agent), the
    rate-style skill metrics return None — render layer prints
    placeholder."""
    _seed_db(tmp_path, agents=[
        dict(agent_id="aaaa", day_pnl=0.0, bet_count=0,
             matured=0, closed=0, stop=0, forced=0, naked=0,
             locked=0.0, naked_pnl=0.0),
    ])
    snaps = collect_snapshots(tmp_path)
    assert snaps[0].maturation_rate is None
    assert snaps[0].naked_open_rate is None
    assert snaps[0].locked_per_matured is None
    # Render shouldn't crash on None.
    out = render_table(snaps)
    assert "Scalping skill" in out


def test_render_json_includes_skill_aggregates(tmp_path: Path):
    """JSON output exposes mean/median maturation_rate, naked_open_rate,
    locked_per_matured, mean_locked_pnl for downstream tooling."""
    _seed_db(tmp_path, agents=[
        dict(agent_id="aaaa", day_pnl=+10.0, bet_count=50,
             matured=10, closed=5, stop=15, forced=0, naked=5,
             locked=30.0),
        dict(agent_id="bbbb", day_pnl=+20.0, bet_count=60,
             matured=12, closed=6, stop=18, forced=0, naked=6,
             locked=40.0),
    ])
    # Set pairs_opened so rates compute.
    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "models.db"))
    conn.execute("UPDATE evaluation_days SET pairs_opened = bet_count")
    conn.commit()
    conn.close()

    snaps = collect_snapshots(tmp_path)
    payload = json.loads(render_json(snaps))
    assert payload["mean_maturation_rate"] is not None
    assert payload["median_naked_open_rate"] is not None
    assert payload["mean_locked_per_matured"] is not None
    assert payload["mean_locked_pnl"] == pytest.approx(35.0)
    assert all("maturation_rate" in a for a in payload["agents"])


def test_per_generation_summary_appears_for_multigen(tmp_path: Path):
    """When agents span multiple generations (read from JSONL
    'generation' field), the table includes a per-generation
    summary section."""
    sb = tmp_path / "scoreboard.jsonl"
    rows = []
    for i in range(4):
        rows.append({
            "schema": "v2_cohort_scoreboard",
            "agent_id": f"a{i:08x}aaa",
            "eval_day": "2026-05-02",
            "generation": i // 2,  # 2 agents in gen 0, 2 in gen 1
            "eval_day_pnl": -10.0 * i,
            "eval_bet_count": 100,
            "eval_arbs_completed": 20,
            "eval_arbs_closed": 5,
            "eval_arbs_stop_closed": 10,
            "eval_arbs_force_closed": 30,
            "eval_arbs_naked": 5,
            "eval_pairs_opened": 70,
            "eval_locked_pnl": 50.0,
            "eval_naked_pnl": 0.0,
        })
    sb.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    snaps = collect_snapshots(tmp_path)
    out = render_table(snaps)
    assert "Per-generation skill trends" in out
    # Both generations should show in the table.
    assert "  0   2" in out  # gen 0, n=2
    assert "  1   2" in out  # gen 1, n=2


def test_render_table_emits_verdict_summary_rows(tmp_path: Path):
    """The text table contains the four canonical verdict-bar
    summary lines so an operator scanning the output can compare
    directly with the AMBER v2 / S01 references in findings.md."""
    _seed_db(tmp_path, agents=[
        dict(agent_id="aaaa", day_pnl=+1.0, bet_count=10,
             matured=2, closed=1, stop=2, forced=0, naked=5),
    ])
    snaps = collect_snapshots(tmp_path)
    out = render_table(snaps)
    assert "mean fc_rate" in out
    assert "median policy-close fraction" in out
    assert "median stop-close fraction" in out
    assert "positive day_pnl" in out
    assert "AMBER v2" in out  # explicit reference visible in summary


def test_jsonl_only_run_falls_back_to_scoreboard(tmp_path: Path):
    """If the run dir has only scoreboard.jsonl (no models.db, e.g.
    a future session writes JSONL-first), the tool reads the JSONL.
    Pre-S01a cohorts that flushed JSONL only at end may now land
    here.
    """
    sb = tmp_path / "scoreboard.jsonl"
    sb.write_text(
        json.dumps({
            "schema": "v2_cohort_scoreboard",
            "agent_id": "x1234567890ab",
            "eval_day": "2026-04-28",
            "eval_day_pnl": 7.5,
            "eval_bet_count": 22,
            "eval_arbs_completed": 4,
            "eval_arbs_closed": 2,
            "eval_arbs_stop_closed": 5,
            "eval_arbs_force_closed": 0,
            "eval_arbs_naked": 1,
            "eval_locked_pnl": 9.0,
            "eval_naked_pnl": -1.5,
        }) + "\n",
        encoding="utf-8",
    )
    snaps = collect_snapshots(tmp_path)
    assert len(snaps) == 1
    s = snaps[0]
    assert s.agent_id == "x1234567890ab"
    assert s.source == "jsonl"
    assert s.day_pnl == pytest.approx(7.5)
    # total = 4+2+5+0+1 = 12; fc_rate = 1/12.
    assert s.fc_rate == pytest.approx(1 / 12)


def test_empty_run_dir_returns_no_snapshots(tmp_path: Path):
    """Brand-new run dir with no completed agents — neither db nor
    JSONL has rows yet. The tool returns an empty list cleanly."""
    snaps = collect_snapshots(tmp_path)
    assert snaps == []
    out = render_table(snaps)
    assert "No agents complete" in out


def test_db_preferred_when_both_sources_present(tmp_path: Path):
    """When db has the full breakdown AND jsonl has it, the tool
    picks whichever has more outcomes (i.e. whichever is fresher).
    On post-S01b cohorts both should match; this test covers the
    tie-break choice.
    """
    _seed_db(tmp_path, agents=[
        dict(agent_id="aaaa", day_pnl=+1.0, bet_count=10,
             matured=2, closed=1, stop=2, forced=0, naked=5),
    ])
    # Add a JSONL row with DIFFERENT counts so we can tell which the
    # tool chose. JSONL has higher matured count → JSONL wins.
    sb = tmp_path / "scoreboard.jsonl"
    sb.write_text(
        json.dumps({
            "schema": "v2_cohort_scoreboard",
            "agent_id": "aaaa",
            "eval_day": "2026-04-28",
            "eval_day_pnl": 1.0,
            "eval_arbs_completed": 999,  # distinct sentinel
            "eval_arbs_closed": 1, "eval_arbs_stop_closed": 2,
            "eval_arbs_force_closed": 0, "eval_arbs_naked": 5,
        }) + "\n",
        encoding="utf-8",
    )
    snaps = collect_snapshots(tmp_path)
    assert len(snaps) == 1
    # JSONL had more total outcomes (999 vs 10) → it wins.
    assert snaps[0].arbs_completed == 999
    assert snaps[0].source == "jsonl"
