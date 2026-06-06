"""Tick-Tock piece D — held-out compare harness.

Unit-gates the pure pieces (sealed-day resolution, champion selection by
locked, per-leg σ_naked from bet-logs, per-era summary, paired delta) and the
orchestration via a STUBBED reeval engine — so no real rollouts run, but the
4-call fan-out (tick/tock × fc0/fc120), the fc=120 reward-override, the
report, and the peek-ledger append are all exercised.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tools import tick_tock_compare as ttc


# ── sealed days ────────────────────────────────────────────────────────────


def test_existing_sealed_days_filters_to_on_disk(tmp_path: Path):
    for d in ("2026-05-20", "2026-05-22", "2026-05-29"):
        (tmp_path / f"{d}.parquet").write_bytes(b"")
    got = ttc.existing_sealed_days(ttc.SEALED_DAYS_NAMED, tmp_path)
    assert got == ["2026-05-20", "2026-05-22", "2026-05-29"]


# ── champion selection ──────────────────────────────────────────────────────


def _sb(agent_id, locked, **extra):
    return {"agent_id": agent_id, "eval_locked_pnl": locked, **extra}


class TestSelectChampions:
    def test_ranks_by_locked_and_caps_top_k(self):
        rows = [_sb("a", 1.0), _sb("b", 9.0), _sb("c", 5.0)]
        assert ttc.select_champions(rows, selectors={}, top_k=2) == ["b", "c"]

    def test_empty_selector_matches_all(self):
        """An untagged full-width campaign selects as the whole tick."""
        rows = [_sb("a", 1.0), _sb("b", 2.0)]
        assert set(ttc.select_champions(rows, selectors={"era_type": None},
                                        top_k=5)) == {"a", "b"}

    def test_selector_filters_by_tag(self):
        rows = [_sb("a", 9.0, hypothesis_id="h1"),
                _sb("b", 8.0, hypothesis_id="h2"),
                _sb("c", 7.0, hypothesis_id="h1")]
        got = ttc.select_champions(
            rows, selectors={"hypothesis_id": "h1"}, top_k=5)
        assert got == ["a", "c"]

    def test_dedupes_agent_keeping_best_locked(self):
        rows = [_sb("a", 1.0), _sb("a", 9.0), _sb("b", 5.0)]
        assert ttc.select_champions(rows, selectors={}, top_k=5) == ["a", "b"]


# ── σ_naked_leg from bet-logs ───────────────────────────────────────────────


def _write_naked_parquet(bet_logs: Path, stem: str, agent: str, day: str,
                         naked_pnls, other_pnls=()):
    run_dir = bet_logs / f"reeval_{stem}_{agent}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"final_outcome": "naked", "pnl": float(p)} for p in naked_pnls]
    rows += [{"final_outcome": "matured", "pnl": float(p)} for p in other_pnls]
    pd.DataFrame(rows).to_parquet(run_dir / f"{day}.parquet")


class TestSigmaLegForAgent:
    def test_population_std_of_naked_legs_only(self, tmp_path: Path):
        bl = tmp_path / "bet_logs"
        # 6 naked legs (>= n_min) + matured legs that must be ignored.
        _write_naked_parquet(bl, "tt_tick_fc0", "agentA", "2026-05-20",
                             naked_pnls=[-10, -5, 0, 5, 10, 0],
                             other_pnls=[100, -100])
        out = ttc.sigma_leg_for_agent(bl, "tt_tick_fc0", "agentA")
        assert out["n_legs"] == 6
        import numpy as np
        assert out["sigma_leg"] == pytest.approx(
            float(np.std([-10, -5, 0, 5, 10, 0], ddof=0)))
        assert out["worst_leg"] == -10.0

    def test_nan_when_below_min_legs(self, tmp_path: Path):
        bl = tmp_path / "bet_logs"
        _write_naked_parquet(bl, "tt_tick_fc0", "agentB", "2026-05-20",
                             naked_pnls=[-1, 1])  # only 2 < n_min=5
        out = ttc.sigma_leg_for_agent(bl, "tt_tick_fc0", "agentB")
        assert out["n_legs"] == 2
        assert out["sigma_leg"] != out["sigma_leg"]  # NaN

    def test_missing_dir_is_nan(self, tmp_path: Path):
        out = ttc.sigma_leg_for_agent(tmp_path / "bet_logs", "x", "nope")
        assert out["n_legs"] == 0 and out["sigma_leg"] != out["sigma_leg"]


# ── summary + paired delta ──────────────────────────────────────────────────


def test_summarise_and_paired_delta(tmp_path: Path):
    bl = tmp_path / "bet_logs"
    for ag, locked in (("a", 4.0), ("b", 6.0)):
        _write_naked_parquet(bl, "tt_tick_fc0", ag, "2026-05-20",
                             naked_pnls=[-2, -1, 0, 1, 2])
    tick_rows = [
        {"agent_id": "a", "reeval_locked_pnl": 4.0, "reeval_naked_pnl": 1.0},
        {"agent_id": "b", "reeval_locked_pnl": 6.0, "reeval_naked_pnl": 2.0},
    ]
    tick = ttc.summarise_era(tick_rows, bl, "tt_tick_fc0")
    assert tick["mean_locked_pnl"] == pytest.approx(5.0)
    assert tick["n_agents"] == 2 and tick["n_agents_with_sigma"] == 2
    assert tick["mean_sigma_leg"] > 0

    tock = dict(tick, mean_locked_pnl=8.0, mean_sigma_leg=tick["mean_sigma_leg"])
    delta = ttc.paired_delta(tick, tock)
    assert delta["delta_mean_locked_pnl"] == pytest.approx(3.0)
    assert delta["delta_mean_sigma_leg"] == pytest.approx(0.0)


# ── orchestration (stubbed reeval engine) ───────────────────────────────────


class _StubReeval:
    """Records each reeval call + writes a reeval JSONL and naked bet-logs so
    the harness's downstream summarise path runs for real."""

    def __init__(self, cohort_dir: Path, *, tick_locked, tock_locked):
        self.cohort_dir = cohort_dir
        self.tick_locked = tick_locked
        self.tock_locked = tock_locked
        self.calls: list[dict] = []

    def __call__(self, argv: list[str]) -> int:
        a = {}
        i = 0
        while i < len(argv):
            tok = argv[i]
            if tok == "--output":
                a["output"] = argv[i + 1]; i += 2; continue
            if tok == "--filter-agent-ids":
                ids = []
                i += 1
                while i < len(argv) and not argv[i].startswith("--"):
                    ids.append(argv[i]); i += 1
                a["ids"] = ids; continue
            if tok == "--reward-overrides":
                a.setdefault("ro", []).append(argv[i + 1]); i += 2; continue
            i += 1
        stem = a["output"][:-len(".jsonl")]
        is_tock = "tock" in stem
        locked = self.tock_locked if is_tock else self.tick_locked
        self.calls.append({"stem": stem, "ids": a["ids"],
                           "ro": a.get("ro", [])})
        # Write the reeval JSONL + 5 naked legs per agent so σ_leg is finite.
        with (self.cohort_dir / a["output"]).open("w", encoding="utf-8") as f:
            for ag in a["ids"]:
                f.write(json.dumps({
                    "agent_id": ag, "reeval_locked_pnl": locked,
                    "reeval_naked_pnl": 1.0}) + "\n")
                _write_naked_parquet(self.cohort_dir / "bet_logs", stem, ag,
                                     "2026-05-20", naked_pnls=[-2, -1, 0, 1, 2])
        return 0


def _make_cohort(tmp_path: Path) -> Path:
    cohort = tmp_path / "cohort"
    cohort.mkdir()
    rows = [
        {"agent_id": "tick_a", "eval_locked_pnl": 5.0, "era_type": "tick"},
        {"agent_id": "tick_b", "eval_locked_pnl": 3.0, "era_type": "tick"},
        {"agent_id": "tock_a", "eval_locked_pnl": 6.0, "era_type": "tock",
         "hypothesis_id": "hypothesis_001"},
        {"agent_id": "tock_b", "eval_locked_pnl": 4.0, "era_type": "tock",
         "hypothesis_id": "hypothesis_001"},
    ]
    with (cohort / "scoreboard.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return cohort


class TestRunCompareOrchestration:
    def test_four_reeval_calls_report_and_ledger(self, tmp_path: Path):
        cohort = _make_cohort(tmp_path)
        stub = _StubReeval(cohort, tick_locked=5.0, tock_locked=9.0)
        ledger = tmp_path / "work" / "peek_ledger.jsonl"
        report = tmp_path / "work" / "compare.md"
        res = ttc.run_compare(
            tick_cohort_dir=cohort, tock_cohort_dir=cohort,
            tick_selectors={"era_type": "tick"},
            tock_selectors={"hypothesis_id": "hypothesis_001"},
            sealed_days=["2026-05-20"], data_dir=tmp_path, device="cpu",
            top_k=5, argmax_eval=True, fc_list=[0, 120],
            extra_reeval_args=["--use-race-outcome-predictor"],
            peek_ledger=ledger, report_out=report,
            reeval_fn=stub, timestamp="2026-06-06T21:00:00+00:00")

        # 4 reeval calls: tick/tock × fc0/fc120.
        assert len(stub.calls) == 4
        stems = sorted(c["stem"] for c in stub.calls)
        assert stems == ["tt_tick_fc0", "tt_tick_fc120",
                         "tt_tock_fc0", "tt_tock_fc120"]
        # fc=120 calls carry the force-close reward-override; fc=0 do not.
        for c in stub.calls:
            has_ro = any("force_close_before_off_seconds=120" in x
                         for x in c["ro"])
            assert has_ro == c["stem"].endswith("fc120")
        # passthrough arg reached every call's argv is implicit (stub ignores
        # it) — assert champion selection picked the tock hypothesis agents.
        assert set(res["tock_ids"]) == {"tock_a", "tock_b"}
        assert set(res["tick_ids"]) == {"tick_a", "tick_b"}

        # Paired delta: tock locked (9) − tick locked (5) = +4 at each fc.
        for fc in (0, 120):
            assert res["by_fc"][fc]["delta"]["delta_mean_locked_pnl"] == \
                pytest.approx(4.0)

        # Report written + mentions both fc sections.
        md = report.read_text(encoding="utf-8")
        assert "force_close = 0s" in md and "force_close = 120s" in md

        # Peek-ledger appended exactly one row with the headline numbers.
        lines = ledger.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["n_sealed"] == 1
        assert row["results"]["0"]["delta_mean_locked"] == pytest.approx(4.0)
        assert row["tock"].endswith('{"hypothesis_id": "hypothesis_001"}')

    def test_raises_when_no_tock_match(self, tmp_path: Path):
        cohort = _make_cohort(tmp_path)
        stub = _StubReeval(cohort, tick_locked=5.0, tock_locked=9.0)
        with pytest.raises(SystemExit, match="tock"):
            ttc.run_compare(
                tick_cohort_dir=cohort, tock_cohort_dir=cohort,
                tick_selectors={}, tock_selectors={"hypothesis_id": "nope"},
                sealed_days=["2026-05-20"], data_dir=tmp_path,
                fc_list=[0], reeval_fn=stub,
                timestamp="2026-06-06T21:00:00+00:00")
