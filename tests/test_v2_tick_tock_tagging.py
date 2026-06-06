"""Tick-Tock piece B — era row tagging (era_id / era_type / hypothesis_id).

Every scoreboard / model_register / hall-of-fame row of a tagged era is
stamped so the ONE shared leaderboard stays sortable by era and the phenotype
tool can filter to tick rows. The contract:

* a TAGGED era stamps only the tags actually set;
* an UNTAGGED (full-width / legacy) era writes byte-identical rows (NO era_*
  keys) — proven via run_cohort with the stub trainer;
* readers are TOLERANT — a missing tag reads as untagged (``.get → None``);
* model_register.csv hoists the tags as columns iff the lineage rows carry
  them (tolerant of legacy registers that omit them).
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.test_v2_cohort_runner import _populate_data_dir, _stub_train_one_agent
from tools import pbt_leaderboard
from training_v2.cohort import runner as runner_mod
from training_v2.cohort.genes import sample_genes
from training_v2.cohort.pbt import PbtAgentSpec
from training_v2.cohort.runner import (
    _agent_result_to_scoreboard_row,
    _pbt_model_row,
)
from training_v2.cohort.worker import AgentResult, EvalSummary, TrainSummary

_TAGS = {"era_id": "era_007", "era_type": "tock",
         "hypothesis_id": "hypothesis_001"}


def _make_result(genes, *, model_id="m", reward=1.0) -> AgentResult:
    train = TrainSummary(
        n_days=1, total_steps=1, total_reward=reward, mean_reward=reward,
        mean_pnl=reward, mean_value_loss=0.0, mean_policy_loss=0.0,
        mean_approx_kl=0.0, wall_time_sec=0.5, per_day_rows=[])
    ev = EvalSummary(
        eval_day="2026-04-23", total_reward=reward, day_pnl=reward, n_steps=1,
        bet_count=1, winning_bets=1, bet_precision=1.0, pnl_per_bet=reward,
        early_picks=0, profitable=True, action_histogram={})
    return AgentResult(
        agent_id="a", model_id=model_id, architecture_name="lstm",
        genes=genes, train=train, eval=ev, weights_path="w.pt", run_id="r")


# ── scoreboard row ─────────────────────────────────────────────────────────


class TestScoreboardRowTagging:
    def _row(self, era_tags):
        return _agent_result_to_scoreboard_row(
            result=_make_result(sample_genes(__import__("random").Random(0))),
            generation=0, agent_idx=0, eval_days=["2026-04-23"],
            training_days=["2026-04-21", "2026-04-22"], era_tags=era_tags)

    def test_tagged_row_carries_all_three(self):
        row = self._row(_TAGS)
        assert row["era_id"] == "era_007"
        assert row["era_type"] == "tock"
        assert row["hypothesis_id"] == "hypothesis_001"

    def test_untagged_row_has_no_era_keys(self):
        """Byte-identity for a full-width / legacy era: NO era_* keys."""
        for era_tags in (None, {}):
            row = self._row(era_tags)
            assert not any(k.startswith("era_") for k in row)
            assert "hypothesis_id" not in row

    def test_partial_tags_only_set_keys_present(self):
        row = self._row({"era_type": "tick"})
        assert row["era_type"] == "tick"
        assert "era_id" not in row and "hypothesis_id" not in row


# ── pbt model row (lineage + hall-of-fame share this) ──────────────────────


class TestPbtModelRowTagging:
    def _spec(self, genes):
        return PbtAgentSpec(
            genes=genes, tier=1, lineage_id="lin", rotations_seen=frozenset({1}),
            init_weights_path=None, parent_model_id=None, role="fresh")

    def test_tagged(self):
        g = sample_genes(__import__("random").Random(1))
        row = _pbt_model_row(self._spec(g), _make_result(g), generation=0,
                             score=1.0, era_tags=_TAGS)
        assert row["era_type"] == "tock"
        assert row["hypothesis_id"] == "hypothesis_001"
        assert row["era_id"] == "era_007"

    def test_untagged_omits(self):
        g = sample_genes(__import__("random").Random(2))
        row = _pbt_model_row(self._spec(g), _make_result(g), generation=0,
                             score=1.0)
        assert not any(k.startswith("era_") for k in row)
        assert "hypothesis_id" not in row


# ── model_register.csv hoisting (tolerant) ─────────────────────────────────


class TestRegisterHoisting:
    def test_tagged_lineage_hoists_era_columns(self, tmp_path: Path):
        rows = [{
            "generation": 0, "model_id": "m1", "lineage_id": "L",
            "tier": 1, "role": "fresh", "rotations_seen": [1],
            "locked_pnl": 5.0, "naked_pnl": 1.0,
            "era_id": "era_007", "era_type": "tock",
            "hypothesis_id": "hypothesis_001",
            "genes": {"learning_rate": 1e-4},
        }]
        reg = pbt_leaderboard.build_register_rows(rows, frozen_keys=set())
        assert reg[0]["era_type"] == "tock"
        assert reg[0]["hypothesis_id"] == "hypothesis_001"
        out = tmp_path / "model_register.csv"
        pbt_leaderboard.write_csv(out, reg)
        header = out.read_text(encoding="utf-8").splitlines()[0]
        assert "era_type" in header and "hypothesis_id" in header

    def test_legacy_lineage_omits_era_columns(self, tmp_path: Path):
        """A legacy register (rows without tags) must NOT invent era columns."""
        rows = [{
            "generation": 0, "model_id": "m1", "lineage_id": "L",
            "tier": 1, "role": "fresh", "rotations_seen": [1],
            "locked_pnl": 5.0, "naked_pnl": 1.0,
            "genes": {"learning_rate": 1e-4},
        }]
        reg = pbt_leaderboard.build_register_rows(rows, frozen_keys=set())
        assert "era_type" not in reg[0]
        # Tolerant reader: a downstream consumer .get()s None.
        assert reg[0].get("era_type") is None


# ── run_cohort integration: end-to-end scoreboard tagging ──────────────────


class TestRunCohortTagging:
    def _run(self, tmp_path: Path, **kw):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _populate_data_dir(data_dir, ["2026-04-21", "2026-04-22", "2026-04-23"])
        out = tmp_path / "out"
        runner_mod.run_cohort(
            n_agents=2, n_generations=1, days=3, data_dir=data_dir,
            device="cpu", seed=42, output_dir=out,
            train_one_agent_fn=_stub_train_one_agent, **kw)
        return [json.loads(x) for x in
                (out / "scoreboard.jsonl").read_text().splitlines()]

    def test_tagged_run_stamps_every_scoreboard_row(self, tmp_path: Path):
        rows = self._run(tmp_path, era_id="era_007", era_type="tick",
                         hypothesis_id="hypothesis_002")
        assert rows and all(r["era_type"] == "tick" for r in rows)
        assert all(r["era_id"] == "era_007" for r in rows)
        assert all(r["hypothesis_id"] == "hypothesis_002" for r in rows)

    def test_untagged_run_is_byte_identical(self, tmp_path: Path):
        rows = self._run(tmp_path)
        assert rows
        for r in rows:
            assert not any(k.startswith("era_") for k in r)
            assert "hypothesis_id" not in r
