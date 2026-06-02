"""Resume/checkpoint for the v2 GA cohort runner (ga-recipe-search §C).

Breeding is purely gene-based, so a checkpoint only needs cohort genes +
parent_ids + RNG state + generation index — NO weights. These guard the
serialize/load round-trip, deterministic RNG restoration, and the
stale-row scoreboard truncation, all without a GPU training run.
"""

from __future__ import annotations

import json
import random

import pytest

from training_v2.cohort.genes import CohortGenes, sample_genes
from training_v2.cohort.runner import (
    RESUME_STATE_FILENAME,
    _load_resume_state,
    _truncate_scoreboard_at_generation,
    _write_resume_state,
)


def _make_cohort(n: int, seed: int = 7) -> list[CohortGenes]:
    rng = random.Random(seed)
    return [sample_genes(rng, enabled_set=frozenset()) for _ in range(n)]


class TestResumeStateRoundTrip:
    def test_write_then_load_restores_cohort_and_generation(self, tmp_path):
        cohort = _make_cohort(4)
        parent_ids = [(None, None), ("a", "b"), ("c", "d"), (None, None)]
        rng = random.Random(123)
        _write_resume_state(
            tmp_path, generation=3, cohort=cohort, parent_ids=parent_ids,
            rng=rng, run_id="run-xyz", n_agents=4, n_generations=10,
        )
        assert (tmp_path / RESUME_STATE_FILENAME).exists()
        state = _load_resume_state(tmp_path)
        assert state["generation"] == 3
        assert state["run_id"] == "run-xyz"
        assert [g.to_dict() for g in state["cohort"]] == [
            g.to_dict() for g in cohort
        ]
        assert state["parent_ids"] == [(None, None), ("a", "b"),
                                       ("c", "d"), (None, None)]

    def test_missing_file_returns_none(self, tmp_path):
        assert _load_resume_state(tmp_path) is None

    def test_rng_state_restored_reproduces_sequence(self, tmp_path):
        # The whole point: breeding after resume must draw the SAME random
        # numbers the original run would have, so the GA trajectory is
        # deterministic across a restart.
        rng = random.Random(999)
        rng.random()  # advance past gen-0 sampling-equivalent draws
        rng.random()
        _write_resume_state(
            tmp_path, generation=2, cohort=_make_cohort(2),
            parent_ids=[(None, None), (None, None)], rng=rng,
            run_id="r", n_agents=2, n_generations=5,
        )
        # What the original rng would produce next:
        expected = [rng.random() for _ in range(5)]
        # A fresh rng restored from the checkpoint must match:
        state = _load_resume_state(tmp_path)
        restored = random.Random()
        restored.setstate(state["rng_state"])
        got = [restored.random() for _ in range(5)]
        assert got == expected

    def test_write_is_atomic_no_leftover_tmp(self, tmp_path):
        _write_resume_state(
            tmp_path, generation=0, cohort=_make_cohort(2),
            parent_ids=[(None, None), (None, None)], rng=random.Random(1),
            run_id="r", n_agents=2, n_generations=2,
        )
        assert not (tmp_path / (RESUME_STATE_FILENAME + ".tmp")).exists()
        assert (tmp_path / RESUME_STATE_FILENAME).exists()


class TestScoreboardTruncation:
    def _write_rows(self, path, gens):
        with path.open("w", encoding="utf-8") as f:
            for g in gens:
                f.write(json.dumps({"generation": g, "agent_id": f"a{g}"}) + "\n")

    def test_keeps_below_drops_at_and_above(self, tmp_path):
        sb = tmp_path / "scoreboard.jsonl"
        self._write_rows(sb, [0, 0, 1, 1, 2, 2, 3])  # gens 0..3
        kept = _truncate_scoreboard_at_generation(sb, min_gen=2)
        assert kept == 4  # the four gen-0/gen-1 rows
        remaining = [json.loads(l)["generation"]
                     for l in sb.read_text().splitlines() if l.strip()]
        assert remaining == [0, 0, 1, 1]
        assert all(g < 2 for g in remaining)

    def test_min_gen_zero_drops_everything(self, tmp_path):
        sb = tmp_path / "scoreboard.jsonl"
        self._write_rows(sb, [0, 1, 2])
        kept = _truncate_scoreboard_at_generation(sb, min_gen=0)
        assert kept == 0
        assert sb.read_text().strip() == ""

    def test_missing_file_is_noop(self, tmp_path):
        assert _truncate_scoreboard_at_generation(
            tmp_path / "nope.jsonl", min_gen=1) == 0

    def test_tolerates_blank_and_malformed_lines(self, tmp_path):
        sb = tmp_path / "scoreboard.jsonl"
        sb.write_text(
            json.dumps({"generation": 0}) + "\n\n"
            + "not json\n"
            + json.dumps({"generation": 5}) + "\n"
        )
        kept = _truncate_scoreboard_at_generation(sb, min_gen=3)
        assert kept == 1  # only the gen-0 row survives
