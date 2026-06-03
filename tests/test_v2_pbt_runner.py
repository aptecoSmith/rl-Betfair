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
