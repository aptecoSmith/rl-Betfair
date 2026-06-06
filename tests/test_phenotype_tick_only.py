"""Tick-Tock piece C — phenotype `--tick-only` discovery filter.

A tock band-seeds (pins) its driver genes, so including its rows would drive
their variance toward zero and corrupt the gene→behaviour correlations the
analysis exists to surface. `--tick-only` drops `era_type=='tock'` rows while
KEEPING untagged/legacy rows (a full-width campaign is a tick by
construction — the build plan pools them in as bonus tick data).
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from tools.phenotype_analysis import filter_tick_only, run


# ── filter_tick_only (pure) ────────────────────────────────────────────────


class TestFilterTickOnly:
    def test_drops_tock_keeps_tick_and_untagged(self):
        df = pd.DataFrame({
            "era_type": ["tick", "tick", "tock", "tock", None, float("nan")],
            "x": [1, 2, 3, 4, 5, 6],
        })
        kept, info = filter_tick_only(df, [])
        assert info["has_col"] is True
        assert info["n_tock"] == 2
        assert info["n_tick"] == 2
        assert info["n_untagged"] == 2
        assert info["n_kept"] == 4
        assert set(kept["x"]) == {1, 2, 5, 6}  # tocks (3,4) dropped

    def test_case_insensitive_tock(self):
        df = pd.DataFrame({"era_type": ["TOCK", "Tick"], "x": [1, 2]})
        kept, _ = filter_tick_only(df, [])
        assert set(kept["x"]) == {2}

    def test_no_column_keeps_all_as_legacy_tick(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        kept, info = filter_tick_only(df, [])
        assert info["has_col"] is False
        assert info["n_kept"] == 3
        assert len(kept) == 3


# ── run() integration over a mixed-era register ────────────────────────────


def _write_register(path: Path, *, n_tick: int, n_tock: int) -> None:
    rng = random.Random(0)
    rows = []

    def _row(mid, era, *, pin_stop=None):
        return dict(
            generation=0, model_id=mid, era_type=era, pairs_opened=20,
            arbs_completed=rng.randint(0, 10), arbs_closed=rng.randint(0, 10),
            arbs_force_closed=0, arbs_naked=rng.randint(0, 5),
            arbs_stop_closed=rng.randint(0, 5),
            locked_pnl=rng.uniform(-5, 5), naked_pnl=rng.uniform(-5, 5),
            naked_std=rng.uniform(0, 3),
            gene_learning_rate=rng.uniform(1e-5, 1e-3),
            gene_stop_loss_pnl_threshold=(
                pin_stop if pin_stop is not None else rng.uniform(0, 0.3)),
        )
    for i in range(n_tick):
        rows.append(_row(f"t{i}", "tick"))
    for i in range(n_tock):
        rows.append(_row(f"k{i}", "tock", pin_stop=0.22))  # PINNED in tock
    pd.DataFrame(rows).to_csv(path, index=False)


def _latest_md(cohort_dir: Path) -> str:
    mds = sorted(cohort_dir.glob("phenotype_analysis_*.md"))
    return mds[-1].read_text(encoding="utf-8")


class TestRunTickOnly:
    def test_tick_only_excludes_tock_rows(self, tmp_path: Path):
        _write_register(tmp_path / "model_register.csv", n_tick=14, n_tock=8)
        rc = run(tmp_path, out=None, source=None, top_k=6, tick_only=True)
        assert rc == 0
        md = _latest_md(tmp_path)
        assert "`--tick-only`" in md
        assert "8 tock rows EXCLUDED" in md
        assert "**Agents (n):** 14" in md  # only the tick rows analysed

    def test_without_tick_only_pools_all(self, tmp_path: Path):
        _write_register(tmp_path / "model_register.csv", n_tick=14, n_tock=8)
        rc = run(tmp_path, out=None, source=None, top_k=6, tick_only=False)
        assert rc == 0
        md = _latest_md(tmp_path)
        assert "**Agents (n):** 22" in md  # 14 + 8 pooled

    def test_all_tock_errors(self, tmp_path: Path):
        _write_register(tmp_path / "model_register.csv", n_tick=0, n_tock=10)
        rc = run(tmp_path, out=None, source=None, top_k=6, tick_only=True)
        assert rc == 2  # nothing left to correlate
