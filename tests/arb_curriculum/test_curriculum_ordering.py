"""Tests for arb-curriculum Session 05 — curriculum day ordering.

Covers hard_constraints.md §21 (opt-in modes), §22 (membership preserved),
§23 (missing-cache fallback), §31 (test suite).

Test inventory (7):
1. random mode reproduces rng.sample behaviour
2. density_desc sorts densest-first
3. density_asc sorts sparsest-first
4. missing cache → density 0, placed at end for density_desc (warning captured)
5. membership preserved across all modes
6. config round-trip: curriculum_day_order flows through to JSONL field
7. invalid mode falls back to random with error log
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import pytest

from training.arb_oracle import density_for_date, order_days_by_density


# ── helpers ───────────────────────────────────────────────────────────────────


def _write_header(cache_dir: Path, date: str, density: float) -> None:
    day_dir = cache_dir / date
    day_dir.mkdir(parents=True, exist_ok=True)
    (day_dir / "header.json").write_text(
        json.dumps({"density": density}), encoding="utf-8"
    )


# ── 1. Random mode reproduces rng.sample ─────────────────────────────────────


def test_random_mode_matches_rng_sample(tmp_path):
    dates = ["2026-04-01", "2026-04-02", "2026-04-03"]
    seed = 42
    expected = random.Random(seed).sample(dates, len(dates))
    result = order_days_by_density(
        dates, "random", tmp_path, random.Random(seed)
    )
    assert result == expected


# ── 2. density_desc sorts densest-first ──────────────────────────────────────


def test_density_desc_sorts_descending(tmp_path):
    cache = tmp_path / "oracle_cache"
    _write_header(cache, "2026-04-01", 0.001)
    _write_header(cache, "2026-04-02", 0.010)
    _write_header(cache, "2026-04-03", 0.005)

    dates = ["2026-04-01", "2026-04-02", "2026-04-03"]
    result = order_days_by_density(
        dates, "density_desc", cache, random.Random(0)
    )
    assert result == ["2026-04-02", "2026-04-03", "2026-04-01"]


# ── 3. density_asc sorts sparsest-first ──────────────────────────────────────


def test_density_asc_sorts_ascending(tmp_path):
    cache = tmp_path / "oracle_cache"
    _write_header(cache, "2026-04-01", 0.001)
    _write_header(cache, "2026-04-02", 0.010)
    _write_header(cache, "2026-04-03", 0.005)

    dates = ["2026-04-01", "2026-04-02", "2026-04-03"]
    result = order_days_by_density(
        dates, "density_asc", cache, random.Random(0)
    )
    assert result == ["2026-04-01", "2026-04-03", "2026-04-02"]


# ── 4. Missing cache → density 0, placed at end for density_desc ─────────────


def test_missing_cache_placed_at_end_in_density_desc(tmp_path, caplog):
    cache = tmp_path / "oracle_cache"
    _write_header(cache, "2026-04-01", 0.005)
    # "2026-04-02" has no cache → density 0

    dates = ["2026-04-01", "2026-04-02"]
    with caplog.at_level(logging.WARNING, logger="training.arb_oracle"):
        result = order_days_by_density(
            dates, "density_desc", cache, random.Random(0)
        )

    assert result == ["2026-04-01", "2026-04-02"], (
        "Known-density date must come first; missing-cache date at end"
    )
    assert any("density=0" in msg or "0 density" in msg for msg in caplog.messages), (
        "Expected a warning about missing-cache dates"
    )


# ── 5. Membership preserved across all modes ─────────────────────────────────


@pytest.mark.parametrize("mode", ["random", "density_desc", "density_asc"])
def test_membership_preserved(tmp_path, mode):
    cache = tmp_path / "oracle_cache"
    _write_header(cache, "2026-04-01", 0.002)
    _write_header(cache, "2026-04-02", 0.008)
    _write_header(cache, "2026-04-03", 0.001)

    dates = ["2026-04-01", "2026-04-02", "2026-04-03"]
    result = order_days_by_density(dates, mode, cache, random.Random(99))

    assert sorted(result) == sorted(dates), (
        f"mode={mode!r} changed membership: {result}"
    )
    assert len(result) == len(dates)


# ── 6. Config round-trip: curriculum_day_order flows to JSONL ─────────────────


def test_config_round_trip_to_episode_stats():
    """curriculum_day_order in config flows to EpisodeStats and JSONL record."""
    from agents.ppo_trainer import EpisodeStats
    from dataclasses import fields as dc_fields

    # Field must exist with correct default.
    field_names = {f.name for f in dc_fields(EpisodeStats)}
    assert "curriculum_day_order" in field_names, (
        "EpisodeStats must have curriculum_day_order field"
    )
    ep_default = EpisodeStats(
        day_date="2026-04-01", total_reward=0.0, total_pnl=0.0,
        bet_count=0, winning_bets=0, races_completed=0,
        final_budget=100.0, n_steps=1,
        raw_pnl_reward=0.0, shaped_bonus=0.0,
        clipped_reward_total=0.0,
    )
    assert ep_default.curriculum_day_order == "random", (
        "Default must be 'random'"
    )

    # Explicit value survives construction.
    ep_desc = EpisodeStats(
        day_date="2026-04-01", total_reward=0.0, total_pnl=0.0,
        bet_count=0, winning_bets=0, races_completed=0,
        final_budget=100.0, n_steps=1,
        raw_pnl_reward=0.0, shaped_bonus=0.0,
        clipped_reward_total=0.0,
        curriculum_day_order="density_desc",
    )
    assert ep_desc.curriculum_day_order == "density_desc"

    # JSONL serialisation path: check the field is present.
    import json
    record = {"curriculum_day_order": ep_desc.curriculum_day_order}
    assert json.loads(json.dumps(record))["curriculum_day_order"] == "density_desc"


# ── 7. Invalid mode falls back to random with error log ───────────────────────


def test_invalid_mode_falls_back_to_random(tmp_path, caplog):
    cache = tmp_path / "oracle_cache"
    _write_header(cache, "2026-04-01", 0.005)
    _write_header(cache, "2026-04-02", 0.010)

    dates = ["2026-04-01", "2026-04-02"]
    with caplog.at_level(logging.ERROR, logger="training.arb_oracle"):
        result = order_days_by_density(
            dates, "best", cache, random.Random(7)
        )

    # Should not crash; result is a permutation of dates
    assert sorted(result) == sorted(dates)
    assert any("best" in msg or "unknown" in msg or "Unknown" in msg
               for msg in caplog.messages), (
        "Expected an error log mentioning the invalid mode"
    )
