"""Train-time column + ``_fmt_hms`` rendering (pbt-breeding, 2026-06-04).

The operator asked the PBT leaderboard to show how long each agent took to
train (it exposes the speed cost of bigger LSTMs / transformers). The wall
clock is already measured by ``train_one_agent`` (TrainSummary.wall_time_sec);
these guard the render + the old-row tolerance.
"""

from __future__ import annotations

import json
from pathlib import Path

from tools.pbt_leaderboard import (
    _fmt_hms,
    build_leaderboard_text,
    infer_top_tier,
    regenerate,
)


class TestFmtHms:
    def test_zero_or_missing_is_dash(self):
        assert _fmt_hms(0) == "-"
        assert _fmt_hms(-5) == "-"
        assert _fmt_hms(0.3) == "-"  # rounds to 0 -> dash, not "0:00"

    def test_sub_hour_is_m_ss(self):
        assert _fmt_hms(5) == "0:05"
        assert _fmt_hms(65) == "1:05"
        assert _fmt_hms(300) == "5:00"

    def test_over_hour_is_h_mm_ss(self):
        assert _fmt_hms(3600) == "1:00:00"
        assert _fmt_hms(4440) == "1:14:00"  # the big-transformer straggler case


class TestLeaderboardTrainColumn:
    def _row(self, secs):
        return {
            "model_id": "abc12345", "architecture": "lstm", "hidden_size": 256,
            "locked_pnl": 10.0, "naked_pnl": -5.0, "train_seconds": secs,
            "genes": {"learning_rate": 1e-4, "entropy_coeff": 0.01},
        }

    def test_header_and_value_render(self):
        txt = build_leaderboard_text([self._row(4440)], "run", frozen=True)
        assert "train" in txt        # column header
        assert "1:14:00" in txt      # rendered value

    def test_missing_train_seconds_renders_without_crash(self):
        # Old rows (pre-2026-06-04) have no train_seconds -> tolerant "-".
        r = self._row(0.0)
        del r["train_seconds"]
        txt = build_leaderboard_text([r], "run", frozen=True)
        assert "train" in txt and "abc12345" in txt

    def test_train_column_present_on_live_tier_leaderboards_too(self):
        # frozen=False is the R1/R2 path (no frozen_at) — train col still shown.
        txt = build_leaderboard_text(
            [self._row(125)], "run", frozen=False, tier_label="R1 TIER")
        assert "train" in txt and "2:05" in txt
        assert "frozen_at" not in txt


class TestTrainedAtColumn:
    """`trained_at` (when the agent trained) shows on R1/R2 so the operator can
    tell which agents are the latest across the multi-day campaign (gen resets
    per run). R3 keeps frozen_at instead (== trained_at for a champion)."""

    def _row(self, trained_at="2026-06-04T13:25:00+00:00", frozen_at=""):
        return {
            "model_id": "abc12345", "architecture": "lstm", "hidden_size": 256,
            "locked_pnl": 10.0, "naked_pnl": -5.0, "train_seconds": 300,
            "trained_at": trained_at, "frozen_at": frozen_at,
            "genes": {"learning_rate": 1e-4, "entropy_coeff": 0.01},
        }

    def test_r1_r2_show_trained_at_not_frozen_at(self):
        txt = build_leaderboard_text(
            [self._row()], "run", frozen=False, tier_label="R1 TIER")
        assert "trained_at" in txt
        assert "2026-06-04T13:25:00" in txt   # rendered value
        assert "frozen_at" not in txt

    def test_r3_shows_frozen_at_not_redundant_trained_at(self):
        txt = build_leaderboard_text(
            [self._row(frozen_at="2026-06-04T14:00:00+00:00")],
            "run", frozen=True)
        assert "frozen_at" in txt and "2026-06-04T14:00:00" in txt
        assert "trained_at" not in txt   # dropped on R3 (redundant)

    def test_missing_trained_at_renders_blank_not_crash(self):
        r = self._row()
        del r["trained_at"]
        txt = build_leaderboard_text([r], "run", frozen=False)
        assert "trained_at" in txt and "abc12345" in txt   # header + row OK


class TestRegenerateNTier:
    """rotation-rework: regenerate writes a board for every non-top tier
    R1..R(N-1) and labels the hall-of-fame R{N}; 3-tier stays R1+R2 + R3 hall."""

    def _row(self, tier, mid, **kw):
        r = {"model_id": mid, "generation": tier - 1, "tier": tier,
             "architecture": "lstm", "hidden_size": 128,
             "locked_pnl": 10.0 - tier, "naked_pnl": -5.0,
             "train_seconds": 100, "trained_at": "2026-06-07T10:00:00+00:00",
             "genes": {"learning_rate": 1e-4, "entropy_coeff": 0.01}}
        r.update(kw)
        return r

    def _write(self, d: Path, lineage, hall):
        (d / "pbt_lineage.jsonl").write_text(
            "\n".join(json.dumps(r) for r in lineage), encoding="utf-8")
        (d / "pbt_hall_of_fame.jsonl").write_text(
            "\n".join(json.dumps(r) for r in hall), encoding="utf-8")

    def test_four_tier_writes_r1_r2_r3_and_r4_hall(self, tmp_path: Path):
        lineage = [self._row(t, f"m{t}") for t in (1, 2, 3, 4)]
        hall = [self._row(4, "m4", frozen_at="2026-06-07T11:00:00+00:00")]
        self._write(tmp_path, lineage, hall)
        regenerate(tmp_path, n_tiers=4)
        lb = (tmp_path / "leaderboard.txt").read_text(encoding="utf-8")
        assert "R4 HALL-OF-FAME" in lb
        for t in (1, 2, 3):
            assert (tmp_path / f"leaderboard_r{t}.txt").exists()
            assert f"R{t} TIER" in (tmp_path / f"leaderboard_r{t}.txt").read_text(
                encoding="utf-8")
        # R4 is the hall-of-fame tier — no separate R4 board.
        assert not (tmp_path / "leaderboard_r4.txt").exists()

    def test_three_tier_unchanged(self, tmp_path: Path):
        lineage = [self._row(t, f"m{t}") for t in (1, 2, 3)]
        hall = [self._row(3, "m3", frozen_at="2026-06-07T11:00:00+00:00")]
        self._write(tmp_path, lineage, hall)
        regenerate(tmp_path, n_tiers=3)
        assert "R3 HALL-OF-FAME" in (tmp_path / "leaderboard.txt").read_text(
            encoding="utf-8")
        assert (tmp_path / "leaderboard_r1.txt").exists()
        assert (tmp_path / "leaderboard_r2.txt").exists()
        assert not (tmp_path / "leaderboard_r3.txt").exists()

    def test_infer_top_tier_precedence(self):
        # explicit wins
        assert infer_top_tier([{"tier": 2}], [], 4) == 4
        # recorded on rows next
        assert infer_top_tier([{"tier": 2, "n_tiers": 5}], [], None) == 5
        # else max observed, floored at 3 (legacy)
        assert infer_top_tier([{"tier": 1}, {"tier": 2}], [], None) == 3
        assert infer_top_tier([{"tier": 4}], [{"tier": 4}], None) == 4
