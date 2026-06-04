"""Train-time column + ``_fmt_hms`` rendering (pbt-breeding, 2026-06-04).

The operator asked the PBT leaderboard to show how long each agent took to
train (it exposes the speed cost of bigger LSTMs / transformers). The wall
clock is already measured by ``train_one_agent`` (TrainSummary.wall_time_sec);
these guard the render + the old-row tolerance.
"""

from __future__ import annotations

from tools.pbt_leaderboard import _fmt_hms, build_leaderboard_text


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
