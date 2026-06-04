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
