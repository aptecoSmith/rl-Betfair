"""Tests for ``select_days(exclude_days=...)`` — scalping-locked-fitness-and-age-obs Phase 0.

The plan adds ``exclude_days`` to ``select_days`` so callers can drop
held-out evaluation dates from the candidate pool BEFORE the
most-recent-N slice. This keeps held-out dates out of the training
pool even when ``--n-days`` extends past the natural leak boundary.
"""

from __future__ import annotations

from pathlib import Path

from training_v2.discrete_ppo import train as train_mod


def _touch_days(tmp_path: Path, dates: list[str]) -> None:
    for d in dates:
        (tmp_path / f"{d}.parquet").write_bytes(b"")


def test_exclude_days_removes_from_pool(tmp_path: Path) -> None:
    """exclude_days drops the listed dates BEFORE the last-N slice.

    Pool: 2026-04-26..2026-05-02 (7 days). Excluding 2026-04-30 +
    2026-05-01 leaves 5 days. Asking for last-4 gives the last-4 of
    THAT filtered pool — none of which contain the excluded dates.
    """
    _touch_days(
        tmp_path,
        [
            "2026-04-26", "2026-04-27", "2026-04-28", "2026-04-29",
            "2026-04-30", "2026-05-01", "2026-05-02",
        ],
    )

    training, eval_days = train_mod.select_days(
        data_dir=tmp_path,
        n_days=4,
        day_shuffle_seed=42,
        n_eval_days=2,
        exclude_days=["2026-04-30", "2026-05-01"],
    )

    selected = set(training) | set(eval_days)
    assert "2026-04-30" not in selected
    assert "2026-05-01" not in selected
    # After excluding, the filtered pool is
    # ['2026-04-26','27','28','29','2026-05-02']; last-4 =
    # ['27','28','29','2026-05-02']; eval = last 2; training = first 2.
    assert eval_days == ["2026-04-29", "2026-05-02"]
    assert sorted(training) == ["2026-04-27", "2026-04-28"]


def test_exclude_days_empty_byte_identical(tmp_path: Path) -> None:
    """exclude_days=None and exclude_days=[] both reproduce pre-flag output."""
    _touch_days(
        tmp_path,
        [
            "2026-04-22", "2026-04-23", "2026-04-24", "2026-04-25",
            "2026-04-26",
        ],
    )

    baseline_training, baseline_eval = train_mod.select_days(
        data_dir=tmp_path, n_days=4, day_shuffle_seed=7, n_eval_days=2,
    )
    none_training, none_eval = train_mod.select_days(
        data_dir=tmp_path, n_days=4, day_shuffle_seed=7, n_eval_days=2,
        exclude_days=None,
    )
    empty_training, empty_eval = train_mod.select_days(
        data_dir=tmp_path, n_days=4, day_shuffle_seed=7, n_eval_days=2,
        exclude_days=[],
    )
    assert (none_training, none_eval) == (baseline_training, baseline_eval)
    assert (empty_training, empty_eval) == (baseline_training, baseline_eval)


def test_exclude_days_works_with_n_days_above_leak_boundary(
    tmp_path: Path,
) -> None:
    """The plan's load-bearing scenario.

    Pool spans 2026-04-06 → 2026-05-13 (36 days). Held-out is
    2026-04-28/29/30 — falling inside the last-14-to-last-16 window.
    With exclude_days the plan can safely request ``n_days=20`` or
    even ``n_days=30`` and never see a held-out day in the training
    or in-sample-eval slice.
    """
    pool = [
        # 2026-04-06..05-13 inclusive (38 days minus weekends/etc).
        # Simulate the real pool's 36-day length explicitly.
        f"2026-04-{day:02d}" for day in range(6, 31)
    ] + [
        f"2026-05-{day:02d}" for day in range(1, 14)
    ]
    assert len(pool) == 38  # 25 April-days + 13 May-days
    _touch_days(tmp_path, pool)

    held_out = ["2026-04-28", "2026-04-29", "2026-04-30"]

    for n_days in (20, 25, 30):
        training, eval_days = train_mod.select_days(
            data_dir=tmp_path,
            n_days=n_days,
            day_shuffle_seed=42,
            n_eval_days=n_days // 2,
            exclude_days=held_out,
        )
        selected = set(training) | set(eval_days)
        for h in held_out:
            assert h not in selected, (
                f"n_days={n_days} leaked {h} into training+eval despite "
                "exclude_days"
            )
        # Sanity: filtered pool size is 38-3=35; last-n_days is selected.
        assert len(training) + len(eval_days) == n_days
