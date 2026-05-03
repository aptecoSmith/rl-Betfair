"""Tests for ``training_v2.discrete_ppo.train`` multi-day loop.

Phase 3, Session 02 deliverable. The tests cover the loop logic
(day selection, deterministic shuffle, per-day env rebind, hidden-
state reset between days). They do NOT cover the gradient pathway —
Session 01's parity test plus Phase 2's ``tests/test_discrete_ppo_*``
suite already cover that.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from env.betfair_env import BetfairEnv
from agents_v2.discrete_policy import DiscreteLSTMPolicy
from agents_v2.env_shim import DiscreteActionShim
from training_v2.discrete_ppo.rollout import RolloutCollector
from training_v2.discrete_ppo.trainer import DiscretePPOTrainer
from training_v2.discrete_ppo import train as train_mod
from tests.test_betfair_env import _make_day


REPO_ROOT = Path(__file__).resolve().parents[1]
SCORER_DIR = REPO_ROOT / "models" / "scorer_v1"


def _scorer_runtime_available() -> tuple[bool, str]:
    if not (SCORER_DIR / "model.lgb").exists():
        return False, (
            f"Scorer artefacts missing under {SCORER_DIR}; "
            "run `python -m training_v2.scorer.train_and_evaluate` first."
        )
    try:
        import lightgbm  # noqa: F401
    except Exception as exc:
        return False, f"lightgbm not importable: {exc!r}"
    try:
        import joblib  # noqa: F401
    except Exception as exc:
        return False, f"joblib not importable: {exc!r}"
    return True, ""


_runtime_ok, _runtime_reason = _scorer_runtime_available()


# ── Pure helpers (no scorer / no env required) ─────────────────────────────


def test_enumerate_day_files_filters_and_sorts(tmp_path: Path) -> None:
    """Only ``YYYY-MM-DD.parquet`` matches; result is lexicographic."""
    # Create the parquet day-files plus some noise the matcher must skip.
    for name in [
        "2026-04-23.parquet",
        "2026-04-21.parquet",
        "2026-04-22.parquet",
        # Noise — must be excluded.
        "2026-04-23_runners.parquet",
        "2026-04-23.parquet.bak",
        "schema.json",
        "README.md",
    ]:
        (tmp_path / name).write_bytes(b"")

    found = train_mod._enumerate_day_files(tmp_path)
    assert found == ["2026-04-21", "2026-04-22", "2026-04-23"]


def test_enumerate_day_files_missing_dir_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        train_mod._enumerate_day_files(missing)


def test_enumerate_day_files_excludes_days_without_winner_data(
    tmp_path: Path,
) -> None:
    """Phase 3 follow-on no-betting-collapse Session 01 (2026-04-30):
    days whose parquet has zero markets with a populated
    ``winner_selection_id`` cause every race to void in the env, so
    they must not be selected for training or evaluation. The
    filter drops them with a warning.

    The 2026-04-29 incident: that day's parquet had 2 markets and
    0 winners populated, producing a Bar-6c FAIL on the AMBER
    cohort regardless of policy."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    def _write_parquet(name: str, winners: list) -> None:
        # Two markets per day (mirrors the 2026-04-29 shape).
        table = pa.table({
            "market_id": ["1.1", "1.1", "1.2", "1.2"],
            "winner_selection_id": pa.array(
                winners, type=pa.float64(),
            ),
        })
        pq.write_table(table, tmp_path / name)

    # Good day — both markets carry a winner.
    _write_parquet("2026-04-28.parquet", [101.0, 101.0, 202.0, 202.0])
    # Bad day — no winners (the 2026-04-29 case).
    _write_parquet("2026-04-29.parquet", [None, None, None, None])
    # Edge case — one market has a winner, one does not. Kept
    # (we only filter the all-null case; partial coverage is the
    # data pipeline's responsibility to surface separately).
    _write_parquet("2026-04-30.parquet", [101.0, 101.0, None, None])

    found = train_mod._enumerate_day_files(tmp_path)
    assert "2026-04-28" in found
    assert "2026-04-29" not in found
    assert "2026-04-30" in found


def test_day_has_any_winner_data_falls_back_to_true_on_unreadable(
    tmp_path: Path,
) -> None:
    """Legacy / pre-column / corrupt parquets are kept (returning
    True from the helper) — the filter is defensive, not strict.
    Day-load downstream will surface real corruption with a clearer
    error than the enumerator can give."""
    bad = tmp_path / "2026-01-01.parquet"
    bad.write_bytes(b"not a parquet")
    assert train_mod._day_has_any_winner_data(bad) is True


def test_select_days_holds_out_last_and_shuffles_rest(tmp_path: Path) -> None:
    """Training = N-1 deterministically-shuffled; eval = most recent."""
    # 8 days; ``--days 5`` → take last 5, hold out 2026-04-25 as eval,
    # shuffle the remaining 4 deterministically.
    for d in [
        "2026-04-18", "2026-04-19", "2026-04-20", "2026-04-21",
        "2026-04-22", "2026-04-23", "2026-04-24", "2026-04-25",
    ]:
        (tmp_path / f"{d}.parquet").write_bytes(b"")

    training, eval_day = train_mod.select_days(
        data_dir=tmp_path, n_days=5, day_shuffle_seed=42,
    )
    assert eval_day == "2026-04-25", "most recent date must be held out"
    assert len(training) == 4, "training set is N-1 days"
    # Eval day never appears in training.
    assert eval_day not in training
    # The four training candidates were 2026-04-21..04-24 — not the older
    # files (which are out of the most-recent-N window).
    assert sorted(training) == [
        "2026-04-21", "2026-04-22", "2026-04-23", "2026-04-24",
    ]
    # Determinism: same seed → same order.
    training_again, _ = train_mod.select_days(
        data_dir=tmp_path, n_days=5, day_shuffle_seed=42,
    )
    assert training == training_again
    # Different seed → at least one different ordering across the
    # 4! = 24 permutations (sanity check that the seed actually drives
    # the shuffle rather than no-op).
    seen = {tuple(training)}
    for s in range(1, 50):
        t, _ = train_mod.select_days(
            data_dir=tmp_path, n_days=5, day_shuffle_seed=s,
        )
        seen.add(tuple(t))
    assert len(seen) > 1, (
        "shuffle is not seed-driven — every seed produced the same order"
    )


def test_select_days_rejects_n_under_2(tmp_path: Path) -> None:
    (tmp_path / "2026-04-25.parquet").write_bytes(b"")
    with pytest.raises(ValueError):
        train_mod.select_days(
            data_dir=tmp_path, n_days=1, day_shuffle_seed=0,
        )


def test_select_days_rejects_insufficient_data(tmp_path: Path) -> None:
    for d in ["2026-04-23", "2026-04-24"]:
        (tmp_path / f"{d}.parquet").write_bytes(b"")
    with pytest.raises(RuntimeError):
        train_mod.select_days(
            data_dir=tmp_path, n_days=5, day_shuffle_seed=0,
        )


# ── Loop integration: day-selection drives env construction order ──────────


def test_multi_day_loop_uses_each_day_once_in_shuffled_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loop calls load_day for each training day exactly once.

    Eval day is held out (never loaded). Order matches the
    deterministic shuffle.

    PPO is short-circuited: ``train_episode`` returns a stub
    EpisodeStats so the test runs in milliseconds.
    """
    # Six dates; ``--days 4`` → eval = last, shuffle 3 training days.
    dates = [
        "2026-04-20", "2026-04-21", "2026-04-22",
        "2026-04-23", "2026-04-24", "2026-04-25",
    ]
    for d in dates:
        (tmp_path / f"{d}.parquet").write_bytes(b"")

    expected_training, expected_eval = train_mod.select_days(
        data_dir=tmp_path, n_days=4, day_shuffle_seed=7,
    )
    assert expected_eval == "2026-04-25"

    # Track every load_day call (the inner _build_env_for_day uses
    # train_mod.load_day, which we monkey-patch directly on the module).
    load_day_calls: list[str] = []

    def _fake_load_day(date_str, data_dir):
        load_day_calls.append(date_str)
        return object()  # opaque stub Day; never inspected

    monkeypatch.setattr(train_mod, "load_day", _fake_load_day)

    # Stub BetfairEnv + DiscreteActionShim + DiscreteLSTMPolicy +
    # DiscretePPOTrainer so nothing real runs. We only care about which
    # dates load_day saw and that the trainer is asked for the right
    # number of episodes per day.
    class _StubShim:
        def __init__(self, *_a, **_kw) -> None:
            self.obs_dim = 4
            self.max_runners = 2

            class _AS:
                n = 5
            self.action_space = _AS()

    class _StubEnv:
        def __init__(self, *_a, **_kw) -> None: ...

    class _StubPolicy:
        def __init__(self, *_a, **_kw) -> None: ...

        def to(self, *_a, **_kw) -> "_StubPolicy":
            return self

        def parameters(self):  # noqa: D401 — torch ducktype
            return iter([torch.zeros(1, requires_grad=True)])

    train_episode_calls: list[tuple[int, int]] = []
    rebind_calls: list[int] = []

    class _StubTrainer:
        def __init__(self, *, policy, shim, device, **_kw) -> None:
            self.policy = policy
            self.shim = shim
            self.device = torch.device(device)
            self._collector = object()
            self.action_space = shim.action_space
            self.max_runners = shim.max_runners
            self._rebind_count = 0

        def train_episode(self):
            train_episode_calls.append(
                (self._rebind_count, len(train_episode_calls)),
            )
            return train_mod.EpisodeStats(
                total_reward=0.0,
                n_steps=1,
                n_updates_run=0,
                policy_loss_mean=0.0,
                value_loss_mean=1.0,
                entropy_mean=0.0,
                approx_kl_mean=0.0,
                approx_kl_max=0.0,
                mini_batches_skipped=0,
                kl_early_stopped=False,
                wall_time_sec=0.0,
                action_histogram={"NOOP": 1},
                advantage_mean=0.0,
                advantage_std=0.0,
                advantage_max_abs=0.0,
                day_pnl=0.0,
            )

    monkeypatch.setattr(train_mod, "BetfairEnv", _StubEnv)
    monkeypatch.setattr(train_mod, "DiscreteActionShim", _StubShim)
    monkeypatch.setattr(train_mod, "DiscreteLSTMPolicy", _StubPolicy)
    monkeypatch.setattr(train_mod, "DiscretePPOTrainer", _StubTrainer)

    # Spy on the rebind helper so we can confirm it fires once per
    # day after the first.
    real_rebind = train_mod._rebind_trainer_for_day

    def _spy_rebind(trainer, shim):
        rebind_calls.append(len(load_day_calls))
        trainer._rebind_count += 1
        # Simulate the real rebind — keep the trainer self-consistent
        # in case downstream code in main() reads any of these.
        trainer.shim = shim
        trainer.action_space = shim.action_space
        trainer.max_runners = shim.max_runners

    monkeypatch.setattr(train_mod, "_rebind_trainer_for_day", _spy_rebind)

    out_path = tmp_path / "out.jsonl"
    rc = train_mod.main(
        days=4,
        data_dir=tmp_path,
        epochs_per_day=2,
        seed=11,
        day_shuffle_seed=7,
        out_path=out_path,
        scorer_dir=tmp_path,  # never read by the stub shim
        device="cpu",
    )
    assert rc == 0

    # 1. load_day was called once per training day, in the shuffled
    #    order, and never for the held-out eval day.
    assert load_day_calls == expected_training
    assert expected_eval not in load_day_calls

    # 2. train_episode fired epochs_per_day × n_training_days times.
    assert len(train_episode_calls) == 2 * len(expected_training)

    # 3. Rebind happened once per day after the first (3 days → 2
    #    rebinds; the first day uses the trainer's construction-time
    #    shim).
    assert len(rebind_calls) == len(expected_training) - 1

    # 4. JSONL rows match the train_episode count and carry day_idx /
    #    epoch_idx / cumulative_episode_idx.
    rows = [
        __import__("json").loads(line)
        for line in out_path.read_text().splitlines()
    ]
    assert len(rows) == 2 * len(expected_training)
    expected_cum = 0
    for day_idx, day_str in enumerate(expected_training):
        for epoch in range(2):
            row = rows[expected_cum]
            assert row["day_str"] == day_str
            assert row["day_idx"] == day_idx
            assert row["epoch_idx"] == epoch
            assert row["cumulative_episode_idx"] == expected_cum
            assert row["episode_idx"] == expected_cum  # backward compat
            expected_cum += 1


def test_main_rejects_both_modes_set() -> None:
    """``day_str`` and ``days`` are mutually exclusive."""
    with pytest.raises(ValueError):
        train_mod.main(
            day_str="2026-04-23",
            days=7,
            data_dir=Path("."),
        )


def test_main_rejects_neither_mode_set() -> None:
    with pytest.raises(ValueError):
        train_mod.main(day_str=None, days=None, data_dir=Path("."))


# ── End-to-end: hidden state reset between days ────────────────────────────


pytestmark_realdata = pytest.mark.skipif(
    not _runtime_ok, reason=_runtime_reason,
)


def _scalping_config(max_runners: int = 4) -> dict:
    return {
        "training": {
            "max_runners": max_runners,
            "starting_budget": 100.0,
            "max_bets_per_race": 20,
            "scalping_mode": True,
            "betting_constraints": {
                "max_back_price": 50.0,
                "max_lay_price": None,
                "min_seconds_before_off": 0,
                "force_close_before_off_seconds": 0,
            },
        },
        "actions": {"force_aggressive": True},
        "reward": {
            "early_pick_bonus_min": 1.2,
            "early_pick_bonus_max": 1.5,
            "early_pick_min_seconds": 300,
            "efficiency_penalty": 0.01,
            "commission": 0.05,
            "mark_to_market_weight": 0.0,
        },
    }


def _build_shim(seed: int, n_races: int = 2) -> DiscreteActionShim:
    torch.manual_seed(seed)
    env = BetfairEnv(
        _make_day(n_races=n_races, n_pre_ticks=5, n_inplay_ticks=2),
        _scalping_config(),
    )
    return DiscreteActionShim(env)


@pytest.mark.slow
@pytest.mark.timeout(120)
@pytestmark_realdata
def test_episode_boundary_is_day_boundary_hidden_reset() -> None:
    """Day 2's first rollout sees zero hidden state.

    The contract: ``RolloutCollector.collect_episode`` calls
    ``policy.init_hidden(batch=1)`` at the start of every rollout
    (Phase 1 hidden-state protocol). Per-day rebind must NOT carry
    hidden state across days. Run two synthetic days, capture the
    first transition's ``hidden_state_in`` from each, and assert
    both are zero.
    """
    shim_day1 = _build_shim(seed=0)
    policy = DiscreteLSTMPolicy(
        obs_dim=shim_day1.obs_dim,
        action_space=shim_day1.action_space,
        hidden_size=32,
    )
    trainer = DiscretePPOTrainer(
        policy=policy, shim=shim_day1,
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        entropy_coeff=0.01, value_coeff=0.5, ppo_epochs=1,
        mini_batch_size=64, max_grad_norm=0.5, device="cpu",
    )

    # Day 1: rollout via the trainer's own collector — first
    # transition's hidden_state_in must be zeros.
    from training_v2.discrete_ppo.transition import (
        rollout_batch_to_transitions,
    )
    transitions_day1 = rollout_batch_to_transitions(
        trainer._collector.collect_episode()
    )
    assert len(transitions_day1) > 0
    h0_d1, c0_d1 = transitions_day1[0].hidden_state_in
    assert torch.equal(h0_d1, torch.zeros_like(h0_d1))
    assert torch.equal(c0_d1, torch.zeros_like(c0_d1))

    # Confirm SOME later transition has non-zero hidden state — proves
    # the LSTM is statefully accumulating within the day (otherwise the
    # day-boundary test below would be vacuous).
    later_nonzero = any(
        float(tr.hidden_state_in[0].abs().max().item()) > 0.0
        for tr in transitions_day1[1:]
    )
    assert later_nonzero

    # Day 2: rebind to a NEW shim; assert the trainer holds a fresh
    # collector instance and its first transition's hidden state is
    # again zero (the per-day reset the multi-day loop relies on).
    shim_day2 = _build_shim(seed=1)
    old_collector_id = id(trainer._collector)
    train_mod._rebind_trainer_for_day(trainer, shim_day2)

    assert id(trainer._collector) != old_collector_id, (
        "rebind must construct a fresh RolloutCollector — otherwise the "
        "Phase 1 hidden-state protocol's per-rollout init_hidden won't "
        "fire on day 2."
    )
    assert trainer.shim is shim_day2
    assert isinstance(trainer._collector, RolloutCollector)

    transitions_day2 = rollout_batch_to_transitions(
        trainer._collector.collect_episode()
    )
    assert len(transitions_day2) > 0
    h0_d2, c0_d2 = transitions_day2[0].hidden_state_in
    assert torch.equal(h0_d2, torch.zeros_like(h0_d2)), (
        "day 2's first transition should have zero hidden state — the "
        "per-day rebind must reset the LSTM hidden, not carry it from "
        "day 1's terminal state."
    )
    assert torch.equal(c0_d2, torch.zeros_like(c0_d2))
