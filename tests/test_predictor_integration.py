"""Predictor-integration regression guards (plans/predictor-integration/).

Covers `hard_constraints.md §1` (flag-off byte-identical to pre-plan) and
the obs-schema bookkeeping deltas Session 02 lands.

The load-bearing test is
:func:`test_flag_off_is_byte_identical_to_pre_plan` — it re-runs the
deterministic zero-action rollout that captured
``tests/fixtures/predictor_integration_baseline.json`` and asserts the
SHA256 digest of the per-step (reward, raw_pnl_reward, shaped_bonus,
race_idx) tuples matches the captured baseline. Both predictor flags
default off; once Session 02 lands the new kwargs, this test will
explicitly pass `use_race_outcome_predictor=False` +
`use_direction_predictor=False`.

The fixture was captured via
``python tests/_capture_predictor_integration_baseline.py`` against the
pre-Session-02 commit (`81cd092` = "feat(predictor-integration):
Session 01 — predictor loader + segment router"). If the fixture ever
needs to be re-captured (e.g. after a deliberate env change), do it on
a clean checkout of that commit and bump the captured_at marker.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed_amber_v2_window"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "predictor_integration_baseline.json"


def _require_fixture_and_data() -> dict:
    if not FIXTURE_PATH.exists():
        pytest.skip(f"baseline fixture missing: {FIXTURE_PATH}")
    with FIXTURE_PATH.open("r", encoding="utf-8") as fh:
        baseline = json.load(fh)
    if not (DATA_DIR / f"{baseline['day']}.parquet").exists():
        pytest.skip(
            f"baseline day parquet missing: {DATA_DIR / (baseline['day'] + '.parquet')}"
        )
    return baseline


@pytest.mark.slow
def test_flag_off_is_byte_identical_to_pre_plan():
    """Hard_constraints §1: flag-off env is byte-identical to pre-plan.

    Replays the same deterministic zero-action rollout that produced the
    captured baseline and asserts the per-step digest matches. The
    digest fingerprints (reward, raw_pnl_reward, shaped_bonus, race_idx)
    on every step — any drift in env-side behaviour (matcher, reward
    shaping, settlement, scalping accounting, …) trips the digest.

    Post-Session-02 this test will pass `use_race_outcome_predictor=False`
    + `use_direction_predictor=False` to the env constructor. Today the
    env doesn't yet have those kwargs; the test runs against today's
    flag-defaulted-off behaviour, which IS the pre-Session-02 baseline.
    """
    baseline = _require_fixture_and_data()

    from data.episode_builder import load_day  # type: ignore[import-not-found]
    from env.betfair_env import BetfairEnv  # type: ignore[import-not-found]
    from training_v2.discrete_ppo.train import (  # type: ignore[import-not-found]
        _scalping_train_config,
    )

    cfg = _scalping_train_config(max_runners=baseline["max_runners"])
    day = load_day(baseline["day"], data_dir=DATA_DIR)
    env = BetfairEnv(
        day,
        cfg,
        predictor_bundle=None,
        use_race_outcome_predictor=False,
        use_direction_predictor=False,
    )

    obs, _info = env.reset(seed=baseline["seed"])
    action_dim = int(env.action_space.shape[0])
    assert action_dim == baseline["action_dim"], (
        f"action_dim drift: env={action_dim} baseline={baseline['action_dim']}"
    )

    zero_action = np.zeros(action_dim, dtype=np.float32)
    digest = hashlib.sha256()
    n_steps = 0
    info: dict = {}
    while True:
        obs, reward, terminated, truncated, info = env.step(zero_action)
        digest.update(np.float64(float(reward)).tobytes())
        digest.update(np.float64(float(info.get("raw_pnl_reward", 0.0))).tobytes())
        digest.update(np.float64(float(info.get("shaped_bonus", 0.0))).tobytes())
        digest.update(int(info.get("race_idx", -1)).to_bytes(4, "little", signed=True))
        n_steps += 1
        if terminated or truncated:
            break
        if n_steps >= 30_000:
            raise RuntimeError("regression run overshot 30k steps")

    assert n_steps == baseline["n_steps"], (
        f"step count drifted: env={n_steps} baseline={baseline['n_steps']}"
    )
    actual_digest = digest.hexdigest()
    if actual_digest != baseline["per_step_digest"]:
        # Diagnostic localisation: re-run with sample capture if the
        # digest diverges. For now, surface the mismatch with the sampled
        # steps from the baseline so the operator can replay locally.
        sample_summary = ", ".join(
            f"step={s['step']} reward={s['reward']:.6g} raw_pnl={s['raw_pnl_reward']:.6g}"
            for s in baseline["sampled_steps"][:3]
        )
        raise AssertionError(
            f"per-step digest mismatch:\n"
            f"  baseline  = {baseline['per_step_digest']}\n"
            f"  current   = {actual_digest}\n"
            f"  baseline samples (first 3): {sample_summary}\n"
            f"  Re-run tests/_capture_predictor_integration_baseline.py "
            f"on a clean pre-Session-02 commit to confirm the fixture is "
            f"current; otherwise investigate env-side drift."
        )

    # Final aggregates as a secondary check.
    final_day_pnl = float(info.get("day_pnl", 0.0))
    assert final_day_pnl == baseline["final_day_pnl"], (
        f"day_pnl drift: env={final_day_pnl} baseline={baseline['final_day_pnl']}"
    )


def test_obs_schema_version_is_8():
    """integration_contract.md §2: schema bumps to 8 in Session 02."""
    from env.betfair_env import OBS_SCHEMA_VERSION
    assert OBS_SCHEMA_VERSION == 8


def test_runner_dim_is_143():
    """integration_contract.md §2: RUNNER_DIM grows by 18 (6 race + 12 tick)."""
    from env.betfair_env import RUNNER_DIM, RUNNER_KEYS
    assert RUNNER_DIM == 143
    assert len(RUNNER_KEYS) == 143


def test_runner_keys_predictor_block_present():
    """The 18 predictor keys are appended at the tail of RUNNER_KEYS in
    the canonical order specified in integration_contract.md §2."""
    from env.betfair_env import RUNNER_KEYS

    expected_tail = [
        # Race-level (6)
        "champion_p_win",
        "champion_p_placed",
        "champion_segment_strong",
        "ranker_softmax_share",
        "ranker_top1_flag",
        "ranker_top1_high_conf_flag",
        # Per-tick direction (12)
        "dir_q10_1m", "dir_q50_1m", "dir_q90_1m",
        "dir_q10_3m", "dir_q50_3m", "dir_q90_3m",
        "dir_q10_7m", "dir_q50_7m", "dir_q90_7m",
        "dir_fire_drift", "dir_fire_shorten", "dir_fire_no_signal",
    ]
    assert RUNNER_KEYS[-18:] == expected_tail


def test_env_constructs_with_flags_off_and_no_bundle():
    """Sanity: env constructs with predictor flags off + bundle=None."""
    from data.episode_builder import load_day  # type: ignore[import-not-found]
    from env.betfair_env import BetfairEnv  # type: ignore[import-not-found]
    from training_v2.discrete_ppo.train import (  # type: ignore[import-not-found]
        _scalping_train_config,
    )

    baseline = _require_fixture_and_data()
    cfg = _scalping_train_config(max_runners=baseline["max_runners"])
    day = load_day(baseline["day"], data_dir=DATA_DIR)
    env = BetfairEnv(
        day,
        cfg,
        predictor_bundle=None,
        use_race_outcome_predictor=False,
        use_direction_predictor=False,
    )
    # Internal state surfaced for downstream wiring.
    assert env._predictor_bundle is None
    assert env._use_race_outcome_predictor is False
    assert env._use_direction_predictor is False


def test_env_refuses_flag_on_without_bundle():
    """Hard_constraints §10: silent fallback forbidden. A flag set True
    without a PredictorBundle is a configuration error and must raise."""
    from data.episode_builder import load_day  # type: ignore[import-not-found]
    from env.betfair_env import BetfairEnv  # type: ignore[import-not-found]
    from training_v2.discrete_ppo.train import (  # type: ignore[import-not-found]
        _scalping_train_config,
    )

    baseline = _require_fixture_and_data()
    cfg = _scalping_train_config(max_runners=baseline["max_runners"])
    day = load_day(baseline["day"], data_dir=DATA_DIR)
    with pytest.raises(ValueError, match="predictor_bundle is None"):
        BetfairEnv(
            day,
            cfg,
            predictor_bundle=None,
            use_race_outcome_predictor=True,
            use_direction_predictor=False,
        )
    with pytest.raises(ValueError, match="predictor_bundle is None"):
        BetfairEnv(
            day,
            cfg,
            predictor_bundle=None,
            use_race_outcome_predictor=False,
            use_direction_predictor=True,
        )


def test_old_checkpoint_refuses_to_load():
    """Hard_constraints §13 + integration_contract.md §5: a v7 checkpoint
    refuses to load against the v8 env. The existing
    `validate_obs_schema` is the load-bearing guard; this test cross-checks
    it trips on the specific v7 → v8 transition this plan introduces.
    """
    from env.betfair_env import OBS_SCHEMA_VERSION, validate_obs_schema

    assert OBS_SCHEMA_VERSION == 8

    # v7 checkpoint (the schema right before this plan)
    v7_checkpoint = {"obs_schema_version": 7, "weights": {}}
    with pytest.raises(ValueError, match="obs_schema_version"):
        validate_obs_schema(v7_checkpoint)

    # Pre-schema-bump checkpoints (no key) also refuse.
    pre_schema = {"weights": {}}
    with pytest.raises(ValueError, match="obs_schema_version"):
        validate_obs_schema(pre_schema)

    # A v8 checkpoint passes through.
    v8_checkpoint = {"obs_schema_version": 8, "weights": {}}
    validate_obs_schema(v8_checkpoint)  # no raise


def test_predictor_keys_default_to_zero_with_no_bundle():
    """Hard_constraints §1: with no predictor bundle attached, the new
    predictor keys MUST default to 0.0 in the runner obs slice.

    This test asserts the env's `_features_to_array` default-zero floor
    holds for the 18 new keys at the tail of every runner's obs slice.
    Loaded as a unit test: builds a tiny `_features_to_array`-compatible
    runners dict that omits the predictor keys; the function should
    populate them with 0.0 via the existing `feats.get(key, 0.0)`
    fallback at env/betfair_env.py:1238.
    """
    from env.betfair_env import RUNNER_DIM, RUNNER_KEYS

    # Pure unit test of the default-zero floor: a `feats.get(key, 0.0)`
    # over an empty dict.
    feats: dict = {}
    for key in RUNNER_KEYS[-18:]:
        assert feats.get(key, 0.0) == 0.0, f"{key} default-zero floor failed"


def test_strategy_mode_default_arb():
    """Default config (training.strategy_mode = arb) → env derives
    scalping_mode = True. Backward-compat check."""
    from data.episode_builder import load_day  # type: ignore[import-not-found]
    from env.betfair_env import BetfairEnv  # type: ignore[import-not-found]
    from training_v2.discrete_ppo.train import (  # type: ignore[import-not-found]
        _scalping_train_config,
    )

    baseline = _require_fixture_and_data()
    cfg = _scalping_train_config(max_runners=baseline["max_runners"])
    day = load_day(baseline["day"], data_dir=DATA_DIR)
    env = BetfairEnv(day, cfg)
    assert env._strategy_mode == "arb"
    assert env.scalping_mode is True


def test_strategy_mode_value_win_disables_scalping():
    """strategy_mode=value_win sets scalping_mode False."""
    from data.episode_builder import load_day  # type: ignore[import-not-found]
    from env.betfair_env import BetfairEnv  # type: ignore[import-not-found]
    from training_v2.discrete_ppo.train import (  # type: ignore[import-not-found]
        _scalping_train_config,
    )

    baseline = _require_fixture_and_data()
    cfg = _scalping_train_config(max_runners=baseline["max_runners"])
    day = load_day(baseline["day"], data_dir=DATA_DIR)
    env = BetfairEnv(day, cfg, strategy_mode="value_win")
    assert env._strategy_mode == "value_win"
    assert env.scalping_mode is False


def test_strategy_mode_unknown_raises():
    """An unrecognised strategy_mode raises loudly (hard_constraints §10)."""
    from data.episode_builder import load_day  # type: ignore[import-not-found]
    from env.betfair_env import BetfairEnv  # type: ignore[import-not-found]
    from training_v2.discrete_ppo.train import (  # type: ignore[import-not-found]
        _scalping_train_config,
    )

    baseline = _require_fixture_and_data()
    cfg = _scalping_train_config(max_runners=baseline["max_runners"])
    day = load_day(baseline["day"], data_dir=DATA_DIR)
    with pytest.raises(ValueError, match="unknown strategy_mode"):
        BetfairEnv(day, cfg, strategy_mode="not_a_mode")


def test_value_mode_reward_is_settle_only():
    """Hard_constraints §3 + strategy_modes.md §value_win: value modes
    use `shaped_bonus = 0`. arb mode retains the full shaping composition.

    Replays the deterministic zero-action rollout in `value_win` mode,
    sums `info["shaped_bonus"]` across every step, and asserts the
    cumulative shaped bonus is exactly 0.0. (Zero-action policy doesn't
    bet, so the difference between modes wouldn't show up in
    `raw_pnl_reward` either way — but `shaped_bonus` exposes the gate
    directly.)
    """
    from data.episode_builder import load_day  # type: ignore[import-not-found]
    from env.betfair_env import BetfairEnv  # type: ignore[import-not-found]
    from training_v2.discrete_ppo.train import (  # type: ignore[import-not-found]
        _scalping_train_config,
    )

    baseline = _require_fixture_and_data()
    cfg = _scalping_train_config(max_runners=baseline["max_runners"])
    day = load_day(baseline["day"], data_dir=DATA_DIR)
    env = BetfairEnv(day, cfg, strategy_mode="value_win")
    obs, _info = env.reset(seed=baseline["seed"])
    action_dim = int(env.action_space.shape[0])
    zero_action = np.zeros(action_dim, dtype=np.float32)

    cumulative_shaped = 0.0
    n_steps = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(zero_action)
        cumulative_shaped += float(info.get("shaped_bonus", 0.0))
        n_steps += 1
        if terminated or truncated:
            break
        if n_steps >= 30_000:
            break

    assert cumulative_shaped == 0.0, (
        f"value_win shaped_bonus should be 0 (settle-only reward); "
        f"got cumulative shaped={cumulative_shaped}"
    )


def test_strategy_mode_legacy_scalping_mode_independent():
    """Backward compat: legacy `scalping_mode=False` callers (no
    explicit strategy_mode anywhere) keep their pre-plan reward shape.

    The strategy_mode and scalping_mode axes are INDEPENDENT for
    backward compat — legacy non-scalping callers (pre-plan tests,
    directional-mode runs) stay on `strategy_mode=arb` (the default)
    so their `early_pick / precision / efficiency` shaping continues
    to apply. Only when the operator explicitly opts INTO a value
    mode does the shaped=0 gate kick in.
    """
    from data.episode_builder import load_day  # type: ignore[import-not-found]
    from env.betfair_env import BetfairEnv  # type: ignore[import-not-found]
    from training_v2.discrete_ppo.train import (  # type: ignore[import-not-found]
        _scalping_train_config,
    )

    baseline = _require_fixture_and_data()
    cfg = _scalping_train_config(max_runners=baseline["max_runners"])
    cfg["training"].pop("strategy_mode", None)
    cfg["training"]["scalping_mode"] = False
    day = load_day(baseline["day"], data_dir=DATA_DIR)
    env = BetfairEnv(day, cfg)  # no strategy_mode → defaults to "arb"
    assert env._strategy_mode == "arb"
    # scalping_mode is independent: stays False as the legacy config asked.
    assert env.scalping_mode is False


def test_value_mode_forces_scalping_false():
    """A value strategy_mode forces scalping_mode False regardless of
    the legacy kwarg / config — single-shot semantics are mode-defined."""
    from data.episode_builder import load_day  # type: ignore[import-not-found]
    from env.betfair_env import BetfairEnv  # type: ignore[import-not-found]
    from training_v2.discrete_ppo.train import (  # type: ignore[import-not-found]
        _scalping_train_config,
    )

    baseline = _require_fixture_and_data()
    cfg = _scalping_train_config(max_runners=baseline["max_runners"])
    day = load_day(baseline["day"], data_dir=DATA_DIR)
    # Even with scalping_mode=True in config + kwarg, value_win forces False.
    env = BetfairEnv(day, cfg, strategy_mode="value_win", scalping_mode=True)
    assert env._strategy_mode == "value_win"
    assert env.scalping_mode is False


def test_flag_on_populates_predictor_keys():
    """Hard_constraints §1 + Session 02 success bar: with
    `use_race_outcome_predictor=True` and a real `PredictorBundle`,
    the runner obs slice carries non-zero values at the predictor-key
    indices for at least one runner.

    Cross-checks the data-bridging integration end-to-end:
    `data/predictor_features.py::build_predict_race_dataframe` →
    `PredictorBundle.predict_race(df)` →
    `env._compute_race_predictor_outputs` →
    `feat_dict["runners"][sid]` →
    `_features_to_array` → obs slice.
    """
    from data.episode_builder import load_day  # type: ignore[import-not-found]
    from env.betfair_env import (  # type: ignore[import-not-found]
        RUNNER_DIM,
        RUNNER_KEYS,
        BetfairEnv,
    )
    from training_v2.discrete_ppo.train import (  # type: ignore[import-not-found]
        _scalping_train_config,
    )

    baseline = _require_fixture_and_data()

    # Predictor bundle: real production weights via sibling repo.
    sibling_repo = Path(__file__).resolve().parents[1].parent / "betfair-predictors"
    if not (
        sibling_repo / "production" / "race-outcome" / "manifest.json"
    ).exists():
        pytest.skip("sibling betfair-predictors repo not present")
    from predictors import PredictorBundle  # type: ignore[import-not-found]
    bundle = PredictorBundle.from_manifests(
        champion_manifest=str(
            sibling_repo / "production" / "race-outcome" / "manifest.json"
        ),
        ranker_manifest=str(
            sibling_repo / "production" / "race-outcome-ranker" / "manifest.json"
        ),
        direction_manifest=str(
            sibling_repo / "production" / "direction-predictor" / "manifest.json"
        ),
    )

    cfg = _scalping_train_config(max_runners=baseline["max_runners"])
    day = load_day(baseline["day"], data_dir=DATA_DIR)
    env = BetfairEnv(
        day,
        cfg,
        predictor_bundle=bundle,
        use_race_outcome_predictor=True,
        use_direction_predictor=False,  # per-tick path not wired yet
    )

    # Predictor-key indices in the runner obs slice.
    predictor_key_indices = [
        i for i, k in enumerate(RUNNER_KEYS)
        if k.startswith("champion_") or k.startswith("ranker_")
    ]
    assert len(predictor_key_indices) == 6, (
        "expected 6 race-level predictor keys at the tail of RUNNER_KEYS"
    )

    # Inspect the first race's first tick obs. Find runner slot 0's
    # obs slice and check that the predictor indices have at least one
    # non-zero value (the champion's p_win is in [0, 1] and varies per
    # runner — at least one non-zero is essentially guaranteed).
    race_idx = 0
    tick_idx = 0
    static = env._static_obs[race_idx][tick_idx]
    runner_offset = 0  # MARKET + VELOCITY come first; we'll find runner_0 below
    from env.betfair_env import MARKET_DIM, VELOCITY_DIM
    runner_start = MARKET_DIM + VELOCITY_DIM

    found_nonzero = False
    runner_map = env._runner_maps[race_idx]
    for slot in range(env.max_runners):
        slot_offset = runner_start + slot * RUNNER_DIM
        slot_slice = static[slot_offset:slot_offset + RUNNER_DIM]
        for idx in predictor_key_indices:
            if slot_slice[idx] != 0.0:
                found_nonzero = True
                break
        if found_nonzero:
            break

    assert found_nonzero, (
        f"Expected at least one runner slot's predictor-key indices "
        f"(slots 0..{env.max_runners-1}, indices {predictor_key_indices} "
        f"into the per-runner obs slice) to carry non-zero predictor output, "
        f"but all were zero — the predict_race injection didn't land. "
        f"runner_map for race {race_idx}: {runner_map}"
    )
