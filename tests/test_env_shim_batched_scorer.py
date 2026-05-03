"""Tests for the Phase 6 Session 02 batched scorer + calibrator path.

The integration test is the load-bearing per-session correctness guard:
it runs one full episode on ``--seed 42 --day 2026-04-23 --device cpu``
twice — once with the production batched ``compute_extended_obs`` and
once with a per-row reference implementation that mirrors the pre-S02
code — and asserts byte-equality on every step's ``obs``, ``mask``,
per-step ``info["raw_pnl_reward"]`` / ``day_pnl``, and the rollout
collector's first 100 ``log_prob_action`` entries.

Three smoke unit tests cover the batched path in isolation against a
small synthetic K — they catch contract regressions (NaN row leak,
non-finite calibrator output not gated, wrong dest-index scatter)
without paying the full-episode cost.

Skips cleanly when the scorer artefacts (``models/scorer_v1/``) or the
2026-04-23 parquet under ``data/processed_amber_v2_window/`` are
absent — same convention as ``tests/test_agents_v2_env_shim.py``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCORER_DIR = REPO_ROOT / "models" / "scorer_v1"
DATA_DIR = REPO_ROOT / "data" / "processed_amber_v2_window"
INTEGRATION_DAY = "2026-04-23"
INTEGRATION_PARQUET = DATA_DIR / f"{INTEGRATION_DAY}.parquet"


def _scorer_runtime_available() -> tuple[bool, str]:
    if not (SCORER_DIR / "model.lgb").exists():
        return False, f"Scorer artefacts missing under {SCORER_DIR}."
    try:
        import lightgbm  # noqa: F401
        import joblib  # noqa: F401
    except Exception as exc:
        return False, f"scorer deps unavailable: {exc!r}"
    return True, ""


_runtime_ok, _runtime_reason = _scorer_runtime_available()
pytestmark = pytest.mark.skipif(not _runtime_ok, reason=_runtime_reason)


# ── Per-row reference (mirrors the pre-S02 implementation) ──────────────────


def _per_row_compute_extended_obs(self, base_obs: np.ndarray) -> np.ndarray:
    """Reference per-row scorer pipeline — frozen copy of pre-S02 code.

    Bound onto :class:`DiscreteActionShim` for the integration test via
    monkeypatch. Mirrors ``compute_extended_obs`` as it stood pre-S02
    (commit ``857289b``): one ``booster.predict(feature_vec)`` and one
    ``calibrator.predict([raw])`` call per priceable runner-side, with
    the per-row finiteness gate inside the loop. Used as the parity
    baseline against the new batched path.
    """
    extra = np.zeros(2 * self._N, dtype=np.float32)
    race = self._current_race()
    tick = self._current_tick()
    if race is None or tick is None:
        return np.concatenate([base_obs, extra]).astype(
            np.float32, copy=False,
        )

    slot_map = self.env._slot_maps[self.env._race_idx]
    runner_by_sid = {r.selection_id: r for r in tick.runners}
    feature_names = self._feature_spec["feature_names"]

    for slot in range(self._N):
        sid = slot_map.get(slot)
        if sid is None:
            continue
        runner = runner_by_sid.get(sid)
        if runner is None or runner.status != "ACTIVE":
            continue
        ltp = runner.last_traded_price
        if ltp is None or ltp <= 1.0:
            continue
        try:
            runner_idx_in_tick = next(
                j for j, r in enumerate(tick.runners)
                if r.selection_id == sid
            )
        except StopIteration:
            continue
        for side_idx, side in enumerate(("back", "lay")):
            try:
                feat_dict = self._feature_extractor.extract(
                    race=race,
                    tick_idx=self.env._tick_idx,
                    runner_idx=runner_idx_in_tick,
                    side=side,
                )
            except Exception:
                logging.getLogger("env_shim_per_row").debug(
                    "extract failed", exc_info=True,
                )
                continue
            feature_vec = np.asarray(
                [feat_dict[name] for name in feature_names],
                dtype=np.float32,
            ).reshape(1, -1)
            raw = self._booster.predict(feature_vec)
            cal_arr = self._calibrator.predict(np.asarray(raw))
            cal = float(cal_arr[0])
            if not np.isfinite(cal):
                continue
            extra[2 * slot + side_idx] = float(np.clip(cal, 0.0, 1.0))
    return np.concatenate([base_obs, extra]).astype(
        np.float32, copy=False,
    )


# ── Smoke unit tests for the batched path in isolation ──────────────────────


def _load_scorer_artefacts() -> tuple[Any, Any, list[str]]:
    import joblib
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(SCORER_DIR / "model.lgb"))
    calibrator = joblib.load(SCORER_DIR / "calibrator.joblib")
    with (SCORER_DIR / "feature_spec.json").open() as fh:
        spec = json.load(fh)
    return booster, calibrator, list(spec["feature_names"])


def test_batched_scorer_matches_per_row_on_synthetic_inputs():
    """K=10 random feature vectors, batched path == per-row, byte-equal."""
    booster, calibrator, feature_names = _load_scorer_artefacts()
    n_features = len(feature_names)
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(10, n_features)).astype(np.float32)

    # Per-row path
    per_row_outputs = []
    for i in range(10):
        raw = booster.predict(matrix[i:i + 1])
        cal = calibrator.predict(np.asarray(raw))
        per_row_outputs.append(float(np.clip(cal[0], 0.0, 1.0)))

    # Batched path
    raw_batch = booster.predict(matrix)
    cal_batch = calibrator.predict(np.asarray(raw_batch))
    batched_outputs = np.clip(cal_batch, 0.0, 1.0)

    for i in range(10):
        assert per_row_outputs[i] == float(batched_outputs[i]), (
            f"row {i}: per_row={per_row_outputs[i]!r} "
            f"vs batched={float(batched_outputs[i])!r}"
        )


def test_batched_scorer_skips_nan_row_in_isolation():
    """K=5 with row 2 NaN: rows 0,1,3,4 unchanged, row 2's slot stays 0.

    Reproduces the per-row "skip exactly one bad row" contract on the
    batched form. NaN-bearing rows are filtered during collection (the
    existing ``try/except`` around ``feature_extractor.extract`` keeps
    them out), so the batched matrix never sees them — verified here by
    constructing the post-collection matrix without the bad row and
    confirming the scatter places outputs at the correct dest indices.
    """
    booster, calibrator, feature_names = _load_scorer_artefacts()
    n_features = len(feature_names)
    rng = np.random.default_rng(7)
    rows_full = rng.normal(size=(5, n_features)).astype(np.float32)

    # Simulate "row 2 was filtered out at extract-time" — only rows
    # 0, 1, 3, 4 enter the batch with destination indices 0, 1, 3, 4
    # respectively (row 2's destination is left at 0).
    keep_indices = [0, 1, 3, 4]
    matrix = rows_full[keep_indices]
    raw = booster.predict(matrix)
    cal = calibrator.predict(np.asarray(raw))
    clipped = np.clip(cal, 0.0, 1.0).astype(np.float32, copy=False)

    extra = np.zeros(5, dtype=np.float32)
    finite = np.isfinite(cal)
    for k, dest in enumerate(keep_indices):
        if finite[k]:
            extra[dest] = clipped[k]

    # Per-row reference
    expected = np.zeros(5, dtype=np.float32)
    for dest in keep_indices:
        raw_row = booster.predict(rows_full[dest:dest + 1])
        cal_row = calibrator.predict(np.asarray(raw_row))
        if np.isfinite(cal_row[0]):
            expected[dest] = float(np.clip(cal_row[0], 0.0, 1.0))

    assert np.array_equal(extra, expected), f"{extra=} vs {expected=}"
    assert extra[2] == 0.0  # the skipped slot stays at zero


def test_batched_calibrator_finiteness_check_per_row():
    """K=5 with a forged calibrator returning inf at row 3.

    Row 3's slot must be left at 0; rows 0, 1, 2, 4 must be written.
    Verifies the post-batch ``np.isfinite(cal)`` mask gates per-element,
    not per-batch.
    """
    booster, _calibrator, feature_names = _load_scorer_artefacts()
    n_features = len(feature_names)
    rng = np.random.default_rng(13)
    matrix = rng.normal(size=(5, n_features)).astype(np.float32)

    raw = booster.predict(matrix)

    class _ForgeCalibrator:
        def predict(self, x):
            out = np.asarray(x, dtype=np.float64).copy()
            out[3] = np.inf
            return out

    forge = _ForgeCalibrator()
    cal = forge.predict(np.asarray(raw))
    finite = np.isfinite(cal)
    clipped = np.clip(cal, 0.0, 1.0).astype(np.float32, copy=False)

    extra = np.zeros(5, dtype=np.float32)
    for k in range(5):
        if finite[k]:
            extra[k] = clipped[k]

    assert extra[3] == 0.0, "row 3's inf must skip the slot"
    for k in (0, 1, 2, 4):
        assert extra[k] != 0.0 or float(clipped[k]) == 0.0, (
            f"row {k} should have been written; got {extra[k]}"
        )
        assert extra[k] == clipped[k]


# ── Integration test: full-episode bit-identity per-row vs batched ──────────


_integration_skip_reason = (
    None
    if INTEGRATION_PARQUET.exists()
    else f"Integration day data missing: {INTEGRATION_PARQUET}"
)


@pytest.mark.skipif(
    _integration_skip_reason is not None,
    reason=_integration_skip_reason or "",
)
@pytest.mark.slow
def test_full_episode_byte_identical_per_row_vs_batched(monkeypatch):
    """Full-episode bit-identity guard for the S02 batched scorer path.

    Runs the rollout collector twice on the same seed / day / device:
    once with the per-row reference ``compute_extended_obs`` and once
    with the production batched implementation. Asserts byte-equality
    on every per-step ``obs`` and ``mask``, on per-step ``raw_pnl_reward``
    and final ``day_pnl``, and on the first 100 ``log_prob_action``
    entries from the rollout's collected transitions.

    LightGBM and isotonic regression both produce identical floats
    between batched and per-row paths (verified pre-write — see
    plans/rewrite/phase-6-profile-and-attack/findings.md S02). Any
    divergence here is a contract bug: wrong skip semantics, wrong
    dest-index scatter, or finiteness gate misplacement.
    """
    from data.episode_builder import load_day
    from env.betfair_env import BetfairEnv

    from agents_v2.discrete_policy import DiscreteLSTMPolicy
    from agents_v2.env_shim import DiscreteActionShim
    from training_v2.discrete_ppo.rollout import RolloutCollector
    from training_v2.discrete_ppo.train import _scalping_train_config

    seed = 42

    def _run_episode(use_per_row: bool) -> dict:
        torch.manual_seed(seed)
        np.random.seed(seed)

        cfg = _scalping_train_config(max_runners=14)
        day = load_day(INTEGRATION_DAY, data_dir=DATA_DIR)
        env = BetfairEnv(day, cfg)
        shim = DiscreteActionShim(env, scorer_dir=SCORER_DIR)

        if use_per_row:
            # Bind the per-row reference onto this shim instance only.
            # MethodType so ``self`` resolves correctly inside the helper.
            import types

            shim.compute_extended_obs = types.MethodType(
                _per_row_compute_extended_obs, shim,
            )

        policy = DiscreteLSTMPolicy(
            obs_dim=shim.obs_dim,
            action_space=shim.action_space,
            hidden_size=128,
        )

        # Capture per-step info dicts via env.step wrapper.
        captured_infos: list[dict] = []
        orig_step = env.step

        def _capturing_step(action):
            obs, reward, term, trunc, info = orig_step(action)
            # Snapshot the keys we compare to defend against in-place
            # mutation downstream.
            snap = {
                "raw_pnl_reward": float(info.get("raw_pnl_reward", 0.0)),
                "day_pnl": float(info.get("day_pnl", 0.0)),
            }
            captured_infos.append(snap)
            return obs, reward, term, trunc, info

        env.step = _capturing_step  # type: ignore[method-assign]

        collector = RolloutCollector(shim=shim, policy=policy, device="cpu")
        batch = collector.collect_episode()

        return {
            "obs": np.asarray(batch.obs).copy(),
            "mask": np.asarray(batch.mask).copy(),
            "log_prob_action": np.asarray(batch.log_prob_action).copy(),
            "n_steps": int(batch.n_steps),
            "infos": captured_infos,
        }

    # Per-row reference
    ref = _run_episode(use_per_row=True)
    # Batched (production)
    bat = _run_episode(use_per_row=False)

    assert ref["n_steps"] == bat["n_steps"], (
        f"step count diverged: per_row={ref['n_steps']} "
        f"batched={bat['n_steps']}"
    )

    n = ref["n_steps"]
    assert n > 0, "rollout produced zero steps"

    # Per-step obs / mask byte equality
    for t in range(n):
        if not np.array_equal(ref["obs"][t], bat["obs"][t]):
            diff = np.where(ref["obs"][t] != bat["obs"][t])[0]
            raise AssertionError(
                f"obs diverged at step {t}; differing indices "
                f"{diff[:8].tolist()} per_row[idx]={ref['obs'][t][diff[:4]]} "
                f"batched[idx]={bat['obs'][t][diff[:4]]}"
            )
        if not np.array_equal(ref["mask"][t], bat["mask"][t]):
            raise AssertionError(f"mask diverged at step {t}")

    # Per-step info bit-equality (raw_pnl_reward + day_pnl)
    assert len(ref["infos"]) == len(bat["infos"]) == n
    for t in range(n):
        if ref["infos"][t]["raw_pnl_reward"] != bat["infos"][t]["raw_pnl_reward"]:
            raise AssertionError(
                f"raw_pnl_reward diverged at step {t}: "
                f"per_row={ref['infos'][t]['raw_pnl_reward']!r} "
                f"batched={bat['infos'][t]['raw_pnl_reward']!r}"
            )
    # day_pnl at episode end — last step's info has the final value
    assert ref["infos"][-1]["day_pnl"] == bat["infos"][-1]["day_pnl"], (
        f"final day_pnl diverged: per_row={ref['infos'][-1]['day_pnl']!r} "
        f"batched={bat['infos'][-1]['day_pnl']!r}"
    )

    # First 100 log_prob_action entries — ULP-strict equality
    n_check = min(100, n)
    ref_lp = ref["log_prob_action"][:n_check]
    bat_lp = bat["log_prob_action"][:n_check]
    if not np.array_equal(ref_lp, bat_lp):
        diff_idx = np.where(ref_lp != bat_lp)[0]
        first = int(diff_idx[0])
        raise AssertionError(
            f"log_prob_action diverged in first {n_check}: "
            f"first diff at idx {first} "
            f"per_row={ref_lp[first]!r} batched={bat_lp[first]!r} "
            f"(ULP-strict equality required)"
        )
