"""Discrete-action shim around ``BetfairEnv``.

Phase 1, Session 01 deliverable. The shim hides the env's 70-dim Box
action vector behind a small discrete + continuous interface and
augments observations with the Phase 0 supervised scorer's calibrated
``P(mature | features)`` per (active runner, side).

Hard constraints (rewrite README §1, §3 and Session 01 prompt):

- The env is **not** modified. We only consume its public surface +
  the per-race slot maps it builds for us.
- The Phase 0 scorer is **frozen**. Its booster + isotonic calibrator
  are loaded once at construction; no parameter sees gradient.
- The shim re-uses
  :class:`training_v2.scorer.feature_extractor.FeatureExtractor`
  verbatim. Re-implementing the rolling-window math is exactly how
  Phase 1 silently miscalibrates the scorer (Phase 0 findings, F7).

Translation rules (LOCKED — see Session 01 prompt §3):

* The env's per-runner action layout in scalping mode is **dim-major**::

      action[          slot] = signal
      action[  N      + slot] = stake
      action[  2*N    + slot] = aggression
      action[  3*N    + slot] = cancel
      action[  4*N    + slot] = arb_spread
      action[  5*N    + slot] = requote_signal
      action[  6*N    + slot] = close_signal

  (Per ``env/betfair_env.py::_process_action``. The Session 01 prompt
  describes a slot-major layout — that's a documentation drift in the
  prompt; the env decoder is dim-major.)
* ``signal``: env decodes ``> 0.33`` ⇒ BACK, ``< −0.33`` ⇒ LAY. We use
  ``+1.0`` / ``−1.0`` so the round-trip is unambiguous.
* ``stake``: env decodes ``stake = ((raw+1)/2) * bm.budget``. Inverse
  ``raw = 2*(stake/budget) - 1`` (clamped to ``[-1, 1]``). This is the
  budget-fraction encoding the env actually uses — NOT the
  ``[MIN_BET_STAKE, max_stake_cap]`` linear map suggested by the
  prompt. Documented in Session 01 findings under "Stake
  normalisation".
* ``aggression``: env decodes ``> 0`` ⇒ aggressive. We use ``+1.0`` to
  guarantee the aggressive path; the scalping reward shape relies on
  the auto-paired passive that only fires from the aggressive path.
* ``cancel``: env decodes ``> 0`` ⇒ cancel oldest open passive on
  this runner. Always ``0`` here — the discrete head doesn't expose
  cancel as a separate action; the close-pass already cancels the
  passive when ``close_signal`` is raised.
* ``arb_spread``: env decodes ``frac=(raw+1)/2; ticks = MIN +
  frac*(MAX-MIN)``. Inverse for the locked default ``arb_ticks=20`` is
  ``raw = 2 * (20-1)/(25-1) - 1 = 0.5833…``.
* ``requote_signal``: always ``0``. The discrete head doesn't expose
  re-quote as a primitive in this session; if Phase 2 wants it,
  Session 01b can extend the action space.
* ``close_signal``: ``+1.0`` for ``CLOSE``, ``0.0`` otherwise.

Observation extension::

    obs_v2 = obs_v1 || scorer_features
    scorer_features[2*i + 0] = calibrated_p_mature_back_i
    scorer_features[2*i + 1] = calibrated_p_mature_lay_i

Both slots are ``0.0`` for runners that are inactive, unpriceable,
or whose feature vector contains a NaN the booster can't consume.
The booster itself handles NaN natively but we drop NaN-feature
predictions defensively to keep the obs vector ``np.isfinite``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from env.bet_manager import MIN_BET_STAKE
from env.betfair_env import (
    MAX_ARB_TICKS,
    MIN_ARB_TICKS,
    SCALPING_ACTIONS_PER_RUNNER,
)
from training_v2.scorer.feature_extractor import (
    FEATURE_NAMES,
    FeatureExtractor,
)

from agents_v2.action_space import (
    ActionType,
    DiscreteActionSpace,
    compute_mask,
)

if TYPE_CHECKING:
    from env.betfair_env import BetfairEnv


logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORER_DIR = REPO_ROOT / "models" / "scorer_v1"

_DEFAULT_ARB_TICKS = 20
_DEFAULT_STAKE = 10.0


__all__ = ["DiscreteActionShim", "DEFAULT_SCORER_DIR"]


class DiscreteActionShim:
    """Translate (discrete + small continuous) policy outputs to env steps.

    The shim is one-directional — it adapts to the env. Any divergence
    between the shim's encoding and the env's decoding is the shim's
    bug, never the env's. If round-trip ever breaks, stop and document
    the discrepancy as a Session 01b finding (Session 01 prompt
    §"Stop conditions").

    Parameters
    ----------
    env:
        A constructed :class:`BetfairEnv` (no need to call ``reset``
        first; the shim's ``reset`` forwards). MUST be in scalping
        mode — the discrete action space is built around the
        7-dim per-runner scalping layout.
    scorer_dir:
        Path to the Phase 0 artefacts (booster ``model.lgb``,
        ``calibrator.joblib``, ``feature_spec.json``).
    arb_ticks:
        Tick count for the auto-paired passive when discrete OPEN_*
        is chosen. Locked at 20 per Phase 0's findings.
    default_stake:
        £ stake used when ``step`` is called without a ``stake``
        override. Clamped to ``MIN_BET_STAKE`` from below.
    """

    def __init__(
        self,
        env: "BetfairEnv",
        scorer_dir: Path = DEFAULT_SCORER_DIR,
        arb_ticks: int = _DEFAULT_ARB_TICKS,
        default_stake: float = _DEFAULT_STAKE,
    ) -> None:
        if not env.scalping_mode:
            raise ValueError(
                "DiscreteActionShim requires scalping_mode=True on the env "
                "(the discrete action space is built around the 7-dim "
                "per-runner scalping layout).",
            )
        self.env = env
        self._scorer_dir = Path(scorer_dir)
        if not (MIN_ARB_TICKS <= arb_ticks <= MAX_ARB_TICKS):
            raise ValueError(
                f"arb_ticks {arb_ticks} outside the env's allowed "
                f"[{MIN_ARB_TICKS}, {MAX_ARB_TICKS}] range",
            )
        self._arb_ticks = int(arb_ticks)
        self._default_stake = max(float(default_stake), MIN_BET_STAKE)

        self._action_space = DiscreteActionSpace(env.max_runners)
        self._N = env.max_runners
        self._actions_per_runner = env._actions_per_runner
        if self._actions_per_runner != SCALPING_ACTIONS_PER_RUNNER:
            raise ValueError(
                "Env reports actions_per_runner="
                f"{self._actions_per_runner}, expected "
                f"{SCALPING_ACTIONS_PER_RUNNER} (scalping mode).",
            )

        # ── Scorer artefacts ────────────────────────────────────────────
        self._feature_extractor = FeatureExtractor()
        self._booster, self._calibrator, self._feature_spec = (
            self._load_scorer_artefacts()
        )
        # Confirm feature contract — same regression guard the
        # standalone scorer uses (tests/test_scorer_v1_inference.py
        # ::test_feature_spec_matches_booster).
        spec_names = list(self._feature_spec["feature_names"])
        if spec_names != list(self._booster.feature_name()):
            raise RuntimeError(
                "scorer_v1 feature_spec.json names diverge from the "
                "booster's feature_name() — refusing to run with a "
                "stale spec; re-run training_v2.scorer.train_and_evaluate.",
            )
        if list(FEATURE_NAMES) != spec_names:
            raise RuntimeError(
                "scorer_v1 feature_spec.json diverges from "
                "training_v2.scorer.feature_extractor.FEATURE_NAMES — "
                "the extractor and the booster were trained on "
                "different feature orderings.",
            )

        # Pre-compute per-step constants used in the action encoder.
        # ``arb_raw`` is the inverse of the env's
        #     frac = (arb_raw + 1) / 2
        #     ticks = MIN + frac * (MAX - MIN)
        # mapping. With arb_ticks=20, MIN=1, MAX=25 ⇒ raw ≈ 0.5833.
        arb_frac = (self._arb_ticks - MIN_ARB_TICKS) / (
            MAX_ARB_TICKS - MIN_ARB_TICKS
        )
        self._arb_raw = float(np.clip(2.0 * arb_frac - 1.0, -1.0, 1.0))

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def action_space(self) -> DiscreteActionSpace:
        return self._action_space

    @property
    def obs_dim(self) -> int:
        """Underlying env obs dim + 2 × max_runners scorer features."""
        return int(self.env.observation_space.shape[0]) + 2 * self._N

    @property
    def max_runners(self) -> int:
        return self._N

    # ── Public API ─────────────────────────────────────────────────────────

    def reset(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray, dict]:
        """Forward to ``env.reset`` and append scorer features to the obs."""
        base_obs, info = self.env.reset(*args, **kwargs)
        # Fresh extractor — drops cross-episode rolling-window state.
        self._feature_extractor = FeatureExtractor()
        self._update_history_for_current_tick()
        extended = self.compute_extended_obs(base_obs)
        return extended, info

    def step(
        self,
        discrete_action: int,
        stake: float | None = None,
        arb_spread: int | None = None,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Translate ``discrete_action`` to a Box action and step the env.

        ``stake`` overrides ``default_stake`` for OPEN_* actions only.
        ``arb_spread`` (in ticks) overrides ``arb_ticks`` for OPEN_*.
        Both are ignored for NOOP / CLOSE.
        """
        action_vec = self.encode_action(
            discrete_action,
            stake=stake,
            arb_spread=arb_spread,
        )
        next_obs, reward, terminated, truncated, info = self.env.step(action_vec)
        # Update rolling-window state for the post-step tick BEFORE
        # we extract scorer features off it. After a race-end step
        # ``_race_idx`` may have advanced past ``_total_races`` — the
        # update helper short-circuits in that case.
        self._update_history_for_current_tick()
        if terminated:
            # ``env.step`` returns its terminal-obs zero vector when the
            # episode is done — we still pad to ``obs_dim`` so downstream
            # consumers can rely on a stable shape.
            extended = np.concatenate([
                next_obs,
                np.zeros(2 * self._N, dtype=np.float32),
            ])
        else:
            extended = self.compute_extended_obs(next_obs)
        return extended, reward, terminated, truncated, info

    def get_action_mask(self) -> np.ndarray:
        return compute_mask(self._action_space, self.env)

    def encode_action(
        self,
        discrete_action: int,
        stake: float | None = None,
        arb_spread: int | None = None,
    ) -> np.ndarray:
        """Pure encoder: discrete + continuous → ``Box(-1, 1, …)`` vector.

        Exposed publicly for tests and for callers that want to
        inspect / log the env-side action without stepping.
        """
        kind, runner_idx = self._action_space.decode(int(discrete_action))
        action = np.zeros(
            self._N * self._actions_per_runner, dtype=np.float32,
        )
        if kind is ActionType.NOOP:
            return action

        slot = int(runner_idx)  # type: ignore[arg-type]
        if slot < 0 or slot >= self._N:
            raise ValueError(
                f"runner_idx {slot} out of [0, {self._N})",
            )

        if kind in (ActionType.OPEN_BACK, ActionType.OPEN_LAY):
            stake_value = (
                self._default_stake if stake is None else float(stake)
            )
            stake_raw = self._encode_stake(stake_value)
            arb_raw = (
                self._arb_raw
                if arb_spread is None
                else self._encode_arb_spread(int(arb_spread))
            )
            sign = +1.0 if kind is ActionType.OPEN_BACK else -1.0
            action[slot] = sign                              # signal
            action[self._N + slot] = stake_raw               # stake
            action[2 * self._N + slot] = +1.0                # aggression
            action[3 * self._N + slot] = 0.0                 # cancel
            action[4 * self._N + slot] = arb_raw             # arb_spread
            action[5 * self._N + slot] = 0.0                 # requote
            action[6 * self._N + slot] = 0.0                 # close
            return action

        # ActionType.CLOSE
        action[6 * self._N + slot] = +1.0
        return action

    def compute_extended_obs(self, base_obs: np.ndarray) -> np.ndarray:
        """Append ``2 * max_runners`` scorer features to ``base_obs``.

        Slot ``2*i + 0`` is the BACK side prediction for slot-i's runner
        in the current race; slot ``2*i + 1`` is LAY. Zeroes for slots
        where the runner is inactive, unpriceable, or whose feature
        vector tripped a NaN guard.

        Two-pass batched scorer (Phase 6 S02). Pass 1 collects the K
        priceable runner-side feature vectors and their destination
        indices in ``extra``; Pass 2 calls ``booster.predict`` and
        ``calibrator.predict`` once each on the stacked ``(K, n_feat)``
        matrix; Pass 3 scatters the calibrated outputs back, skipping
        any non-finite slot. Bit-identical to the per-row path
        (LightGBM and isotonic regression both produce identical floats
        between batched and per-row evaluation — verified pre-write,
        see plans/rewrite/phase-6-profile-and-attack/findings.md S02).
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

        # Pass 1: collect priceable runner-side feature vectors + their
        # destination indices in ``extra``. Skip semantics are identical
        # to the per-row path — runners that are inactive, unpriceable,
        # or whose feature_extractor.extract raises never enter the
        # batch in the first place.
        rows: list[np.ndarray] = []
        extra_idx: list[int] = []
        for slot in range(self._N):
            sid = slot_map.get(slot)
            if sid is None:
                continue
            runner = runner_by_sid.get(sid)
            if runner is None:
                continue
            if runner.status != "ACTIVE":
                continue
            ltp = runner.last_traded_price
            if ltp is None or ltp <= 1.0:
                continue
            try:
                runner_idx_in_tick = next(
                    j for j, r in enumerate(tick.runners)
                    if r.selection_id == sid
                )
            except StopIteration:  # pragma: no cover — defensive
                continue
            for side_idx, side in enumerate(("back", "lay")):
                try:
                    feat_dict = self._feature_extractor.extract(
                        race=race,
                        tick_idx=self.env._tick_idx,
                        runner_idx=runner_idx_in_tick,
                        side=side,
                    )
                except Exception:  # pragma: no cover
                    logger.debug(
                        "FeatureExtractor.extract failed for race=%s "
                        "tick=%d runner_idx=%d side=%s",
                        race.market_id, self.env._tick_idx,
                        runner_idx_in_tick, side,
                        exc_info=True,
                    )
                    continue
                # NaN is expected here for the F7-dead velocity
                # features. LightGBM handles NaN natively (see Phase 0
                # findings); the post-calibrator finiteness gate in
                # Pass 3 catches any genuine downstream blow-up.
                row = np.asarray(
                    [feat_dict[name] for name in feature_names],
                    dtype=np.float32,
                )
                rows.append(row)
                extra_idx.append(2 * slot + side_idx)

        if not rows:
            return np.concatenate([base_obs, extra]).astype(
                np.float32, copy=False,
            )

        # Pass 2: one batched booster call, one batched calibrator call.
        matrix = np.stack(rows, axis=0)
        raw = self._booster.predict(matrix)
        cal = self._calibrator.predict(np.asarray(raw))

        # Pass 3: scatter back. Per-row finiteness gate moves to a
        # vectorised np.isfinite mask — non-finite slots stay at zero,
        # preserving the per-row contract that one bad slot never
        # poisons the others.
        finite = np.isfinite(cal)
        clipped = np.clip(cal, 0.0, 1.0).astype(np.float32, copy=False)
        for k, dest in enumerate(extra_idx):
            if not finite[k]:
                continue
            extra[dest] = clipped[k]

        return np.concatenate([base_obs, extra]).astype(
            np.float32, copy=False,
        )

    # ── Internals ──────────────────────────────────────────────────────────

    def _load_scorer_artefacts(self) -> tuple[Any, Any, dict]:
        import joblib
        import lightgbm as lgb

        model_path = self._scorer_dir / "model.lgb"
        cal_path = self._scorer_dir / "calibrator.joblib"
        spec_path = self._scorer_dir / "feature_spec.json"
        for p in (model_path, cal_path, spec_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Phase 0 scorer artefact missing: {p}. Run "
                    "`python -m training_v2.scorer.train_and_evaluate` "
                    "first.",
                )
        booster = lgb.Booster(model_file=str(model_path))
        calibrator = joblib.load(cal_path)
        with spec_path.open() as fh:
            spec = json.load(fh)
        return booster, calibrator, spec

    def _current_race(self):
        if self.env._race_idx >= self.env._total_races:
            return None
        return self.env.day.races[self.env._race_idx]

    def _current_tick(self):
        race = self._current_race()
        if race is None:
            return None
        if self.env._tick_idx >= len(race.ticks):
            return None
        return race.ticks[self.env._tick_idx]

    def _update_history_for_current_tick(self) -> None:
        race = self._current_race()
        tick = self._current_tick()
        if race is None or tick is None:
            return
        try:
            self._feature_extractor.update_history(race, tick)
        except Exception:  # pragma: no cover — defensive
            logger.debug(
                "FeatureExtractor.update_history failed for race=%s tick=%d",
                race.market_id, self.env._tick_idx,
                exc_info=True,
            )

    def _encode_stake(self, stake_value: float) -> float:
        """Inverse of env's ``stake = ((raw+1)/2) * bm.budget``.

        Uses the env's *current* budget — which the env reads at
        decode time. If the current budget is zero (race not yet
        reset), the helper returns ``-1.0`` (the env decodes that as
        zero stake, which falls below ``MIN_BET_STAKE`` and refuses
        the bet — the correct behaviour).
        """
        bm = self.env.bet_manager
        budget = bm.budget if bm is not None else 0.0
        if budget <= 0.0:
            return -1.0
        # Bound the stake to [MIN_BET_STAKE, budget] before encoding —
        # values above budget round-trip to budget anyway after the
        # env's [-1, 1] clip, but explicit clamping makes the
        # encoder side-effect-free for callers that want to spy on
        # the raw value.
        clamped = max(MIN_BET_STAKE, min(stake_value, budget))
        raw = 2.0 * (clamped / budget) - 1.0
        return float(np.clip(raw, -1.0, 1.0))

    def _encode_arb_spread(self, ticks: int) -> float:
        """Inverse of env's frac/raw_ticks mapping at default scale=1."""
        ticks = int(np.clip(ticks, MIN_ARB_TICKS, MAX_ARB_TICKS))
        frac = (ticks - MIN_ARB_TICKS) / (MAX_ARB_TICKS - MIN_ARB_TICKS)
        return float(np.clip(2.0 * frac - 1.0, -1.0, 1.0))
