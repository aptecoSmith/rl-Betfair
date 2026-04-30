# Session prompt — Phase 1, Session 01: discrete action space + env shim

Use this prompt to open a new session in a fresh context. The prompt
is self-contained — it briefs you on the task, the design decisions
already locked, and the constraints. Do not require any context from
the session that scaffolded this prompt.

---

## The task

Build the discrete action space, the masking helpers, and the env
shim that translates discrete-policy outputs → the existing
`BetfairEnv`'s 70-dim Box action vector. Wire the Phase 0 supervised
scorer into the observation extension. **No policy class in this
session.** No training. No env edits.

Output: a working `agents_v2/` package that another session
(Session 02) can import and drive end-to-end with random tensors.

## What you need to read first

1. `plans/rewrite/README.md` — rewrite plan overview, hard
   constraints (env stays untouched, parallel tree, no shaping).
2. `plans/rewrite/phase-1-policy-and-env-wiring/purpose.md` —
   locked action space, locked obs extension, locked scorer
   integration. **The shape decisions are not up for debate in
   this session; if you find a reason to change them, file it as
   a finding and stop.**
3. `plans/rewrite/phase-0-supervised-scorer/findings.md` — Phase
   0's GREEN verdict. Pay attention to the F7 velocity-feature
   dead-data note — your shim must compute features the same way
   the dataset pipeline did.
4. `models/scorer_v1/feature_spec.json` — feature contract.
5. `training_v2/scorer/feature_extractor.py` — **reuse this
   verbatim. Do not re-implement.** Re-implementing the rolling
   windows wrong is the obvious failure mode.
6. `tests/test_scorer_v1_inference.py` — the booster + calibrator
   load contract. Your shim's loader should pass the same
   regression guards.
7. `env/betfair_env.py:104, 124` — `ACTIONS_PER_RUNNER = 4` and
   `SCALPING_ACTIONS_PER_RUNNER = 7`. The scalping per-runner
   layout is `[signal, stake, aggression, cancel, arb_spread,
   requote_signal, close_signal]`. Your shim writes into this
   70-dim vector.
8. `env/betfair_env.py:880-898` — observation_space and
   action_space construction; copy the dim arithmetic into the
   shim's obs extension.
9. `CLAUDE.md` sections "Bet accounting", "Order matching",
   "Equal-profit pair sizing". The shim's translation must
   respect these — don't pick stake/spread choices that the env
   will silently refuse.

## What to do

### 1. `agents_v2/action_space.py` (~30 min)

`DiscreteActionSpace` — pure-Python (no torch) class that owns
the index math:

```python
class DiscreteActionSpace:
    """Index layout (locked):
        0                          → no-op
        1     .. max_runners       → open_back_i  for i in [0, max_runners)
        max_runners + 1 .. 2*max_runners → open_lay_i
        2*max_runners + 1 .. 3*max_runners → close_i
    Total size = 1 + 3 * max_runners.
    """
    def __init__(self, max_runners: int) -> None: ...

    @property
    def n(self) -> int: ...   # total action count

    def decode(self, idx: int) -> tuple[ActionType, int | None]:
        """idx → (kind, runner_idx). runner_idx is None for no-op."""

    def encode(self, kind: ActionType, runner_idx: int | None) -> int:
        """Round-trip inverse of decode."""
```

`ActionType` is an `IntEnum` with values `NOOP=0, OPEN_BACK=1,
OPEN_LAY=2, CLOSE=3`.

### 2. `agents_v2/action_space.py::compute_mask` (~30 min)

```python
def compute_mask(
    space: DiscreteActionSpace,
    env: BetfairEnv,
) -> np.ndarray:
    """Return a (space.n,) bool mask: True = legal, False = illegal.

    No-op is always legal. open_back_i / open_lay_i illegal when:
        - runner i is INACTIVE / has no LTP / hard cap exceeded
        - runner i already has an open pair (BetManager has any
          unsettled bet with selection_id matching runner i)
        - bet_manager.budget < MIN_BET_STAKE
    close_i illegal when:
        - runner i has no open pair to close (no unsettled aggressive
          leg with the runner's selection_id whose passive hasn't
          filled)
    """
```

Tests in `tests/test_agents_v2_action_space.py`:

- `test_index_layout_round_trip` — `encode(decode(i)) == i` for all
  i in `[0, n)`.
- `test_noop_always_legal` — mask[0] == True regardless of env
  state.
- `test_open_masked_when_runner_inactive` — set runner status to
  REMOVED, verify both open_back_i and open_lay_i mask False.
- `test_open_masked_when_no_ltp` — runner with `last_traded_price
  is None` → both opens masked.
- `test_close_masked_when_no_open_pair` — fresh BetManager, all
  close_i masked False.
- `test_close_unmasked_after_open` — place an aggressive on
  runner_i via the helper, verify close_i unmasks.

### 3. `agents_v2/env_shim.py::DiscreteActionShim` (~90 min)

The translator. Constructor takes the env + scorer artefact paths.
Loads scorer + calibrator once. Owns the per-market
`FeatureExtractor` instance(s).

```python
class DiscreteActionShim:
    def __init__(
        self,
        env: BetfairEnv,
        scorer_dir: Path = REPO_ROOT / "models" / "scorer_v1",
        arb_ticks: int = 20,            # locked from Phase 0 findings
        default_stake: float = 10.0,    # MIN_BET_STAKE-clear default
    ) -> None: ...

    @property
    def action_space(self) -> DiscreteActionSpace: ...

    @property
    def obs_dim(self) -> int:
        """Underlying env obs dim + 2 × max_runners scorer features."""

    def reset(self, *args, **kwargs) -> tuple[np.ndarray, dict]:
        """Pass-through to env.reset(); returns extended obs."""

    def step(
        self,
        discrete_action: int,
        stake: float | None = None,
        arb_spread: int | None = None,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Translate discrete action to a 70-dim Box, step env, return
        (extended_obs, reward, terminated, truncated, info).

        The continuous heads (stake, arb_spread) are optional — when
        omitted the shim uses (default_stake, arb_ticks) so the
        smoke test can drive with discrete-only inputs.
        """

    def compute_extended_obs(self, base_obs: np.ndarray) -> np.ndarray:
        """Append 2×max_runners scorer-prediction features to base_obs."""

    def get_action_mask(self) -> np.ndarray:
        """Forward to action_space.compute_mask(self.env)."""
```

**Translation rules (locked):**

For each per-runner slot in the 70-dim Box vector
(`[signal, stake, aggression, cancel, arb_spread, requote_signal,
close_signal]`):

| Discrete action chosen | Per-runner i values written |
|---|---|
| `NOOP` | All zeros for all runners |
| `OPEN_BACK` runner=i | `signal=+1.0, stake=normalised(stake), aggression=+1.0, arb_spread=normalised(arb_ticks), close_signal=0`, all other dims 0 |
| `OPEN_LAY` runner=i | `signal=-1.0, stake=normalised(stake), aggression=+1.0, arb_spread=normalised(arb_ticks), close_signal=0`, all other dims 0 |
| `CLOSE` runner=i | `close_signal=+1.0`, all other dims 0 |

Where `normalised(stake)` maps the [`MIN_BET_STAKE`,
`max_stake_cap`] range to `[-1, 1]` per the env's existing
expectation, and `normalised(arb_ticks)` maps `[MIN_ARB_TICKS,
MAX_ARB_TICKS]` to `[-1, 1]`. **Look up the exact normalisation
the env uses** — do NOT guess. Read `env/betfair_env.py` for the
inverse mapping (the code that decodes the action vector); your
shim's encoding must be its inverse to round-trip cleanly.

All non-target runners receive zero on every dim. The env
interprets `signal in [-0.5, 0.5]` as no-op for that runner, so
zero is safe.

**Scorer wiring (locked):**

```python
def compute_extended_obs(self, base_obs):
    extra = np.zeros(2 * self.env.max_runners, dtype=np.float32)
    current_tick = self._current_tick()
    for i, runner_snap in enumerate(current_tick.runners[:self.env.max_runners]):
        if runner_snap.status != "ACTIVE" or runner_snap.last_traded_price is None:
            continue  # leave zeros — sentinel for "scorer not applicable"
        for side_idx, side in enumerate(("back", "lay")):
            try:
                features = self._feature_extractor.extract(
                    race=self._current_race(),
                    tick_idx=self._current_tick_idx(),
                    runner_idx=i,
                    side=side,
                )
            except Exception:
                continue  # leave zeros
            if not np.isfinite(features).all():
                continue  # NaN feature → leave zero
            raw = self._booster.predict(features.reshape(1, -1))
            cal = float(self._calibrator.predict(raw)[0])
            extra[2 * i + side_idx] = cal
    return np.concatenate([base_obs, extra])
```

(The `_current_*` helpers are bookkeeping into `env`'s internal
state — read `env/betfair_env.py:_build_observation` to see how
the env itself reaches into the current race / tick.)

### 4. Tests for the shim (~60 min)

`tests/test_agents_v2_env_shim.py`:

- `test_obs_dim_is_base_plus_2x_runners` — `shim.obs_dim ==
  env.observation_space.shape[0] + 2 * env.max_runners`.
- `test_reset_returns_extended_obs_of_correct_shape`.
- `test_step_with_noop_writes_zeros_to_box_action` — instrument
  the env to capture the action vector it received; assert all
  70 dims are 0 when discrete=NOOP.
- `test_step_with_open_back_writes_correct_per_runner_slot` —
  pick a runner_i, send OPEN_BACK, assert the i-th slot has
  `signal>0`, `stake>0`, `aggression>0`, `close_signal=0`; all
  other runners' slots are zero.
- `test_step_with_close_writes_close_signal` — symmetric.
- `test_scorer_predictions_packed_at_correct_indices` —
  monkey-patch `shim._booster.predict` to return a deterministic
  fixed value, step, assert obs[base_dim + 2*i + side_idx] equals
  the calibrated version for ACTIVE runners.
- `test_scorer_returns_zero_for_inactive_runner` — REMOVED runner
  → both scorer slots zero.
- `test_action_mask_blocks_open_on_inactive_runner` — call
  `get_action_mask()`, assert open_back/open_lay for an inactive
  runner are False.

### 5. Re-export contract (~5 min)

`agents_v2/__init__.py`:

```python
from agents_v2.action_space import (
    ActionType, DiscreteActionSpace, compute_mask,
)
from agents_v2.env_shim import DiscreteActionShim

__all__ = [
    "ActionType", "DiscreteActionSpace", "compute_mask",
    "DiscreteActionShim",
]
```

## Stop conditions

- All tests pass (`pytest tests/test_agents_v2_action_space.py
  tests/test_agents_v2_env_shim.py`) → write Session 01 findings,
  stop. Session 02 takes the shim from here.
- Any test reveals the env normalisation isn't a clean inverse
  → stop, document the discrepancy, do NOT modify the env. The
  fix may need to live in the shim (e.g. asymmetric stake
  encoding) or as a Phase −1 follow-on.
- Any failure in scorer loading → stop, check that
  `models/scorer_v1/` exists; if not, Phase 0 wasn't run, escalate.

## Hard constraints

- **No env edits.** Even if you find a bug. The shim is
  one-directional: it adapts to the env, not vice versa.
- **No policy class.** Session 02 owns that. If you need a
  "policy" to test with, drive the shim with random `int` indices
  and random stakes from numpy directly.
- **No training-related code.** No optimiser, no loss, no GAE,
  no value-function code. The per-runner value head is Session
  02's job; it goes ON the policy class, not the shim.
- **Reuse `training_v2.scorer.feature_extractor.FeatureExtractor`
  verbatim.** Do not re-implement it. Do not "improve" it.
- **Stake / spread normalisation: round-trip with the env, not
  with intuition.** Read the env's decoder and inverse it.

## Out of scope

- Policy class (Session 02).
- PPO / training (Phase 2).
- Frontend wiring (Phase 3).
- Performance profiling (the shim runs once per env step; even
  a slow scorer prediction is dwarfed by env step cost — don't
  optimise prematurely).
- Per-runner velocity-feature fixes for F7 (Phase 0 finding;
  out of scope for Phase 1).

## Useful pointers

- `env/betfair_env.py:104, 124, 130, 135` — action layout
  constants.
- `env/betfair_env.py:880-898` — obs / action space construction.
- `env/betfair_env.py::_decode_actions` (or equivalent) — read
  this to derive your encoder's inverse.
- `env/bet_manager.py::MIN_BET_STAKE` — the minimum stake
  the env will accept.
- `env/scalping_math.py::min_arb_ticks_for_profit` — the
  spread floor that clears commission.
- `training_v2/scorer/feature_extractor.py::FeatureExtractor` —
  reuse for the shim's per-tick feature computation.
- `models/scorer_v1/feature_spec.json` — feature contract.
- `tests/test_scorer_v1_inference.py` — patterns for loading the
  booster + calibrator the same way the regression guards do.

## Estimate

3–5 hours.

- 30 min: `DiscreteActionSpace` + index round-trip tests.
- 30 min: `compute_mask` + masking tests.
- 90 min: `DiscreteActionShim` (loader + translation + obs
  extension).
- 60 min: shim tests.
- 30 min: smoke run a fresh env-shim cycle by hand to confirm
  end-to-end before declaring done.
- 30 min: findings writeup
  (`plans/rewrite/phase-1-policy-and-env-wiring/session_01_findings.md`).

If past 6 hours, stop and check scope. The most likely overrun
is the env decoder reverse-engineering (constraint #2 above:
read the env's decoder, don't guess).
