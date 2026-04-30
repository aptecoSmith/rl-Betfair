---
plan: rewrite/phase-1-policy-and-env-wiring
session: 01
status: complete — DiscreteActionSpace + DiscreteActionShim + scorer wiring shipped
opened: 2026-04-27
---

# Session 01 — discrete action space + env shim findings

## TL;DR

Shipped `agents_v2/action_space.py` (`ActionType`, `DiscreteActionSpace`,
`compute_mask`), `agents_v2/env_shim.py` (`DiscreteActionShim`), and
`agents_v2/__init__.py`. **32 tests pass** end-to-end across
`tests/test_agents_v2_action_space.py` (17) and
`tests/test_agents_v2_env_shim.py` (15). End-to-end smoke run: shim
drives a random masked policy through a 2-race / ~24-tick episode
with no crashes and a sane non-zero reward, obs shape stable at
`base + 2 × max_runners`.

Three findings worth pinning before Session 02 starts:

1. **Env action layout is dim-major, not slot-major.** The Session 01
   prompt described a per-runner layout
   `[signal, stake, aggression, cancel, arb_spread, requote, close]`.
   The env decoder
   ([env/betfair_env.py:1794-2063](../../../env/betfair_env.py))
   actually reads
   `action[d * max_runners + slot]` for each dim `d ∈ [0, 7)` —
   i.e. all signals first, then all stakes, then all aggressions,
   etc. The shim writes to dim-major indices. **No env edit, no
   policy class change — just documentation drift in the prompt.**
   See `agents_v2/env_shim.py` module docstring for the corrected
   layout.

2. **Stake normalisation maps `[0, budget]` → `[-1, 1]`, not
   `[MIN_BET_STAKE, max_stake_cap]` → `[-1, 1]`.** The prompt's
   "look up the exact normalisation" instruction is the right rule;
   the suggested `[MIN_BET_STAKE, max_stake_cap]` framing was a red
   herring. The env's actual decode is
   `stake = ((raw + 1) / 2) × bm.budget` (env line 1834-1835).
   The shim inverts to `raw = 2 × (stake / budget) - 1`, clamped to
   `[-1, 1]`, and round-trips exactly for the test cases (default
   stake £10 on a £100 budget → raw = -0.8 → decoded stake £10).
   Logged in the shim docstring under "Translation rules" so
   Session 02 doesn't re-litigate.

3. **NaN feature vectors must NOT be dropped.** The Session 01 prompt's
   reference pseudocode in `purpose.md` short-circuits when
   `np.isfinite(features).all()` is False. Phase 0's findings
   (`phase-0-supervised-scorer/findings.md` "F7 limitations on
   per-runner velocity") confirm the booster was trained with
   `time_since_last_trade_seconds` and `traded_volume_last_30s` at
   100 % NaN. LightGBM handles NaN natively and the booster's
   feature importance reflects that — those features are rank 0.
   The shim therefore passes feature vectors through (NaN included)
   and only drops the prediction if the calibrator output itself
   isn't finite. Without this fix, the
   `test_scorer_predictions_packed_at_correct_indices` regression
   guard fails because the input-side NaN guard never lets the
   booster fire.

## What landed

### `agents_v2/action_space.py`

```
ActionType                       — IntEnum: NOOP=0, OPEN_BACK=1, OPEN_LAY=2, CLOSE=3
DiscreteActionSpace(max_runners) — encode/decode + n
compute_mask(space, env)         — bool mask over space.n; True = legal
```

Locked layout (n = 1 + 3 × max_runners):

```
0                                   → NOOP
1                .. max_runners     → open_back_i   (i ∈ [0, max_runners))
max_runners+1    .. 2×max_runners   → open_lay_i
2×max_runners+1  .. 3×max_runners   → close_i
```

Mask rules:

- NOOP always legal (even when `bm is None` or env unreset — `mask[0] = True`).
- `open_*_i` legal iff
  `runner.status == "ACTIVE" AND runner.last_traded_price > 1.0
   AND no unsettled bet on that selection_id
   AND bm.budget >= MIN_BET_STAKE
   AND race_bet_count < max_bets_per_race`.
- `close_i` legal iff there's an open pair on that runner whose
  `complete=False` (aggressive matched, passive still resting).
  Read off `bm.get_paired_positions(market_id=…)`.

### `agents_v2/env_shim.py`

```
DiscreteActionShim(env, scorer_dir, arb_ticks=20, default_stake=10.0)
    .action_space         → DiscreteActionSpace
    .obs_dim              → env base + 2 × max_runners
    .reset(*a, **kw)      → (extended_obs, info)
    .step(idx, stake?, arb_spread?) → (extended_obs, reward, term, trunc, info)
    .compute_extended_obs(base_obs)  → obs || scorer features
    .encode_action(idx, stake?, arb_spread?) → 70-dim action vector
    .get_action_mask()    → forwards to compute_mask
```

Constructor refuses non-scalping envs and out-of-range `arb_ticks`
(the env's decode clips to `[MIN_ARB_TICKS, MAX_ARB_TICKS]` so an
out-of-range value would silently round to the cap; better to
fail loud).

The shim re-uses `training_v2.scorer.feature_extractor.FeatureExtractor`
verbatim per Phase 1 hard constraint #2. It calls
`update_history(race, tick)` on every reset and step so the
rolling-window state is current; on race rollover the velocities
take a few ticks to populate (NaN until then), matching how the
Phase 0 dataset pipeline saw the data.

Architecture-hash boundary: nothing — Session 02's policy class
will introduce a new variant via `model_store.py`'s existing
shape check, but Session 01 ships no torch state.

### Tests

| File | Tests | Notes |
|---|---:|---|
| `tests/test_agents_v2_action_space.py` | 17 | Pure-Python — no scorer deps. Always runs. |
| `tests/test_agents_v2_env_shim.py` | 15 | Skip cleanly when `lightgbm` / `joblib` / `models/scorer_v1/` missing. |

The env-shim suite covers the prompt's full checklist:
`obs_dim`, `reset` shape, NOOP / OPEN_BACK / OPEN_LAY / CLOSE
encoding, scorer index packing, scorer zero on inactive runners,
mask forwarding, arb_spread + stake round-trip, constructor guards.

`compute_mask`'s "open masked when budget below MIN_BET_STAKE" and
"NOOP legal even without reset" aren't on the prompt's checklist
but they're cheap regression guards for the two cases I felt the
hardest to spot at review time, so I added them.

## Smoke run

Driver script (~30 lines, run from repo root):

```python
env = BetfairEnv(_make_day(n_races=2, n_pre_ticks=10, n_inplay_ticks=2), scalping_cfg)
shim = DiscreteActionShim(env)
obs, info = shim.reset()  # obs.shape = (552,), action_space.n = 13
rng = np.random.default_rng(0)
done = False
while not done:
    mask = shim.get_action_mask()
    legal = np.flatnonzero(mask)
    obs, reward, terminated, truncated, info = shim.step(int(rng.choice(legal)))
    done = terminated or truncated
# 24 steps, total_reward=3.402, day_pnl=28.31
```

obs shape stable, `np.isfinite(obs).all()` on every non-terminal
step, and the env runs through 2 race transitions without the
shim's history reset breaking. End-to-end is healthy.

## Open items / non-blocking concerns

1. **Runtime dependencies on this machine.** `lightgbm`, `joblib`,
   and `scikit-learn` weren't installed when the session opened —
   the existing `tests/test_scorer_v1_inference.py` errors on
   collection in that state (the `pytestmark` only checks the
   `model.lgb` file path, not the import). The new shim test file
   improves on this with an explicit "import lightgbm + joblib"
   check inside the `pytestmark` predicate, so it skips cleanly
   on a fresh checkout. **Suggested follow-on:** mirror the same
   check into `tests/test_scorer_v1_inference.py` so Phase 0's
   guards skip cleanly too. Out of scope for Session 01 (touching a
   Phase 0 file).

2. **Race rollover and the FeatureExtractor.** The shim re-creates
   `FeatureExtractor` on `reset()` only. Mid-episode race transitions
   (race A finishes, race B begins) reuse the same extractor —
   it's keyed by `market_id` internally so cross-race interference
   is avoided, but the per-race `_RunnerHistory` objects accumulate
   across the day. Memory is bounded (deques are pruned on each
   `update_history` call), but if Phase 2 finds memory growth on
   long days, calling `extractor.forget_market(prev_market_id)`
   on race rollover is the cheap fix.

3. **`requote_signal` is unreachable from the discrete head.** By
   design — the prompt's locked action set is `{NOOP, OPEN_BACK,
   OPEN_LAY, CLOSE}`. If Phase 2 wants the policy to re-quote, the
   discrete space needs widening. Not a Session 01 concern; logging
   here so Session 02 has it on the radar before the policy class
   gets too rigid.

4. **`cancel` flag is also unused.** Same reason. The env's
   passive-cancel pathway only fires from `cancel_signal > 0`;
   the discrete head never raises it. `CLOSE` cancels the passive
   *and* crosses the spread, which is the only "cancel" path the
   v2 design exposes. Documented in the env_shim docstring.

5. **No follow-on env audit findings.** Phase 1 hard constraint #1
   is "don't touch the env". Session 01 didn't find anything worth
   filing as a Phase −1 follow-on — the env's decoder, scalping
   layout, and pair tracking were all internally consistent.

## Numbers (regression guards live in tests/)

| Metric | Value |
|---|---:|
| `DiscreteActionSpace(14).n` | 43 |
| `DiscreteActionShim` env-action shape (max_runners=4, scalping) | (28,) |
| `DiscreteActionShim` obs_dim (max_runners=4, scalping) | 552 |
| Smoke run total reward (2 races, deterministic seed) | +3.402 |
| Smoke run final `info["day_pnl"]` | +£28.31 |
| Tests pass / total | 32 / 32 |

## Hand-off to Session 02

Session 02 imports `DiscreteActionShim` and `DiscreteActionSpace`
verbatim. The shim's `obs_dim`, `action_space.n`, and
`get_action_mask()` are the only surface the policy class needs.
The continuous heads (`stake`, `arb_spread`) are already plumbed
through `step(stake=…, arb_spread=…)` so Session 02 can wire its
Beta heads straight in without any shim revision.

If Session 02 surfaces a problem that wants a Session 01 change,
**stop and revisit Session 01**, don't patch in Session 02 — per
`purpose.md` "Each session is independently re-runnable" /
"if Session 01's design needs to change after Session 02 surfaces a
problem, that's a Session 01 revisit".
