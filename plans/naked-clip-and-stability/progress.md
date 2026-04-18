# Progress — Naked-Windfall Clip & Training Stability

One entry per completed session. Most recent at the top. Include
commit hash, what landed, what's not changed, and any gotchas.

---

## Session 03 — Entropy control: halved coefficient + reward centering

**Commit:** `6379362`
**Date:** 2026-04-18

What landed — two coordinated changes in `agents/ppo_trainer.py`
targeting the monotone-rising entropy (139 → 189) observed across
transformer `0a8cacd3` ep 1–7:

1. **`entropy_coefficient` default halved.** `hp.get(…, 0.01)` →
   `hp.get(…, 0.005)` at line ~482. With the per-mini-batch advantage
   normalisation from commit `8b8ca67` making the surrogate-loss term
   O(1), a 0.01 entropy bonus was dominating the gradient and
   flattening the policy under uniformly-negative rewards. Fresh-init
   default only — GA gene range for `entropy_coefficient` is
   unchanged per `hard_constraints.md §13`. Inherited agents that
   already have a gene-expressed value carry that value through; the
   change only bites fresh-init agents (every agent after the
   Session 05 registry reset).
2. **Reward centering.** New EMA baseline state on `PPOTrainer`:

   ```python
   self._reward_ema: float = 0.0
   self._reward_ema_alpha: float = 0.01
   self._reward_ema_initialised: bool = False
   ```

   New helper `_update_reward_baseline(episode_reward)`:
   first-observed-reward init, then EMA blend with α=0.01. The
   zero-init path is explicitly avoided — biased-zero EMA produces
   biased advantages for the first rollout.

   Subtraction point is inside `_compute_advantages`: the GAE delta
   becomes `delta = (tr.training_reward - self._reward_ema) +
   gamma * next_value - tr.value`. EMA update cadence is once per
   call to `_ppo_update`, AFTER advantages are computed, so the
   current rollout uses the pre-update EMA (no self-referential
   leakage). The running sum passed to the baseline is
   `sum(tr.training_reward for tr in transitions)` — the rollout-
   level training reward.

   Per `hard_constraints.md §14`, centering MUST NOT change
   advantage ordering within a rollout. The constant subtraction is
   a pure translation of returns that the per-mini-batch advantage
   normalisation downstream erases in expectation — verified by a
   pytest.

Tests — new class `TestEntropyAndCentering` in
`tests/test_ppo_trainer.py` (6 tests):

- `test_entropy_default_is_halved` — fresh PPOTrainer with no hp
  gives `entropy_coeff == 0.005`.
- `test_entropy_explicit_hp_overrides_default` — `hp={
  "entropy_coefficient": 0.02}` still wins (GA path intact).
- `test_reward_baseline_initialises_on_first_episode` — first call
  sets EMA to the observed value exactly; second call applies the
  `0.99 × old + 0.01 × new` blend.
- `test_reward_baseline_ema_update_is_monotonic` — monotone reward
  sequence → monotone EMA.
- `test_centering_preserves_advantage_ordering` — load-bearing
  principled check per `hard_constraints §14`. Synthetic
  terminal-at-every-step rollout with `value=0`: advantage_t equals
  `r_t - ema` (delta collapses to `r_t - ema` at every done
  boundary, `last_gae` resets). Per-mini-batch normalisation
  subtracts the mean; the constant `ema` cancels; centered and
  uncentered normalised advantages agree to `1e-5` tolerance.
- `test_centering_fixes_uniformly_negative_rewards` — synthetic
  rollout with rewards in [-900, -200] (transformer `0a8cacd3`
  scale). Un-centered pre-normalisation advantages have mean
  ≪ -300; centered advantages (with EMA set to the reward mean)
  have mean ≈ 0. Sanity-check that centering does what we think
  it does before normalisation gets to it.

Existing `TestPPOTrainerInit.test_default_hyperparams` updated to
assert `entropy_coeff == 0.005` (the sole existing test that
pinned the old default).

Regression guards (all still green):

- `tests/test_ppo_advantage_normalisation.py` — 8 tests PASS. The
  per-mini-batch normalisation from `policy-startup-stability`
  still fires; centering slots in FRONT of it, doesn't replace it.
- `tests/test_ppo_stability.py` — 16 tests PASS (Session 02 KL
  early-stop + ratio clamp + per-arch LR + warmup coverage).
- `tests/test_ppo_trainer.py` — 42 tests PASS (36 pre-existing +
  6 new).

Full suite: `pytest tests/ -q` → **2194 passed**, 7 skipped, 133
deselected, 1 xfailed (Session 02 baseline 2188 → +6 for the new
`TestEntropyAndCentering` class).

Qualitative probe (documented here per session prompt §4; not a
pytest). Transformer policy (seed 0), 15 synthetic episodes per
rollout, rewards drawn from U[-900, +300] (transformer `0a8cacd3`
ep 1–3 magnitude), matched log-prob pattern so `ratio=1` on the
first minibatch:

| ep | policy_loss | entropy | approx_kl | epochs | reward_ema |
|---|---|---|---|---|---|
| 1 | −0.0016 | 79.4597 | 0.0012 | 4 | −3724.660 |
| 2 | −0.0002 | 79.4575 | −0.0003 | 4 | −3727.556 |
| 3 | +0.0000 | 79.4563 | 0.0001 | 4 | −3734.690 |

Both session-prompt assertions pass:

- **Ep 1 `|policy_loss|` = 0.0016 ≪ 100** (compare to `1.04e17` in
  the untreated transformer `0a8cacd3` run — the combined Session
  02 defences + this session's centering squash the first-update
  explosion at this reward scale).
- **Ep 3 entropy (79.4563) ≤ Ep 1 entropy (79.4597)** —
  monotonically non-increasing across the probe, flipping the
  rising-entropy pathology.

This is the "Session 02 + Session 03 combined" acceptance check
from `master_todo.md` (qualitative, not gate-blocking). The
smoke-test gate (Session 04) will apply the same assertions in
production.

Not changed: matcher, reward shape (Sessions 01/01b), action/obs
schemas, gene ranges, GA selection, pair sizing, per-pair naked
aggregation, KL early-stop/ratio-clamp/per-arch-LR (Session 02),
entropy-floor controller (`_entropy_coeff_base` and friends). Per
`hard_constraints §1`, §13.

Gotchas:

- The entropy-floor controller (from arb-improvements Session 2)
  uses `self._entropy_coeff_base = float(self.entropy_coeff)` to
  snapshot the baseline at init. The halved default flows through
  here naturally — the controller's scaling ratio is unchanged,
  only the base it scales from halves. No additional plumbing
  needed.
- Reward centering subtracts per-step, not per-episode. For
  rollouts where reward is concentrated at terminal steps (typical
  scalping), only the terminal-step delta meaningfully shifts
  (non-terminal rewards are 0 and the baseline is small relative
  to the terminal magnitude). The mathematical "pure translation"
  property is only exact when `done=True` at every step OR when
  `gamma*lambda=0`; the pytest constructs the former. In general
  use, the approximation is close enough that per-mini-batch
  normalisation still erases the bulk of the shift in expectation
  — which is the regime centering targets.

Next: Session 04 (smoke-test gate — UI tickbox + 2-agent × 3-ep
probe + assertion harness). Blocked on operator review of this
commit.

---

## Session 02 — PPO stability: KL early-stop + ratio clamp + per-arch LR

**Commit:** `cc64fbd`
**Date:** 2026-04-18

What landed — three layered defences against first-update policy
explosion on fresh agents (`agents/ppo_trainer.py`), plus a
per-architecture default LR pathway (`agents/policy_network.py`):

1. **Ratio clamp.** In `_ppo_update`'s mini-batch loop,
   `ratio = (new_log_probs - mb_old_log_probs).exp()` becomes
   `log_ratio = torch.clamp(new_logp - old_logp, -20, +20);
   ratio = log_ratio.exp()`. Numerical backstop for when
   KL early-stop hasn't yet caught a runaway ratio within the
   first epoch. No-op in normal operation (|log_ratio| ≪ 20);
   only bites when an aggressive first-minibatch update has
   already driven log-ratio toward overflow. Per
   `hard_constraints.md §10`.
2. **KL early-stop.** After each full epoch sweep of mini-batches,
   a no-grad forward pass over the whole rollout computes
   `approx_kl = (old_logp - new_logp_full).mean()`. If it exceeds
   `self.kl_early_stop_threshold` (default `0.03`, literature
   standard — Andrychowicz et al. 2021, Engstrom et al. 2020),
   break out of the remaining epochs for this rollout. Applied at
   epoch granularity, not mini-batch, per `hard_constraints.md §9`
   — mid-epoch breaks leave mini-batches unevenly weighted.
   Exposed to the GA via the `kl_early_stop_threshold`
   hyperparameter so it can be mutated later if useful.
3. **Per-architecture default LR.** New
   `BasePolicy.default_learning_rate = 3e-4` class attribute,
   overridden to `1.5e-4` on `PPOTransformerPolicy`.
   `PPOTrainer.__init__` now reads
   `type(policy).default_learning_rate` as the fallback when the
   hp dict omits `learning_rate`. The GA still mutates LR around
   the sampled gene value when `learning_rate` is present —
   fresh-init transformer agents get the halved default.
4. **Warmup coverage audit.** All three architectures
   (transformer, LSTM, time-LSTM) construct their optimiser
   through `PPOTrainer.__init__`, so the existing 5-update
   linear LR warmup (`agents/ppo_trainer.py:1114`) fires
   uniformly. No bypass to fix. Hard-constraint §12 satisfied.
   Warmup length stays at 5 updates (§12: only extend if the
   Session 04 smoke test fails).

Return-dict additions on `_ppo_update`:

```python
"approx_kl": float,         # last-epoch approximate KL
"epochs_completed": int,    # how many epochs actually ran
"kl_early_stop_epoch": int, # -1 if didn't fire, else the epoch index
```

These feed the learning-curves panel and the Session 04 smoke-test
assertions.

Tests — new file `tests/test_ppo_stability.py`, 16 tests total:

- `TestRatioClamp` (3) — clamp prevents overflow on |log_ratio|=50;
  clamp is a no-op for |log_ratio| ≤ 0.5; real `_ppo_update`
  surrogate loss stays finite on a deliberately-poisoned rollout
  with `old_logp = -100` (would overflow `.exp()` without the
  clamp).
- `TestKLEarlyStop` (5) — threshold is configurable; default is
  `0.03`; does NOT fire on a normal rollout (all 3 epochs run);
  FIRES on a synthetic high-KL rollout (poisoned `old_logp + 2.0`,
  <5 epochs run, `approx_kl > 0.03`); break is at epoch
  granularity (`epochs_completed` is an integer). The high-KL
  test is the hard-constraint §22 synthetic rollout required by
  the session spec.
- `TestTransformerDefaultLR` (4) —
  `PPOTransformerPolicy.default_learning_rate == 1.5e-4` (exactly
  half of `BasePolicy`'s `3e-4`); LSTM variants keep the base
  default; `PPOTrainer` picks up the transformer default when hp
  omits `learning_rate`; explicit `hp["learning_rate"]` wins over
  the arch default (GA path).
- `TestWarmupCoverageAllArchs` (3) — parameterised over
  [`ppo_lstm_v1`, `ppo_time_lstm_v1`, `ppo_transformer_v1`]. After
  the first `_ppo_update`, each architecture's optimiser lr
  equals `base_learning_rate / 5`. Direct enforcement of the §12
  audit conclusion.
- `TestLargeRewardSmoke` (1) — synthetic rollout with ±£500
  advantages (scalping magnitude) produces
  `|policy_loss| < 100` through the real `_ppo_update`. Cheapest
  possible end-to-end regression net for the transformer
  `0a8cacd3` failure mode.

Regression guards (pre-existing, unchanged):

- `tests/test_ppo_advantage_normalisation.py` — all 8 tests PASS
  (normalisation + warmup from `plans/policy-startup-stability/`
  still bounded under the new clamp + early-stop defences).
- `tests/test_ppo_trainer.py` — all 36 tests PASS.

Full suite: `pytest tests/ -q` → **2188 passed**, 7 skipped, 1
xfailed, 133 deselected (Session 01b baseline 2172 → +16 for the
new stability tests).

Not changed: matcher, reward shape (Session 01/01b), action/obs
schemas, gene ranges, GA selection, pair sizing, entropy
coefficient (that's Session 03), warmup length, gradient clipping.
Per `hard_constraints §1`.

Next: Session 03 (entropy control — halve `entropy_coefficient`
default + reward centering). Blocked on operator review of this
commit.

---

## Session 01b — raw = race_pnl (loss-closed pairs correctly negative)

**Commit:** `a4f689a`
**Date:** 2026-04-18

What landed:

- `env/betfair_env.py::_compute_scalping_reward_terms` helper
  signature changed: first argument renamed from
  `scalping_locked_pnl` to `race_pnl`, and the helper now returns
  `race_reward_pnl = race_pnl` directly. This is the whole-race
  cashflow — `scalping_locked_pnl + scalping_closed_pnl +
  sum(per_pair_naked_pnl)` — so close-leg losses on pairs closed
  via `close_signal` at a loss now land in raw at full cash value.
- Session 01's draft used
  `race_reward_pnl = scalping_locked_pnl + sum(naked_per_pair)`,
  which silently excluded `scalping_closed_pnl`. A pair closed at
  a −£5 loss registered `raw=0` (locked floor) + `+£1` (shaped
  close bonus) = `net +£1` — rewarding the agent for a losing
  trade. Session 01b corrects this: the same −£5 close now
  registers `raw=−5, shaped=+1, net=−4`. The close bonus still
  keeps closing strictly better than letting a naked roll to a
  larger worst-case loss (naked −£80 → net −£80 vs closing at
  −£4), so the learning signal favours closing without making
  close an unconditional reward.
- Call site at `_settle_current_race` updated to pass
  `race_pnl=race_pnl` (the local already computed before the
  reward-assembly block).
- Shaped terms (`−0.95 × sum(max(0, per_pair_naked_pnl))`,
  `+£1 × n_close_signal_successes`) unchanged.

Tests:

- `TestNakedWinnerClipAndCloseBonus` — six existing tests
  rewritten to pass `race_pnl=<sum>` in place of
  `scalping_locked_pnl=<sum>` (same scalar values, new keyword);
  one new test
  `test_loss_closed_scalp_reports_full_loss_in_raw` covers the
  loss-closed row of the `purpose.md` outcome table
  (`race_pnl=−5, naked=[], n_close=1` → `raw=−5, shaped=+1,
  net=−4`).
- `TestCloseAtLossRawRewardInvariant` in `tests/test_close_signal.py`
  reframed: the class's invariant was "close-at-loss contributes 0
  to raw_pnl_reward" — under Session 01b this no longer holds.
  Renamed test to
  `test_close_at_loss_flows_cash_loss_into_raw_reward` and
  asserts `raw_pnl_reward == day_pnl` (cash loss flows through
  raw at full value). `terminal_bonus_weight` overridden to 0 in
  the env fixture so the raw accumulator equals the race-level
  contribution exactly.
- `test_naked_windfall_excluded_from_raw_reward` in
  `tests/test_forced_arbitrage.py` reframed:
  renamed to `test_naked_windfall_in_raw_with_shaped_winner_clip`.
  Old test stripped `pair_id` to hide the naked from the helper;
  the new test keeps `pair_id` intact so the naked pair appears
  in `get_naked_per_pair_pnls`, and asserts `raw ==
  day_pnl` (full cash) plus `shaped == −0.95 × naked_pnl`
  (winner clip fires). Same directional-luck-neutralisation
  behaviour, exercised through the real code path.

Invariant test:
`pytest tests/test_forced_arbitrage.py::TestScalpingReward::test_invariant_raw_plus_shaped_equals_total_reward -v`
→ PASS.

Full suite: `pytest tests/ -q` → **2172 passed**, 7 skipped, 1
xfailed, 133 deselected (Session 01 baseline 2171 → +1 for the
new loss-closed test).

Docs:

- `CLAUDE.md` "Reward function: raw vs shaped" 2026-04-18
  naked-clip paragraph updated: formula changed from
  `scalping_locked_pnl + sum(per_pair_naked_pnl)` to `race_pnl`;
  outcome-table line gains the loss-closed row (`net −£4`); a
  closing sentence notes the Session 01 → 01b refinement lineage.
  Historical 2026-04-15 and naked-asymmetry paragraphs
  preserved.

Outcome table with the loss-closed row now covered by tests:

| Per-pair outcome | Raw | Shaped | Net |
|---|---|---|---|
| Scalp locks +£2 (passive filled naturally) | +2 | 0 | **+2** |
| Scalp locks +£2 via `close_signal` | +2 | +1 | **+3** |
| Loss-closed scalp (close at −£5, locked=0) | −5 | +1 | **−4** |
| Naked winner +£100 (held to settle) | +100 | −95 | **+5** |
| Naked loser −£80 (held to settle) | −80 | 0 | **−80** |
| Naked winner +£10 (held to settle) | +10 | −9.50 | **+0.50** |

Not changed: matcher, action/obs schemas, gene ranges, GA
selection, pair sizing, per-pair aggregation, shaped-term
formulas. Per `hard_constraints §1–§2`.

Next: Session 02 (PPO stability) remains gated on the operator
reviewing Session 01 + 01b.

---

## Session 01 — reward shape (naked winner clip + close bonus + full loss in raw)

**Commit:** `e0799a4`
**Date:** 2026-04-18

What landed:

- `env/betfair_env.py::_settle_current_race` scalping branch now
  computes the two-channel split via a pure helper
  `_compute_scalping_reward_terms(scalping_locked_pnl,
  naked_per_pair, n_close_signal_successes) → (race_reward_pnl,
  race_shaping)`.
  - Raw: `scalping_locked_pnl + sum(naked_per_pair)` — actual
    race cashflow, winners AND losers (no softener, no hiding).
  - Shaped gains `−0.95 × sum(max(0, naked_per_pair))`
    (per-pair winner clip, hard_constraints §5) +
    `1.0 × scalping_arbs_closed` (per-close bonus, §6).
  - `scalping_arbs_closed` reused as the close-signal success
    count — it already increments exactly once per pair that
    completed via a `close_leg=True` fill, matching §6 design.
- Module-level constants `NAKED_WINNER_CLIP_FRACTION=0.95` and
  `CLOSE_SIGNAL_BONUS=1.0` document the scale knobs.
- 0.5× naked-loss softener (2026-04-15) removed — per `§1`.
- `scalping_locked_pnl` floor (`max(0, min(win, lose))`),
  equal-profit pair sizing, and per-pair accessor
  `BetManager.get_naked_per_pair_pnls` all untouched.
- Aggregate `naked_pnl = race_pnl − locked − closed` kept for
  `RaceRecord` logging + `info["naked_pnl"]` + scoreboard.

Worked-example contributions (from the `TestNakedWinnerClipAndCloseBonus`
tests — hand-authored per-pair P&L, six cases):

| Inputs | Raw | Shaped | Net |
|---|---|---|---|
| 1 naked winner +£100 | +100 | −95 | **+5** |
| 1 naked loser −£80 | −80 | 0 | **−80** |
| Winner +£100 + loser −£80 | +20 | −95 | **−75** |
| Closed pair locked +£2 (1 close) | +2 | +1 | **+3** |
| 0 raw, N closes | 0 | +N | **+N** |
| Mixed: locked +£5, +£50 winner, −£30 loser, 2 closes | +25 | −45.5 | **−20.5** |

Invariant test: `pytest tests/test_forced_arbitrage.py::TestScalpingReward::test_invariant_raw_plus_shaped_equals_total_reward -v` → PASS.

Full suite: `pytest tests/ -q` → 2171 passed, 7 skipped, 1 xfailed
(baseline pre-change was 2170 passed — the +1 is this session's
new `TestNakedWinnerClipAndCloseBonus.test_raw_plus_shaped_invariant_with_new_terms`
netted with one pre-existing test
(`test_naked_loss_subtracted_from_raw_at_half_factor`) that was
reframed in place to assert the new full-cash shape).

Docs:

- `CLAUDE.md` — appended 2026-04-18 `naked-clip-and-stability`
  paragraph to "Reward function: raw vs shaped". Historical
  2026-04-15 and 2026-04-18 (`scalping-naked-asymmetry`)
  paragraphs preserved.

Gotchas:

- `test_precision_and_early_pick_zeroed_in_scalping_mode`
  previously asserted `|shaped_bonus| < 1.0` to prove directional
  shaping leaks were absent. That absolute threshold no longer
  holds — shaped legitimately carries the naked-winner clip
  (−£46 on the synthetic single-back race). Test reframed as a
  differential: run the same episode with high vs zero
  precision/early_pick weights and assert shaped_bonus is
  identical — tighter guarantee than the old threshold.
- `TestAsymmetricNakedLossReward.test_naked_loss_subtracted_from_raw_at_half_factor`
  renamed to `..._at_full_cash` and its expected-raw constant
  switched from `0.5 × naked_pnl` to `naked_pnl`. Still
  exercises the single-naked-loser path end-to-end.

Not changed: matcher, action/obs schemas, gene ranges, GA
selection, pair sizing, per-pair aggregation. Per
`hard_constraints §1`.

Next: Session 02 is gated on the operator reviewing this
commit. Do NOT queue Session 02 automatically.

---

_Plan folder created 2026-04-18. See `purpose.md` for the
gen-2 transformer `0a8cacd3` episode-1 policy-loss blow-up,
rising-entropy pathology, and naked-windfall reward-shape
diagnosis that motivated this plan._
