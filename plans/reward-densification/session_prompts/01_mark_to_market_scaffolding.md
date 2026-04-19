# Session 01 prompt — Mark-to-market scaffolding (knob at 0, byte-identical default)

## PREREQUISITE — read first

- [`../purpose.md`](../purpose.md) — diagnosis from
  entropy-control-v2 + fill-prob-aux-probe, MTM design
  sketch, success criteria.
- [`../hard_constraints.md`](../hard_constraints.md). §5–§9
  (semantics + formulas), §10–§11 (knob and default),
  §12–§14 (telemetry + invariant), §15–§16 (tests), §19–§20
  (reward-scale change).
- [`../master_todo.md`](../master_todo.md) — Session 01
  deliverables.
- `env/betfair_env.py` — the file being edited. Key
  existing reference points to locate before editing:
  - `_settle_current_race` — where race-level raw P&L is
    accumulated. MTM does NOT plumb through here; it
    drives `shaped_bonus` on a per-step path.
  - `BetManager.bets` — iterate open bets at each step.
    Resolved bets are excluded from MTM automatically
    because their `is_matched` / settle state removes them
    from the MTM computation (see §5 / §9 of
    hard_constraints).
  - `shaped_bonus` / `raw_pnl_reward` accumulators on
    `EpisodeStats` (or the equivalent container the env
    writes to on each step).
  - The existing `info` dict construction on each env
    step. We add `mtm_delta` there.
  - CLAUDE.md "Order matching: single-price, no walking"
    — do NOT use this to compute MTM. MTM reads LTP
    only, per §5 of hard_constraints.
- `plans/entropy-control-v2/progress.md` — the 2026-04-19
  Validation entry evidencing why this work exists.
- `plans/naked-clip-and-stability/lessons_learnt.md`
  2026-04-18 entry on "reward centering: units mismatch"
  — parallel failure mode; the integration-test pattern
  (spy on the real call site, not a mocked helper) is the
  guard we need for the MTM telescope invariant.

## Locate the code

```
grep -n "shaped_bonus\|raw_pnl_reward\|shaped_reward" env/betfair_env.py
grep -n "bet_manager\|BetManager\|bets\b" env/betfair_env.py
grep -n "_settle_current_race\|settle_" env/betfair_env.py
grep -n "info\[" env/betfair_env.py | head -30
```

Confirm before editing:

1. The per-step accumulator for shaped contributions lives
   on the env (typically an `EpisodeStats`-like dataclass
   or direct attribute). The mark-to-market delta adds to
   the SAME accumulator that `early_pick_bonus`,
   `precision_bonus`, `efficiency_penalty`, and (in
   scalping mode) the close-signal bonus / naked-winner
   clip contribute to.
2. `self.bet_manager.bets` at each step contains only
   currently-open bets; resolved bets have been moved to
   `self.all_settled_bets` (per CLAUDE.md "info['realised_pnl']
   is last-race-only" — the same invariant applies here).
3. LTP per runner is accessible on the current tick via
   the RunnerSnap records. The matcher already reads LTP
   for its junk-filter; the code path for "get current LTP
   for a runner" exists and can be reused.

## What to do

### 1. Add the MTM computation method

New helper on `BetfairEnv`:

```python
def _compute_portfolio_mtm(
    self, current_ltps: dict[int, float],
) -> float:
    """Sum mark-to-market P&L across all currently-open bets.

    Uses LTP as the current market reference price (per
    plans/reward-densification/hard_constraints.md §5). A
    bet with no LTP available (runner withdrawn,
    unpriceable) contributes zero. Resolved bets are
    excluded — ``self.bet_manager.bets`` only contains
    open positions.

    Formulas (hard_constraints §6 / §7):

    - Back: ``S * (P_matched - LTP) / LTP``
    - Lay:  ``S * (LTP - P_matched) / LTP``

    Returns the portfolio-level sum (pounds).
    """
    total = 0.0
    for bet in self.bet_manager.bets:
        # Only open, matched bets with known LTP contribute.
        if bet.matched_stake <= 0.0:
            continue
        ltp = current_ltps.get(bet.selection_id)
        if ltp is None or ltp <= 1.0:
            continue  # unpriceable
        if bet.side == "back":
            total += bet.matched_stake * (
                bet.average_price - ltp
            ) / ltp
        else:  # "lay"
            total += bet.matched_stake * (
                ltp - bet.average_price
            ) / ltp
    return total
```

Notes:

- `bet.side` attribute exists on `Bet` (either "back" or
  "lay"); verify at grep time.
- The `matched_stake <= 0.0` guard handles in-flight orders
  with no fill yet — their MTM is zero by definition.
- The `ltp <= 1.0` guard is belt-and-braces — exchange
  prices must exceed 1.0; anything at or below is noise
  from a junk-filtered tick.

### 2. Wire the knob into `__init__`

```python
# Reward config read (same pattern as existing knobs)
reward_cfg = config.get("reward", {})
self.mark_to_market_weight: float = float(
    reward_cfg.get("mark_to_market_weight", 0.0)
)
# Per-race running snapshot — last tick's portfolio MTM.
# Reset on race-start; see _start_new_race (or equivalent).
self._mtm_prev: float = 0.0
# Per-episode cumulative shaped contribution from MTM —
# telemetry only (hard_constraints §13).
self._cumulative_mtm_shaped: float = 0.0
```

Comment cross-link:

```python
# Per-step mark-to-market shaping
# (plans/reward-densification, Session 01, 2026-04-19).
# Default 0.0 => no-op; rollouts byte-identical to
# pre-change. When > 0, emits a shaped contribution
# proportional to the delta in open-position MTM
# between consecutive ticks. Cumulative shaped MTM
# across a race telescopes to zero at settle (resolved
# bets drop out of the MTM sum; the final delta unwinds
# whatever was on the books). See
# hard_constraints.md §5-§9 for the formal semantics.
```

### 3. Emit per-step delta on the main step path

Inside `step()` (or whichever method drives the per-tick
reward assembly), AFTER the bet-matching logic updates
`self.bet_manager.bets` for the current tick, BEFORE the
shaped-reward accumulation:

```python
current_ltps = self._current_ltps()  # helper may need
                                      # extracting — see
                                      # existing LTP
                                      # access patterns
mtm_now = self._compute_portfolio_mtm(current_ltps)
mtm_delta = mtm_now - self._mtm_prev
self._mtm_prev = mtm_now

# Shaped contribution. When weight == 0 (default), this
# adds 0.0 to the bucket and the branch is a no-op.
mtm_shaped = self.mark_to_market_weight * mtm_delta
shaped_bonus_this_step += mtm_shaped
self._cumulative_mtm_shaped += mtm_shaped

info["mtm_delta"] = float(mtm_delta)  # pre-weight; §14
```

Naming: `shaped_bonus_this_step` is illustrative — use the
actual accumulator name in the code. Grep for it first.

### 4. Reset on race boundary

In `_start_new_race` (or wherever the env initialises a
fresh `BetManager` per race):

```python
self._mtm_prev = 0.0
# NB: do NOT reset _cumulative_mtm_shaped here — it's a
# per-episode total that spans all races in the episode.
# Reset that at episode start instead.
```

At episode start (`reset()` / wherever
`EpisodeStats`-level state is initialised):

```python
self._cumulative_mtm_shaped = 0.0
```

### 5. Expose in rollup + JSONL

`EpisodeStats` (or equivalent dataclass) gains:

```python
cumulative_mtm_shaped: float = 0.0
mtm_weight_active: float = 0.0
```

At episode end (the same place `total_reward`, `day_pnl`,
`arbs_closed` etc. get finalised):

```python
ep_stats.cumulative_mtm_shaped = self._cumulative_mtm_shaped
ep_stats.mtm_weight_active = self.mark_to_market_weight
```

`agents/ppo_trainer.py::_log_episode` gains:

```python
record["mtm_weight_active"] = round(
    float(ep.mtm_weight_active), 6,
)
record["cumulative_mtm_shaped"] = round(
    float(ep.cumulative_mtm_shaped), 6,
)
```

Existing JSONL readers must tolerate absence on pre-change
rows (same backward-compat pattern as `alpha` /
`log_alpha` in entropy-control-v2).

### 6. Race-boundary telescope check

At the end of each race (inside `_settle_current_race`,
right after settling all open bets), `self._mtm_prev` should
equal zero (no bets open means portfolio MTM is zero).
Assert this in an internal consistency check ONLY when a
debug flag is set (don't burn cycles in production). For
the tests, the `test_mtm_telescopes_to_zero_at_settle` test
verifies this externally — the internal check is optional.

### 7. Tests

New test class `TestMarkToMarketShaping` in
`tests/test_betfair_env.py` (or a new
`tests/test_mark_to_market.py` — choose based on file
size / ownership):

```python
class TestMarkToMarketShaping:
    def test_mark_to_market_weight_default_is_zero(self):
        """Fresh BetfairEnv from default config has
        ``mark_to_market_weight == 0.0`` — the byte-
        identical-migration guarantee."""

    def test_mtm_delta_zero_when_no_open_bets(self):
        """With no open bets on any step, ``mtm_delta``
        is 0 and the shaped accumulator is unchanged."""

    def test_mtm_back_formula_matches_spec(self):
        """Stub with one open back bet, stake 10.0,
        average price 8.0. Feed LTP=6.0: expected
        ``mtm = 10 * (8-6) / 6 = +3.333...``"""

    def test_mtm_lay_formula_matches_spec(self):
        """Symmetric. One open lay, stake 10, price 4.0,
        LTP=5.0: expected ``mtm = 10 * (5-4) / 5 = +2.0``."""

    def test_mtm_zero_when_ltp_missing(self):
        """A bet on a runner with no LTP on the current
        tick contributes 0. No crash, no NaN."""

    def test_mtm_telescopes_to_zero_at_settle(self):
        """Scripted 5-tick race: open a back bet at tick 1,
        LTP drifts over ticks 2..4, race settles at tick 5.
        Cumulative shaped MTM across the 5 ticks must be
        zero within float tolerance — every gain accrued
        to ``_cumulative_mtm_shaped`` is unwound when the
        position resolves on settle (the last mtm_delta
        is minus the running MTM, not zero)."""

    def test_invariant_raw_plus_shaped_with_nonzero_weight(self):
        """NEW integration-style guard. Run a full
        scripted rollout (3 races) with
        ``mark_to_market_weight=0.05``; assert
        ``ep.raw_pnl_reward + ep.shaped_bonus ≈
        ep.total_reward`` within float tolerance.
        Catches telescope-break bugs that unit tests on
        the formula alone would miss — the same pattern
        as
        ``test_real_ppo_update_feeds_per_step_mean_to_baseline``
        guards the reward-centering units contract per the
        2026-04-18 units-mismatch lesson
        (plans/naked-clip-and-stability/lessons_learnt.md)."""

    def test_mtm_weight_zero_byte_identical_rollout(self):
        """Construct two identical rollouts — one via the
        pre-change code path simulated by setting
        ``mark_to_market_weight=0``, one via the new
        code path — and assert per-episode
        ``(raw_pnl_reward, shaped_bonus, total_reward)``
        match to floating-point epsilon. When weight is 0
        there must be no observable change."""

    def test_info_mtm_delta_field_present(self):
        """Env step ``info`` dict carries ``mtm_delta`` as
        a float whenever any open position exists."""
```

Extend
`tests/test_forced_arbitrage.py::TestScalpingReward::test_invariant_raw_plus_shaped_equals_total_reward`:

Add a second parametrisation (or a new test method) that
runs the same scenario with `mark_to_market_weight=0.05`;
assert the invariant still holds.

The `test_invariant_raw_plus_shaped_with_nonzero_weight` in
the new test class is the load-bearing regression guard per
the 2026-04-18 units-mismatch lesson. Unit tests on the
MTM formula alone are insufficient — they can't catch a
telescope-break that only surfaces in a full rollout.

### 8. Scripted-rollout qualitative probe

Documented in `progress.md`, not a pytest. Construct:

- `BetfairEnv` with a scripted 3-race day,
  `mark_to_market_weight=0.05`.
- Script tick sequences where the agent holds an open back
  at price 8.0, LTP drifts 8.0 → 7.0 → 6.5 → 7.5 → 8.0
  across 5 ticks, settles on tick 5.
- Record per-step `mtm_delta`, cumulative
  `_cumulative_mtm_shaped`, and `ep_stats.shaped_bonus` at
  race end.

Assert in the progress-entry text:

- Cumulative `mtm_shaped` across the 5-tick race is zero
  to floating-point tolerance.
- `ep_stats.raw_pnl_reward + ep_stats.shaped_bonus ==
  ep_stats.total_reward` for the 3-race episode.
- `info["mtm_delta"]` on the step where LTP went 8.0 → 7.0
  is positive (the back bet's position value grew as LTP
  fell).

This is the smoke check that the mechanism actually
works in a real env loop. Don't gate the commit on the
probe; gate on the pytest suite. But document the result in
the progress entry.

### 9. CLAUDE.md

Add a new paragraph under "Reward function: raw vs
shaped":

```
### Per-step mark-to-market shaping (2026-04-19)

``shaped_bonus`` also accumulates a per-tick contribution
proportional to the delta in open-position mark-to-market
P&L. For each open back bet of stake ``S`` at average
matched price ``P_matched``, with current runner LTP
``P_current``:

    mtm_back = S * (P_matched - P_current) / P_current
    mtm_lay  = S * (P_current - P_matched) / P_current

Portfolio MTM = sum across open bets. The per-step
shaped contribution is
``mark_to_market_weight * (MTM_t - MTM_{t-1})``. Default
weight ``0.0`` (no-op — byte-identical to pre-change).
Project-wide default lives in ``config.reward.mark_to_market_weight``.

Key property: cumulative ``shaped_mtm`` across a race
telescopes to zero at settle (resolved bets drop out of
the MTM sum; the last mtm_delta unwinds the running
portfolio value). The ``raw + shaped ≈ total`` invariant
holds episode-by-episode; the shaping only redistributes
existing race-level P&L signal through the steps that
caused it. See
``plans/reward-densification/purpose.md`` for the
motivation (entropy-control-v2 Validation concluded
entropy isn't the training-signal lever; reward sparsity
is). ``test_invariant_raw_plus_shaped_with_nonzero_weight``
in ``tests/test_mark_to_market.py`` is the load-bearing
regression guard per the 2026-04-18 units-mismatch lesson.
```

Historical entries (2026-04-15, 2026-04-18) stay preserved.

### 10. Full suite

```
pytest tests/ -q
```

Must be green. Regression guards:

- `tests/test_ppo_trainer.py::TestTargetEntropyController`
  — untouched; reward-path changes don't affect the
  controller.
- `tests/test_forced_arbitrage.py::TestScalpingReward::
  test_invariant_raw_plus_shaped_equals_total_reward` —
  must stay green at weight=0 (byte-identical) AND
  weight=0.05 (telescope closes at settle).
- `tests/test_ppo_trainer.py::TestEntropyAndCentering::
  test_real_ppo_update_feeds_per_step_mean_to_baseline`
  — untouched.
- `tests/test_smoke_test.py` — untouched; the
  tracking-error gate doesn't care about the reward
  shape.

### 11. Commit

```
feat(env): per-step mark-to-market shaping (weight=0 default)

Add per-tick shaped-reward contribution proportional to the
delta in open-position mark-to-market P&L. For each open
back bet of stake S at average matched price P_matched with
current LTP P_current:

  mtm_back = S * (P_matched - P_current) / P_current
  mtm_lay  = S * (P_current - P_matched) / P_current

Portfolio MTM sums across open bets; the per-step shaped
contribution is ``mark_to_market_weight * (MTM_t - MTM_{t-1})``.
Default weight 0.0 -- rollouts byte-identical to pre-change.

Why: entropy-control-v2's 2026-04-19 Validation concluded
the target-entropy controller works correctly but entropy
isn't the training-signal lever. The follow-on
``fill-prob-aux-probe`` confirmed aux-head supervised signal
alone doesn't break the bifurcation either. The diagnosis
is reward sparsity: the policy sees ~0 gradient on 99% of
its steps because settle P&L only arrives hundreds-to-
thousands of ticks after the decisions that caused it. MTM
surfaces the market's own instantaneous valuation of open
positions as a per-tick shaped signal, without changing the
raw P&L accumulator -- the cumulative shaped MTM
telescopes to zero at settle (resolved bets drop out of the
MTM sum).

Key property: raw + shaped ~= total holds episode-by-
episode. The shaping only redistributes existing race-level
P&L signal through the steps that caused it; total reward
per race is unchanged (to floating-point tolerance).

Changes:
- BetfairEnv._compute_portfolio_mtm helper sums mark-to-
  market across open bets using current LTPs; missing LTP
  => that bet contributes 0 (matches the matcher's
  unpriceable rule).
- Per-step reward assembly emits
  ``mtm_delta = MTM_t - MTM_{t-1}``; multiplied by weight
  and added to the shaped-bonus accumulator.
- ``_mtm_prev`` reset on race boundary;
  ``_cumulative_mtm_shaped`` reset on episode boundary.
- Per-episode JSONL row gains optional ``mtm_weight_active``
  and ``cumulative_mtm_shaped`` fields (downstream readers
  tolerate absence on pre-change rows).
- ``info["mtm_delta"]`` on every env step carries the
  pre-weight delta for diagnostics.

Tests: N new in tests/test_mark_to_market.py (or
test_betfair_env.py). ``test_mtm_telescopes_to_zero_at_settle``
and ``test_invariant_raw_plus_shaped_with_nonzero_weight``
are the load-bearing regression guards per the 2026-04-18
units-mismatch lesson (plans/naked-clip-and-stability/
lessons_learnt.md) -- unit tests on the formula alone can't
catch a telescope-break that only surfaces in a full rollout.

Not changed: matcher, controller, PPO stability defences,
action/obs schemas, gene ranges, smoke-gate assertion, raw
P&L accounting. Per
plans/reward-densification/hard_constraints.md s1-s4.

pytest tests/ -q: <delta>.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## Cross-session rules

- No matcher changes.
- No controller changes (entropy-control-v2 stays
  untouched).
- No smoke-gate changes.
- No GA gene-range changes.
- No PPO-stability changes (those landed in
  `naked-clip-and-stability` Session 02).
- Default weight is 0.0 for Session 01. Non-zero default
  is Session 02.

## After Session 01

1. Append a `progress.md` entry: commit hash, the MTM
   scaffolding, test counts, scripted-rollout probe
   results (cumulative shaped MTM ≈ 0 at settle; invariant
   holds at weight=0.05).
2. Hand back for Session 02 (plan-level default weight).
