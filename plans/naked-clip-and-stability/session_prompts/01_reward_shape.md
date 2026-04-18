# Session 01 prompt — Reward shape: full cash in raw, 95% winner clip + close bonus in shaped

## PREREQUISITE — read first

- [`../purpose.md`](../purpose.md) — gen-2 `0a8cacd3` evidence
  table and the outcome-table showing the new per-pair reward
  contributions (naked winner +£100 → raw+100/shaped−95,
  naked loser −£80 → raw−80/shaped 0, closed scalp +£2 →
  raw+2/shaped+1).
- [`../hard_constraints.md`](../hard_constraints.md) — all 28
  non-negotiables. The ones that will bite this session:
  §3 (raw+shaped invariant), §4 (raw reports actual cash),
  §5–§6 (shaped channel gets the clip and bonus), §7 (design
  intent: no reward for directional luck — now in shaped
  instead of raw-masking), §20 (tests green on commit), §23
  (worked-example coverage), §24 (reward-scale change
  protocol), §25 (CLAUDE.md update).
- `CLAUDE.md` — "Reward function: raw vs shaped" and
  "Bet accounting: matched orders, not netted positions".
- `env/betfair_env.py::_settle_current_race` — search for
  `scalping_locked_pnl` (the scalping reward branch starts
  around line 1989 in the current file).
- `env/bet_manager.py::get_naked_per_pair_pnls` — accessor
  this session builds on; read once so you know its
  semantics (insertion order, skips complete pairs, skips
  unsettled aggressives).

## Locate the code

```
grep -n "scalping_locked_pnl\|scalping_naked_pnl\|naked_loss_term\|scalping_closed_pnl" env/betfair_env.py
grep -n "get_naked_per_pair_pnls" env/bet_manager.py
grep -n "test_invariant_raw_plus_shaped" tests/test_forced_arbitrage.py
```

Confirm before editing:
1. There's exactly ONE call site for `naked_loss_term` in
   `_settle_current_race`. If grep shows two, both must move
   together to preserve §3.
2. `scalping_closed_pnl` is the correct source for counting
   close-signal successes (§6). If it's not — i.e. if there's
   a separate counter somewhere else — use the one that
   reflects pairs that actually went `incomplete → complete`
   via `close_signal`, not action-emission counts.

## What to do

### 1. Refactor the scalping reward branch

Current shape (around line 2307–2308 in `env/betfair_env.py`):

```python
naked_per_pair = bm.get_naked_per_pair_pnls(
    market_id=race.market_id,
)
naked_loss_term = sum(min(0.0, p) for p in naked_per_pair)
race_reward_pnl = scalping_locked_pnl + 0.5 * naked_loss_term
```

Replace with two separate terms:

```python
naked_per_pair = bm.get_naked_per_pair_pnls(
    market_id=race.market_id,
)

# Raw: full naked cash (winners + losers) — truthful race P&L.
naked_full_term = sum(naked_per_pair)
race_reward_pnl = scalping_locked_pnl + naked_full_term

# Shaped: 95% clip on naked winners (neutralises training
# incentive for directional luck) + per-close bonus
# (positive gradient for close_signal substitution).
n_close_signal_successes = _count_close_signal_successes(
    bm, race.market_id,  # or equivalent — see §6 in hard_constraints
)
naked_winner_clip = -0.95 * sum(max(0.0, p) for p in naked_per_pair)
close_bonus = 1.0 * n_close_signal_successes

race_shaping = naked_winner_clip + close_bonus
```

Whatever accumulator you have for `shaped_bonus` in the scalping
branch gains `race_shaping` added to it. Don't create a NEW
shaped accumulator — add to whatever existing one already feeds
`info["shaped_bonus"]`.

### 2. Count close-signal successes

A "close-signal success" is a pair that went `incomplete →
complete` because of a `close_signal` action. Two plausible
source signals:

- **`scalping_closed_pnl` construction block** (around line
  2249–2261 in `env/betfair_env.py`). Iterates paired
  positions, sums `b.pnl` for closing-leg bets. Count the
  pairs visited instead of summing the P&L — that count is
  `n_close_signal_successes`.
- **`BetManager` or the action handler**, if either already
  tracks a "closes executed this race" counter.

Prefer reusing an existing counter over adding one. If nothing
exists and you must add, put it on `RaceRecord` (the
`_race_records` element) alongside `naked_pnl` and
`locked_pnl` — that keeps it in the same accounting family
and available to downstream logging.

Do NOT count close_signal ACTIONS that didn't reduce exposure
(no-op emissions). The bonus is for successful closes, not
for the agent spamming the action button.

### 3. Remove the 0.5× softener

The existing `race_reward_pnl = scalping_locked_pnl + 0.5 *
naked_loss_term` drops the `0.5`. Naked losers now enter raw
at 100%. Check `scalping-naked-asymmetry/progress.md` for the
context — that 0.5× was explicitly preserved "per
`hard_constraints.md §1` (ONE thing changed: aggregation
level)". This plan's §1 authorises removing it.

### 4. Preserve aggregate `naked_pnl` for logging

The aggregate `naked_pnl = race_pnl - scalping_locked_pnl -
scalping_closed_pnl` (line 2268) still feeds `info["naked_pnl"]`,
the scoreboard, and the evaluator. Keep it — it's informational,
not reward-path. Same treatment as `scalping-naked-asymmetry`.

### 5. Update CLAUDE.md

Per `hard_constraints.md §25`. In the "Reward function: raw vs
shaped" section, add a new paragraph after the 2026-04-18
`scalping-naked-asymmetry` paragraph:

> **Scalping mode (2026-04-18 — `naked-clip-and-stability`):**
> the reward shape now splits naked handling across the two
> channels. Raw becomes `scalping_locked_pnl +
> sum(per_pair_naked_pnl)` — actual race cashflow, winners
> AND losers. Shaped absorbs the training-signal adjustments:
> `shaped += −0.95 × sum(max(0, per_pair_naked_pnl))` neuters
> 95% of any naked windfall, and `shaped += +£1 per
> close_signal success` gives a positive gradient for
> substituting closes for naked rolls. The 0.5× softener from
> 2026-04-15 is removed — naked losses now land at full cash
> value in raw. Net effect per per-pair outcome: scalp locks
> +£2 → net +£3 reward, naked winner +£100 → net +£5 reward,
> naked loser −£80 → net −£80 reward. Reward-scale change;
> scoreboard rows from before this fix are not directly
> comparable.

Do NOT delete the 2026-04-15 or 2026-04-18 (naked-asymmetry)
historical paragraphs — they stay as record.

### 6. Tests

New class `TestNakedWinnerClipAndCloseBonus` in
`tests/test_forced_arbitrage.py`. The six cases from
`master_todo.md` Session 01:

```python
class TestNakedWinnerClipAndCloseBonus:
    def test_single_naked_winner_raw_full_shaped_clipped(self):
        """Naked winner +£100 → raw=+100, shaped=−95, net=+5."""

    def test_single_naked_loser_raw_full_shaped_zero(self):
        """Naked loser −£80 → raw=−80, shaped=0, net=−80."""

    def test_mixed_win_and_loss_per_pair_aggregation(self):
        """Winner +£100, loser −£80 in same race →
        raw=+20 (sum), shaped=−95 (clip on winner only),
        net=−75."""

    def test_scalp_using_close_signal_earns_bonus(self):
        """Closed pair with locked_pnl=+£2 → raw=+2,
        shaped=+1 (close bonus), net=+3."""

    def test_multiple_close_signal_successes_accumulate(self):
        """N closes in one race → shaped += N × 1.0."""

    def test_raw_plus_shaped_invariant_with_new_terms(self):
        """raw + shaped ≈ total_reward across a
        mixed-outcome race. Exercises all four new terms
        simultaneously."""
```

For the random-EV sanity (no-luck-reward invariant): the new
shape is strictly stronger than the old — naked winners
contribute +5% net reward, so a random-EV naked population now
produces a slightly negative net reward in expectation
(−0.95 × E[max(0, X)] + E[X] ≈ −0.5 × 0.95 × E[|X|] +
0 = −0.475 × E[|X|]). That's consistent with §7 intent and
doesn't need its own test — the per-pair asymmetry test from
`scalping-naked-asymmetry` already guarantees the loss side
lands.

### 7. Run the invariant test explicitly

After the change:

```
pytest tests/test_forced_arbitrage.py::TestScalpingReward::test_invariant_raw_plus_shaped_equals_total_reward -v
```

Must pass. If it fails, the new shaped terms didn't land in
the shaped accumulator — trace where `race_shaping` is added
and fix the plumbing. DO NOT relax the invariant.

### 8. Full suite

```
pytest tests/ -q
```

Must be green. Frontend `ng test` NOT required for this
session (no UI change).

### 9. Commit

One commit per `hard_constraints.md §24`. Template:

```
fix(env): naked-winner clip (95%) in shaped + close bonus + full loss in raw

Reshapes the scalping reward split across raw and shaped
channels. Raw now reports actual race cashflow — full
per-pair naked P&L (winners AND losers). Shaped absorbs the
training-signal clip on naked winners (−95%) and a per-close
bonus (+£1).

Reward-scale change. Post-fix scoreboard P&L is not directly
comparable to pre-fix; training signal is uniformly stronger
on both the closure-reward and the naked-loss sides.

Worked examples (per-pair contributions):
  scalp locks +£2 (used close_signal):
    raw=+2,  shaped=+1 (close bonus),     net=+3
  naked winner +£100 (held to settle):
    raw=+100, shaped=−95 (95% clip),       net=+5
  naked loser −£80 (held to settle):
    raw=−80,  shaped=0,                     net=−80

Motivation: gen-2 activation-A training, transformer
`0a8cacd3-3c44-47d1-a1c3-15791862a4e6`, ep 1–7 (2026-04-18):
arbs_closed collapsed from 5 (ep 1) to 0 (ep 2–7) while
naked_pnl dominated day_pnl (+£455 vs locked +£10 on ep 7).
The 0.5× softener on naked losses + min(0, …) hiding of naked
winners made naked gambling positive-EV for training even
under per-pair aggregation. This shape makes naked winners
uninteresting (+5% net) and naked losers catastrophic (full
loss), giving `close_signal` the gradient it was missing.

See plans/naked-clip-and-stability/.

Tests: N new in tests/test_forced_arbitrage.py
(TestNakedWinnerClipAndCloseBonus). pytest tests/ -q:
<delta from baseline>.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

Worked-example numbers MAY adjust based on actual test
fixture prices; keep them realistic and illustrative.

## Cross-session rules

- Full pytest green on commit (`hard_constraints.md §20`).
- No schema bumps (implicitly — this plan doesn't touch
  schemas).
- No matcher changes.
- No new shaped terms beyond the clip and bonus
  (`hard_constraints.md §2`).
- The per-pair naked accessor from
  `scalping-naked-asymmetry` stays read-only and
  deterministic.

## After Session 01

1. Append a `progress.md` entry: commit hash, worked-example
   numbers from actual tests, invariant-test pass
   confirmation, pytest delta.
2. Hand back to the operator for Session 02 (PPO stability).
   Do NOT queue Session 02 automatically — the operator
   reviews Session 01's commit first.
