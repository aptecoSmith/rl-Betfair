# Session 01b prompt — Raw = race_pnl (loss-closed pairs correctly negative)

## Why this session exists

Session 01 was drafted with:

```python
race_reward_pnl = scalping_locked_pnl + naked_full_term
# where naked_full_term = sum(per_pair_naked_pnl)
```

Review caught that this silently drops `scalping_closed_pnl`
— the cash contribution from `close_signal` legs that closed
otherwise-naked positions. That's fine for profitable
closes (they register in `scalping_locked_pnl`), but a pair
closed at a LOSS has:

- `scalping_locked_pnl` → 0 (floor: `max(0, min(win, lose))`)
- `scalping_closed_pnl` → −£5 (actual cash loss)
- `sum(per_pair_naked_pnl)` → 0 (the pair is COMPLETE; the
  naked accessor skips complete pairs)

Under Session 01's formula: `raw = 0 + 0 = 0`. With the
shaped `+£1` close bonus that Session 01 also landed, the
agent sees `net = +£1` reward for losing £5 of real cash.
That's exactly the pathology the plan exists to prevent.

The fix: set raw to the whole-race cashflow. Every £ that
moved in or out of the wallet lands in raw, including
close-leg losses.

## PREREQUISITE — read first

- [`../purpose.md`](../purpose.md) — "Reward-shape details"
  section (sessions 01 + 01b substance) and the outcome
  table with the loss-closed row.
- [`../hard_constraints.md`](../hard_constraints.md) §4
  (raw = race_pnl), §4a (the bug-and-refinement context),
  §3 (raw+shaped invariant — still holds), §5–§7 (shaped
  terms unchanged), §20 (pytest green), §24 (reward-scale
  change protocol — this is a follow-up delta on Session
  01's scale change; commit body notes the refinement
  rather than re-announcing a fresh scale change).
- [`../master_todo.md`](../master_todo.md) — Session 01b
  deliverables.
- `env/betfair_env.py::_settle_current_race` — specifically
  the scalping reward branch that Session 01 edited. The
  change here is a ONE-LINE substitution on top of Session
  01's work.
- **Session 01's commit** — read the landed diff first.
  Session 01b's implementation is a refinement of whatever
  Session 01 actually shipped; do not re-do Session 01's
  work.

## Locate the code

```
git log --oneline -10 env/betfair_env.py | head
grep -n "race_reward_pnl\|scalping_locked_pnl\|scalping_closed_pnl\|naked_full_term" env/betfair_env.py
grep -n "TestNakedWinnerClipAndCloseBonus" tests/test_forced_arbitrage.py
```

Before editing, confirm the state Session 01 landed in:

1. **Has Session 01 committed?** If yes, read the commit
   hash and diff so Session 01b's commit can reference it.
   If no (Session 01 still drafting), this session blocks
   until it does — §27 in `hard_constraints.md`.
2. **What formula is currently in `_settle_current_race`?**
   Two cases:
   - Session 01 landed `scalping_locked_pnl + naked_full_term`
     — the buggy draft. This session replaces it with
     `race_pnl`. Full code + tests + CLAUDE.md work below.
   - Session 01 already landed `race_pnl` (e.g. the other
     session's agent caught the bug mid-implementation).
     This session becomes tests-only + CLAUDE.md
     verification; note that in `progress.md`.
3. **What does CLAUDE.md currently say?** Session 01
   should have added a 2026-04-18 paragraph under "Reward
   function: raw vs shaped". Read it; this session updates
   the formula line inside that paragraph.

## What to do (assuming Session 01 landed the buggy draft)

### 1. Replace the scalping raw-reward line

Current (Session 01):

```python
naked_per_pair = bm.get_naked_per_pair_pnls(
    market_id=race.market_id,
)

naked_full_term = sum(naked_per_pair)
race_reward_pnl = scalping_locked_pnl + naked_full_term
```

New (Session 01b):

```python
naked_per_pair = bm.get_naked_per_pair_pnls(
    market_id=race.market_id,
)

# Raw is the whole-race cashflow. race_pnl already
# sums scalping_locked_pnl + scalping_closed_pnl +
# sum(per_pair_naked_pnl); setting race_reward_pnl to it
# ensures close-leg losses (which Session 01's draft
# silently excluded via scalping_closed_pnl) are honestly
# reported in raw. See plans/naked-clip-and-stability/
# purpose.md and session_prompts/01b_*.md.
race_reward_pnl = race_pnl
```

`naked_per_pair` is still needed — the shaped winner clip
reads it. Don't delete that line. The shaped terms Session
01 added stay exactly as they are:

```python
naked_winner_clip = -0.95 * sum(max(0.0, p) for p in naked_per_pair)
close_bonus = 1.0 * n_close_signal_successes
```

Only `race_reward_pnl` changes.

### 2. Verify `race_pnl` is available at this point in the function

Session 01's code already references `race_pnl` (line
around 2268 in the pre-Session-01 file:
`naked_pnl = race_pnl - scalping_locked_pnl -
scalping_closed_pnl`). So `race_pnl` is computed before the
reward-assembly block. Confirm this still holds after
Session 01's landing — if Session 01 moved code around
such that `race_pnl` isn't computed yet at the assignment
point, reorder as needed.

### 3. Update CLAUDE.md

Find the paragraph Session 01 added ("Scalping mode
(2026-04-18 — `naked-clip-and-stability`)") under "Reward
function: raw vs shaped". Replace the current formula
line with the corrected one. One-line patch:

```diff
- Raw becomes `scalping_locked_pnl + sum(per_pair_naked_pnl)` — actual race cashflow, winners AND losers.
+ Raw becomes `race_pnl` — the whole-race cashflow (`scalping_locked_pnl + scalping_closed_pnl + sum(per_pair_naked_pnl)`), truthful about every £ that moved including close-leg losses.
```

Add one sentence at the end of the paragraph noting the
refinement lineage so future readers can trace it:

> (Initial Session 01 draft excluded `scalping_closed_pnl`;
> Session 01b corrected this so loss-closed pairs report
> their actual loss in raw rather than netting `+£1` via the
> close bonus — see `plans/naked-clip-and-stability/`.)

Do NOT delete any historical (2026-04-15, 2026-04-18
naked-asymmetry) paragraphs. They stay.

### 4. Update `TestNakedWinnerClipAndCloseBonus`

Session 01 added six tests. Review each:

- **`test_single_naked_winner_raw_full_shaped_clipped`** —
  no close_signal involved, no closed_pnl contribution.
  Expected values (`raw=+100, shaped=−95, net=+5`)
  unchanged. Verify the test still passes unmodified.
- **`test_single_naked_loser_raw_full_shaped_zero`** — same
  reasoning; unchanged.
- **`test_mixed_win_and_loss_per_pair_aggregation`** — if
  the fixture has NO close_signal activity,
  `scalping_closed_pnl = 0` and expected `raw=+20`
  unchanged. If the fixture happens to involve a close,
  update expected raw by `+scalping_closed_pnl`. Read the
  test source to decide.
- **`test_scalp_using_close_signal_earns_bonus`** — this
  test's scalp closed at a PROFIT (`locked_pnl=+£2`). Under
  Session 01: `raw = locked_pnl + naked_term = 2 + 0 = 2`.
  Under Session 01b: `raw = race_pnl = locked_pnl +
  closed_pnl + naked_term`. Whether raw stays `+£2`
  depends on whether the profitable close's cash outcome
  lives in `locked_pnl` or `closed_pnl`. Read
  `env/betfair_env.py` around `scalping_closed_pnl =
  sum(b.pnl ...)` (pre-Session-01 line ~2249–2261) to
  determine the accounting. If `closed_pnl` carries the
  close-leg's cash and `locked_pnl` carries the locked
  spread, a profitable close would have both non-zero —
  adjust the expected `raw` accordingly. Target: ensure the
  test matches reality, not a cooked-up number.
- **`test_multiple_close_signal_successes_accumulate`** —
  only asserts shaped accumulates, so raw changes don't
  affect it. Unchanged.
- **`test_raw_plus_shaped_invariant_with_new_terms`** —
  still passes as long as both raw and shaped are computed
  consistently by `_settle_current_race`. Unchanged.

### 5. Add the loss-closed test

New test in `TestNakedWinnerClipAndCloseBonus`:

```python
def test_loss_closed_scalp_reports_full_loss_in_raw(self):
    """Close_signal closes a pair at a loss:
    locked_pnl is floored to 0, closed_pnl is the actual
    cash loss, no naked contribution. Under Session 01's
    formula (`locked + naked_sum`) this would register
    raw=0 and net=+£1 (close bonus) — rewarding a losing
    trade. Under Session 01b (raw=race_pnl) the close-leg
    loss flows through raw and net is correctly negative.

    Fixture: a scalp that opens back @ P1, passive never
    fills, close_signal fires at P2 where P2 makes the
    close unprofitable by £5.

    Expected:
      raw    = −£5   (scalping_closed_pnl contribution)
      shaped = +£1   (close bonus)
      net    = −£4
    """
```

Implement the fixture by adapting existing close_signal
fixtures (grep the scalping-close-signal test suite for
examples). If an exact-cash-value close is hard to
arrange synthetically, use a fixture where the close cost
is deterministic and adjust the expected numbers
accordingly — the constraint is `raw=−close_cost`,
`shaped=+1`, `net=−(close_cost − 1)`.

### 6. Run the invariant test explicitly

```
pytest tests/test_forced_arbitrage.py::TestScalpingReward::test_invariant_raw_plus_shaped_equals_total_reward -v
```

Must pass. `race_pnl = locked + closed + naked` by
construction, so raw + shaped should still sum to
total_reward. If it fails, trace the plumbing — do NOT
relax the invariant (`hard_constraints.md §3`).

### 7. Full suite

```
pytest tests/ -q
```

Must be green.

### 8. Commit

```
fix(env): naked-clip raw = race_pnl (loss-closed pairs correctly negative)

Refines Session 01's raw-reward formula from
`scalping_locked_pnl + sum(per_pair_naked_pnl)` to
`race_pnl`. Session 01's formula silently dropped
`scalping_closed_pnl` — the cash contribution from
close_signal legs — so a pair closed at a loss via
close_signal would contribute raw=0 (locked floor) + £1
(shaped close bonus) = +£1 net reward, rewarding the agent
for a trade that actually lost real cash.

Under this fix a £5-loss close registers:
  raw    = −£5   (real cash loss flows through race_pnl)
  shaped = +£1   (close bonus, unchanged)
  net    = −£4   (correctly negative)

The close bonus still provides a positive gradient for
close_signal over holding-to-settle (a naked loss of £80
would net −£80 vs close-at-loss's −£4), so the learning
signal favours closing without letting close be an
unconditional reward.

No shaped-term changes. No matcher/schema changes. The
raw+shaped invariant continues to hold (race_pnl is the
exact sum of its three components by construction).

Cross-refs: Session 01 commit <hash>, plans/
naked-clip-and-stability/ (sessions 01 and 01b).

Tests: 1 new (test_loss_closed_scalp_reports_full_loss_in_raw);
N existing updated if fixtures had close_signal activity.
pytest tests/ -q: <delta>.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

Replace `<hash>` with the actual Session 01 commit hash.
Replace `<delta>` with the pytest count diff.

## Cross-session rules

- One-line implementation change to
  `_settle_current_race`. Any broader rewrite gets kicked
  back to review — the point of 01b is a targeted fix.
- No shaped-term changes. `−0.95 × max(0, …)` and `+£1 ×
  n_close` are untouched.
- No GA gene-range or default-hyperparameter changes.
- No matcher, schema, or obs changes.
- CLAUDE.md update is a patch to Session 01's paragraph,
  not a new paragraph.

## After Session 01b

1. Append a `progress.md` entry: commit hash, the one-line
   code change, the six-row outcome table (with the
   loss-closed row) reflected in tests, invariant-test
   pass confirmation.
2. Hand back to the operator for Session 02 (PPO stability).

## If Session 01 already landed the correct formula

If the other session caught the bug mid-implementation and
landed `race_pnl` directly:

1. Session 01b is tests-only + CLAUDE.md verification.
2. Add the loss-closed test (step 5 above) — the test
   should already pass against the Session-01 landing.
3. Verify CLAUDE.md's formula line reads `race_pnl`, not
   `scalping_locked_pnl + sum(per_pair_naked_pnl)`. Patch
   if needed.
4. Commit as `docs(env): add loss-closed test and
   CLAUDE.md refinement note`. Body notes that Session 01
   landed the correct formula directly and 01b is a
   docs/test completeness pass.
