# Progress — Scalping Equal-Profit Sizing

One entry per completed session. Most recent at the top.

---

## Session 03 — CLAUDE.md + cross-plan notes (2026-04-18)

**Landed.** Docs-only commit (no code, no tests touched).

- `CLAUDE.md` gains a new "Equal-profit pair sizing (scalping)"
  section after "Order matching: single-price, no walking".
  Includes the commission-aware formula, the canonical worked
  example (Back £16 @ 8.20 / Lay @ 6.00 / c=5% → locked £4.03),
  and a historical audit-trail note pointing at Session 02's
  commit (`f7a09fc`) where the env switched off the old
  zero-commission-only formula.
- `plans/scalping-active-management/lessons_learnt.md` appended
  with a one-paragraph entry recording that the original
  Session-01 sizing comment ("derived from demanding equal P&L
  in win and lose outcomes") was correct in intent but wrong in
  math — the derivation only holds at c=0. Cross-link to
  `plans/scalping-equal-profit-sizing/` Session 02 (commit
  `f7a09fc`).
- `progress.md` (this file) gets an operator-facing "Reading the
  new locked numbers" paragraph immediately below this entry,
  explaining the pre-vs-post-fix scoreboard comparability cliff.

No code changes this session. Test suite untouched.

### Reading the new locked numbers (operator note)

After Session 02 (commit `f7a09fc`), `scalping_locked_pnl` values
from new training runs are NOT directly comparable to scoreboard
rows from runs before that commit. The new values reflect
equal-profit-balanced pair P&L; the old values reflected the
worst-case floor of over-laid pairs.

Rule of thumb:

- Old `locked_pnl` ≈ new `locked_pnl` × `(1 − c)` for tight
  spreads. Roughly: take the new number and multiply by 0.95 to
  approximate the comparable old-formula floor, OR take an old
  number and divide by 0.95 (then add roughly the win-side cliff)
  to estimate the equal-profit-equivalent.
- For wide spreads (well into the profitable zone) the two
  formulas converge; the difference is largest right at the
  commission edge, where the old over-lay was collapsing the
  win-side payoff toward zero.

When in doubt: the post-fix number is the one a real-world
scalping calculator (greenupgreen, Bet Angel, etc.) would
produce.

---

## Session 02 — Wire helper into all three placement paths (2026-04-18)

**Status:** complete. **Reward-scale change has landed.** The env's
scalping placement code now sizes passive / close / requote legs
with the commission-aware equal-profit helper. Scoreboard rows
from before this commit are not directly comparable to post-fix;
new training is the comparison surface.

**Call sites switched to the helper** (`env/betfair_env.py`):

- `_maybe_place_paired` — auto-paired passive after aggressive
  fill. Picks `equal_profit_lay_stake` (aggressive BACK) or
  `equal_profit_back_stake` (aggressive LAY). Old comment block
  that described the formula ("derived from demanding equal P&L
  in win and lose outcomes") is now CORRECT in its description
  of behaviour; preserved with a sentence noting the fix landed
  in this plan.
- `_attempt_close` — close-at-loss from `scalping-close-signal`.
  Same helper selection (close leg is opposite side of agg).
  Docstring updated: no longer advertises the buggy
  `S_close = S_agg × P_agg / P_close` formula.
- `_attempt_requote` — previously carried `target.requested_stake`
  forward. That stake was sized for the OLD passive price; at a
  new lay price it re-introduced the same asymmetric-payoff bug.
  Now RE-SIZES via the helper at the new price (hard_constraints §8).

Import block at the top of `betfair_env.py` expanded to pull
`equal_profit_back_stake`, `equal_profit_lay_stake` alongside
the existing `locked_pnl_per_unit_stake` / `min_arb_ticks_for_profit`.

**Worked example confirmation** (test
`test_canonical_worked_example_locks_4_03`): Back £16 @ 8.20 /
Lay @ 6.00 / c=5% locks £4.03 (old sizing reported £0.08 on the
same trade). Invariant `|win_pnl − lose_pnl| < 0.01` holds on
the synthesised pair.

**Tests:**

- Updated `test_paired_passive_stake_sized_asymmetrically` to
  assert the new helper's output (old formula expectation was
  wrong post-fix). Docstring rewritten to cite the correct
  commission-aware formula; still asserts the passive-lay stake
  is strictly larger than the aggressive-back stake for BACK→LAY.
- New `TestEqualProfitSizingEndToEnd` class: 4 tests (delta: +4,
  full suite now 2157 passed from 2153).
  1. `test_paired_passive_stake_uses_equal_profit_formula` —
     `_maybe_place_paired` output matches the helper.
  2. `test_close_leg_stake_uses_equal_profit_formula` —
     `_attempt_close` output matches the helper.
  3. `test_requote_resizes_at_new_lay_price` — requote's new
     stake differs from the old AND matches the helper at the
     new price.
  4. `test_canonical_worked_example_locks_4_03` — end-to-end
     settlement-style check via `get_paired_positions`.
- Pre-existing `test_invariant_raw_plus_shaped_equals_total_reward`
  stays green (sizing change touches stakes, not the raw+shaped
  accumulator).
- No pre-existing specific `locked_pnl` assertions needed
  updating — `test_completed_arb_locks_real_pnl_via_race_pnl`
  uses `> 0.0`, and the `test_early_lock_bonus_*` harness tests
  synthesise their own stakes manually.

**Test results:**
- `pytest tests/test_forced_arbitrage.py::TestEqualProfitSizingEndToEnd -q` → 4 passed.
- `pytest tests/ -q` → 2157 passed, 7 skipped, 133 deselected,
  1 xfailed.

**Activation plans reset to draft:** `activation-A-baseline`
was `running` from the frozen-fitness 2026-04-18 launch. Reset
via the same JSON-edit pattern used 2026-04-17: `status=draft`,
`started_at=None`, `completed_at=None`, `current_generation=None`,
`current_session=0`, `outcomes=[]`. The other three B plans
(`activation-B-100`, `-010`, `-001`) were already draft/outcomes=[]
and were left as-is. All four plans now show `status=draft,
outcomes=[]` — the env is ready for post-fix re-launch.

---

## Session 01 — Equal-profit sizing helper + tests (2026-04-18)

**Status:** complete. No env wiring yet — the helper is pure math,
live code paths still use the old `S_b × P_b / P_l` formula.
Wiring lands atomically in Session 02 (the reward-scale-change
commit).

**Delivered:**

- `env/scalping_math.py`: two new closed-form helpers, matching
  the existing "pure, dependency-free, vendorable" style.
  - `equal_profit_lay_stake(back_stake, back_price, lay_price, commission)`
    — forward formula `S_l = S_b × [P_b(1 − c) + c] / (P_l − c)`
    from `purpose.md`'s derivation. Collapses to the pre-fix
    `S_b × P_b / P_l` at `c = 0`.
  - `equal_profit_back_stake(lay_stake, lay_price, back_price, commission)`
    — **algebraic inverse** of the same balance equation:
    `S_b = S_l × (P_l − c) / [P_b(1 − c) + c]`. Back and lay legs
    are not algebraically symmetric (different P&L shapes), so
    the naive label-swap form that appeared in earlier drafts of
    `hard_constraints.md §4` and the session prompt is wrong.
    Corrected in both files this session; see the "Plan-doc
    correction" note below.
- `tests/test_scalping_math.py`: new `TestEqualProfitSizing`
  class, eight tests (delta: +8, full suite now 2153 passed
  from 2145).
  - c=0 collapses to old formula.
  - c=5% worked example (`S_l ≈ 21.083` for Back £16 @ 8.20 /
    Lay @ 6.00) matches `purpose.md`'s canonical figure.
  - c=10% alternate recomputed by hand.
  - Equal-profit invariant over a 4×4 price grid: every sized
    pair satisfies `|win_pnl − lose_pnl| < 0.01`.
  - Symmetric lay-first helper: output satisfies the same
    equal-profit property (this is the test that caught the
    label-swap bug in the plan's original spec).
  - Tiny-stake (£0.01) stability.
  - `ValueError` on `back_price <= 1.0` and
    `lay_price <= commission` for the forward helper.

**Plan-doc correction:** `hard_constraints.md §4`,
`purpose.md`'s §"What this plan delivers", and
`session_prompts/01_math_helper.md` all originally specified
`S_back = S_lay × [P_lay × (1 − c) + c] / (P_back − c)` for the
symmetric helper. That's the forward formula with labels
mechanically swapped — it does not solve the balance equation.
The implementation follows the genuine algebraic inverse; all
three plan files updated in-place so Session 02 can't
re-introduce the wrong form by reading them.

**Test results:**
- `pytest tests/test_scalping_math.py -q` → 22 passed.
- `pytest tests/ -q` → 2153 passed, 7 skipped, 133 deselected,
  1 xfailed.

**Reward landscape:** unchanged. The env's three placement paths
(`_maybe_place_paired`, `_attempt_close`, `_attempt_requote`)
still use the old formula. No training needs re-running after
this commit. Session 02 is where the reward scale actually
shifts.

---

_(Plan folder created 2026-04-18 in
response to the operator spotting that an activity-log line
"Arb completed: Back £8.20 / Lay £6.00 → locked £+0.08" was
reporting a wildly under-locked balance for what should have been
a healthy scalp. Diagnosis: the sizing formula
`S_lay = S_back × P_back / P_lay` is correct only at zero
commission; at Betfair's 5 % it equalises *exposure*, not P&L,
producing pairs whose worst-case floor (which `locked_pnl`
reports) collapses near zero. The correct formula derives from
setting `total_win == total_lose` after commission and produces
genuinely balanced pairs. See `purpose.md` for the full
derivation, worked example, and reward-scale-change protocol.)_
