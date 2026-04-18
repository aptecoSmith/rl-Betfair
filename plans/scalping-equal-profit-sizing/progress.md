# Progress — Scalping Equal-Profit Sizing

One entry per completed session. Most recent at the top.

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
