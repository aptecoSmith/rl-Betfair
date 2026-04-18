# Hard constraints — Scalping Equal-Profit Sizing

Non-negotiable rules. Anything that violates one gets rejected in
review before destabilising production.

## Scope

**§1** This plan changes ONE conceptual thing: the formula used to
size the passive (or closing) leg of a paired scalp so that both
race outcomes net the same profit after commission. Every code
change must trace back to that single intent. Anything else
(reward shape, action schema, observation schema, matcher,
budget-reservation policy, naked-penalty shaping, locked-pnl
floor, commission rate semantics) stays as it is.

**§2** No new shaped-reward terms. The asymmetric raw reward
(`scalping_locked_pnl + sum(min(0, per_pair_naked_pnl))`, post
plan #13) stays. The locked numbers it consumes will move because
sizing changes; that's the entire reward-scale change being
flagged in §11–§13. No compensating shaping is added.

**§3** No schema bumps. `OBS_SCHEMA_VERSION`,
`ACTION_SCHEMA_VERSION`, and `SCALPING_ACTIONS_PER_RUNNER` stay
as they are. Pre-fix checkpoints continue to load without
migration; only their reward landscape during *new training*
shifts.

## Math

**§4** The new sizing formula MUST be:

```
S_lay = S_back × [P_back × (1 − c) + c] / (P_lay − c)
```

derived from setting `total_win == total_lose` after commission.
No "approximation", no "near-equal" simplification. Closed-form,
exactly equal P&L either way (modulo float rounding). The
symmetric helper for lay-first scalps:

```
S_back = S_lay × [P_lay × (1 − c) + c] / (P_back − c)
```

(swap the labels; same derivation).

**§5** The formula MUST collapse to `S_lay = S_back × P_back / P_lay`
exactly when `c = 0`. A unit test asserts this — guards against
introducing the wrong commission term direction.

**§6** When the lay price approaches `1/c` (Betfair would never
actually let this happen — at c=5 % that's 20.0; at c=10 % that's
10.0), the denominator `P_lay − c` is fine but the win-side
constraint of profitability has already failed. The helper does
NOT handle "is this trade profitable at all" — that's the
commission-aware tick floor's job (`min_arb_ticks_for_profit`).
The sizing helper just sizes; profitability is a separate
question, separately gated.

**§7** Integer rounding rules. Stakes computed by the helper are
floats. Downstream callers MUST NOT round to whole pence before
placement — Betfair accepts `S` to 2 d.p. and our placement code
already does the right thing (passes the float to
`PassiveOrderBook.place` / `BetManager.place_back`). Don't
introduce a `round(stake, 2)` call inside the sizing helper.

## Implementation

**§8** Three call sites in `env/betfair_env.py` MUST switch to
the new helper atomically in Session 02:

- `_maybe_place_paired` (auto-paired passive after aggressive
  fill).
- `_attempt_close` (close-at-loss from `scalping-close-signal`).
- `_attempt_requote` (re-quote of an existing passive — currently
  carries the old stake forward, which compounds the bug; MUST
  re-size at the new lay price).

Atomic = same commit. Doing two of three leaves the env in a
state where some pairs are equal-profit and some aren't, making
debugging much harder.

**§9** No changes to `env/exchange_matcher.py`. The
single-price-no-walking rule (CLAUDE.md → "Order matching") is
load-bearing. The new sizing only changes WHAT stake we offer;
the matching logic that decides whether and at what price the
offer fills is untouched.

**§10** No changes to `env/bet_manager.py::get_paired_positions`'s
`locked_pnl` formula. That formula computes
`max(0, min(win_pnl, lose_pnl))` from the actual matched legs and
their actual prices/stakes. Its OUTPUT will move because the
INPUTS (stakes) change, but the formula itself is correct and
stays.

## Reward-scale change protocol

**§11** Per CLAUDE.md and the convention from
`scalping-active-management/activation_playbook.md` Step E and
`scalping-naked-asymmetry` Session 01, this is a reward-scale
change. The Session-02 commit message (the one that wires the
new sizing into the env, where the reward shift actually lands)
MUST:

- Name the change in the first line.
- Include a worked numerical example: pre-fix vs post-fix
  outcomes for the canonical Back £16 @ 8.20 / Lay @ 6.00 trade.
- State explicitly: "scoreboard rows from before this commit are
  not directly comparable; new training is the comparison
  surface".

**§12** Update CLAUDE.md's "Order matching: single-price, no
walking" section (and any adjacent scalping-specific notes) to
cite the new sizing formula. Preserve the historical context —
the old formula's rationale was wrong, but the comment lived for
weeks; explain when and why it changed. Don't delete the
2026-04-15 / asymmetric-hedging notes; they remain part of the
audit trail.

**§13** Garaged models from previous training runs are NOT
migrated. Their existing scoreboard rows reflect what they
actually did under the old sizing — those numbers are valid AS
PRE-FIX REFERENCES. Don't rewrite history. New training runs
populate the comparison surface.

## Testing

**§14** Pre-existing
`test_invariant_raw_plus_shaped_equals_total_reward` (in
`tests/test_forced_arbitrage.py`) MUST stay green. The sizing
change moves stakes; raw + shaped accounting is untouched.

**§15** Pre-existing
`test_paired_passive_stake_sized_asymmetrically` (in
`tests/test_forced_arbitrage.py`) WILL need updating — its
fixture asserts the old `S = S_b × P_b / P_l` formula. Update
it to assert the new formula's expected stake. The test name
stays appropriate (the sizing IS still asymmetric — just
correctly so).

**§16** New tests, minimum eight, covering:

1. Helper closed-form: c=0 collapses to `S_b × P_b / P_l`.
2. Helper closed-form: c=0.05 worked example matches £21.08 for
   the canonical trade.
3. Helper closed-form: c=0.10 stake calculation for the same
   trade (different value, exact match to the formula).
4. Helper symmetric: lay-first scalp produces the analogous
   back-side stake.
5. Equal-profit invariant: for a sized pair, `win_pnl` and
   `lose_pnl` differ by at most £0.01 (rounding tolerance).
6. End-to-end: `_maybe_place_paired` produces a passive whose
   stake matches the helper's output for the same prices.
7. End-to-end: `_attempt_close` produces a closing leg whose
   stake matches the helper's output.
8. End-to-end: `_attempt_requote` re-sizes the passive at the new
   lay price (NOT carries the old stake).

**§17** Pre-existing tests that depend on a SPECIFIC numerical
locked_pnl value (e.g. the `test_completed_arb_locks_real_pnl_via_race_pnl`
family) MAY need their expected values updated. Each such update
MUST recompute the expected value from the new sizing formula in
the test's docstring; no "bumping the magic number until green"
allowed. Rationale: the locked_pnl values these tests assert are
the very numbers the operator will see in the activity log; they
need to be derivable from the formula by inspection.

**§18** Full `pytest tests/ -q` green on each session's commit.
Frontend `ng test` green on Session 04's commit (the UI fix).

## UI display fix (Session 04)

**§19** Drop `£` from any UI string that displays a Betfair price
(decimal odds). Replace with `@` to make the unit visually
distinct from monetary values. Sweep with `grep -rn "Back £\|Lay £\|@ £"`
across `frontend/src/`, `agents/`, `env/`, `api/`. A `£` on a
*locked_pnl*, *realised_pnl*, *stake*, or any other monetary
field stays — those genuinely are pounds.

**§20** No changes to JSON schemas, Parquet column names, or any
machine-readable interface. The fix is presentation-only — the
underlying field names (`back_price`, `lay_price`) are correct
and unambiguous; only the human-rendered display strings change.

## Cross-session

**§21** Do NOT bundle the activation re-run into any
implementation commit. The activation re-run is a separate
operator action documented in Session 03's `progress.md` entry
with the new run_id.

**§22** Do NOT pre-emptively prune any models from the registry
as part of this plan. Pruning is its own operation
(`scripts/prune_non_garaged.py`); it's orthogonal and the
operator invokes it when ready.

**§23** Do NOT touch `plans/scalping-naked-asymmetry/` — that
plan is closed. Reference it where useful in this plan's prose;
don't edit its files.
