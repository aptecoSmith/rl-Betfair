# Master TODO — Scalping Equal-Profit Sizing

Four sessions in dependency order. Each session must land green
(full `pytest tests/ -q`, frontend `ng test` where relevant) before
the next begins. Sessions 01–02 are the critical fix; 03 is the
documentation + reset; 04 is the parked UI display cleanup.

## Session 01 — Equal-profit sizing helper + tests

**Status:** pending

**Deliverables:**
- `env/scalping_math.py`: new pure functions
  - `equal_profit_lay_stake(back_stake, back_price, lay_price, commission) -> float`
  - `equal_profit_back_stake(lay_stake, lay_price, back_price, commission) -> float`
- Both follow the closed-form derivation in `purpose.md`. Pure,
  dependency-free, vendorable into `ai-betfair`.
- New tests in `tests/test_scalping_math.py`:
  - `c=0` collapses to `S × P_agg / P_pass`.
  - `c=0.05` worked example: `S_back=16, P_back=8.20, P_lay=6.00`
    produces `S_lay ≈ 21.083`.
  - `c=0.10` worked example: same trade produces a different (and
    documented) lay stake.
  - Equal-profit invariant: helper-sized pair has
    `|win_pnl − lose_pnl| < 0.01` for a representative price grid.
  - Symmetric (lay-first) helper produces the analogous back stake.
  - Edge case: very small back stake (£0.01) — no division
    instability.
  - Edge case: lay price approaches `1/c` — denominator stays
    positive; helper does NOT pre-empt the profitability check
    (that's a separate concern).
- The existing `min_arb_ticks_for_profit` and
  `locked_pnl_per_unit_stake` helpers in `env/scalping_math.py`
  STAY UNCHANGED — they compute correctly for any stake/price
  pair; the only thing wrong was upstream sizing.

**Exit criteria:**
- `pytest tests/test_scalping_math.py -q` green.
- `pytest tests/ -q` green (no other tests broken; the helper
  is new and not yet wired).

**Acceptance:** the equal-profit invariant test scans a 4-by-4
grid of `(P_back, P_lay)` pairs across mid-range odds, asserts
`abs(win_pnl − lose_pnl) < 0.01` for each.

**Commit:** one commit. First line: "feat(scalping): add equal-
profit lay-stake sizing helper". No reward-scale change yet
(helper isn't wired); commit body says so explicitly so the
reader knows the wiring lands in Session 02.

## Session 02 — Wire helper into all three placement paths

**Status:** pending

**This is the reward-scale-change commit.** Reward landscape
shifts when this lands.

**Deliverables:**
- `env/betfair_env.py::_maybe_place_paired`: replace
  `passive_stake = aggressive_bet.matched_stake * aggressive_bet.average_price / passive_price`
  with a call to `equal_profit_lay_stake` (or the symmetric
  `equal_profit_back_stake` when the aggressive is a lay).
- `env/betfair_env.py::_attempt_close`: same replacement on the
  closing-leg sizing.
- `env/betfair_env.py::_attempt_requote`: replace the
  `stake_to_replace = target.requested_stake` line — re-quote
  must RE-SIZE at the new lay price using the helper, NOT carry
  the old (wrong-priced) stake forward.
- Update existing tests:
  - `tests/test_forced_arbitrage.py::test_paired_passive_stake_sized_asymmetrically`:
    expected stake matches the new helper.
  - Any test asserting a specific `locked_pnl` value where the
    pair was sized via the env's automatic path — recompute the
    expected value from the new sizing in the test docstring
    (per `hard_constraints.md §17`).
- New end-to-end tests:
  - `_maybe_place_paired` produces a passive whose stake matches
    `equal_profit_lay_stake(...)` for the same prices.
  - `_attempt_close` produces a closing leg whose stake matches
    the helper's output.
  - `_attempt_requote` re-sizes at the new lay price (assert the
    new stake differs from the old when the lay price differs).
- The activation-A-baseline plan's stale-state JSON gets reset
  to `draft` (same procedure used 2026-04-17): its
  `started_at`, `completed_at`, `current_generation`,
  `current_session`, and `outcomes[]` fields. This is in scope
  for Session 02 because the reset MUST follow the env change
  atomically (operator should never be able to launch the plan
  against a half-fixed env).

**Exit criteria:**
- `pytest tests/ -q` green.
- All three placement-path tests pass.
- The pre-existing
  `test_invariant_raw_plus_shaped_equals_total_reward` green
  (the sizing change does NOT touch the reward accumulation
  invariant).
- A 1-agent 5-episode smoke run (manual) produces
  `Arb completed: ...` log lines whose `locked` value is
  materially larger than pre-fix runs at comparable prices.

**Acceptance:** the worked example from `purpose.md` (Back £16
@ 8.20 / Lay @ 6.00 / c=5%) MUST produce `locked_pnl ≈ £4.03`
in an end-to-end test that drives the env through a fake-day
fixture (similar pattern to the existing
`test_completed_arb_locks_real_pnl_via_race_pnl`).

**Commit:** one commit, the named reward-scale-change commit
(per `hard_constraints.md §11`). Worked example in the body.
First line names the change loudly.

## Session 03 — CLAUDE.md + activation re-run prep

**Status:** pending

**Deliverables:**
- CLAUDE.md "Order matching: single-price, no walking" section
  augmented (or a new sub-section added) with:
  - The equal-profit sizing formula.
  - Worked example from `purpose.md`.
  - Historical note: the previous `S × P_b / P_l` form was wrong
    when commission was non-zero; preserved as audit trail.
  - Cross-link to this plan folder.
- `progress.md` of this plan gets a Session 03 entry
  cross-linking to commits from Sessions 01 + 02 + 03.
- `plans/scalping-active-management/lessons_learnt.md`: brief
  append-only entry noting that the original Session-01 sizing
  comment ("derived from demanding equal P&L in win and lose
  outcomes") was true in intent but wrong in math, fixed by this
  plan.
- A short "Reading the new locked numbers" section in
  `purpose.md` (or this plan's `progress.md` — operator's
  preference at the time): one paragraph explaining the
  pre-fix-vs-post-fix comparability cliff for any operator
  later trying to compare scoreboard rows across the cutover.

**Exit criteria:**
- Prose merge, no code changes (commit type: `docs`).
- Operator confirms the explanation reads cleanly cold.

**Acceptance:** opening CLAUDE.md fresh, the "scalping mode"
sub-section makes it obvious which formula is currently in use
and why.

**Commit:** one commit, type `docs`. References Sessions 01 +
02 commit hashes.

## Session 04 — UI display: drop £ from odds

**Status:** pending

The UI bug the operator parked on 2026-04-18, folded into this
plan because the fix is visually next to the locked-pnl numbers
that this plan is making meaningful.

**Deliverables:**
- Sweep: `grep -rn "Back £\|Lay £" frontend/src/ agents/ env/ api/`.
- Replace `£` with `@` on every match where the value being
  displayed is a Betfair price (decimal odds). Keep `£` on
  values that are genuinely pounds (locked_pnl, realised_pnl,
  stake, day_pnl, etc.).
- Sites known in advance:
  - `agents/ppo_trainer.py:~1605` — `arb_completed` event format.
  - `agents/ppo_trainer.py:~?` — `pair_closed` event format
    (added by `scalping-close-signal` Session 01; find via grep).
  - Likely also: calibration card, bet explorer chips,
    scoreboard tooltips. Each found via the sweep.
- Spec / test updates wherever a spec asserts the format string
  text. No logic change.

**Exit criteria:**
- `pytest tests/ -q` green.
- `cd frontend && npx ng test --watch=false` green.
- Browser-verify per the
  `Verify frontend in browser before done` memory entry: open
  Training Monitor, look at the activity log, confirm "Back @
  8.20 / Lay @ 6.00 → locked £+X.XX" reads correctly.

**Commit:** one commit, type `fix(ui)`. Cross-references the
sites changed.

---

## After Session 04: re-run activation-A-baseline

Once all four sessions land:

1. Reset all four activation plans to `draft` (Session 02 already
   reset activation-A-baseline; Sessions 03–04 don't touch them).
   If the other three drifted state, reset them too via the same
   JSON-edit pattern.
2. Operator launches activation-A-baseline.
3. Watch the learning-curves panel. Expected differences from
   the 2026-04-18 frozen-fitness run:
   - `Arb completed: ... → locked £+X.XX` lines show **materially
     larger** locked values for the same back/lay prices.
   - `best_fitness` per generation moves (no longer frozen at a
     single value across multiple gens).
   - High-volume scalpers RANK HIGHER than low-volume cautious
     agents (locked numbers correctly reflect their work).
4. Capture findings in `progress.md` under a "Validation" entry
   following the convention in
   `scalping-active-management/progress.md`.

If validation succeeds: green light to run the activation
playbook's B sweeps. If the same frozen-fitness pattern reappears
post-fix, the next layer is `naked_penalty_weight` gene range
(per `scalping-naked-asymmetry/purpose.md`'s Failure Mode).
