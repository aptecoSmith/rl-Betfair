# Purpose — Scalping Equal-Profit Sizing

## Why this work exists

A live-data observation made by the operator on 2026-04-18 exposed
a long-standing bug in how the env sizes the passive leg of a
scalp pair.

> **Activity log line that triggered the dig:**
> ```
> Arb completed: Back £8.20 / Lay £6.00 on runner 28642168 → locked £+0.08
> ```
>
> Operator's intuition: "this looks like a good scalp, but we only
> made 0.08p". Real-world scalping calculators (e.g. greenupgreen,
> Bet Angel, dozens of others) all balance the lay stake so that
> BOTH outcomes net the same profit after commission. The above
> trade nets £0.08 if the runner wins and £4.78 if it loses — wildly
> unbalanced.

The math is the bug. We use the formula:

```
S_lay  =  S_back × P_back / P_lay
```

The comment in [`env/betfair_env.py:1474`](../../env/betfair_env.py#L1474)
says "derived from demanding equal P&L in win and lose outcomes" —
but that derivation **only holds when commission is zero**. With
Betfair's 5 % commission (or 10 %, or any non-zero value), the
formula equalises *exposure* (`stake × (price − 1)`) — not P&L.

## The math

Let:
- `S_b` = back stake, `P_b` = back price (decimal odds)
- `S_l` = lay stake, `P_l` = lay price
- `c` = commission rate (Betfair charges only on winning leg's net
  profit)

Win outcome (back wins):
```
back_pnl  = +S_b × (P_b − 1) × (1 − c)     [winnings × after-commission]
lay_pnl   = −S_l × (P_l − 1)                [layer pays the backer]
total_win = S_b × (P_b − 1) × (1 − c) − S_l × (P_l − 1)
```

Lose outcome (back loses):
```
back_pnl   = −S_b                            [stake gone]
lay_pnl    = +S_l × (1 − c)                  [keeps backer's stake, pays commission]
total_lose = −S_b + S_l × (1 − c)
```

Set `total_win == total_lose`:
```
S_b × (P_b − 1) × (1 − c) − S_l × (P_l − 1)  =  −S_b + S_l × (1 − c)
S_b × [(P_b − 1)(1 − c) + 1]                  =  S_l × [(P_l − 1) + (1 − c)]
S_b × [P_b × (1 − c) + c]                     =  S_l × (P_l − c)
```

Solve for `S_l`:

```
S_l  =  S_b × [P_b × (1 − c) + c] / (P_l − c)         ← CORRECT formula
```

When `c = 0`: collapses to `S_b × P_b / P_l` — same as today's code.
When `c > 0`: produces a **smaller** lay stake than today's formula
yields, which moves the lay-side payoff up and the win-side payoff
down until both equal.

## Worked example — the £8.20 / £6.00 / £16 trade

| Quantity | Today's formula (`S_l = S_b × P_b / P_l`) | Correct formula (`S_l = S_b × [P_b(1−c) + c] / (P_l − c)`) |
|---|---:|---:|
| Lay stake | £21.87 | **£21.08** |
| Net on back wins | +£0.08 | **+£4.04** |
| Net on back loses | +£4.78 | **+£4.03** |
| `locked_pnl = min(win, lose)` | £0.08 | **£4.03** |
| EV at p=0.5 | +£2.43 | **+£4.03** |
| EV at p=0.122 (market-implied for 8.20) | +£4.21 | +£4.03 |

Three consequences worth naming:

1. **`locked_pnl` is systematically understated.** What we report as
   the worst-case floor is not the true floor of an equal-profit
   scalp; it's the *near-zero edge* of an over-laid trade. Real
   scalpers achieve much more than the locked figures suggest.
2. **EV at fair-market probability is comparable** between the two
   formulas. The over-lay accidentally biases the payoff toward the
   high-probability outcome (loss for a longshot). It "works" in EV
   terms by accident, but it's not what scalping calculators do and
   not what we tell the operator we do.
3. **The reward signal sees the wrong number.** `scalping_locked_pnl`
   in the env's reward path uses `min(win, lose)`. With over-laid
   pairs, `min` is almost always the win-side cliff — meaning the
   reward almost always sees a tiny lock even when the trade is
   genuinely profitable.

## What this plan delivers

A targeted three-session change to the sizing formula and every
caller that depends on it.

### 1. Math helper

A new pure function in `env/scalping_math.py`:
`equal_profit_lay_stake(back_stake, back_price, lay_price, commission)
-> float`. Closed-form, dependency-free, vendorable into the
`ai-betfair` live-inference repo.

A second helper for the symmetric case (lay-first scalp where the
"passive" is a back leg):
`equal_profit_back_stake(lay_stake, lay_price, back_price, commission)
-> float`. Derived by algebraically inverting the same balance
equation (back and lay legs have different P&L shapes, so the
formula is **not** a label-swap):

```
S_back = S_lay × (P_lay − c) / [P_back × (1 − c) + c]
```

Both are unit-tested with the worked example above + edge cases
(c=0 collapses to old formula; high odds; ladder boundaries; very
small stakes).

### 2. Wire into all three placement paths

Three call sites in [`env/betfair_env.py`](../../env/betfair_env.py)
currently use `S_passive = S_aggressive × P_aggressive / P_passive`:

- `_maybe_place_paired` (placement of the auto-paired passive after
  an aggressive fill).
- `_attempt_close` (the close-at-loss mechanic from
  `scalping-close-signal` Session 01).
- `_attempt_requote` (re-quote of an existing passive at a new
  price; currently keeps the old stake, which compounds the bug
  when the new lay price differs from the original).

All three switch to the new helper. Re-quote also re-sizes (it
was carrying the old, mis-sized stake forward).

### 3. Recompute the commission-aware tick floor

`min_arb_ticks_for_profit` in `env/scalping_math.py` was derived
against the old (wrong) sizing. It still uses
`locked_pnl_per_unit_stake` internally, which is correct (it just
computes `min(win, lose)` for given prices). But the *interpretation*
of "minimum tick offset to lock > 0" was that the OLD formula's
locked floor crossed zero — under the new formula, the locked floor
is uniformly higher, so the minimum tick offset will be the same
or smaller (better). Confirm with tests, no expected behaviour
change beyond "the floor still works correctly".

### 4. Update CLAUDE.md

CLAUDE.md "Order matching: single-price, no walking" stays. New
section under "Scalping mode" capturing the equal-profit sizing
formula + the historical note that the old form was wrong.

### 5. UI display cleanup — drop £ from odds

The activity-log emitter today reads
`"Arb completed: Back £X / Lay £Y → locked £Z"` — the `£` on the
back/lay values is misleading because they're decimal **odds**, not
stakes. Same bug in the `pair_closed` event format added by
`scalping-close-signal` Session 01 and likely propagates to other
UI surfaces (calibration card, bet explorer chips, scoreboard
tooltips).

Sweep + replace: grep `"Back £"`, `"Lay £"`, and similar patterns
across `frontend/src/`, `agents/`, `env/`, `api/`. Replace `£` on
price values with `@`; keep `£` on monetary values (locked_pnl,
realised_pnl, stake). Tests for the format strings get a small
update; no logic change.

## What this plan does NOT cover

- **Re-tuning the `profit_floor` parameter** in
  `min_arb_ticks_for_profit`. Today's default of `0.0` is preserved.
  Once the locked numbers are accurate, the operator may want to
  raise the floor (e.g. to 0.01 = "must lock 1 % of stake"), but
  that's a tuning decision for after this fix lands.
- **Reward shaping changes.** No new shaped terms. Per-pair naked
  asymmetry (`scalping-naked-asymmetry`, plan #13) already shipped
  and stays as-is.
- **Migration of historical scoreboard / locked figures.** Pre-fix
  bets retain their pre-fix locked values in the DB; the value is
  what was actually computed at settlement time. We don't rewrite
  history. New runs will show the corrected values.

## Reward-scale change — call out loudly

This **is** a reward-scale change, per CLAUDE.md's "Reward function:
raw vs shaped" rules. `scalping_locked_pnl` magnitudes will
materially shift upward for any agent placing balanced scalps.
Operators comparing post-fix scoreboard rows against pre-fix runs
must know the raw signal changed. Same protocol the activation
playbook's Step E spells out:

- The Session-02 commit (the one that wires the new sizing into the
  env, where the actual reward shift lands) MUST name the change in
  its first line.
- The body MUST include the worked example from this purpose.md
  showing old vs new numbers for a canonical trade.
- The garaged models from previous training runs are NOT migrated;
  they keep their pre-fix scoreboards and remain valid AS PRE-FIX
  references. New training is the comparison surface.

## What success looks like

- The next activation-A-baseline run, post-fix, reports
  `scalping_locked_pnl` values **at least 5×** what the equivalent
  agent would have shown pre-fix on the same trades.
- The `Arb completed: ... → locked £+X` activity log lines show
  values that match a quick calculator check (back stake × the
  per-unit lock). Operator can sanity-check live.
- The selection-signal-broken pattern from gen-2 of the previous
  activation-A-baseline run (high-volume scalpers ranked WORST
  because their tiny per-trade locks looked like nothing) goes
  away — high-volume scalpers should now register as the top
  agents by `avg_reward` because their locked totals correctly
  reflect their balanced-pair skill.
- The `scalping-naked-asymmetry` plan's success criteria (top
  agent has `arbs_closed > 0` AND `arbs_closed / arbs_naked > 0.3`)
  remain valid metrics on top.

If post-fix activation-A-baseline still shows the same frozen-
fitness pattern, the next layer to investigate is the
`naked_penalty_weight` gene range (per `scalping-naked-asymmetry/
purpose.md`'s Failure Mode section).

## Relationship to upstream plans

- **`scalping-asymmetric-hedging`** (plan #10) introduced
  `min(0, naked_pnl)` as the asymmetric raw term. Untouched here;
  this plan only changes sizing.
- **`scalping-active-management`** (plan #11) introduced re-quote
  + aux heads. Re-quote sizing is fixed by this plan
  (`_attempt_requote`); aux-head plumbing untouched.
- **`scalping-close-signal`** (plan #12) added the close mechanic.
  Close sizing is fixed by this plan (`_attempt_close`).
- **`scalping-naked-asymmetry`** (plan #13) made the asymmetric
  naked penalty per-pair. The locked term it interacts with is
  what this plan fixes — the two stack cleanly: per-pair penalty
  applied to true equal-profit locks.
- **Implication for `scalping-active-management/Session 07`
  (validation report).** Yet another mechanism between the
  original baseline and Session 07's validation. The operator-
  facing lesson once Session 07 eventually runs: there were
  THREE intervening fixes since the original Gen 1 baseline
  (close_signal, naked-asymmetry, equal-profit sizing). Document
  in Session 07's progress entry; not a blocker for this plan.

## Folder layout

Standard, per the convention established in
[`scalping-active-management/`](../scalping-active-management/) and
[`scalping-close-signal/`](../scalping-close-signal/):

```
plans/scalping-equal-profit-sizing/
  purpose.md              <- this file
  hard_constraints.md     <- non-negotiables
  master_todo.md          <- ordered session list
  progress.md             <- one entry per completed session
  lessons_learnt.md       <- append-only
  session_prompt.md       <- pointer to the current session
  session_prompts/
    01_math_helper.md       <- session 01 detailed prompt
    02_wire_placement.md    <- session 02 detailed prompt
    03_docs_and_reset.md    <- session 03 detailed prompt
    04_ui_display_fix.md    <- session 04 detailed prompt
```
