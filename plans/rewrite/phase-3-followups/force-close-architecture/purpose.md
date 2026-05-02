---
plan: rewrite/phase-3-followups/force-close-architecture
status: design-locked
opened: 2026-05-01
depends_on: rewrite/phase-3-followups/no-betting-collapse (GREEN-with-caveat, 2026-05-01)
---

# Force-close architecture — fix the crutch

## Purpose

The `no-betting-collapse` follow-on shipped GREEN on Bar 6c (2/12
agents positive on raw P&L), but the AMBER v2 baseline measured
**mean force-close rate = 0.809** — slightly worse than the v1
baseline (~0.75) the rewrite was supposed to improve on. The
rewrite's central architectural claim ("per-runner credit +
new training stack reduces fc rate") is not supported by this
data.

The operator review on 2026-05-01 reframed the problem
(`no-betting-collapse/findings.md` §"Operator review (2026-05-01)
— force-close is a crutch"). Quotes:

> Force close is a serious fail. A human scalper does all they
> can to close trades that are not going the way they want. This
> force close is a crutch we put in because the models weren't
> closing trades. Perhaps that itself points to an architectural
> issue?
>
> If I were opening a trade, I'd have some idea of how much money
> I wanted to make … In general, I'm looking to make a single £1
> of profit per trade. … If we put on the close trade to make a
> £1 it would stand more chance of closing.
>
> Also, if we look like we would lose £1 because the price has
> not gone the way we were expecting, we close and take the loss.
> What we definitely don't do is 'leave it' unless the only bets
> we had on were lay bets for long odds runners.

The reframe says: don't tune coefficients on the existing shaping
terms, fix the underlying mechanics. The original
`no-betting-collapse/purpose.md` §"Ablation order is locked"
(matured_arb_bonus → naked_loss_anneal → mark_to_market) is
deferred indefinitely — those terms shape *incentives* on top of
mechanics that may themselves be wrong.

## What the data and code say about the mechanics

Three concrete observations from AMBER v2 + a code read of
[`env/betfair_env.py`](../../../../env/betfair_env.py) on
2026-05-01:

1. **Auto-pair places passive at `back_price ± arb_ticks`** where
   `arb_ticks` comes from the agent's per-runner action dim
   `arb_spread` (mapped 0..1 → 0..MAX_ARB_TICKS). The lay price
   is in tick-space; the £-target falls out as
   `stake × (price_diff / current_price)` after the equal-profit
   sizing. The agent has no first-class £-target action; it has
   to learn the mapping
   `(stake, back_price, arb_ticks) → expected_£_profit` from
   delayed cash signal alone.

2. **`close_signal` closes at the current top opposite-side
   price.** No target, no stop-loss anchor. The agent fires and
   takes whatever the book offers. Reward shaping doesn't push
   the policy to fire on projected loss thresholds — the MTM
   shaping term is symmetric (gradient on every tick) where
   trading wants it asymmetric (much stronger gradient when
   exposure is bleeding past a tolerable loss).

3. **Force-close is a T−N safety net** that was added because
   policies weren't learning closes on their own. AMBER v2 has
   80 % of pairs ending via env-initiated bail-out. The two
   profitable agents in the cohort have the *highest* fc rates
   (0.850, 0.862) — within this architecture, fc rate is
   positively correlated with eval P&L. That's the smell that
   the safety net is masking the failure mode rather than fixing
   it: a policy that can't close on its own settles via the
   crutch, and we've stopped seeing that as a problem because
   the cash buckets reconcile.

## What this plan tests

A single, mechanics-level hypothesis:

> If the policy is given a first-class £-target on each open
> (instead of a tick spread) AND a stop-loss mechanism that fires
> at projected loss thresholds (instead of relying on
> end-of-race force-close), the policy will learn closes on its
> own; force-close rate falls toward 0 ; eval P&L stops being
> dominated by the naked-loss tail.

The hypothesis predicts three behavioural changes vs AMBER v2:

- mean fc_rate < 0.30 (vs 0.809; vs v1 ~0.75)
- locked-via-policy / (locked-via-policy + force_closed) > 0.7
- agents with positive eval `naked_pnl` outnumber agents with
  negative eval `naked_pnl` (vs current 2/10)

If the hypothesis fails, the rewrite's "no shaping" bet is
refuted at the mechanics level (not just the coefficient level)
and we revert to v1 + write the post-mortem.

## What's locked

### Phase-3 cohort protocol stays locked

Same `select_days(seed=42)` / 7+1 day split / 12 agents / 1
generation / no other shaping. Each cohort run is ~3.5 h GPU.
Cross-cohort comparison against AMBER v2
(`registry/v2_amber_v2_baseline_1777577990/`) is the load-bearing
mechanism for any verdict; differing seeds invalidate
comparison.

### Eval day stays 2026-04-28

(69/69 winner coverage, full settlement.) Don't re-process or
substitute eval days within this plan; that adds variance
unrelated to the mechanics question.

### One mechanics change per cohort

The original `no-betting-collapse` plan's hard constraint
("one ablation at a time") survives **at the cohort level** —
Sessions 01 and 02 each test one mechanics change in their own
cohort, so the per-mechanics behavioural contribution is
attributable. Session 03's stacked cohort then turns both flags
on together; this is a **deliberate combination**, not a third
new mechanic. See §"Stacking is a legitimate ship configuration"
for the reframe (2026-05-02): stacking is the expected ship
configuration, not a degraded outcome.

### No GA gene additions for as long as possible

Mechanics changes are landed as plan-level config knobs on the
existing `reward_overrides` / env-init paths. If a knob *must*
become a gene (e.g. per-agent `target_pnl_per_pair`), that's a
SEPARATE follow-on plan, not bundled here.

### Force-close stays on as a backstop

`force_close_before_off_seconds` does NOT go to 0 in any session
of this plan. The safety net stays as an envelope; the goal is
that the policy stops *needing* it, measured by fc rate falling.
Removing the net wholesale risks naked-back catastrophe on a
day where the new mechanics fail.

## Success bar

The plan ships GREEN iff **any** cohort — including a stacked
Session 01 + Session 02 cohort — produces:

1. **mean fc_rate ≤ 0.30** on eval day 2026-04-28, AND
2. **≥ 4/12 agents with positive eval `day_pnl`** (vs AMBER v2's
   2/12), AND
3. cohort wall ≤ 4 h (no throughput regression beyond the
   AMBER v2 envelope).

If a cohort hits 1+2 but at higher wall, log it as
**GREEN-with-throughput-caveat** and open a throughput-fix
follow-on.

### Stacking is a legitimate ship configuration (2026-05-02 reframe)

The original framing of this plan required EACH session to clear
the bar individually, with stacked-only GREEN treated as a
degraded outcome requiring an explicit operator decision. That
constraint reflected scientific discipline (isolate per-mechanics
contribution) but conflicted with the structural reality of
scalping: **profit-taking and stop-loss are the minimum viable
toolkit, not alternatives.** A real human scalper always uses
both — neither alone is a sustainable strategy.

The reframe (2026-05-02): if Session 01's £-target mechanism
demonstrably moves a behavioural metric (e.g. policy-close
fraction) AND Session 02's stop-close mechanism demonstrably
moves a different metric (e.g. fc_rate / stop-close fraction)
AND the stacked cohort clears the bar, the plan ships **GREEN
without asterisk**. This is the expected outcome from real
scalping mechanics: each tool does one job, you need both
together to ship.

The methodological discipline is preserved by the per-session
attribution evidence: Session 01 alone documented its
behavioural shift (median pcf 0 → 0.255); Session 02 alone
documented its (fc_rate 0.82 → x, scf 0 → x). When stacked
clears the bar, the writeup shows each piece pulling its own
weight without requiring each to clear the macro bar in
isolation.

If both sessions individually have non-trivial behavioural
effects but neither clears the bar alone, AND stacked still
fails: ship **RED** — the rewrite's "no shaping" premise needs
fundamental rework. The decision to step back to v1 vs attempt
further architectural changes lives with the operator at that
point.

## Sessions

### Session 01 — target-£-pair sizing

Replace the agent's tick-space `arb_spread` action with a
£-target. The agent's per-runner output dim `arb_spread` (0..1)
re-interprets as `target_pair_pnl ∈ [£0.20, £5.00]` (linear).
Env solves for the lay-price that, given the equal-profit sizing
math + the agent's chosen stake, produces the target P&L on lock.

Then:

```
P_lay_target = solve_for_lay_price(
    back_stake, back_price, target_pnl=action_target,
    commission=self._commission,
)
passive_price = quantise_to_tick(P_lay_target, side="lay")
```

Hard constraints:

- The action *space* doesn't change — same dim, same range,
  same gene schema. Only the env's interpretation of the dim
  changes. Pre-plan policies cannot meaningfully cross-load
  (the dim's semantics differ), so this is treated as a
  weight-architecture-hash break per the same convention as
  `fill-prob-in-actor`.
- If the solved lay-price lies inside the matcher's ±50 % junk
  filter or outside the runner's available_to_lay top, fall
  back to refusing the open (don't silently swap to the
  current arb_ticks behaviour). The refusal is the signal.
- Force-close stays on. T−N still flattens any unmatched
  passive.

Cohort: 12 agents / 1 gen / `--seed 42` / eval 2026-04-28.
Output dir
`registry/v2_force_close_arch_session01_target_pnl_<ts>/`.

End-of-session check: Bar 6 trio + a NEW per-agent metric:

- **policy-close fraction** = `arbs_closed / (arbs_closed +
  arbs_force_closed)` on eval day. AMBER v2 baseline
  (computed at session start as a pre-flight): typically
  0.05–0.20. Target ≥ 0.50.

### Session 02 — projected-loss stop-close

Add an env-side auto-close when an open pair's MTM crosses
`-stop_loss_pnl_threshold` (£). Plan-level knob, default `0.0`
= no-op = byte-identical to pre-plan. When > 0, the env auto-
fires the same close path as `close_signal` (NOT the relaxed
force-close path) so the close lands inside the strict matcher.

Why env-initiated rather than agent-learned:

- Agent-learned stop-loss requires the policy to develop the
  abstraction "if MTM is bleeding past £X, close" *before*
  positive cash signal arrives — the same chicken-and-egg that
  the original rewrite ran into. An env-side auto-close makes
  the abstraction structural; the policy learns when *not*
  to open instead of when to close.
- The env-initiated stop-close is still distinct from
  force-close: it's mid-race (not T−N), targeted (not
  blanket), and uses the strict matcher (not relaxed). It
  reflects what a human scalper does, not what the safety net
  does.
- A naked back at race-end is unbounded directional risk; the
  stop-close caps it at `stop_loss_pnl_threshold + spread`.

Hard constraints:

- Stop-close fires through the same `_attempt_close` path as
  `close_signal`. Don't add a third placement path.
- The pair's `Bet.force_close` flag is NOT set on stop-close
  legs. They land in `scalping_arbs_closed`, not
  `scalping_arbs_force_closed`. The matured-arb / close-signal
  shaped bonuses do NOT count stop-closes (they weren't
  policy-initiated). New counter `scalping_arbs_stop_closed`
  exposed on info dict for diagnostics.
- Per the operator review's "leave only long-odds lays naked"
  principle: stop-close fires on naked-back exposures
  unconditionally; for naked-lay exposures, fires only when
  `back_price < lay_only_naked_price_threshold` (gene-mutable,
  default 4.0). Above that price, the lay carries naked.
- Threshold default `0.0` = disabled. Tested values:
  `0.50`, `1.00`, `2.00`.

Cohort: same protocol; output dir
`registry/v2_force_close_arch_session02_stop_close_<ts>/`.

End-of-session check: Bar 6 + policy-close fraction (Session 01
metric) + a NEW metric:

- **stop-close fraction** = `arbs_stop_closed / (arbs_closed
  + arbs_stop_closed + arbs_force_closed)`. Useful range is
  0.10–0.30; if > 0.50, threshold is too tight (stop-closing
  pairs that would have matured).

### Session 03 — verdict + writeup

Three outcomes per the 2026-05-02 reframe (see §"Success bar"):

1. **Session 01 OR Session 02 individually meets the bar** →
   ship GREEN; the single mechanics change is the answer.
   Scale-up gate unblocks.

2. **Stacked Session 01 + Session 02 meets the bar** → ship
   GREEN. Profit-taking and stop-loss are the minimum scalping
   toolkit; stacking IS the expected ship configuration. The
   writeup documents per-session behavioural attribution
   (S01's pcf shift, S02's scf shift) so the per-mechanics
   contribution is on the record.

3. **Neither alone nor stacked meets the bar, but each
   demonstrably moves a behavioural metric** → ship AMBER
   with explicit operator decision required. The mechanism
   layer works but isn't enough; the next plan considers
   additional structural mechanics (e.g. lay-first preference,
   per-runner risk gating).

4. **Neither alone nor stacked meets the bar AND behavioural
   metrics don't move** → RED. Post-mortem, decision on v1
   revert or further architectural rethink.

A stacked cohort is run when needed (after individual S01 + S02
verdicts are in). Output dir
`registry/v2_force_close_arch_session03_stacked_<ts>/`.

## Hard constraints

Inherited from `purpose.md` §"Hard constraints" of
`no-betting-collapse` plus rewrite-level + phase-3-cohort, and:

1. **No env-mechanics change beyond the two named in Sessions
   01 and 02.** No bundled refactors.
2. **No removal of force-close as the T−N backstop.** It stays
   on. The goal is that the policy stops needing it.
3. **Same `--seed 42` for every cohort.** Cross-cohort
   comparison is the load-bearing mechanism.
4. **NEW output dirs for every run.** Don't overwrite AMBER v2.
5. **Pair-sizing math stays equal-profit.** Don't open a
   sizing-philosophy fork in this plan; the question here is
   the *price target*, not the stake-balance formula.
6. **Bar 6c interpretation is unchanged.** "≥ 1 agent
   individually positive". The success bar in this plan tightens
   to ≥ 4/12 to match the operator's fc-rate-and-tail-risk
   reframe.

## Out of scope

- Throughput fix (separate follow-on,
  `plans/rewrite/phase-3-followups/throughput-fix/`).
- 66-agent scale-up (gated on this plan's GREEN verdict).
- v1 deletion (gated on the rewrite's overall PASS).
- New genes / schema changes — the action-dim semantics shift
  in Session 01 is NOT a schema change; the dim count and
  ranges are unchanged.
- Reward-shaping coefficient tuning — the original
  matured_arb_bonus / naked_loss_anneal / mark_to_market tree
  from `no-betting-collapse` is deferred indefinitely.
- Multi-generation cohorts.
- BC pretrain.
- Re-running the AMBER v2 baseline (it stands as the comparison
  floor, captured in
  `registry/v2_amber_v2_baseline_1777577990/`).

## Useful pointers

- AMBER v2 baseline:
  `registry/v2_amber_v2_baseline_1777577990/scoreboard.jsonl`.
- Bar 6 analysis tool: `C:/tmp/v2_phase3_bar6.py`.
- Pair-placement code path:
  [`env/betfair_env.py::_maybe_place_paired`](../../../../env/betfair_env.py)
  (~line 2087).
- Close-leg code path:
  [`env/betfair_env.py::_attempt_close`](../../../../env/betfair_env.py)
  (~line 2395).
- Equal-profit sizing math:
  [`env/scalping_math.py`](../../../../env/scalping_math.py).
- Force-close hard constraints:
  `plans/arb-signal-cleanup/hard_constraints.md`.

## Estimate

Per session: ~4 h (3.5 h GPU + 0.5 h scoring + writeup).

- Session 01: ~4 h
- Session 02: ~4 h (only if 01 doesn't ship GREEN alone)
- Session 03: ~1 h writeup

Best case (Session 01 GREEN): ~5 h.
Worst case (both individually fail): ~9 h + post-mortem.
