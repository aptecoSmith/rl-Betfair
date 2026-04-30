---
plan: rewrite/phase-3-followups/no-betting-collapse
status: design-locked
opened: 2026-04-30
depends_on: rewrite/phase-3-cohort (AMBER, 2026-04-30 — Bar 6c FAIL)
---

# No-betting-collapse follow-on — Phase 3 AMBER recovery

## Purpose

Phase 3's first 12-agent cohort produced an AMBER verdict
(`plans/rewrite/phase-3-cohort/findings.md` Session 04 live-run
results). Two of three Bar-6 architectural metrics PASSED:

- **Bar 6a:** mean force-close rate **0.308** (< 0.50 threshold; vs
  v1 ~0.75).
- **Bar 6b:** ρ(entropy_coeff, fc_rate) = **−0.517** (≤ −0.5
  threshold; vs v1 ~0).

The third FAILED:

- **Bar 6c:** **0/12 agents positive on raw P&L** on the held-out
  eval day. Every agent's `eval_day_pnl` was exactly £0.00 with 6–9
  bets and 0 winning bets — the policy had collapsed to a
  near-NOOP regime by eval time, almost certainly because:

  > Train-side `total_reward` was −£1000 to −£2200 on early
  > episodes (driven by the per-pair naked-loss term in
  > `shaped_bonus`); the policy responded by collapsing to NOOP —
  > the only action that doesn't accumulate negative reward.
  > (findings.md Session 04 §"Diagnostic")

The rewrite's bet was **"no shaping at all"** (rewrite README
§"What survives", phase-3-cohort/purpose.md §"What does NOT
change"). The result here says that bet does not pay on cash —
but the architecture itself works. The follow-on tests two
minimum-viable hypotheses:

1. **One reward-shaping term is load-bearing for "agents bet at
   all".** A one-term-at-a-time ablation isolates which one.
2. **The `locked_pnl + naked_pnl = 0.00` accounting pattern
   observed in every eval row is a bug, not a feature.** A
   targeted micro-investigation either fixes the accounting
   (and possibly flips Bar 6c without any reward-shape change)
   or proves it's correct-by-design and only `close_signal` can
   produce positive cash.

Both hypotheses are session-sized; both are required before
deciding whether the rewrite premise survives.

## Why one plan, not two

- The accounting micro-investigation (Session 01) is a
  prerequisite for the shaping ablation (Session 02): if Bar
  6c was failing because the env silently zeros out cash P&L
  from matured arbs, no amount of reward shaping would have
  recovered it.
- Outcome-coupled: if Session 01 fixes the accounting AND
  agents have non-zero cash, Session 02's shaping ablation
  scope shrinks.
- Single-phase keeps the AMBER verdict's investigation under
  one verdict header for the rewrite README.

## What's locked

### The 7-gene Phase 3 schema is NOT extended

Per phase-3-cohort hard constraint §5 ("No GA gene additions
beyond the locked schema"). This follow-on tests one shaping
*plan-level* knob at a time, NOT new genes. Any positive finding
that warrants a new gene goes into a SEPARATE follow-on plan.

### The cohort-run protocol is locked

Same `select_days` / 7 train + 1 eval split / `--seed 42` /
12 random-init agents (no breeding) — so cross-comparisons
against the AMBER baseline are clean. **`--generations 1` is
locked** for this follow-on; the GA breeding question is
deferred until Bar 6c is recoverable.

### Ablation order is locked

Session 02 ablates the v1 shaping terms in the order most
likely to be load-bearing for bet-starvation:

1. **Matured-arb bonus** (`reward.matured_arb_weight > 0`) —
   directly rewards completing pair lifecycles, which is
   structurally what the agent stopped doing.
2. **Naked-loss anneal** (`naked_loss_anneal: {start, end}`) —
   softens the early-episode penalty that drove the collapse.
3. **Mark-to-market shaping** (`mark_to_market_weight > 0`) —
   surfaces per-tick value gradient so the policy doesn't have
   to wait for race-end to learn.

ONE at a time. ONE cohort per ablation. Cohort cost is fixed
(~3.1 h GPU per 12-agent / 1-gen run). Three cohorts ≈ ~10 h
GPU total, plus the accounting fix in Session 01.

If a session can be skipped because the previous one's verdict
is already a clear PASS on Bar 6c, do skip — the goal is the
verdict, not running every ablation.

## Success bar (this follow-on)

The plan ships iff:

1. **Accounting investigation lands a verdict.** Either:
   (a) the `locked_pnl + naked_pnl = 0.00` pattern is fixed and
       eval `day_pnl` reflects per-pair locked spreads, OR
   (b) the pattern is proven correct-by-design with a documented
       trace (e.g. "matured-arb cash flow is by-design zero
       under the equal-profit lock; only `close_signal` produces
       cash") and the original Bar 6c interpretation gets
       restated.
2. **At least one shaping ablation produces ≥ 1/12 agent
   positive on raw P&L** on the same eval day with the same
   seed. OR — if all three ablations fail — the rewrite
   premise's "no shaping" bet is documented as definitively
   refuted.
3. **No throughput regression.** Each ablation cohort runs in
   the same ~3.1 h envelope as the AMBER baseline. (If
   throughput-fix follow-on lands first, that envelope shrinks
   and this bar adjusts proportionally.)

If 1 PASSES with verdict (a) and 2 PASSES on any term →
**phase-3-followups GREEN**, the rewrite ships with one
restored shaping term and a documented cash-accounting
correctness fix.

If 1 PASSES with verdict (b) and 2 FAILS on every term →
**phase-3-followups RED**, the rewrite is refuted. Step back
to v1 baseline and write the post-mortem.

If 1 PASSES with verdict (a) and 2 still FAILS after all
three shaping ablations → **phase-3-followups RED with
caveat**: the no-shaping bet AND the minimum-viable shaping
suite both fail. Stronger refutation of the rewrite.

## Deliverables

### Session 01 — locked/naked accounting investigation

- A one-page trace of where `eval_day_pnl` is computed in the
  env (`env/betfair_env.py::_settle_current_race`) and where
  `eval_locked_pnl` / `eval_naked_pnl` come from (the env's
  `info["locked_pnl"]` / `info["naked_pnl"]`).
- A unit test (`tests/test_v2_eval_pnl_accounting.py`) that
  builds a synthetic 1-race day, opens one matured arb pair,
  and asserts `day_pnl == locked_pnl > 0` (NOT zero).
- Verdict: bug or by-design.
  - If bug: minimal fix in env, regression test, **re-run**
    the AMBER baseline cohort with the fix to see if Bar 6c
    flips on its own. If yes — Phase 3 follow-on is GREEN
    with verdict (a).i
  - If by-design: a docs change (CLAUDE.md "Reward function:
    raw vs shaped" + the cohort findings) clarifying that
    matured-arb cash is by-design zero, and Bar 6c's intent
    becomes "≥1 agent positive via close_signal-driven
    cash" — possibly requiring an env config change to
    incentivise close_signal usage.

### Session 02 — minimal shaping ablations

ONLY runs if Session 01 verdict is (b) (by-design zero) OR
verdict is (a) but the post-fix re-run still shows 0/12
positive.

Three sub-runs (each is a 12-agent / 1-gen cohort, ~3.1 h):

1. `matured_arb_weight = 1.0` (everything else default).
2. `naked_loss_anneal = {start_gen: 0, end_gen: 4}`
   (everything else default).
3. `mark_to_market_weight = 0.05` (everything else default).

Each sub-run produces:

- A scoreboard.jsonl + the same Bar 6 metric trio.
- A side-by-side row in this plan's findings.md table:

  | Cohort | mean fc_rate | ρ(ent, fc) | positive on P&L |
  |---|---|---|---|
  | AMBER baseline | 0.308 | −0.517 | 0/12 |
  | + matured_arb | ? | ? | ? |
  | + naked_loss_anneal | ? | ? | ? |
  | + mark_to_market | ? | ? | ? |

- A verdict line: did this ablation flip Bar 6c?

Stop early as soon as one ablation flips Bar 6c — the cheapest
ablation that flips it is the answer.

### Session 03 — verdict + write-up

- `plans/rewrite/phase-3-followups/no-betting-collapse/findings.md`
  with the table above + the success-bar verdict
  (GREEN / RED / RED-with-caveat) + the next step (proceed to
  scale-up if GREEN; revert to v1 if RED).
- An entry in `plans/rewrite/README.md` linking the verdict
  back to the rewrite's "Phase 3 success bar" and updating
  the rewrite's overall status.

## Hard constraints

In addition to all rewrite hard constraints (README §"Hard
constraints") and phase-3-cohort hard constraints:

1. **No GA gene additions.** Even if a shaping term works,
   it stays plan-level until a SEPARATE plan promotes it to
   a gene. Mid-plan gene-schema changes invalidate breeding.
2. **One ablation at a time.** Stacking shaping terms
   produces no per-term signal. If two terms are needed for
   Bar 6c, that's a SEPARATE finding requiring its own
   investigation.
3. **No env edits beyond the accounting fix in Session 01.**
   The shaping ablations operate ENTIRELY through plan-level
   `reward.*` config knobs that the env already supports.
   If a shaping term doesn't have an existing knob, it stays
   out of scope.
4. **Same seed (42) for every cohort.** Cross-cohort
   comparison is the load-bearing mechanism for the
   ablation verdict; differing seeds invalidate comparison.
5. **Each cohort's verdict is honest.** "Mean P&L > 0 across
   the cohort" is NOT Bar 6c — Bar 6c is "≥ 1 agent
   individually positive on raw P&L". Don't grade on
   averages.
6. **Stop conditions inherit from phase-3-cohort.** Don't
   chain ablations after a verdict is in.

## Out of scope

- Throughput fix (separate plan
  `phase-3-followups/throughput-fix/`).
- 66-agent scale-up (gated on this plan's GREEN verdict).
- v1 deletion (gated on the rewrite's overall PASS, not on
  this follow-on alone).
- New gene additions of any kind.
- BC pretrain (rewrite removes it).
- Curriculum changes.
- UI-driven cohort launch (separate follow-on, not gated on
  Bar 6).

## Phase-3-cohort hand-offs

From `plans/rewrite/phase-3-cohort/findings.md` Session 04
live-run results:

1. **AMBER baseline numbers** (mean fc=0.308, ρ=−0.517,
   0/12 positive) are the comparison floor for every
   ablation. Document them in this plan's findings.md
   intro so future readers don't have to cross-reference.
2. **Cohort run dir**
   `registry/v2_first_cohort_1777499178/` is the canonical
   AMBER baseline. Do NOT delete it before the follow-on
   ships.
3. **Per-day wall envelope** is ~140–160 s/ep on a 3090.
   12 agents × 7 days × 1 gen ≈ 11 200 s = 3.1 h. Plan
   sessions around this (no Session 02 sub-run runs in
   under 3 h).
4. **`select_days(seed=42)` produces a deterministic
   training-day order** per agent — same days = same agents.
   Re-using the seed makes cross-cohort comparison apples-
   to-apples.

## Sessions

1. **`01_accounting_investigation.md`** — trace `day_pnl`
   computation, write the synthetic-arb regression test,
   land bug-or-by-design verdict. End-of-session check:
   either a fix is committed + re-run flips Bar 6c, OR
   a docs change is committed + Bar 6c interpretation is
   restated.
2. **`02_shaping_ablations.md`** — run the three named
   ablations sequentially, stop at the first one that
   flips Bar 6c. End-of-session check: at least one
   ablation has a complete scoreboard + verdict.
3. **`03_verdict_writeup.md`** — fill in findings.md table,
   update rewrite README, decide next step.

Each session is independently re-runnable. Session 02
imports Session 01's accounting fix verbatim if there is
one; Session 03 imports Session 02's results.

This plan is the rewrite's recovery-or-refute test. After:

- **GREEN verdict** → green-light scale-up (66 agents) and
  v1-deletion plan; rewrite premise survives with one
  restored shaping term.
- **RED verdict** → revert to v1, write rewrite post-mortem.
- **RED-with-caveat** → step back further; either v1 revert
  or a more fundamental rethink of the per-runner-credit
  + entropy-controller-free architecture.
