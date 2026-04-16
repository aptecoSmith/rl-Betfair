# Scalping Active Management — Session 07 prompt

Final session of the plan. This is a **measurement**
session: no new architecture, no new UI. Train a scalping
population with all features active, compare against the
Gen 1 baseline, write the story.

## PREREQUISITE — activation playbook complete

**Do not start this session until `activation_playbook.md`
Steps A–E have completed.** If you run Session 07 with
`fill_prob_loss_weight=0.0` and `risk_loss_weight=0.0`, you
are measuring only Session 01's re-quote mechanic, not the
plan's net effect. The activation playbook's Step E commits
non-zero defaults into `config/scalping_gen1.yaml`; that
commit is the green light for this session.

Verify the prereq with:

```
git log --oneline -- config/scalping_gen1.yaml | head -5
```

and confirm the most recent commit bumps
`fill_prob_loss_weight` and/or `risk_loss_weight` off 0.0.

## Before you start — read these

- `plans/scalping-active-management/purpose.md` §"What
  success looks like" — the four target outcomes:
  1. Active re-quote lifts Gen 1 fill rate from 14.5 %
     toward 50 %+ on the best models.
  2. Fill-prob predictions calibrated within ± 5 % per
     bucket.
  3. Risk predictions correlate with realised locked-P&L
     variance (Spearman ρ > 0.3).
  4. Top model's `arbs_naked` count trends down across
     generations.
  This session's job is to **measure whether those
  targets were hit**, not to fix them if they weren't.
- `plans/scalping-active-management/hard_constraints.md` —
  in particular §14 (composite ranking still on L/N >
  composite) and §20 (reward-scale changes called out
  loudly). Any comparison-run headline that looks wrong
  needs to be cross-checked against known reward-scale
  changes.
- `plans/scalping-active-management/activation_playbook.md`
  — the weights you're training with are whatever Step E
  promoted. Reference those values in the progress.md entry
  so readers can reproduce.
- `plans/scalping-active-management/progress.md` — every
  prior session. In particular the Session-02 + 03
  progress entries describe what the aux heads output, and
  Session 05/06 describe where to read the calibration
  numbers from. Those are your inputs.
- Commit `7a3968a` (the Gen 1 baseline run referenced in
  master_todo.md) — git-show it, read the population
  config and results summary. This is the "before" side of
  your comparison.
- `CLAUDE.md` — in particular the "Live training
  learning-curve" memory entry. If the training monitor
  now shows a useful per-episode curve, use it to
  smoke-check the run's health while it's running.

## Before you touch anything — locate the code

```
grep -rn "scripts/scalping\|comparison" scripts/
grep -rn "L/N ratio\|ln_ratio" registry/ api/
ls plans/scalping-active-management/
```

Identify:

1. `scripts/` — the home for one-off analysis scripts. If
   there's already a comparable before/after script
   (e.g. for the reward-signal fix or the scalping-asymmetric
   fix), mirror its shape. If not, create
   `scripts/scalping_active_comparison.py` from scratch.
2. `registry/scoreboard.py` — the source of L/N ratio,
   composite score, and (post-Session-06) MACE. Read rows
   via the existing API or direct DB query — whichever is
   less code in the script.
3. `registry/model_store.py::get_evaluation_bets` — the
   bet-log reader. You'll call this per top-K model to
   produce the calibration numbers.

## Session 07 — Training run + analysis

### Context

The plan has landed six sessions of mechanism:

- Session 01: active re-quote.
- Sessions 02 + 03: fill-prob + risk auxiliary heads.
- Sessions 04–06: UI surfaces for the predictions.

Plus the `activation_playbook.md` has turned the aux heads
on at promoted weights. Now we need one concrete answer:
**is the whole plan net-positive against Gen 1?**

### What to do

#### 1. Train

Run a full scalping population training run on the same
day range as Gen 1, with current `config/scalping_gen1.yaml`
(which now has non-zero aux weights per Step E of the
activation playbook). Same population size (16), same
generation count (4). Tag the run
`scalping-active-mgmt-validation`.

```bash
python -m training.run_training \
  --config config/scalping_gen1.yaml \
  --tag scalping-active-mgmt-validation \
  --seed <seed used for Gen 1 if recorded, else 42>
```

While the run is going, sanity-check the live training
monitor: does the per-episode fill-rate curve trend up
generation over generation? If it's flat or trending down,
something is wrong — stop, diagnose, re-read the activation
playbook's failure-modes table.

Post-run: record the run_id (or tag → run_id mapping) in
scratch notes; you need it for the next step.

#### 2. Build the comparison script

`scripts/scalping_active_comparison.py`. Output: two CSV
files and a markdown summary.

**CSV 1 — per-generation fill-rate evolution.**

Columns: `generation`, `gen_1_top_model_fill_rate`,
`validation_top_model_fill_rate`, `gen_1_median_fill_rate`,
`validation_median_fill_rate`. One row per generation
(0..3). Uses `registry/scoreboard.py` or the underlying DB
to pull top-model-per-generation metrics. Fill rate =
`arbs_completed / (arbs_completed + arbs_naked)`.

**CSV 2 — top-model-per-run comparison.**

Columns: `metric`, `gen_1_value`, `validation_value`,
`delta`, `pct_change`. Rows:
- `ln_ratio` (locked-over-naked ratio; this is the
  scoreboard's primary ranking metric).
- `composite_score`.
- `arbs_completed`.
- `arbs_naked`.
- `fill_rate` (computed).
- `mean_pnl`.
- `fill_prob_mace` — `None` for Gen 1 (head didn't exist),
  float for validation.
- `risk_spearman_rho` — correlation between
  `predicted_locked_stddev_at_placement` and
  `abs(realised_locked_pnl - predicted_locked_pnl)` on the
  validation run's completed pairs. `None` for Gen 1.

**Markdown summary.**

`plans/scalping-active-management/validation_report.md`.
Structure:

```markdown
# Scalping Active Management — Validation Report

## Headline

(One-line verdict: "Plan is net-positive / neutral /
negative vs Gen 1." Followed by two or three numbers.)

## Targets from purpose.md

| # | Target | Measured | Hit? |
|---|---|---|---|
| 1 | Fill rate lifts from 14.5 % toward 50 %+ | {top_model_fill_rate} % | ✅ / ❌ |
| 2 | Fill-prob MACE ≤ 5 % per bucket | {per_bucket_table} | ✅ / ❌ |
| 3 | Risk Spearman ρ > 0.3 | {rho} | ✅ / ❌ |
| 4 | `arbs_naked` trends down across generations | {gen_by_gen_trend} | ✅ / ❌ |

## Generation-over-generation trend

(Embed CSV 1 as a markdown table or link to the file.)

## Top-model comparison

(Embed CSV 2 similarly.)

## What didn't work

(Any of the four targets that missed, with your best guess
at why. This is NOT a place to propose fixes — just honest
accounting. Fixes go in a follow-on plan.)

## Reproducing

- Gen 1 baseline: commit `7a3968a`, tag
  `<gen 1 training tag>`, config X.
- Validation run: commit `<current HEAD>`, tag
  `scalping-active-mgmt-validation`, config
  `config/scalping_gen1.yaml` at weights
  `fill_prob_loss_weight={value}`,
  `risk_loss_weight={value}` (as promoted by activation
  playbook Step E in commit `<activation commit hash>`).
- Same day range, same seed, same population size.
```

#### 3. Analyse and write

Run the comparison script, produce the three artefacts
(two CSVs + the markdown report). Commit all three under
`plans/scalping-active-management/`.

Append to `lessons_learnt.md` anything surprising:

- Did `arbs_naked` actually trend down, or did the agent
  discover a trickier failure mode?
- Does the risk head's stddev predict realised variance,
  or did it collapse to the marginal stddev for everything?
- Is there a generation-over-generation interaction (e.g.
  the aux heads help early generations but plateau by
  Gen 3)?
- Any reward-scale ghost chases — cases where P&L looked
  lower but the operator chasing the delta missed an
  activation-playbook config commit?

If the plan didn't hit some targets, **do not** open a
Session 08 to fix them in this plan. The plan's scope was
the seven sessions. Follow-up work opens a new plan folder
under `plans/` with its own purpose.md / master_todo.md.

### Tests

This session is measurement, not mechanism — there's no
new code path to unit-test. But:

1. The comparison script itself needs at least smoke tests:
   - Runs against a synthetic scoreboard fixture without
     crashing.
   - Produces both CSVs with the right column sets.
   - Handles a model with zero scalping bets
     (validation_run missing values, not crash).
2. A full-suite `pytest tests/ -q` MUST still be green on
   HEAD — this session must not regress the existing 1968+
   tests. No silent schema drift in the scoreboard /
   bet-log paths.

### Exit criteria

- `scripts/scalping_active_comparison.py` exists and runs.
- `plans/scalping-active-management/validation_report.md`
  exists and is honest (not cherry-picked; all four targets
  rated ✅ or ❌ explicitly).
- Both CSVs committed alongside the report.
- Headline verdict is documented in `progress.md` under a
  Session 07 entry — reader gets the one-line summary
  without having to open the report file.
- `lessons_learnt.md` appended with the surprises.
- Full suite green.
- Final commit referencing `plans/scalping-active-management/`
  + session 07. If the plan was net-positive, say so in the
  commit message. If it wasn't, say that clearly too —
  operators need to know whether to keep the feature on
  for subsequent plans.

---

## Cross-session rules for the whole plan

- `pytest tests/ -q` green.
- Do NOT open follow-up sessions under this plan folder to
  address measurement failures — start a new plan.
- Do NOT touch env / trainer / aux heads / UI to "make the
  numbers look better". If the numbers aren't there, they
  aren't there; document and move on.
- Final commit message: **even more important than usual**
  to be loud about any reward-scale changes or unexpected
  dynamics. Future plans will cite this commit as the
  reference for "what the stack looked like post-scalping-
  active-mgmt".
- Knock-on work for `ai-betfair`: drop a final summary
  note in `ai-betfair/incoming/` pointing to the
  validation report — the live-inference repo needs to
  know whether to rely on the aux-head outputs or treat
  them as diagnostic-only.

---

## After Session 07

This plan folder is **closed**. Further scalping work
(cross-market arbitrage, partial-fill-aware sizing, CVaR
on the risk head — all listed as out-of-scope in
`purpose.md`) opens new plan folders. Leave
`plans/scalping-active-management/` in place as a reference;
don't archive it.
