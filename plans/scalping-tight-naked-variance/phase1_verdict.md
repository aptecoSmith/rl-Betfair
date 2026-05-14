# Phase 1 verdict — scalping-tight-naked-variance

**Status: BLOCKED on Phase 1c (held-out reeval). Phase 0 + Phase 1b
completed.**

---

## Headline

The autonomous run produced the per-cohort naked-variance reports and
union-of-top-5 selectors per the plan, but **held-out reeval is
blocked by stop condition #3**: the cohort weights are saved at
`obs_schema_version=8`, and master HEAD (commit `045174d`,
2026-05-14 14:54) bumped the env to `obs_schema_version=9` with a new
per-runner `seconds_since_aggressive_placed` observation column.
Loading raceconf or layq checkpoints into the current env fails with
shape mismatch on `input_proj.0.weight`
(`[64, 504]` saved vs `[64, 574]` current).

No held-out reeval rows were produced. Operator decision is required
before any cohort × selector × window × fc cell can be evaluated.

---

## What was done

### Phase 0 — Variance reporting tool ✅

- `tools/build_naked_variance_report.py` — per-cohort report tool,
  module-level constants per [hard_constraints.md §5](hard_constraints.md).
- `tests/test_naked_variance_report.py` — all 5 tests pass:
  - `test_recovers_known_values_on_synthetic_data`
  - `test_score_e_boundary` (σ_leg=30 + daily=100 KEEP)
  - `test_falls_back_to_db_when_no_per_leg_data`
  - `test_nan_when_sample_too_small`
  - `test_empty_cohort_produces_empty_csv`
- Committed as `83a21a6 feat(scalping-tight-naked-variance):
  variance report tool`.

### Phase 1a — `--filter-agent-ids` ✅ no-op

`tools/reevaluate_cohort.py` already accepts
`--filter-agent-ids <id1> <id2> ...`. No change required.

### Phase 1b — Reports + union-top-5 ✅

Ran the report on both cohorts. Outputs:

- `registry/_predictor_SCALPING_raceconf_1778661062/naked_variance_report.csv`
- `registry/_predictor_SCALPING_raceconf_1778661062/phase1_top5_union.txt`
  (14 unique agents)
- `registry/_predictor_SCALPING_layq_1778712871/naked_variance_report.csv`
- `registry/_predictor_SCALPING_layq_1778712871/phase1_top5_union.txt`
  (14 unique agents)

Both cohorts use a 3-day in-sample-eval window (the same data
`compare_naked_variance_cohorts.py` pulled day 1 of). σ_leg is
populated on every agent with `n_naked_legs ≥ 5`; per-day stats
populated on every agent (n_eval_days = 3).

#### Raceconf — TOP-5 by selector (in-sample-eval only)

Highlights from the printed top-15s:

- `score_a (pure_locked)` top: `e716b410-539` (£127), `609bf1a7-768` (£121),
  `58a28e87-5ee` (£114), `4b4101b5-9ff` (£113), `0de125f5-648` (£112).
- `score_b (per-leg sharpe)` top: `30017150-c46` (σ_leg £0.45 → 68.7),
  `eb4c22b7-b42` (σ_leg £2.60 → 23.4), `f096b9c3-7f2` (σ_leg £8.08 → 8.9).
- `score_d (daily_vol_penalty)` top is `30017150-c46` (+£99.28 score),
  `eb4c22b7-b42` (+£81.22), `cf5975e5-3dc` (+£75.53).
- `score_e (combined filter, σ_leg ≤ £30 AND daily_vol ≤ £100)` keeps
  17 agents above zero; top: `f5001118-0e5` (£100.36), `30017150-c46`
  (£99.65), `cf5975e5-3dc` (£91.35).

#### Layq — TOP-5 by selector (in-sample-eval only)

- `score_a (pure_locked)` top: `abdfa0f3-3ac`, `2e92886c-f87`, `9b3a2b39-ab6`,
  `d66d78ab-eb2`, `c8c92859-be8`.
- `score_b (per-leg sharpe)` top: `9394c439-576` (σ £8.92 → 8.7),
  `f1a118cf-c8c` (σ £9.54 → 7.3).
- `score_d` top: `3a81bb45-172` (+£38.17), `595435ea-885` (+£33.55).
- `score_e (combined filter)` only keeps 2 layq agents above zero:
  `9394c439-576` (£85.67) and `f1a118cf-c8c` (£76.89). Most of layq's
  high-locked population has `daily_naked_vol > £100` — the structural
  reason the layq population sits at higher leg counts than raceconf
  per the README cross-cohort scan.

#### Cross-cohort sanity check

The two cohorts produce comparable σ_leg distributions in-sample
(consistent with `compare_naked_variance_cohorts.py`'s day-1 reading
of σ_leg ≈ £36 across both). The selectors that bite are score_d and
score_e — they surface 17 raceconf and 2 layq agents that combine
locked floor with σ_leg ≤ £30 and daily_vol ≤ £100. These are the
in-sample tight-variance candidates. **In-sample only** — the
deployment question requires held-out reeval, which is blocked
(see below).

### Phase 1c — Held-out reeval ✗ BLOCKED

**Stop condition #3 fired.**

The autonomous run launched all 8 reeval combinations
(2 cohorts × 2 windows × 2 fc settings) with predictor-bundle flags
mirrored from the prior `reeval_LOCKED5_fc0_NEW7DAYS_…`
invocation. Every agent failed weight load on the first eval day:

```
[1/14] 58a28e87-5ee: failed to load weights (Error(s) in loading
state_dict for DiscreteLSTMPolicy:
size mismatch for input_proj.0.weight: copying a param with shape
torch.Size([64, 504]) from checkpoint, the shape in current model
is torch.Size([64, 574]).)
```

Direct verification of a sample checkpoint:

```python
>>> torch.load('weights/0117f71d-….pt')['obs_schema_version']
8
```

`env/betfair_env.py::OBS_SCHEMA_VERSION` is currently 9 (commit
`045174d`, today at 14:54). Commit message:

> Bump SCALPING_POSITION_DIM 8 → 9 and OBS_SCHEMA_VERSION 8 → 9.
> Add a per-runner age column populated from the matched aggressive
> leg of every open pair (matched aggressive + unmatched passive
> partner — same predicate as the Phase-2b naked-leg loop).

This breaks weight cross-load with raceconf, layq, AND every other
cohort trained at schema 8. Plan [hard_constraints.md §2](hard_constraints.md)
called this out for the lockfit cohort but expected raceconf/layq to
remain loadable — they're not, because the env schema bump applies
tree-wide, not just to lockfit-shape weights.

---

## Operator decision required

Three viable paths:

### Option 1 — Pin reevals to a pre-bump worktree

Create a git worktree at commit `2c03503` (the parent of `045174d`,
last commit at obs_schema_version=8) and run the 8 reevals from
there. Predictor bundles and reevaluator code are otherwise compatible.

Pros: zero code churn on master; surgical to the problem.
Cons: any plan iteration that needs both lockfit's age-obs feature
AND raceconf/layq reeval has to juggle worktrees.

### Option 2 — Add a "legacy obs schema" runtime guard

Modify `env/betfair_env.py` so that `obs_schema_version=8` checkpoints
load against an env that ZEROS the `seconds_since_aggressive_placed`
column (or, equivalently, the env exposes a `--legacy-obs-schema=8`
flag that subtracts the new column before computing observations).

Pros: solves this and future schema-bump frictions.
Cons: feature creep relative to this plan. Probably belongs in its
own micro-plan or as a follow-on to
`scalping-locked-fitness-and-age-obs`.

### Option 3 — Revert `045174d`

Roll back the obs schema bump. The locked-fitness-and-age-obs plan's
phase 3+4 cohort that depends on schema=9 would lose its weight
compatibility. Per this plan's
[hard_constraints.md §2](hard_constraints.md), lockfit is already
out of scope; the cohort that ran on it had only 8 gen-0 agents.

Pros: cleanest for this plan's Phase 1.
Cons: discards predecessor plan's evidence about the new obs's
value, even if that evidence was thin.

**Default recommendation (per the verdict-ambiguous rule): Option 1.**
It is reversible, scoped, requires no code change on master, and
leaves the operator free to merge Option 2 or Option 3 later.

---

## Phase 1 verdict against the bands

**Not assessable from in-sample data alone.** Per
[hard_constraints.md §23](hard_constraints.md) σ_leg is the primary
deployment metric — but the success bands in the README all reference
**fc=120 held-out 7-day** numbers, which require the 8 reevals.

The in-sample report does narrow the candidate set: 14 agents per
cohort survive the union-top-5 filter, score_e (combined filter,
σ_leg ≤ £30 AND daily_vol ≤ £100) only keeps 17 raceconf + 2 layq.
The held-out reeval (once unblocked) should focus on those candidates
and the union-top-5 list already on disk.

---

## Phase 2 status

**NOT triggered.** Per
[hard_constraints.md §22](hard_constraints.md), Phase 2 is gated on
Phase 1 verdict + operator sign-off. Phase 1 is incomplete, so the
gate is closed.

---

## Files committed

- `tools/build_naked_variance_report.py`
- `tests/test_naked_variance_report.py`
- `plans/scalping-tight-naked-variance/phase1_verdict.md` (this)
- `plans/scalping-tight-naked-variance/autonomous_run_log.md`

Per-cohort artefacts (not committed; live in `registry/`):

- `registry/_predictor_SCALPING_raceconf_1778661062/naked_variance_report.csv`
- `registry/_predictor_SCALPING_raceconf_1778661062/phase1_top5_union.txt`
- `registry/_predictor_SCALPING_layq_1778712871/naked_variance_report.csv`
- `registry/_predictor_SCALPING_layq_1778712871/phase1_top5_union.txt`
