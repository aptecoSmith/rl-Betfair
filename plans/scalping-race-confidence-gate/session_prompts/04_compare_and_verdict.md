# Session 04 — Compare + verdict

Final session. Read held-out reeval JSONL, compute A/B vs
pwin-gate cohort, write `findings.md`, commit, stop.

## Pre-checks

1. Watcher log shows "reeval done." —
   `<cohort_dir>/auto_reeval_2026-04-28_30.log` ends with that
   line
2. Reeval JSONL exists with 5 rows:
   `<cohort_dir>/reeval_held_out_2026-04-28_30.jsonl`
3. Predecessor pwin-gate's reeval exists for comparison:
   `registry/_predictor_SCALPING_pwingate_1778571007/reeval_held_out_2026-04-28_30.jsonl`

## findings.md template

Write `plans/scalping-race-confidence-gate/findings.md`:

```markdown
# Scalping race-confidence-gate cohort — findings

**Cohort**: `_predictor_SCALPING_raceconf_<TS>`
**Launched**: <date>
**Completed**: <date>
**Wall clock**: <h>h<m>m

## In-sample generation trajectory

| gen | n | mean | median | best | profitable | locked | naked |
|---:|---:|---:|---:|---:|---:|---:|---:|

## Held-out reeval (2026-04-28/29/30, top-5)

| agent | gen | in-sample | held-out | d28 | d29 | d30 | locked | naked | mr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

**Mean held-out: X**
**Median held-out: X**
**Profitable: X/5**

## Comparison vs predecessor cohorts

| metric | safety-gene | pwin-gate | **raceconf** |
|---|---:|---:|---:|
| held-out mean | -£69 | -£13 | <X> |
| held-out median | -£69 | +£48 | <X> |
| profitable | 2/5 | 3/5 | <X/5> |
| mean locked | +£167 | +£144 | <X> |
| mean naked | -£196 | -£130 | <X> |
| mean mr | 0.235 | 0.334 | <X> |

## Verdict against success bar

Success bar (README):
- ≥3/5 profitable AND mean > pwin-gate's -£13

Result: <PASS / FAIL>

## Branch interpretation

State which branch from README "What success looks like":

- **Strong**: mean > +£30 AND ≥4/5 profitable
- **Modest**: mean > 0 AND ≥3/5 profitable
- **No improvement**: mean ~= -£13
- **Regression**: mean < -£13

## What worked / what didn't

(1-3 paragraphs)

## Next plan recommendation

(One paragraph naming the specific next move.)
```

## Commit message

```
docs(scalping-race-confidence-gate): held-out reeval verdict

Cohort _predictor_SCALPING_raceconf_<TS> ran 12 agents x 8
generations stacking the per-race confidence gate (threshold
0.30) on top of the pwin-gate's per-runner filter. Held-out
reeval on 2026-04-28/29/30.

Result: <verdict>
  mean: £X (vs pwin-gate £-13, safety-gene £-69)
  profitable: X/5 (vs 3/5, vs 2/5)

<one-sentence interpretation>

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## After commit

1. Add final entry to `autonomous_run_log.md` declaring loop
   terminated.
2. **DO NOT call ScheduleWakeup.** Loop ends.
3. Surface a one-paragraph summary in the iteration's output.
   The operator reads findings.md on next interaction.
