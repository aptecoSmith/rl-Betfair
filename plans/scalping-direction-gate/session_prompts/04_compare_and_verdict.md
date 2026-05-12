# Session 04 — Compare + verdict

Final session. Read the held-out reeval JSONL, compute the A/B
against the pwin-gate cohort, write `findings.md`, commit, stop.

## Pre-checks

1. The watcher's log must show "reeval done." — confirm
   `<cohort_dir>/auto_reeval_2026-04-28_30.log` ends with that
   line.
2. The reeval JSONL must exist:
   `<cohort_dir>/reeval_held_out_2026-04-28_30.jsonl`. It should
   have 5 rows (one per top-5 agent).
3. The pwin-gate predecessor's reeval must also exist for the
   comparison — at
   `registry/_predictor_SCALPING_pwingate_1778571007/reeval_held_out_2026-04-28_30.jsonl`.

## Required computation

Build this comparison table (Python helper inline):

```python
# Per-agent held-out
for r in reeval_rows:
    aid = r['agent_id'][:8]
    in_sample = scoreboard[r['agent_id']]['eval_day_pnl']
    held = r['reeval_day_pnl']
    locked = r['reeval_locked_pnl']
    naked = r['reeval_naked_pnl']
    mr = r['reeval_maturation_rate']
    per_day = {d['eval_day']: d['day_pnl'] for d in r['reeval_per_day']}
    # print into the table
```

Aggregates needed:
- mean held-out pnl
- median held-out pnl
- profitable count (X / 5)
- mean locked
- mean naked
- mean mr

## findings.md template

Write `plans/scalping-direction-gate/findings.md`:

```markdown
# Scalping direction-gate cohort — findings

**Cohort**: `_predictor_SCALPING_dirgate_<TIMESTAMP>`
**Launched**: <date>
**Completed**: <date>
**Wall clock**: <h>h<m>m

## In-sample generation trajectory

| gen | n | mean | median | best | profitable | mean locked | mean naked |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 12 | ... | ... | ... | ... | ... | ... |
| ... |
| 7 | 12 | ... | ... | ... | ... | ... | ... |

## Held-out reeval (2026-04-28/29/30, top-5)

| agent | gen | in-sample | held-out | day_04-28 | day_04-29 | day_04-30 | locked | naked | mr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ... |

**Mean held-out: £+/- X**
**Median held-out: £+/- X**
**Profitable: X/5**
**Mean locked: £X**
**Mean naked: £X**
**Mean mr: 0.X**

## Comparison vs predecessor cohorts

| metric | safety-gene | pwin-gate | **dirgate** |
|---|---:|---:|---:|
| held-out mean | -£69 | <pwin result> | <dirgate result> |
| held-out median | -£69 | <...> | <...> |
| profitable | 2/5 | <...> | <...> |
| mean locked | +£167 | <...> | <...> |
| mean naked | -£196 | <...> | <...> |
| mean mr | 0.235 | <...> | <...> |

## Verdict against success bar

Success bar (README): ≥3 of top-5 profitable on held-out.

Result: <PASS / FAIL> — actual <X/5>.

## Branch interpretation

(One of the four outcomes from README "What success looks like".
State which applies and why.)

- **Strong success** (mean held-out > +£20, ≥4/5 profitable):
  ...
- **Modest success** (mean held-out > 0, ≥3/5 profitable):
  ...
- **No improvement** (mean held-out ~= pwin-gate cohort):
  ...
- **Regression** (mean held-out < pwin-gate cohort):
  ...

## What worked / what didn't

(1-3 paragraphs of plain English diagnosis.)

## Next plan recommendation

(Based on the branch above. One short paragraph naming the
specific next move — e.g. "stack race-confidence gate" or
"reduce pwin lay threshold to 0.30" or "begin shadow-trading
top agent in ai-betfair".)
```

## Commit message

```
docs(scalping-direction-gate): held-out reeval verdict

Cohort _predictor_SCALPING_dirgate_<TS> ran 12 agents x 8
generations on the same configuration as the pwin-gate cohort
plus --direction-gate-enabled. Held-out reeval on 2026-04-28/29/30.

Result: <verdict>
  mean: £X (vs pwin-gate £Y, vs safety-gene -£69)
  profitable: X/5 (vs Y/5, vs 2/5)
  mr: 0.X (vs 0.Y, vs 0.235)

<one-sentence interpretation>

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## After commit

1. Add a final entry to `autonomous_run_log.md` declaring the
   loop terminated.
2. **DO NOT call ScheduleWakeup.** Loop ends.
3. Surface a one-paragraph summary in the iteration's text
   output. The operator will read findings.md when they next
   interact.
