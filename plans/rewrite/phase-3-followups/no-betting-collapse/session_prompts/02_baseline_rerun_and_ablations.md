# Session prompt — no-betting-collapse Session 02: baseline re-run + shaping ablations

Use this prompt to open a new session in a fresh context. The prompt
is self-contained — it briefs you on the task, the design decisions
already locked from Session 01, and the constraints. Do not require
any context from the session that scaffolded this prompt.

---

## The task

Session 01 (2026-04-30, commit `ace20c2`) shipped two changes that
together make the AMBER baseline at
`registry/v2_first_cohort_1777499178/` no longer the right
comparison floor:

1. **Env accounting fix** —
   `env/betfair_env.py::_settle_current_race` now zeros
   `scalping_locked_pnl` and `scalping_early_lock_bonus` on the
   `void_race` branch. Pre-fix the cohort showed `eval_day_pnl ==
   0` with `eval_locked_pnl + eval_naked_pnl == 0` exactly across
   all 12 agents; the trace found that on void races
   `get_paired_positions` had already accumulated the
   would-have-been lock cash before the void path returned
   `race_pnl == 0`, leaving phantom positives in `locked_pnl`
   that the residual `naked_pnl` cancelled. The fix makes
   void-race cash buckets honestly report 0.

2. **Day-selection filter** —
   `training_v2/discrete_ppo/train.py::_enumerate_day_files` now
   drops any `YYYY-MM-DD.parquet` whose `winner_selection_id`
   column is fully null, with a warning. The AMBER baseline's
   eval day `2026-04-29` had 0/2 markets carrying winners; every
   race voided regardless of policy, producing a Bar-6c FAIL
   (0/12 positive) that no policy could have avoided. The
   parquet was deleted on 2026-04-30; the filter ensures any
   future day with the same defect can never be selected again.

The combined effect: `select_days(seed=42)` in the post-fix code
will pick a different eval day from the pre-fix code (because
`2026-04-29` is now filter-excluded, AND the most-recent-day
ordering shifts). So the AMBER baseline cannot be meaningfully
compared against any post-fix cohort run — the day windows
differ.

**The first thing this session does is re-run the baseline.**
Same protocol (12 agents / 1 generation / `--seed 42` / 7+1
day split / no shaping), but on the post-fix code. Call this the
"AMBER v2 baseline". Bar 6c on AMBER v2 is the load-bearing
question:

- If AMBER v2 PASSES Bar 6c (≥ 1/12 positive on raw P&L) → the
  rewrite is GREEN. The original AMBER FAIL was a data + telemetry
  artefact, not an architectural failure. Plan ships GREEN; skip
  the shaping ablations entirely.
- If AMBER v2 FAILS Bar 6c (0/12 positive) → the architectural
  problem the original plan suspected is real. Proceed with the
  ablation cycle as originally specified, but measured against
  AMBER v2 not the discarded original AMBER baseline.

End-of-session bar:

1. **AMBER v2 baseline run**: cohort run completes in
   `registry/v2_amber_v2_baseline_<ts>/`. Scoreboard inspected;
   Bar 6 (a, b, c) computed and recorded in this plan's
   findings.md.
2. Verdict on Bar 6c logged as either:
   - **GREEN**: at least one agent positive on raw P&L. Plan
     marked complete; no ablations run; Session 03 (writeup)
     loaded next.
   - **FAIL**: 0/12 positive. Run the first shaping ablation
     (`matured_arb_bonus_weight = 1.0`) to a new output dir and
     score it against AMBER v2. Document the comparison in
     findings.md.
3. If the first ablation also FAILS Bar 6c → stop. Don't
   automatically chain — the operator decides whether to spend
   another ~3.1 h GPU on the next ablation
   (`naked_loss_anneal`).

## What you need to read first

1. `plans/rewrite/phase-3-followups/no-betting-collapse/purpose.md`
   — this plan's purpose, success bar, hard constraints.
2. `plans/rewrite/phase-3-followups/no-betting-collapse/findings.md`
   — Session 01's verdict and the trace that landed the env fix.
3. `plans/rewrite/phase-3-cohort/findings.md` Session 04
   "Live-run results 2026-04-29 → 2026-04-30" — the raw AMBER
   observation that motivated the follow-on. **Note**: every
   number in this section came from the pre-fix code on the
   bad-data eval day; it is no longer the comparison floor.
4. `env/betfair_env.py:3000-3030` — the void-branch fix landed
   in Session 01.
5. `training_v2/discrete_ppo/train.py::_enumerate_day_files` and
   `_day_has_any_winner_data` — the day-selection filter.
6. `training_v2/cohort/runner.py::main` — the cohort CLI used
   below.

## What to do

### 1. Pre-flight (~10 min)

Confirm:

- `data/processed/2026-04-29.parquet` is absent (deleted in
  Session 01). It will be re-created the moment the data
  pipeline runs again — by design the filter catches it
  regardless, but you should know whether it's there.
- `_day_has_any_winner_data` excludes 2026-04-29 if it returns
  with 0 winners. Synthetic check:

  ```python
  from pathlib import Path
  import pyarrow.parquet as pq, pyarrow as pa
  from training_v2.discrete_ppo.train import _enumerate_day_files
  print(_enumerate_day_files(Path("data/processed")))
  ```

- The current `select_days(data_dir=Path("data/processed"),
  n_days=8, day_shuffle_seed=42)` returns 7 training days plus
  the held-out eval day. Note which date is the new eval. **If
  the new eval day matches an old training day from the original
  AMBER cohort, that's expected — the day window shifted by one.
  Document the new days in findings.md before launching the
  cohort.**

### 2. AMBER v2 baseline cohort (~3.1 h GPU)

Same flags as the original AMBER baseline (per
`plans/rewrite/phase-3-cohort/session_prompts/04_frontend_and_first_cohort.md`):

```
python -m training_v2.cohort.runner \
    --n-agents 12 --generations 1 --days 8 \
    --device cuda --seed 42 \
    --output-dir registry/v2_amber_v2_baseline_$(date +%s)
```

Wall-time envelope: ~3.1 h on a 3090. Watch for crashes; the
new winner-coverage filter does not change throughput, but if
the day window picked has different per-day market counts the
per-day wall could shift up to ~10 %.

### 3. Score Bar 6 (~15 min)

The Bar-6 analysis tool (`C:/tmp/v2_phase3_bar6.py` per Session
01 prompt §"Useful pointers") consumes a registry directory and
prints:

- Bar 6a: mean force-close rate across cohort. PASS < 0.50.
- Bar 6b: ρ(entropy_coeff, fc_rate) across cohort. PASS ≤ −0.5.
- Bar 6c: count of agents with `eval_day_pnl > 0`. PASS ≥ 1/12.

Run it on the new output dir:

```
python C:/tmp/v2_phase3_bar6.py registry/v2_amber_v2_baseline_<ts>
```

Record all three numbers in findings.md alongside the original
AMBER (now-discarded) row, with a header note: "Original AMBER
baseline (eval-day 2026-04-29) discarded post Session 01 fix —
contaminated by void_race telemetry bug + missing winner data."

Per-agent eval P&L pattern check — verify the void-race fix is
working in production:

```python
import json
rows = [json.loads(l) for l in
    open("registry/v2_amber_v2_baseline_<ts>/scoreboard.jsonl")
    .read().splitlines() if l.strip()]
for r in rows:
    locked = r.get("eval_locked_pnl", 0.0)
    naked = r.get("eval_naked_pnl", 0.0)
    day = r.get("eval_day_pnl", 0.0)
    print(r["agent_id"][:12],
          f"locked={locked:+.2f} naked={naked:+.2f} day={day:+.2f}",
          "void-pattern!" if abs(locked + naked) < 1e-6 and abs(day) < 1e-6 else "")
```

If any row prints "void-pattern!" — the fix didn't take. Stop
and re-investigate. (Should not happen; the unit test
`test_matured_pair_on_void_race_reports_zero_cash_buckets`
covers this.)

### 4. Branch on Bar 6c

#### 4a. GREEN — Bar 6c ≥ 1/12 positive

Plan ships GREEN. Update findings.md "Verdict summary" table:

| Item | Status |
|---|---|
| AMBER v2 baseline | ✓ done |
| Bar 6c | PASS |
| Verdict | GREEN — rewrite premise survives |
| Shaping ablations | not needed |

Mark this plan `status: green; complete`. Load Session 03
(writeup) next. **Do not proceed to ablations.**

#### 4b. FAIL — Bar 6c == 0/12

The architectural problem is real (not a data artefact).
Proceed to the first shaping ablation:

**Ablation 1 — `matured_arb_bonus_weight = 1.0`**

Per `purpose.md` §"Ablation order is locked". This term most
directly rewards completing pair lifecycles (which is what the
agent stopped doing). One config knob, no gene change.

```
python -m training_v2.cohort.runner \
    --n-agents 12 --generations 1 --days 8 \
    --device cuda --seed 42 \
    --reward-overrides matured_arb_bonus_weight=1.0 \
    --output-dir registry/v2_ablation_matured_arb_$(date +%s)
```

(If the runner doesn't accept `--reward-overrides`, check the
gene-schema config plumbing in
`training_v2/cohort/genes.py::CohortGenes` — the ablation needs
a plan-level reward-override path, not a gene addition. **Hard
constraint §1: no gene additions.** If no plan-level path
exists, that's a small implementation-prep step inside this
session before launching.)

Wall ~3.1 h. Score Bar 6 on the new dir. Compare side-by-side
in findings.md:

| Cohort | mean fc | ρ(ent, fc) | positive on P&L |
|---|---|---|---|
| AMBER v2 baseline | … | … | 0/12 |
| + matured_arb_bonus | … | … | ? |

#### 4c. Stop after the first ablation

If ablation 1 PASSES Bar 6c → plan ships GREEN with one
restored shaping term. Document; load Session 03.

If ablation 1 FAILS Bar 6c → stop. Don't auto-launch ablation 2.
The operator decides whether the next ~3.1 h GPU on
`naked_loss_anneal` is justified, or whether the verdict is
RED-with-caveat (purpose.md §"Success bar"). Document the
state honestly and end the session.

## Stop conditions

- **AMBER v2 cohort crashes mid-run** → triage; mechanical bugs
  (GPU OOM, file lock) are cheap to retry but DO NOT bundle a
  fix into this session — the rewrite hard-constraint §3 says
  parallel-tree work; if the bug is in `training_v2/`, fix it
  in its own commit before retrying. If the bug is in `env/`,
  STOP — env edits are out of scope and might invalidate the
  Session 01 fix's regression coverage.
- **AMBER v2 wall > 5 hours** → kill, file
  `phase-3-followups/throughput-fix/` (the workstream named in
  `phase-3-cohort/findings.md` line 754 but never written).
  Throughput regression here means something unrelated broke;
  do NOT debug it inside this session.
- **Cannot construct the day-selection filter check** → the
  `_day_has_any_winner_data` helper or `_enumerate_day_files`
  is broken. Stop, investigate, re-run
  `tests/test_v2_multi_day_train.py::test_enumerate_day_files_excludes_days_without_winner_data`
  to triage.
- **Both AMBER v2 baseline and the first ablation FAIL Bar 6c**
  → stop. The session ships with the documented FAIL; the
  operator decides whether to chain ablation 2 or call
  RED-with-caveat.

## Hard constraints

Inherited from `purpose.md` §"Hard constraints" plus:

1. **No env edits.** The Session 01 fix is final. Any env
   regression discovered in this session goes into a SEPARATE
   plan / commit, not bundled.
2. **No GA gene additions.** Shaping ablations operate ENTIRELY
   through plan-level `reward.*` config knobs that the env
   already supports (per existing `reward_overrides` mechanism
   in `env/betfair_env.py`). If a knob is missing, that's a
   reason to stop, not to add it inline.
3. **Same `--seed 42` for every cohort.** Cross-cohort
   comparison is the load-bearing mechanism for any verdict.
4. **NEW output dirs for every run.** Don't overwrite
   `registry/v2_first_cohort_1777499178/` (kept as the original
   AMBER artefact, with the caveat that its eval-day data was
   bad). Don't overwrite the AMBER v2 dir with ablation runs.
5. **One ablation at a time.** Stacking shaping terms produces
   no per-term signal.
6. **Bar 6c is "≥ 1 agent individually positive".** Don't grade
   on cohort means.

## Out of scope

- Throughput fix (separate plan named but not yet written:
  `plans/rewrite/phase-3-followups/throughput-fix/`).
- 66-agent scale-up (gated on this plan's GREEN verdict; comes
  in a Phase-4 follow-on, not here).
- v1 deletion (gated on the rewrite's overall PASS).
- Multi-generation cohorts (locked at `--generations 1` for
  this whole follow-on).
- New genes / schema changes.
- Re-processing the deleted 2026-04-29 parquet (the day filter
  handles re-emergence; no need to chase the source data).

## Useful pointers

- AMBER v1 baseline (now discarded as a comparison floor):
  `registry/v2_first_cohort_1777499178/scoreboard.jsonl`. Keep
  for historical reference.
- Session 01 commit: `ace20c2` (env fix + day filter +
  accounting tests).
- Session 01 findings:
  `plans/rewrite/phase-3-followups/no-betting-collapse/findings.md`.
- Bar 6 analysis tool: `C:/tmp/v2_phase3_bar6.py`.
- Cohort entry point: `training_v2.cohort.runner`.
- Scoreboard schema: `training_v2/cohort/worker.py:436` (the
  registry-write block).
- Reward override mechanism: search `env/betfair_env.py` for
  `reward_overrides` — that's the plan-level knob path.
- The test that pins the void-fix:
  `tests/test_v2_eval_pnl_accounting.py::test_matured_pair_on_void_race_reports_zero_cash_buckets`.
- The test that pins the day filter:
  `tests/test_v2_multi_day_train.py::test_enumerate_day_files_excludes_days_without_winner_data`.

## Estimate

3.5–7 h, of which ~3–6 h is GPU wall:

- 30 min: pre-flight + day-window inspection + filter
  verification.
- 3.1 h: AMBER v2 baseline cohort.
- 15 min: Bar 6 scoring.
- 30 min: findings.md write-up of baseline.
- (only if Bar 6c FAILS) +3.1 h: first ablation.
- (only if both FAIL) +30 min: write-up + stop verdict.

Best case (AMBER v2 GREEN): ~4 h. Worst case (both FAIL): ~7 h.

If past 8 h excluding cohort wall time, stop and check scope —
something other than waiting for GPU is taking time, which is a
sign the hard constraints are slipping.
