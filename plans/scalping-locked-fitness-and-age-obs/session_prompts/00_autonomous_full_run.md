# 00 — Autonomous full run — scalping-locked-fitness-and-age-obs

You are driving this plan to completion: stack a locked-weighted
GA selection metric + a pair-age obs feature on top of the
scalping-lay-quality-gate cohort, on the bigger 2026-05-14 data
pool. **No operator interaction.** Make every decision yourself
using the documents + defaults below.

## Deliverable

Held-out reeval verdict on **2026-04-28/29/30** (hard-coded —
same surface as every predecessor). Success bar (raised vs
predecessor):

- **Modest**: mean > +£70/day (clear beat) AND ≥ 4/5 profitable
- **Strong**: mean > +£100/day AND ≥ 5/5 profitable

Report BOTH `force_close=0` and `force_close=120` reeval numbers
per `memory/project_force_close_train_vs_deploy.md`.

Predecessor baseline (lay-quality-gate, 2026-05-14):
fc=0 mean +£192.53/day, 5/5 profitable; fc=120 mean +£25.74,
3/5 profitable. Anything above these on fc=0 is a hit; the
fc=120 number is the deployment-realistic comparison.

## Read FIRST every iteration

1. `plans/scalping-locked-fitness-and-age-obs/README.md`
2. `plans/scalping-locked-fitness-and-age-obs/hard_constraints.md`
3. `plans/scalping-locked-fitness-and-age-obs/master_todo.md`
4. `plans/scalping-locked-fitness-and-age-obs/session_handoff_2026-05-14.md`
   — the load-bearing "where to start" doc.
5. `plans/scalping-locked-fitness-and-age-obs/autonomous_run_log.md`
   — you append to this every iteration.
6. The relevant memory entries (loaded automatically):
   - `project_select_days_data_dir_dependence.md` — leak
     boundary check is load-bearing.
   - `feedback_sort_top_by_locked_not_total.md` — selection rule.
   - `reference_cohort_metrics_panel.md` — metric definitions.
   - `reference_phenotype_analysis_methodology.md` — analysis pipeline.
   - `project_two_cohort_diversification.md` — predecessor + this
     plan are deployable side-by-side.
   - `project_force_close_train_vs_deploy.md` — dual reeval discipline.
7. Predecessor `plans/scalping-lay-quality-gate/findings.md` +
   `phenotype_analysis.md`.

## Phases

### Phase 0 — Pre-launch hygiene (small env code change)

**Add `--exclude-days` flag to `select_days`** so future cohorts
can NEVER leak held-out into training, even when the data pool
extends past the leak boundary.

- `training_v2/discrete_ppo/train.py::select_days` — add kwarg
  `exclude_days: list[str] | None = None`. Filter before the
  `available[-n_days:]` slice:
  ```python
  if exclude_days:
      available = [d for d in available if d not in exclude_days]
  ```
- `training_v2/cohort/runner.py` — add CLI flag
  `--exclude-days YYYY-MM-DD [YYYY-MM-DD ...]` and thread to
  `select_days`. Default: empty list (byte-identical to pre-plan).
- Test in `tests/test_v2_select_days.py` (create if needed):
  - `test_exclude_days_removes_from_pool`
  - `test_exclude_days_empty_byte_identical`
  - `test_exclude_days_works_with_n_days_above_leak_boundary`
- Acceptance: tests pass; `--help` shows the new flag.

**Run the leak-boundary check** to confirm current state:

```python
from training_v2.discrete_ppo.train import _enumerate_day_files
from pathlib import Path
days = _enumerate_day_files(Path('data/processed'))
held_out = {'2026-04-28', '2026-04-29', '2026-04-30'}
for n in range(10, 30):
    leak = held_out & set(days[-n:])
    print(f"n_days={n}: {'LEAK' if leak else 'SAFE'} {sorted(leak)}")
```

Record the safe ceiling in `autonomous_run_log.md`. With
`--exclude-days` you can ignore the ceiling; without it, cap
`--n-days` at the safe value.

Commit: `feat(scalping-locked-fitness-and-age-obs): --exclude-days
for select_days; safe n_days unbounded`.

### Phase 1 — Locked-weighted composite_score

`training_v2/cohort/worker.py::train_one_agent` near
`model_store.update_composite_score(...)`:

```python
# Replace:
# score = float(eval_summary.total_reward)
# With:
score = (
    float(eval_summary.locked_pnl)
    + 0.25 * float(eval_summary.naked_pnl)
)
```

Optional CLI flag: `--composite-score-mode {locked_weighted,
total_reward}` defaulting to `total_reward` so byte-identical
behaviour is preserved for any legacy plan. Set to
`locked_weighted` via the launch script for this plan.

Tests in `tests/test_v2_cohort_worker.py`:
- `test_locked_weighted_score_formula` — `EvalSummary(locked=100,
  naked=200)` → 150.
- `test_locked_weighted_handles_negative_naked` — locked=100,
  naked=−100 → 75.
- `test_total_reward_mode_unchanged` — default flag → score equals
  total_reward (byte-identity guard).

Acceptance: tests pass; flag visible in `runner.py --help`.

Commit: `feat(scalping-locked-fitness-and-age-obs):
locked-weighted composite_score`.

### Phase 2 — `seconds_since_aggressive_placed` obs

`env/betfair_env.py`:

- Bump `SCALPING_POSITION_DIM` from 8 → 9.
- In `_get_position_vector` for each per-runner slot with an
  open pair (matched aggressive leg, unmatched passive partner):
  compute `(current_time_to_off - aggressive_placed_time_to_off)
  / race_duration`, clamp [0, 1]. Zero otherwise.
- Reuse the open-pair detection logic from Phase 2b of
  lay-quality-gate (already present at the leverage-obs computation).

Tests in `tests/test_betfair_env.py::TestAggLegAgeObs`:
- `test_obs_dim_increases_by_1_per_runner` (8 → 9).
- `test_zero_when_no_open_pair`.
- `test_increases_monotonically_within_race`.
- `test_normalised_to_race_duration`.
- `test_pre_plan_weights_fail_strict_load`.

Architecture-hash break is expected and correct (pre-plan
weights have one-narrower `lstm_input_proj.0.weight`). Mirrors
Phase 2b pattern.

Commit: `feat(scalping-locked-fitness-and-age-obs):
seconds_since_aggressive_placed obs`.

### Phase 3 — Re-probe with the larger data window (recommended)

The 7-new-days refresh changed the predictor's data. Re-run
`tools/probe_lay_outcome_distribution.py` on the held-out window
WITH the full new gate to verify the structural EV survives:

```
python -m tools.probe_lay_outcome_distribution \
    --days 2026-04-28 2026-04-29 2026-04-30 \
    --race-confidence-threshold 0.50 \
    --lay-threshold 0.20 --lay-price-max 20 \
    --device cuda
```

Expected: EV/£ ≈ +£0.10 (matches lay-quality-gate's Phase 1
result, since the predictor and held-out data haven't changed —
only the training pool did). If EV drops materially, surface
that as a stop and don't launch.

Commit probe output to `autonomous_run_log.md`.

### Phase 4 — Pre-flight smoke

Reuse `tools/smoke_lay_quality_gate.py` on 2026-05-04 (or the
new most-recent training day if 2026-05-04 isn't in the safe
last-N window any more). The obs-shape change in Phase 2 doesn't
break the smoke (smoke constructs a fresh uniform-random rollout;
no policy load).

Four §3 thresholds, ALL must PASS:
- `race_qualification_rate` ≥ 30%
- `legal_ratio` ≤ 80%
- `expected_per_£_lay_EV` ≥ −£0.05 (CONSISTENT version per
  smoke v2 methodology fix)
- `bets_matched` ≥ 50

ANY FAIL → STOP, write diagnostic. Reference
`plans/scalping-lay-quality-gate/smoke_2026-05-04_v2_PASS.txt`
for what the predecessor's PASS looked like.

### Phase 5 — Launch cohort + dual reeval watchers

**Concrete launch flags** — read the hand-off doc for the
day-window context:

```bash
TAG="_predictor_SCALPING_lockfit_$(date +%s)"
LOG="registry/${TAG}.log"
python -m training_v2.cohort.runner \
  --n-agents 12 --generations 8 \
  --days 13 \                          # ← max safe; bump up if you added --exclude-days in Phase 0
  --data-dir data/processed \
  --device cuda --seed 42 \
  --output-dir "registry/${TAG}" \
  --mutation-rate 0.2 \
  --strategy-mode arb \
  --predictor-bundle-manifests \
    ../betfair-predictors/production/race-outcome/manifest.json \
    ../betfair-predictors/production/race-outcome-ranker/manifest.json \
    ../betfair-predictors/production/direction-predictor/manifest.json \
  --use-race-outcome-predictor --use-direction-predictor --predictor-lean-obs \
  --predictor-p-win-back-threshold 0.20 \
  --predictor-p-win-lay-threshold 0.20 \
  --race-confidence-threshold 0.50 \
  --lay-price-max 20 \
  --composite-score-mode locked_weighted \
  --enable-gene stop_loss_pnl_threshold \
  --enable-gene open_cost \
  --enable-gene matured_arb_bonus_weight \
  --enable-gene naked_loss_scale \
  --enable-gene mature_prob_loss_weight \
  --enable-gene fill_prob_loss_weight \
  > "$LOG" 2>&1 &
disown
```

If you implemented `--exclude-days` in Phase 0, add
`--exclude-days 2026-04-28 2026-04-29 2026-04-30` and you can
safely raise `--n-days` arbitrarily high.

**`force_close_before_off_seconds = 0` during training.** Do NOT
set the override — preserves naked-variance signal per
`memory/project_force_close_train_vs_deploy.md`.

Arm TWO watchers (96 rows = 12 agents × 8 generations):

1. `/tmp/auto_reeval_lockfit_no_forceclose.sh` — fires reeval
   with `force_close = 0` (apples-to-apples vs predecessor).
2. `/tmp/auto_reeval_lockfit_forceclose120.sh` — fires reeval
   with `--reward-overrides force_close_before_off_seconds=120`.

**CRITICAL watcher path — predecessor had a path bug**: use bare
filename for `--output`, NOT cohort-dir-prefixed:

```bash
--output reeval_fc0_2026-04-28_30.jsonl      # CORRECT
# NOT: --output ${COHORT_DIR}/reeval_fc0_*.jsonl  ← double-prefix bug
```

Also: held-out window MUST be hard-coded:

```bash
--eval-days 2026-04-28 2026-04-29 2026-04-30
```

Heartbeat 1 h until both reevals complete (~12 h cohort + ~40
min for the two reevals).

### Phase 6 — Compare + verdict

Read BOTH reeval JSONL files. Compute the table:

```
                          force_close=0     force_close=120
mean per-day pnl          £X.X              £Y.Y
median per-day pnl        £X.X              £Y.Y
profitable / 5            N                 M
locked / naked split      ...               ...
```

Write `findings.md` with:

- Both verdicts side-by-side.
- vs lay-quality-gate baseline (fc=0 +£192.53, 5/5; fc=120
  +£25.74, 3/5).
- vs success bar (Modest > +£70 & 4/5; Strong > +£100 & 5/5).
- **Lever 1 validation:** final-gen `agg_back_pct` mean.
  Lay-quality-gate hit 0.11. If this plan's final-gen
  `agg_back_pct ≥ 0.4`, Lever 1 worked.
- **Lever 2 validation:** final-gen `n_closed` mean.
  Lay-quality-gate hit 4.5/3-day. If this plan's mean ≥ 6,
  Lever 2 worked.
- **Held-out vs in-sample locked-floor stability check** per
  the predecessor's pattern.
- Lessons learnt.
- Recommended next plan (likely
  `scalping-train-with-force-close` if both levers landed).

Commit. STOP.

## Stop conditions

1. Phase 6 findings.md committed → plan complete.
2. Phase 4 smoke FAILS any §3 threshold → STOP, write diagnostic.
3. Phase 3 probe shows EV/£ on held-out admitted set < −£0.05 →
   STOP, the data refresh may have broken the gate.
4. Cohort process crashed (Traceback in the log).
5. A hard_constraint is about to be violated.
6. Three consecutive iterations on the same sub-step without
   progress.

## Pacing

- 60-270 s during active code / tests
- 900-1800 s waiting on smoke / cohort partial
- 3600 s max heartbeat during cohort mid-flight
- 1800 s when waiting for the dual reeval to finish

Re-fire prompt verbatim each iteration:

`/loop @plans/scalping-locked-fitness-and-age-obs/session_prompts/00_autonomous_full_run.md`

## Default decisions (no operator)

| Question | Default |
|---|---|
| Plan name | `scalping-locked-fitness-and-age-obs` |
| Composite-score weight on naked | 0.25 (locked in hard_constraints) |
| `--n-days` | 13 (if `--exclude-days` skipped in Phase 0); else 20+ |
| Held-out window | 2026-04-28/29/30 (locked, hard-coded in watchers) |
| `race_confidence_threshold` | 0.50 (inherited) |
| `predictor_p_win_back_threshold` | 0.20 (inherited) |
| `predictor_p_win_lay_threshold` | 0.20 (inherited) |
| `lay_price_max` | 20 (inherited) |
| `force_close_before_off_seconds` | 0 train, 0 AND 120 reeval |
| Cohort size | 12 × 8 × `<n_days from above>` |
| Seed | 42 |
| Mutation rate | 0.2 |
| Enabled genes | same 6 as lay-quality-gate |
| Smoke day | 2026-05-04 (or latest training day in the safe window) |
| If a test fails | fix in same iter; one retry; stop on third |
| If launch fails | one fix retry; stop if still failing |

## What NOT to do

- Do NOT skip Phase 0's `--exclude-days` work AND set
  `--n-days ≥ 14` — that's data leakage.
- Do NOT tune the 0.25 composite-score weight mid-flight. Hard
  constraint #9 locks it.
- Do NOT enable `force_close` during training.
- Do NOT bundle Phase 1 and Phase 2 in one commit (variables
  must be separable).
- Do NOT skip the pre-flight smoke.
- Do NOT push to origin; commit locally only.
- Do NOT change the gate config — this plan is orthogonal to
  the gate (hard constraint #4).

## What you SHOULD do

- Commit at clean phase boundaries.
- Log per iteration to `autonomous_run_log.md`.
- Use specific `git add <file>`, never `.`.
- Use `run_in_background=True` for the cohort + watchers.
- Read `status.txt` rather than re-running ad-hoc python.

## Log entry template

```markdown
## YYYY-MM-DD HH:MM — Phase N, iteration M

**State entering iteration:** one sentence.
**Work done:** bullet list with file paths / test names.
**Tests run:** what was run, what passed/failed.
**Decisions made:** any defaults applied.
**Outstanding for this phase:** what's left.
**Next iteration's focus:** specific concrete next step.
```

## After plan exit

When Phase 6 commits `findings.md`:

1. Surface the headline result in one paragraph (both reeval
   numbers vs lay-quality-gate's fc=0 +£192.53 / fc=120 +£25.74
   baselines).
2. Surface the Lever 1 (`agg_back_pct`) and Lever 2 (`n_closed`)
   validation outcomes.
3. Recommend next plan. If Lever 2 worked, the next plan is
   likely `scalping-train-with-force-close` (the deferred Lever
   from the 2026-05-14 analysis). If Lever 2 didn't work, queue
   a diagnostic plan investigating why the obs feature didn't
   improve close discipline.
4. Stop scheduling.
