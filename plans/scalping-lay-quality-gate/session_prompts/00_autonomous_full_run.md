# 00 — Autonomous full run — scalping-lay-quality-gate

You are driving this plan to completion: tighten the lay side of
the scalping gate based on the 2026-05-13 lay-EV probe, and make
the agent's close discipline learnable by giving it the
observations a human trader uses. **No operator interaction.**
Make every decision yourself using the documents + defaults
below.

## Deliverable

Held-out reeval verdict on 2026-04-28/29/30 (same window as
predecessors). Success bar:

- **Modest**: mean > +£39/day (beats race-confidence-gate's
  +£39.40) AND ≥ 3/5 profitable
- **Strong**: mean > +£70/day AND ≥ 4/5 profitable

Report BOTH `force_close=0` and `force_close=120` reeval numbers
per `memory/project_force_close_train_vs_deploy.md`.

## Read FIRST every iteration

1. `plans/scalping-lay-quality-gate/README.md` (you create this
   on iteration 1)
2. `plans/scalping-lay-quality-gate/hard_constraints.md`
3. `plans/scalping-lay-quality-gate/autonomous_run_log.md`
4. The relevant memory entries (loaded automatically):
   - `project_lay_ev_calibration_findings.md` — the lay-EV
     findings that motivate this plan
   - `project_force_close_train_vs_deploy.md` — asymmetric
     train/eval
   - `feedback_reliability_over_upside.md` — scalping not
     value-bet
   - `feedback_per_bet_logging.md` — log per-bet detail
5. Predecessor `plans/scalping-race-confidence-gate/findings.md`

## Phases

### Phase 0 — Scaffold the plan (iter 1 only)

Create the rest of `plans/scalping-lay-quality-gate/`:

- `README.md` — purpose, hypothesis, success bar (above),
  configuration locked (12×8×6, seed 42, mutation 0.2, same 6
  safety genes), inherits `race_confidence_threshold = 0.50`.
- `hard_constraints.md` — default-off byte-identical for every
  new knob; smoke gates the cohort; threshold-locking
  discipline; the new "+£EV per £stake on admitted lay set" §3
  threshold (≥ −£0.05).
- `master_todo.md` — session table mirroring the 6 phases below.
- `session_prompts/01_probe.md`, `02_obs_and_logging.md`,
  `03_gate_code.md`, `04_smoke.md`, `05_launch_cohort.md`,
  `06_compare_and_verdict.md` — one per phase, terse pointers
  back to this driver.
- `autonomous_run_log.md` — empty header, you fill it per
  iteration.

Commit Phase 0 scaffolding with message
`plan(scalping-lay-quality-gate): scaffold next stack-on plan`.

### Phase 1 — Re-run the lay-EV probe with fresh data

Run `tools/probe_lay_outcome_distribution.py` on 2026-04-28/29/30
with current pwin lay threshold 0.40 to confirm the calibration
profile hasn't shifted. Use the per-bucket EV table to set the
new defaults for Phase 3:

- `predictor_p_win_lay_threshold` (target: where EV/£ is positive
  across at least n=100 admitted runners — likely 0.20 but
  verify against the fresh probe).
- `lay_price_max` (target: max bucket where EV/£ ≥ −£0.05 —
  likely 20 but verify).

Commit the probe output to `autonomous_run_log.md`. If the
profile has materially shifted (calibration hole moved, or no
bucket is positive), STOP and surface — that's a new plan, not
a knob retune.

### Phase 2 — Foundational obs / logging changes (no env knobs)

Two pure-additive code changes, **tested independently**,
**committed separately**:

**2a. Per-bet logging during training-eval rollouts.** The eval
rollout that writes scoreboard.jsonl rows currently doesn't
populate `registry/<TAG>/bet_logs/`. Wire bet-log capture on
that path so every agent's eval-day bets land on disk. Schema
per bet (minimum):

```
agent_id, generation, bet_id, market_id, selection_id,
side ("back"/"lay"), price_matched, stake_matched, pair_id,
runner_champion_p_win, race_max_pwin, tick_time_to_off_s,
final_outcome (matured/agent_closed/force_closed/stop_closed/
naked), final_pnl
```

See `memory/feedback_per_bet_logging.md`. Tests in
`tests/test_cohort_worker.py` or equivalent. Acceptance: a 2
agents × 1 gen × 1 day smoke cohort writes `bet_logs/` with all
fields parseable and joinable to scoreboard.jsonl.

**2b. Per-runner leverage + close-cost observation features.**
Extend `SCALPING_POSITION_DIM` by 4 fields per runner:
- `naked_downside_if_runner_wins`
- `naked_downside_if_runner_loses`
- `cost_to_close_now`
- `worst_case_naked_pnl`

Zero when no open leg on that runner; computed from `bm.bets`
filtered to open legs (outcome == UNSETTLED, complete=False) +
current opposite-side LTP. Architecture-hash will break — that's
correct-by-default and matches the pattern set by
`fill_prob_in_actor` and `mature_prob_in_actor`.

Tests in `tests/test_betfair_env.py::TestLeverageObsFeatures`
mirroring the pattern of existing per-runner-feature tests.
Required test names per the spawn-task brief:
- `test_naked_downside_zero_when_no_open_leg`
- `test_naked_downside_back_leg_correct_arithmetic`
- `test_naked_downside_lay_leg_correct_arithmetic`
- `test_cost_to_close_reflects_opposite_side_book`
- `test_worst_case_naked_pnl_is_min_of_two_downsides`
- `test_obs_dim_increases_by_4_per_runner`
- `test_pre_plan_weights_fail_strict_load`

Both changes maintain a byte-identical contract when the new
fields are zero (no open positions) and when bet-log writing is
disabled.

### Phase 3 — Gate-tuning code

Add to `env/betfair_env.py`:

- New kwarg `lay_price_max: float = 0.0` (0 = disabled). When
  `> 0`, `compute_mask` refuses OPEN_LAY on runners whose current
  LTP exceeds the cap.
- Validation: `lay_price_max` in `[0, 1000]`; loud-fail if `> 0`
  but `use_race_outcome_predictor = False` (the cap composes with
  the pwin gate per hard_constraints).

Plumb the new flag through `worker.py`, `runner.py`, and
`tools/reevaluate_cohort.py` — mirror the
`predictor_p_win_back_threshold` plumbing verbatim.

Tests in `tests/test_agents_v2_action_space.py::
TestLayPriceCapGate` mirroring `TestRaceConfidenceGate`'s six
tests. Acceptance: all tests pass; new flag visible in
`python -m training_v2.cohort.runner --help`.

### Phase 4 — Pre-flight smoke (binary PASS / FAIL)

Extend `tools/smoke_race_confidence_gate.py` (or write a
sibling) to also report the lay-bucket impact of the new caps.
Four §3-style thresholds, ALL must PASS:

| Threshold | Bar |
|---|---|
| `race_qualification_rate` | ≥ 30 % |
| `legal_ratio` | ≤ 80 % |
| **`expected_per_£_lay_EV` on admitted set** | **≥ −£0.05** (NEW) |
| `bets_matched` (uniform-random rollout, full-day est.) | ≥ 50 |

ANY FAIL → STOP, write diagnostic. The new EV threshold is the
whole point of this plan; if the gate-tuned admitted set isn't
+EV (or close), the lay-quality-gate hypothesis is wrong.

### Phase 5 — Launch cohort

Mirror predecessor launch verbatim with:

- `--race-confidence-threshold 0.50` (inherited)
- `--predictor-p-win-lay-threshold <Phase 1 value>` (expect 0.20)
- `--lay-price-max <Phase 1 value>` (expect 20)
- All else identical (12 × 8 × 6, seed 42, same 6 safety genes).

**`force_close_before_off_seconds = 0` during training.** Do NOT
set the override — preserves naked-variance signal per
`memory/project_force_close_train_vs_deploy.md`.

Arm TWO watchers (both fire at 96 rows on the same scoreboard):

1. `/tmp/auto_reeval_layq_no_forceclose.sh` — fires reeval with
   `force_close = 0` (apples-to-apples vs predecessor).
2. `/tmp/auto_reeval_layq_forceclose120.sh` — fires reeval with
   `--reward-overrides force_close_before_off_seconds=120`
   (deployment-realistic).

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
- vs race-confidence-gate baseline (+£39.40/day, 3/5 profitable).
- vs success bar (Modest > +£39/day & 3/5; Strong > +£70/day &
  4/5).
- Lessons learnt — especially whether per-bet logs + obs features
  changed close-discipline behaviour (compare close_signal fire
  rate per agent per day vs predecessor's ~3-4 / day).
- Recommended next plan from the README's branches.

Commit. STOP.

## Stop conditions

1. Phase 6 findings.md committed → plan complete.
2. Phase 4 smoke FAILS any §3 threshold → STOP, write diagnostic.
3. Phase 1 probe shows no positive lay bucket exists → STOP.
4. Cohort process crashed (Traceback in the log).
5. A hard_constraint is about to be violated to make progress.
6. Three consecutive iterations on same sub-step without
   progress.

## Pacing

- 60–270 s during active code / tests
- 900–1800 s waiting on smoke / cohort partial
- 3600 s max heartbeat during cohort mid-flight
- 1800 s when waiting for the dual reeval to finish

Re-fire prompt verbatim each iteration:

`/loop @plans/scalping-lay-quality-gate/session_prompts/00_autonomous_full_run.md`

## Default decisions (no operator)

| Question | Default |
|---|---|
| Plan name | `scalping-lay-quality-gate` |
| `race_confidence_threshold` | 0.50 (inherited, locked) |
| `predictor_p_win_back_threshold` | 0.20 (unchanged) |
| `predictor_p_win_lay_threshold` | from Phase 1 probe (~0.20) |
| `lay_price_max` | from Phase 1 probe (~20) |
| `force_close_before_off_seconds` | 0 train, 0 AND 120 reeval |
| Cohort size | 12 × 8 × 6 |
| Seed | 42 |
| Mutation rate | 0.2 |
| Enabled genes | same 6 safety genes as predecessor |
| Eval window | 2026-04-28/29/30 (locked) |
| Smoke day | 2026-05-04 |
| If a test fails | fix in same iter; one retry; stop on third |
| If a launch fails | one fix retry; stop if still failing |

## What NOT to do

- Do NOT enable `force_close` during training (kills naked-
  variance signal — see force-close memory).
- Do NOT bundle Phase 2a + 2b with Phase 3 in one commit
  (variables must be separable for analysis).
- Do NOT skip Phase 1 — calibration MAY have shifted, and locking
  thresholds without re-probing is the same mistake the
  predecessor made with the 0.30 race-confidence threshold.
- Do NOT skip the pre-flight smoke.
- Do NOT push to origin; commit locally only.
- Do NOT add new genes or reward shaping in this plan — it's a
  gate + obs plan.
- Do NOT increase the close_signal bonus or any other reward
  shape; the obs features are the lever this plan uses.
- Do NOT scaffold this plan while the predecessor cohort is
  mid-flight (already complete — but if you find an active
  cohort process, STOP and surface).

## What you SHOULD do

- Commit at clean phase boundaries (one commit per phase, plus
  one each for 2a and 2b separately).
- Log per iteration to `autonomous_run_log.md`.
- Use specific `git add <file>`, never `.`.
- Use `run_in_background=True` for the cohort + watchers.
- Read `status.txt` rather than re-running ad-hoc python.
- Surface the probe's empirical thresholds before locking them.

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
   numbers, vs predecessor's +£39/day baseline).
2. Recommend next plan from `findings.md`'s branches.
3. Stop scheduling.
