# Autonomous run log — scalping-locked-fitness-and-age-obs

Per-iteration log of the autonomous run. Each iteration appends one
entry using the template at the bottom of
`session_prompts/00_autonomous_full_run.md`.

## 2026-05-14 14:30 — Phase 0, iteration 1

**State entering iteration:** Fresh start. Predecessor
`scalping-lay-quality-gate/findings.md` exists (pre-launch gate
satisfied). 36-day data pool: 2026-04-06..2026-05-13.

**Work done:**
- Ran leak-boundary check (`_enumerate_day_files`). Confirmed
  `--n-days ≤ 13` SAFE; `--n-days ≥ 14` leaks 2026-04-30+.
  Held-out 2026-04-28/29/30 sits at indices [-16..-14].
- Added `exclude_days: list[str] | None = None` kwarg to
  `training_v2/discrete_ppo/train.py::select_days`. Filters
  `available` BEFORE the last-N slice.
- Threaded `exclude_days` through
  `training_v2/cohort/runner.py::run_cohort` and added CLI
  flag `--exclude-days YYYY-MM-DD [YYYY-MM-DD ...]` with help
  text. Default `[]` = byte-identical.
- Created `tests/test_v2_select_days.py` with three tests
  per the driver:
  - `test_exclude_days_removes_from_pool`
  - `test_exclude_days_empty_byte_identical`
  - `test_exclude_days_works_with_n_days_above_leak_boundary`

**Tests run:**
- `pytest tests/test_v2_select_days.py -v` → **3/3 PASS**.
- `pytest tests/test_v2_multi_day_train.py -q` → **12/12 PASS**
  (no regression on existing select_days tests).
- `python -m training_v2.cohort.runner --help` shows the new
  flag (verified with `PYTHONIOENCODING=utf-8`; pre-existing
  `√` glyph in another help string breaks default cp1252 — not
  related to this change).

**Decisions made:**
- Use exclude_days at launch — Phase 5 will pass
  `--exclude-days 2026-04-28 2026-04-29 2026-04-30` so we can
  raise `--n-days` past 13 safely. With exclude active, the
  ceiling becomes the size of the post-filter pool minus 1
  (currently 35).

**Outstanding for this phase:** Commit.

**Next iteration's focus:** Commit Phase 0; start Phase 1
(locked-weighted composite_score in worker.py + tests).

## 2026-05-14 14:50 — Phase 1, iteration 2

**State entering iteration:** Phase 0 committed (`d6702b9`).
Working tree clean.

**Work done:**
- Added `composite_score_mode: str = "total_reward"` kwarg to
  `training_v2/cohort/runner.py::_composite_score`. New
  `locked_weighted` branch returns `locked_pnl + 0.25 *
  naked_pnl` (constant `LOCKED_WEIGHTED_NAKED_COEFFICIENT`).
- Threaded the mode through `run_cohort` →
  `train_one_agent_fn(...)` → `_agent_result_to_scoreboard_row(...)`
  → all four `_composite_score` call sites (sort key, generation
  log line, top_5 + best_model events, scoreboard row).
- Added validation: `composite_score_mode not in
  COMPOSITE_SCORE_MODES` → ValueError at the top of `run_cohort`.
- Added matching `composite_score_mode: str = "total_reward"`
  kwarg to `training_v2/cohort/worker.py::train_one_agent` so
  the registry's `models.composite_score` column also reflects
  the active formula (consistent with the scoreboard row).
- Added `--composite-score-mode {total_reward, locked_weighted}`
  CLI flag on the cohort runner.
- Scoreboard JSONL row gains `"composite_score_mode"` field so
  downstream tooling can disambiguate the active formula.
- Created `tests/test_v2_composite_score_mode.py` with four
  tests:
  - `test_locked_weighted_score_formula` — locked=100, naked=200
    → 150
  - `test_locked_weighted_handles_negative_naked` — locked=100,
    naked=-100 → 75
  - `test_total_reward_mode_unchanged` — default kwarg + explicit
    `total_reward` reproduce pre-plan formula; maturation bonus
    still applies
  - `test_locked_weighted_ignores_maturation_bonus_weight` —
    hard_constraints §9 invariant guard

**Tests run:**
- `pytest tests/test_v2_composite_score_mode.py -v` → **4/4 PASS**.
- `pytest tests/test_v2_cohort_runner.py tests/test_v2_cohort_worker.py
  -q` → 39/40 pass; 1 failure
  (`test_run_cohort_writes_scoreboard_and_registry`) is
  PRE-EXISTING on master (gene-dict key drift —
  `value_edge_threshold`, `each_way_*`, `predictor_feature_gain`,
  `value_kelly_fraction`). Confirmed via `git stash` rerun. NOT
  introduced by Phase 1.
- `--help` shows `--composite-score-mode` with both choices.

**Decisions made:**
- 0.25 weight is a module-level constant
  (`LOCKED_WEIGHTED_NAKED_COEFFICIENT`) for grep-ability — locked
  per hard_constraints §9.
- Batched runner path (`train_cluster_batched`) is NOT wired
  through; this plan uses the sequential path so the gap is
  inert. Documented as a known limitation if the operator
  experiments with `--batched`.
- Pre-existing scoreboard-schema test failure NOT fixed in this
  commit — out of scope. Could be queued as a separate cleanup
  task if it starts blocking work.

**Outstanding for this phase:** Commit.

**Next iteration's focus:** Commit Phase 1; start Phase 2
(`seconds_since_aggressive_placed` obs + 5 tests; bump
`SCALPING_POSITION_DIM` 8 → 9).

## 2026-05-14 15:15 — Phase 2, iteration 3

**State entering iteration:** Phases 0+1 committed. Working tree
clean. Env file untouched.

**Work done:**
- Bumped `SCALPING_POSITION_DIM` 8 → 9 in `env/betfair_env.py`.
  Bumped `OBS_SCHEMA_VERSION` 8 → 9 with a Version-9 note.
  Updated the SCALPING_POSITION block comment to list the new
  9th feature.
- Extended `_get_position_vector` docstring with the new feature
  description.
- Added agg-leg-age detection loop right after the existing
  Phase-2b naked-leg loop. Reuses `unfilled_pair_ids` for the
  "open pair" definition (matched aggressive + unmatched passive
  partner). Computes `placed_time_to_off` from
  `race.ticks[bet.tick_index].timestamp`. Per-runner aggregator
  takes the OLDEST aggressive leg's age (max), so multi-leg
  pairs surface the most-stale signal.
- Wrote the new column at `POSITION_DIM + 8` in the per-slot
  loop. Default 0.0 when no open pair on the runner — preserves
  Phase 2b's "byte-identical-when-no-position" guarantee
  (hard_constraints §2).
- Updated existing leverage test to assert
  `SCALPING_POSITION_DIM >= 8` instead of `== 8` (the Phase-2b
  invariant — additive features only).
- Added `tests/test_betfair_env.py::TestAggLegAgeObs` with the
  five tests required by the driver:
  - `test_obs_dim_increases_by_1_per_runner` (asserts == 9)
  - `test_zero_when_no_open_pair`
  - `test_increases_monotonically_within_race`
  - `test_normalised_to_race_duration`
  - `test_pre_plan_weights_fail_strict_load`

**Tests run:**
- `pytest tests/test_betfair_env.py::TestAggLegAgeObs
  tests/test_betfair_env.py::TestLeverageObsFeatures -v` →
  **12/12 PASS**.
- `pytest tests/test_betfair_env.py -q` → **74/74 PASS**
  (no regression on the broader env suite).

**Decisions made:**
- "Aggressive leg" detected via the SAME `unfilled_pair_ids`
  predicate as the naked-leg loop. Same definition of "open
  pair" — matched aggressive + unmatched passive partner.
  Inherits the Phase-2b detection guarantees.
- Per-runner aggregation = MAX(age) across legs. The policy
  needs to act on the most-stale pair, not an averaged value.
- Bets with `tick_index < 0` (never recorded) contribute 0 —
  they have no placement timestamp to anchor age. Also guards
  against the test-stub case where `tick_index` was forgotten.

**Outstanding for this phase:** Commit.

**Next iteration's focus:** Commit Phase 2; run Phase 3 — the
held-out lay-EV re-probe on the new data pool (verifies the
gate's structural EV survived the data refresh).

## 2026-05-14 15:00 — Phases 3+4, iteration 4

**State entering iteration:** Phases 0–2 committed
(`d6702b9` → `2c03503` → `045174d`). Working tree clean.

**Phase 3 — held-out lay-EV re-probe.** Ran on the new 36-day
data pool against the locked held-out window:

```
python -m tools.probe_lay_outcome_distribution \
    --days 2026-04-28 2026-04-29 2026-04-30 \
    --race-confidence-threshold 0.50 \
    --lay-threshold 0.20 --lay-price-max 20 \
    --device cuda
```

Verdict: **EV/£ = +£0.0984** on 572 gate-eligible (race, runner)
tuples. **PASS** (matches lay-quality-gate Phase 1's ~+£0.10 to
within noise — predictor + held-out data unchanged, only the
training pool grew).

Per-day:
| day | n | lay_winrate | EV/£ |
|---|---:|---:|---:|
| 2026-04-28 | 223 | 92.4% | +£0.366 |
| 2026-04-29 | 193 | 88.1% | -£0.086 |
| 2026-04-30 | 156 | 89.1% | -£0.055 |

Bucket sanity:
| price band | n | EV/£ |
|---|---:|---:|
| 2-5 | 59 | -£0.080 |
| 5-10 | 216 | +£0.102 |
| 10-20 | 297 | +£0.131 |

The cap at 20 keeps the +EV territory intact — same shape as
predecessor.

**Phase 4 — pre-flight smoke** (`tools/smoke_lay_quality_gate.py`)
on 2026-05-04. The agg-leg-age obs change (Phase 2) doesn't
break the smoke — the smoke uses uniform-random rollouts at
fresh policy, no policy-load involved.

```
LAY-QUALITY-GATE SMOKE — 2026-05-04
race-confident races ............... 65/118 (55.08%)
LAY legal-tick ratio (full/race-only) 63.30%
CONSISTENT admitted set EV ......... £+0.3185 (n=260)
matched bets/race .................. 32.67  → full day est 3854

VERDICT vs hard_constraints §3:
  race_qualification_rate >= 30%   PASS (55.08%)
  legal_ratio <= 80% material work PASS (63.30%)
  EV per £ admitted >= -£0.05      PASS (+£0.3185)
  bets_matched >= 50 estimate      PASS (3854)
OVERALL: PASS — proceed to Phase 5
```

All four thresholds pass with material headroom. Smoke logs at
`/c/tmp/lockfit_phase4_smoke.log`; probe at
`/c/tmp/lockfit_phase3_probe.log`.

**Outstanding:** Phase 5 launch + dual reeval watchers.

**Next iteration's focus:** Build launch script with
`--composite-score-mode locked_weighted`, `--exclude-days
2026-04-28 2026-04-29 2026-04-30`, `--n-days 20` (raised from
13 because Phase 0 added --exclude-days). Build and arm both
reeval watchers (fc=0 + fc=120) using the predecessor's
`auto_reeval_raceconf.sh` template (NOT the buggy
`auto_reeval_layq_*.sh`). Launch cohort with `run_in_background`.

## 2026-05-14 15:01 — Phase 5 launch, iteration 5

**State entering iteration:** Phases 0-4 done. Cohort dir
`registry/_predictor_SCALPING_lockfit_1778767165` created.
Predecessor's lay-quality-gate watchers carry the
double-prefix `--output` path bug — NOT copied directly.

**Work done:**
- Wrote `/c/tmp/auto_reeval_lockfit_no_forceclose.sh` and
  `/c/tmp/auto_reeval_lockfit_forceclose120.sh`. BOTH use BARE
  filename in `--output` to dodge the predecessor's path bug
  (`reevaluate_cohort.py` prepends `cohort_dir` itself —
  verified at `tools/reevaluate_cohort.py:208-211`).
- fc=120 watcher passes
  `--reward-overrides force_close_before_off_seconds=120` to
  `reevaluate_cohort.py` (memory:
  `project_force_close_train_vs_deploy.md`).
- Held-out window hard-coded to `2026-04-28 2026-04-29
  2026-04-30` in BOTH watchers.
- Launched the cohort with `run_in_background=true` (Bash task
  ID `by248wlkp`). Both watchers also armed in background
  (`bwq6aicit` fc=0, `bgu9u935r` fc=120).

**Launch flags (active):**
- `--n-agents 12 --generations 8` (96 rows total)
- `--days 20 --exclude-days 2026-04-28 2026-04-29 2026-04-30`
  → 10 train + 10 in-sample-eval = 2026-04-22..2026-05-03 train
  / 2026-05-04..2026-05-13 in-sample-eval (verified in cohort
  log)
- `--composite-score-mode locked_weighted` (this plan's Lever 1)
- `--seed 42 --mutation-rate 0.2 --strategy-mode arb`
- 6 enabled genes: stop_loss_pnl_threshold, open_cost,
  matured_arb_bonus_weight, naked_loss_scale,
  mature_prob_loss_weight, fill_prob_loss_weight (same set as
  predecessor)
- Gate: race_confidence_threshold=0.50, both pwin thresholds=
  0.20, lay_price_max=20 (inherited unchanged)
- `force_close_before_off_seconds=0` during training (no
  override)

**Initial cohort log evidence (15:00:23):**
- Predictor bundle loaded (champion / ranker / direction)
- `Cohort: 12 agents × 8 generations on 10 training days
  (eval=2026-05-04..2026-05-13 (10 days)); device=cuda`
- Held-out 2026-04-28/29/30 NOT in either training or
  in-sample-eval window (excluded as designed)
- Generation 1 starting; Agent 1 loading 2026-05-01

**Risk noted:** Bash tool's `run_in_background` carries up to a
600s timeout per the schema; unclear whether that times out the
synchronous wait or kills the OS process. Predecessor cohorts
were launched the same way and survived 12h, so the empirical
evidence is the timeout doesn't kill OS-detached processes. If
later iterations find the cohort/watchers dead, will relaunch
them via PowerShell Start-Process.

**Outstanding:** Wait for the cohort to fill scoreboard.jsonl
to 96 rows (~12h estimate); watchers will fire reevals when
they see TARGET_ROWS=96. Then write findings.md + verdict.

**Next iteration's focus:** Heartbeat the cohort progress.
First check at ~25 min to verify that all three background
processes survived past the 10-min Bash-timeout threshold.

