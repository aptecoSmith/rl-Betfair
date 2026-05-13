# Autonomous run log — scalping-race-confidence-gate

One entry per `/loop` iteration. Append-only. The loop's stop
conditions are evaluated against this file.

## 2026-05-13 07:53 — Session 01, iteration 1

**State entering iteration:** plan freshly scaffolded; no code
changes yet. Sibling direction-gate plan's smoke previously FAILED;
this plan substitutes a per-race confidence filter built on the
champion `p_win` cache (already populated by the predictor-
integration path).

**Work done (full Session 01 — single iteration):**
- `env/betfair_env.py`: added `race_confidence_threshold: float =
  0.0` kwarg, validation block (loud-fail when threshold > 0 but
  `use_race_outcome_predictor=False` or `predictor_bundle is None`,
  range guard `[0, 1]`), `_race_confidence_gate_active` flag, and
  `_race_is_confident_by_race: list[bool]` cache. Populated in
  `_precompute` immediately after `_race_p_win_by_race.append(...)`
  using `max(p_wins.values()) >= threshold`.
- `agents_v2/action_space.py::compute_mask`: added race-level
  short-circuit immediately after the `bm is None` / `race_idx` /
  `tick_idx` early-outs. Default-off via `getattr(..., False)` so
  the mask is byte-identical to pre-plan when the flag is unset.
  When active and the current race is non-confident, returns the
  NOOP-only mask before any per-slot iteration.
- `training_v2/cohort/worker.py`: threaded
  `race_confidence_threshold` through `_build_env_for_day`,
  `train_one_agent`, and all three `_build_env_for_day` call sites
  (sizing env, per-day train env rebuild, held-out eval env).
- `training_v2/cohort/runner.py`: added `--race-confidence-threshold`
  CLI flag and threaded through `run_cohort()` → `train_one_agent`
  call.
- `tools/reevaluate_cohort.py`: added matching CLI flag + threaded
  through both `_build_env_for_day` calls (sizing env + per-eval-day
  env).
- `tests/test_agents_v2_action_space.py`: appended
  `TestRaceConfidenceGate` class with 7 tests:
  - `test_gate_disabled_by_default`
  - `test_confident_race_passes_through_unchanged`
  - `test_non_confident_race_masks_all_opens_and_closes`
  - `test_byte_identical_when_disabled`
  - `test_raises_without_use_race_outcome_predictor`
  - `test_invalid_threshold_raises`
  - `test_composes_with_pwin_gate` (additive composition with the
    pwin gate — verifies non-confident races mask OPEN_LAY
    regardless of pwin, AND confident races still see the pwin gate
    apply on high-pwin runners).

**Tests run:**
- `tests/test_agents_v2_action_space.py` — 45 passed (38 pre-existing
  + 7 new).
- `tests/test_betfair_env.py` — 62 passed (no regressions).
- `python -m training_v2.cohort.runner --help` confirms
  `--race-confidence-threshold` registers with the expected help
  text.

**Decisions made:** applied defaults from `master_todo.md`
(`race_confidence_threshold=0.30` is the cohort flag value, but the
code-side default stays 0.0 = disabled per hard_constraints §1).

**Outstanding for this session:** none — Session 01 complete.

**Next iteration's focus:** Session 02 (pre-flight smoke). Write
`tools/smoke_race_confidence_gate.py` against 2026-05-04. Three
threshold checks per hard_constraints §3.

## 2026-05-13 07:59 — Session 02, iteration 1 — SMOKE FAIL — STOP

**State entering iteration:** Session 01 committed (`cccb8ad`).
Gate code complete + tested. Move to pre-flight smoke per
hard_constraints §3.

**Work done:**
- Wrote `tools/smoke_race_confidence_gate.py` modelled on
  `tools/smoke_direction_gate.py`. Adapts the population statistic
  to per-race `max(p_win) >= threshold` and the legal-ratio
  threshold to ≤ 80% (per `hard_constraints.md §3`).
- Ran the smoke on `2026-05-04` with the locked defaults
  (`--race-confidence-threshold 0.30`, pwin back/lay = 0.20/0.40,
  device cuda). Bundle: champion + ranker + direction production
  manifests.

**Smoke result on 2026-05-04 (118 races, 8925 ticks):**

```
POPULATION:
  total races ........................... 118
  races confident (max p_win >= 0.30) ... 118 (100.00%)
  races skipped ......................... 0

LEGAL ACTIONS (post-mask):
  baseline (pwin only):
    OPEN_BACK ........................... 13890
    OPEN_LAY  ........................... 71700
  with race-confidence gate:
    OPEN_BACK ........................... 13890 (delta: +0)
    OPEN_LAY  ........................... 71700 (delta: +0)
    legal-tick ratio (with/no race gate)  100.00%

POLICY ROLLOUT (uniform-random over legal, 3 races):
  attempted opens BACK / LAY ............ 1296 / 6176
  matched bets .......................... 178
  → bets/race ........................... 59.33
  → full-day estimate (118 races) ....... 7001

VERDICT vs hard_constraints §3:
  race_qualification_rate >= 30%        PASS (actual 100.00%)
  legal_ratio <= 80% (material work)    FAIL (actual 100.00%)
  bets_matched >= 50 (full day est.)    PASS (estimate 7001)

OVERALL: FAIL — STOP loop
```

**Why this fails:** Every single one of the 118 races on
2026-05-04 has at least one runner the champion predictor scored
`p_win ≥ 0.30`. The race-confidence gate at threshold 0.30
refuses zero actions — it is structurally inert on this day. Per
hard_constraints §3, `legal_ratio ≤ 80%` is required for the
gate to do material work; we're at 100%. The gate is a no-op
under these inputs.

**Decisions made:**
- Per hard_constraints §3, `legal_ratio > 80%` is a binary FAIL.
- Per hard_constraints §9, `race_confidence_threshold` is LOCKED
  at 0.30 — the autonomous loop cannot tune it. Any threshold
  change is a new plan.
- Per stop condition §1 (`session_prompts/00_autonomous_full_
  run.md`): "Session 02 pre-flight smoke FAILS any of three
  thresholds (hard_constraints §3). Stop, write diagnostic. Do
  NOT launch the 12h cohort."
- STOP the autonomous loop. Do NOT schedule the next wakeup.
  Surface the headline to the operator.

**Outstanding for this session:** none — gated correctly by the
smoke discipline. No 12h cohort wasted. Same outcome shape as the
direction-gate plan's smoke save.

**Next plan (operator-decision, NOT autonomous):** the data tells
us that on a representative day, racing favorites virtually always
have champion `p_win ≥ 0.30`. The threshold is too permissive to
function as a race-selection filter. Two branches the operator
might consider:

1. **Higher threshold** (e.g. 0.40 or 0.50). Per the smoke's
   POPULATION block, a tighter threshold would actually shrink
   the legal-action surface and let the gate function as intended.
   This requires a new plan because §9 locks 0.30 for this plan;
   re-opening "race-confidence-gate-v2" with a higher default
   would not violate the spirit of the locked threshold.
2. **Different per-race signal**. `segment_strong_flag` from the
   deterministic baseline already encodes "this race has a runner
   the predictor strongly favors" — it might capture race-level
   confidence more selectively than a raw `max(p_win)` cutoff.
   Per §9 this is also a new plan, not a mid-flight tweak.

Recommendation: a follow-on `scalping-race-confidence-gate-v2`
plan with `threshold=0.50` default, modelled on this plan but
using the smoke's per-day max-p_win distribution to set a value
that produces a meaningful split. The current cohort budget (12 ×
8 × 6) is unchanged — only the gate value differs.

## 2026-05-13 08:51 — operator override: pick a new threshold in-plan

**Operator instruction** (this conversation): "yes, use different
defaults. See if you can find a sensible figure that works".

The follow-on-plan path is dropped; the threshold is re-tuned
in-place. The autonomous loop resumes with the operator-authorised
new default.

## 2026-05-13 08:51 — Session 02b, iteration 1 (post-override probe)

**State entering iteration:** smoke FAILED at 0.30 (gate inert).
Operator authorised a new threshold; I need to pick one supported
by the per-race max-p_win distribution.

**Work done:**
- Wrote `tools/probe_race_confidence_distribution.py`. Reads
  `_race_p_win_by_race` after env construction across N days,
  prints quantiles + threshold candidate table.
- Ran the probe across 5 days
  (2026-05-01, 02, 04, 05, 06 — 434 races total).
- Distribution (max per-race champion p_win):

```
QUANTILES:
  p10 = 0.3865
  p25 = 0.4258
  p50 = 0.5338
  p75 = 0.6140
  p90 = 0.7069

THRESHOLD → races SKIPPED (qualification rate):
  0.30  →   0%  (100% qualify) ← FAIL'd smoke; gate inert
  0.40  →  16%  ( 84%)
  0.45  →  30%  ( 70%)
  0.47  →  33%  ( 68%) ← probe's suggestion
  0.50  →  40%  ( 60%) ← chosen default
  0.55  →  58%  ( 42%)
  0.60  →  72%  ( 28%) ← would FAIL §3 qualification floor
```

- Per-day medians confirm the distribution is stable across days:
  2026-05-01 = 0.59, 2026-05-02 = 0.52, 2026-05-04 = 0.53,
  2026-05-05 = 0.51, 2026-05-06 = 0.52.

**Decision: `race_confidence_threshold = 0.50`.** Sits at the
median of the observed distribution — natural "above-median
favorite" cut. Skips ~40% of races (legal_ratio ~60%, comfortably
under §3's 80% bar) without starving training (race
qualification rate ~60%, well above the 30% floor).

The probe also tested 0.47 (its automatic suggestion based on
landing in the 25-40% target band) but 0.50 is a cleaner
round-number choice and sits exactly at the population median.

**Plan documents updated (in-place):**
- `README.md` — threshold-choice section rewritten with the probe
  evidence + new default.
- `hard_constraints.md` §4 — `--race-confidence-threshold 0.30`
  → `0.50`.
- `master_todo.md` — Session 03 launch command threshold.
- `session_prompts/00_autonomous_full_run.md` — defaults table
  entry + "What NOT to do" lock.
- `session_prompts/03_launch_cohort.md` — replaced all 0.30
  references with 0.50 + revision note.

**Outstanding:** re-run smoke at the new threshold.

**Next iteration's focus:** smoke at 0.50, verify all three §3
PASS, then Session 03 launch.

## 2026-05-13 08:53 — Session 02b, iteration 2 — SMOKE PASS

**State entering iteration:** plan docs updated to threshold 0.50.

**Work done:**
- Ran `tools.smoke_race_confidence_gate --day 2026-05-04
  --race-confidence-threshold 0.50 --device cuda`.

**Smoke result on 2026-05-04 (118 races, 8925 ticks) at
threshold 0.50:**

```
POPULATION:
  total races ........................... 118
  races confident (max p_win >= 0.50) ... 65 (55.08%)
  races skipped ......................... 53

LEGAL ACTIONS (post-mask):
  baseline (pwin only):
    OPEN_BACK ........................... 13890
    OPEN_LAY  ........................... 71700
  with race-confidence gate:
    OPEN_BACK ........................... 8133  (delta: -5757)
    OPEN_LAY  ........................... 33287 (delta: -38413)
    legal-tick ratio (with/no race gate)  48.39%

POLICY ROLLOUT (uniform-random over legal, 3 races):
  attempted opens BACK / LAY ............ 750 / 3109
  matched bets .......................... 98
  → bets/race ........................... 32.67
  → full-day estimate (118 races) ....... 3854

VERDICT vs hard_constraints §3:
  race_qualification_rate >= 30%        PASS (actual 55.08%)
  legal_ratio <= 80% (material work)    PASS (actual 48.39%)
  bets_matched >= 50 (full day est.)    PASS (estimate 3854)

OVERALL: PASS — proceed to Session 03
```

All three §3 thresholds PASS with healthy margin:
- 55% qualification rate (1.8× the 30% floor)
- 48% legal-ratio (gate refusing 52% of action surface — substantial work)
- 3854 estimated full-day matched bets (76× the 50 floor)

**Decisions made:** proceed to Session 03 launch.

**Outstanding for this session:** none — Session 02b complete.

**Next iteration's focus:** Session 03 — launch the 12h cohort
with the locked-by-operator threshold 0.50. Verify Gen 1 starts
within 5 min, arm watcher, sleep into 1h heartbeat mode.

## 2026-05-13 09:32 — Session 03, iteration 1 — COHORT LAUNCHED

**State entering iteration:** smoke PASS at threshold 0.50;
operator confirmed launch via AskUserQuestion.

**Work done:**
- Pre-checks: GPU 22 GB free (RTX 3090), all three predictor
  manifests present, prior cohort code untouched.
- Launched cohort:
  - TAG = `_predictor_SCALPING_raceconf_1778661062`
  - PID 635
  - Log: `registry/_predictor_SCALPING_raceconf_1778661062.log`
  - Command: `training_v2.cohort.runner --n-agents 12
    --generations 8 --days 6 --seed 42 --mutation-rate 0.2
    --strategy-mode arb --predictor-p-win-back-threshold 0.20
    --predictor-p-win-lay-threshold 0.40
    --race-confidence-threshold 0.50` + the 6 safety genes +
    three production manifests + lean obs.
- Verified Gen 1 started (log shows "Cohort: 12 agents × 8
  generations on 3 training days (eval=2026-05-04…2026-05-06 (3
  days)); device=cuda" and agents 1-12 enumerated with their
  starting genes).
- Wrote watcher `/tmp/auto_reeval_raceconf.sh` (mirror of the
  pwin-gate watcher template; threshold 0.50 in the reeval call;
  tag substituted). Launched in background, PID 657. Output to
  `/tmp/auto_reeval_raceconf.log`; first lines confirm "waiting
  for 96 rows".
- Launched `tools.show_cohort_status --watch 60`, PID 668.
- Verified cohort artefacts present: `models.db`,
  `scoreboard.jsonl` (empty so far), `status.txt`
  ("Progress: 0/96 agents"), `weights/`, `bet_logs/`.

**Tests run:** none in this iteration — cohort itself is the
test surface. Pre-flight smoke covered correctness.

**Decisions made:** none new; mirroring predecessor pwin-gate
launch verbatim except for the new threshold.

**Outstanding for this session:** ~12h wall-time waiting for the
cohort to fill 96 scoreboard rows. Sleep into 1h heartbeat mode.

**Next iteration's focus:** heartbeat — read `status.txt`, log a
single-line entry "heartbeat N/96 rows", check for Traceback. If
N=96, jump to Session 04.
