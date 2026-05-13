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
