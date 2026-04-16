# Scalping Active Management — Session 06 prompt

Work through session 06 fully (code + tests + progress.md
entry) before moving on. Commit after the session.

**Third and final UI-facing session.** Scope: add a MACE
(mean absolute calibration error) column to the Scoreboard's
Scalping tab. Sortable. Diagnostic only — does NOT feed the
composite score.

## Before you start — read these

- `plans/scalping-active-management/purpose.md` §"4. UI
  surfaces" — the scoreboard calibration column spec.
- `plans/scalping-active-management/hard_constraints.md` §14
  (READ THIS TWICE):
  > The "is this a good scalper?" headline metric is
  > unchanged. The scoreboard still ranks by L/N ratio >
  > composite. Adding a calibration column is diagnostic; it
  > does not feed the composite score. Rank on realised
  > outcome, not on self-reported confidence.
  This is the central constraint of Session 06. If you find
  yourself editing `Scoreboard.compute_score`'s weighting
  math, stop — that's out of scope.
- `plans/scalping-active-management/progress.md` —
  Session 05's entry, especially the `CalibrationStats`
  computation pattern. You'll be reusing the same
  `EvaluationBetRecord` → bucket → MACE logic, but moving
  it to a shared location so the scoreboard service can
  call it without importing from an API router.
- `plans/scalping-active-management/activation_playbook.md`
  §"Step B — Fill-prob weight sweep" — the activation
  playbook uses MACE as one of its pass/fail metrics. Once
  Session 06 lands, operators running the activation
  playbook can read MACE straight off the scoreboard
  instead of running the comparison script. Make sure the
  column presents the same number the playbook expects
  (same bucket thresholds, same `count < 20` exclusion
  rule).
- `CLAUDE.md` — "Verify frontend in browser before done"
  and "Full stack up for UI verify". Required.

## Before you touch anything — locate the code

```
grep -rn "class ModelScore\|class ScoreboardEntry\|compute_score" registry/ api/
grep -rn "ln_ratio\|composite_score\|L/N" registry/ api/ frontend/src
grep -rn "scoreboard\|Scoreboard" frontend/src/app
```

Identify:

1. `registry/scoreboard.py::ModelScore` (and
   `ScoreboardEntry` if that's a separate view model) — the
   dataclass holding per-model scoreboard values. You're
   adding one field: `mean_absolute_calibration_error:
   float | None`.
2. `registry/scoreboard.py::Scoreboard.compute_score` — the
   function that builds `ModelScore` rows. You're adding
   MACE computation (reusing Session 05's logic); you are
   NOT changing the composite-score formula.
3. Wherever Session 05's MACE computation lives — if it's
   inside an API router, lift it to a shared helper in
   `registry/calibration.py` (new module) so both the API
   and the scoreboard can call it. If it already lives in a
   shared location, reuse directly.
4. `frontend/src/app/scoreboard/*` — the scoreboard
   component + template. You'll add one column, visible
   only when the active tab is "Scalping".

## Session 06 — Scoreboard MACE column

### Context

Session 05 added a calibration card to the model-detail page
so operators can see whether an individual model is
calibrated. This session surfaces the summary number (MACE)
on the scoreboard itself so operators can **rank by
calibration diagnostically** — "which of my top-L/N models
also have the best-calibrated fill-prob head?" — without
opening each model's detail page.

Critical constraint: this is **diagnostic only**. The
scoreboard's primary ordering stays L/N ratio > composite
score. MACE is a sortable column, not a ranking input.

### What to do

1. **Shared MACE helper.**
   - If Session 05's MACE computation is still inline in an
     API router, extract it to a new module
     `registry/calibration.py` with a single function:
     ```python
     def compute_mace(
         bets: list[EvaluationBetRecord],
         *, min_bucket_size: int = 20,
     ) -> float | None:
         """Return MACE, or None if fewer than 2 buckets
         clear ``min_bucket_size``.

         Shared between the model-detail calibration card
         (session 05) and the scoreboard column (session 06)
         — keep the math in one place so the two surfaces
         can never drift.
         """
         ...
     ```
   - Export bucket thresholds and the min-bucket-size
     constant from the module so they're documented and
     greppable.
   - Update the Session-05 API code to import and call this
     helper instead of inlining the math. Tests from
     Session 05 should still pass unchanged.

2. **Extend `ModelScore`.**
   - Add `mean_absolute_calibration_error: float | None =
     None` to `registry/scoreboard.py::ModelScore`. Default
     `None` so legacy paths / directional models have a
     sensible value.
   - If `ScoreboardEntry` is a separate view-model class
     (API response shape), mirror the field there as an
     optional float.

3. **Populate in `compute_score`.**
   - Inside `Scoreboard.compute_score`, after the existing
     L/N and composite computations, for each scalping run:
     - Call `model_store.get_evaluation_bets(eval_run_id)`
       for the model's latest eval run.
     - Pass the result to `compute_mace`.
     - Assign to the new `ModelScore` field.
   - Directional runs (no scalping bets): leave the field
     at `None`.
   - Graceful degradation: if `get_evaluation_bets` raises
     (missing parquet file, etc.), set the field to `None`
     and carry on. Do NOT crash the scoreboard pipeline on
     a single model's bet-log issue.
   - **Do NOT modify `compute_score`'s composite math.**
     Add the MACE computation as a sibling call; don't fold
     it into any weighted sum. Grep your diff for anything
     that changes the existing `ln_ratio` or
     `composite_score` lines — if anything does, revert it.

4. **Frontend column.**
   - Add the column to the scoreboard table template,
     positioned after the existing calibration-relevant
     columns (or at the end if there's no natural home).
   - Column header: `"MACE"` with a tooltip explaining
     "Mean absolute calibration error on the latest eval
     run. Lower is better. Null → insufficient eval-day
     data."
   - Cell rendering:
     - `null` → dash (`"—"`) — matches the convention used
       by other optional columns.
     - Number → two decimals with traffic-light colour
       (green < 0.1, amber 0.1–0.2, red > 0.2) — same
       thresholds as the Session-05 card's MACE badge.
   - **Sorting.** The column is sortable. Null values sort
     last regardless of ascending/descending (same as other
     optional columns — grep for the existing null-handling
     pattern before implementing).
   - **Tab visibility.** The column is visible only when
     the active tab is "Scalping". Directional tab doesn't
     show it (MACE is always null there). Follow the
     existing tab-specific-column pattern if one exists; if
     not, add a minimal column-visibility map keyed by tab
     name.

### Tests

- **Server-side**:
  1. `compute_mace` returns `None` for an empty list.
  2. `compute_mace` returns `None` when every bucket has
     < 20 records.
  3. `compute_mace` returns `0.0` for perfect predictions
     (contrived records where bucket midpoint == observed
     rate exactly).
  4. `compute_mace` respects the `min_bucket_size` override
     — override to 5, sparse buckets now clear the
     threshold, different value returned.
  5. `ModelScore.mean_absolute_calibration_error` is
     populated for a scalping run with sufficient eval bets.
  6. `ModelScore.mean_absolute_calibration_error` is `None`
     for a directional run.
  7. `ModelScore.mean_absolute_calibration_error` is `None`
     when `get_evaluation_bets` raises
     `FileNotFoundError`.
  8. **Ranking invariant (critical).** Construct a
     synthetic set of models where MACE values would
     reorder them if MACE was feeding the composite.
     Assert that the scoreboard's returned ordering matches
     the pre-Session-06 ordering exactly. If this test
     fails, someone fed MACE into ranking — revert.

- **Frontend**:
  1. Column renders on the Scalping tab, absent on the
     directional tab.
  2. Null cell renders as a dash, not as `0.00`.
  3. Number cell renders with two decimals.
  4. Traffic-light classes apply at the right thresholds.
  5. Sorting ascending: nulls sort last.
  6. Sorting descending: nulls sort last (still).
  7. Header tooltip text matches spec.

### Browser verification

1. Full stack up.
2. Fixture: at least two scalping models with different
   MACE values (e.g. 0.05 and 0.25) and one directional
   model.
3. Load scoreboard, switch to Scalping tab. Verify the MACE
   column appears, green-on-0.05 and red-on-0.25, dash for
   any run that has no eval data.
4. Click the column header — sort ascending. Verify the
   0.05 model sits above the 0.25 model; null values at the
   bottom.
5. Click again — sort descending. Verify 0.25 above 0.05;
   null values still at the bottom (not at the top).
6. Switch to directional tab. Verify the MACE column is
   absent.
7. Confirm the top row (highest L/N ratio) is the SAME
   model on this tab as it was before Session 06 landed.
   If ranking changed, you touched composite math — revert.

### Exit criteria

- All new tests pass, including the ranking-invariant test.
- Browser verification complete (all 7 steps above).
- `progress.md` entry covers: the shared-helper refactor
  (Session 05's math moved to `registry/calibration.py`),
  the ranking-invariant test and its purpose, the tab-
  visibility scheme chosen, and the null-sort-last
  behaviour.
- `lessons_learnt.md` appended if anything surprising about
  sharing the helper between API + scoreboard emerges
  (circular imports, caching behaviour, etc.).
- Full suite green.
- Commit referencing `plans/scalping-active-management/` +
  session 06. Explicit line in the message: "composite
  ranking unchanged; MACE is diagnostic-only per
  hard_constraints §14."

---

## Cross-session rules for the whole plan

- Run `pytest tests/ -q` + `ng test --watch=false`. Both
  green.
- DO NOT modify `Scoreboard.compute_score`'s composite-score
  formula. The ranking-invariant test is the tripwire;
  treat it as sacred.
- Do NOT touch env / aux heads / action space.
- Commit after the session. No reward-scale change.
- Knock-on work for `ai-betfair`: the live-inference UI
  won't have a scoreboard (there's no population, just one
  trained model) so no cross-repo note needed unless a
  multi-model A/B surface gets built there later.

---

## After Session 06

Sessions 04 / 05 / 06 have now surfaced all the UI hooks the
plan committed to. Before running Session 07 (validation
training run + Gen 1 comparison), the `activation_playbook.md`
Steps A–E should run to completion — otherwise Session 07
measures the re-quote mechanic alone rather than the plan's
net effect. The scoreboard's new MACE column is one of the
artefacts the activation playbook uses to judge when the aux
heads are "trained enough to promote", so there's a natural
feedback loop: land Session 06, then run activation,
watching MACE trend down as weights come up.
