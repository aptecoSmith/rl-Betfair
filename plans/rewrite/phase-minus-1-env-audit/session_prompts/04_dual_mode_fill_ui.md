# Session prompt — surface fill-mode (volume / pragmatic) across the UI

Use this prompt to open a new session in a fresh context. The
prompt is self-contained — it briefs you on the question, the
context, the design, and the constraints. Do not require any
context from the session that scaffolded it.

---

## The question

**Surface the per-day passive-fill mode (volume-faithful vs
pragmatic) loudly enough across the rl-betfair UI that no
operator can lose track of which fill model a run was trained
against, and so cross-mode metric comparisons are visually
flagged rather than silently misleading.**

## Why

Phase −1 env audit finding F7
(`plans/rewrite/phase-minus-1-env-audit/audit_findings.md`)
discovered that historical training data has zero per-runner
cumulative volume. Session 03 (`03_dual_mode_fill_env.md`)
adapted the env to run in two modes — spec-faithful "volume"
when the data has per-runner volumes, "pragmatic" when it
doesn't (prorates market-level traded_volume by ladder size).
The mode is auto-detected per day at load.

Session 03 wires telemetry into the env:

- `info["fill_mode_active"]` per step
- `RaceRecord.fill_mode` per race
- `episodes.jsonl` `fill_mode` per episode
- Operator log line `mode=...` per episode summary

What's missing: **none of this surfaces in the UI**. The
operator sees scoreboards / training monitor / replay /
model-detail screens that look identical regardless of mode.
This is dangerous in two specific ways:

1. **Long-term forgetting.** Once the StreamRecorder1 fix
   (Session 02) lands and fresh data starts arriving, the
   curriculum will gradually shift from pragmatic-only to
   mixed to volume-only over weeks. Without a persistent UI
   indicator, operators looking at an old run six months from
   now won't remember it was pragmatic-mode.
2. **Silent cross-mode comparisons.** Cohort scoreboards
   sort by reward / pnl / fc_rate across runs. Two runs in
   different modes are NOT comparable on those metrics. The
   UI must visually flag mixed-mode comparisons or operators
   will draw wrong conclusions.

This session adds visible surfacing across five UI surfaces +
the API layer that feeds them.

## Five UI surfaces, in priority order

### 1. Training monitor — loud, always-visible badge

`frontend/src/app/training-monitor/`

A coloured badge near the top of the live training screen,
next to the existing run metadata (agent count, current
generation, etc.):

- 🟢 **Volume-faithful** — 100% of curriculum days run in
  volume mode.
- 🟡 **Pragmatic** — 100% of curriculum days run in
  pragmatic mode.
- 🔴 **Mixed: 12 volume / 18 pragmatic** — curriculum spans
  both. **This should be visually loud** because it indicates
  the trainer is learning two fill models simultaneously,
  which corrupts training signal. The trainer SHOULD pin to
  one mode per run; a mixed badge is a config error to flag,
  not a normal state.

Tooltip / info icon on hover:
> Pragmatic mode prorates market-level traded volume across
> runners by ladder size as a fallback for historical data
> that lacks per-runner cumulative volume. Spec-faithful
> volume gating activates automatically once StreamRecorder1
> per-runner totalMatched data flows in. See [link to
> audit_findings.md F7].

### 2. Scoreboard — cohort comparability gate

`frontend/src/app/scoreboard/`

Add a `Fill mode` column. Each run row shows its dominant
mode. **Cross-mode comparisons in the same view should be
visually flagged.** Implementation options (pick the cleanest):

- Group rows by mode with a divider; the user sees pragmatic
  runs and volume runs in separate sections by default.
- Or: chip-level styling — pragmatic chip is amber, volume is
  green. When sort order interleaves them, a visual warning
  banner appears: "⚠ This view mixes 3 pragmatic runs with 5
  volume runs. Reward / PnL / fc_rate metrics are NOT directly
  comparable across modes."
- Or: filter chips at the top of the scoreboard, with the
  default filter set to "current dominant mode". Operator can
  cross-mode if they explicitly opt in.

Choose the design that's clearest in your judgment — the
constraint is that mixed-mode comparison must NEVER be silent.

### 3. Race replay — per-race truth

`frontend/src/app/race-replay/`

Header strip shows `Fill mode: pragmatic` (or `volume`) next
to the date, venue, market_id. When the operator is debugging
"why did this passive fill at this tick?", they need to know
they're looking at synthetic-volume attribution if mode is
pragmatic, real-volume gating if volume.

Optional but recommended: any UI element rendering
`info["fill_mode_active"]` in the replay timeline (e.g. a
fill event's tooltip) should show the mode that produced it.

### 4. Model detail — training provenance

`frontend/src/app/model-detail/` and
`frontend/src/app/models/`

Each model's detail page lists training history. Add a
"Curriculum fill mode" row showing the breakdown of days the
model was trained on:

- "100% volume (180 days)" — clean.
- "100% pragmatic (180 days)" — fully on historical fallback.
- "Mixed: 12 volume + 168 pragmatic" — flag for review.

This is **load-bearing for live trading.** When a model is
lifted into the `ai-betfair` live-inference project, the very
first thing the eval should check is "does this model's
curriculum fill mode match what live trading uses (always
volume)?". If a model trained on pragmatic-only data is
deployed live, fill mechanics differ and the policy's
behaviour is partly out-of-distribution. The model-detail
page should make this visible at a glance.

### 5. Header / global status bar — persistent reminder

`frontend/src/app/header/`

A small persistent indicator (right-hand side of the header,
clickable to expand) showing **today's data state** (not
per-run, but the world the operator is in):

- "Today's data: volume-faithful (StreamRecorder1 fix active
  since 2026-MM-DD)"
- or "Today's data: pragmatic (StreamRecorder1 fix not yet
  deployed)"
- or "Today's data: mixed-period — 30+ volume-faithful days
  available, 200+ pragmatic-only days in archive"

This is the "don't lose sight" surface. Even on pages that
don't directly show fill mode, the header is a constant
reminder of the data world.

Click-through opens a small panel summarising:
- Last day with non-zero per-runner volume
- Total volume-faithful days available
- Total pragmatic-only days in archive
- Link to F7 audit findings + dual-mode design doc

## Backend / API additions

The frontend reads from these. Wire them in this order:

### `GET /api/days/{date}` — extend response

```json
{
  "date": "2026-04-11",
  "race_count": 8,
  "fill_mode": "pragmatic",
  ...existing fields
}
```

Read from `Day.fill_mode` (added in Session 03).

### `GET /api/runs/{id}` — extend response

```json
{
  "run_id": "abc-123",
  "agent_count": 64,
  "fill_mode_breakdown": {
    "volume": 12,
    "pragmatic": 18,
    "total_days": 30
  },
  "dominant_fill_mode": "pragmatic",
  "is_mixed_mode": true,
  ...existing fields
}
```

`dominant_fill_mode` = whichever mode has more days.
`is_mixed_mode` = true iff both counts > 0.

### `GET /api/runs/list` — extend each row

Each run summary in the list response carries
`dominant_fill_mode` + `is_mixed_mode`. Scoreboard reads from
this.

### `GET /api/models/{id}` — extend response

```json
{
  "model_id": "model-xyz",
  "curriculum_fill_mode_breakdown": {
    "volume": 12,
    "pragmatic": 168,
    "total_days": 180
  },
  ...existing fields
}
```

Aggregated across all training runs that produced the model.

### `GET /api/system/data-state` — new endpoint

```json
{
  "latest_volume_mode_date": "2026-05-15",
  "first_volume_mode_date": "2026-04-30",
  "total_volume_mode_days": 16,
  "total_pragmatic_mode_days": 207,
  "streamrecorder_fix_deployed": true,
  "streamrecorder_fix_deployed_date": "2026-04-30"
}
```

Header indicator reads from this. Can be cached (regenerated
once per day).

## CLAUDE.md note

Add a paragraph under a new section "Fill model: pragmatic vs
volume-faithful" in `CLAUDE.md`:

> The env supports two passive-fill modes per day, auto-
> detected at `Day` load. **Volume mode** uses per-runner
> cumulative `RunnerSnap.total_matched` deltas (the spec-
> faithful Betfair behaviour, available on days polled after
> the StreamRecorder1 F7 fix landed). **Pragmatic mode** uses
> market-level `traded_volume` deltas prorated across runners
> by ladder size (a fallback for historical data that has
> per-runner volumes pinned at zero). Both modes share the
> same crossability gate, junk-band filter, and threshold
> check; only the source of `delta` differs.
>
> **Load-bearing rule: training runs must pin to one mode.**
> Mixed-mode curriculums teach the agent two fill mechanics
> simultaneously and corrupt training signal. The trainer
> filters curriculum days to one mode by default; the UI
> flags any mixed-mode run loudly.
>
> Every reward / cohort metric is mode-specific. Cross-mode
> comparisons (`cohort-X vs cohort-Y` scoreboards spanning
> the boundary) are NOT directly comparable. See
> `plans/rewrite/phase-minus-1-env-audit/audit_findings.md`
> finding F7 for the design rationale, and the per-episode
> `episodes.jsonl` `fill_mode` field for per-row
> identification.

## What to do

### 1. Read the surrounding code (~30 min)

- Session 03's write-up
  (`plans/rewrite/phase-minus-1-env-audit/session_03_findings.md`)
  — confirm the telemetry fields are exactly what the API
  layer expects. If they renamed anything, adapt this
  session's API contract to match.
- `frontend/src/app/services/api.service.ts` — existing API
  client. Note the patterns for typed request/response.
- `frontend/src/app/scoreboard/` — current scoreboard layout
  + Material table use.
- `frontend/src/app/training-monitor/` — current run-status
  badges. The scalping_mode chip pattern is a good template.
- The Python API server (likely `api/main.py` or similar) —
  find existing endpoints for `/api/runs`, `/api/models`,
  `/api/days` and see how they read from the JSONL / model
  store.

### 2. Implement backend (~90 min)

a. **`Day.fill_mode` exposure.** Already on the `Day` object
   from Session 03; just include it in the
   `GET /api/days/{date}` JSON.
b. **Run breakdown.** API aggregates `episodes.jsonl`'s
   `fill_mode` field across all episodes in the run. Cache
   the aggregate on the `Run` object after run completion.
c. **Model breakdown.** Aggregate across all runs that
   produced the model. The model store (`registry/`)
   already tracks training-run lineage.
d. **`/api/system/data-state`.** Scan `data/processed/` for
   parquets, compute `Day.fill_mode` for each, return the
   summary. Cache for 1 hour.

### 3. Implement frontend (~3 hours)

In order of priority — ship surfaces 1 and 2 first; if you
run short on time, surfaces 3–5 can sequence next session.

a. **Training-monitor badge** (Surface 1).
b. **Scoreboard column + mixed-mode warning** (Surface 2).
c. **Race-replay header** (Surface 3).
d. **Model-detail provenance** (Surface 4).
e. **Global header indicator** (Surface 5).

Each surface needs:
- TypeScript types in `services/` for the API additions
- Component template + styling
- Unit tests where the existing pattern has them

Use existing chip / badge / banner components from Angular
Material. Don't invent new visual primitives unless the
existing ones are insufficient.

### 4. CLAUDE.md note (~10 min)

Add the section above. Place it near the existing "Order
matching" section (they're related — both describe load-
bearing env behaviour).

### 5. Tests (~60 min)

- **Backend pytests.** API endpoint tests for the new fields:
  - `test_days_endpoint_includes_fill_mode`
  - `test_runs_endpoint_includes_fill_mode_breakdown`
  - `test_models_endpoint_includes_curriculum_breakdown`
  - `test_system_data_state_endpoint_returns_summary`
- **Frontend Playwright e2e** in `frontend/e2e/`:
  - `dual-mode-badge.spec.ts` — load training-monitor with a
    pragmatic-mode run, assert the amber badge renders.
    Same for volume mode (green) and mixed (red + warning).
  - `scoreboard-cross-mode-warning.spec.ts` — populate
    scoreboard with mixed-mode runs, assert the warning
    banner appears on cross-mode sort.
  - `header-data-state.spec.ts` — assert the header
    indicator renders the data-state summary.

### 6. Manual QA (~30 min)

- Walk through all five surfaces with one of each scenario
  (volume-only run, pragmatic-only run, mixed run if you can
  fabricate one in test data).
- Confirm tooltips read correctly.
- Confirm the cross-mode warning is loud enough to actually
  catch the eye.
- Confirm clicking through from header indicator → data-state
  panel → audit_findings.md link works.

### 7. Write up (~15 min)

`plans/rewrite/phase-minus-1-env-audit/session_04_findings.md`:

- Implementation summary (file:line of each change).
- Screenshots (or descriptions if screenshots aren't trivial)
  of each of the five surfaces in each of three modes.
- Test results.
- Hand-off note: any `ai-betfair` cross-repo postbox drop if
  the live-inference project should also surface fill mode
  on its dry-run dashboard.

## Hard constraints

- **Don't ship without Session 03 landed.** Surface
  rendering depends on the telemetry fields. If the env
  isn't emitting `fill_mode_active`, the badge has nothing
  to read.
- **Don't quietly handle mixed-mode comparisons.** If a UI
  surface allows comparing runs across modes (sort,
  filter, group), it MUST visibly warn. A small grey label
  isn't enough — the operator must HAVE to notice.
- **Don't change reward / metric calculations.** This
  session is presentation-only. The cohort scoreboard
  shows the same numbers it shows today; what changes is
  whether the operator can tell which fill model produced
  them.
- **Don't widen scope to "redesign the scoreboard"** even
  if it's tempting. Add the column + warning, leave the
  rest.
- **Don't forget the CLAUDE.md note.** Future Claude
  sessions will work on this codebase; without the note,
  the next session has to re-discover the dual-mode design
  from scratch.
- **Style mismatched modes RED, not amber.** Mixed-mode is
  a real failure mode that corrupts training. Don't
  understate it.

## Out of scope

- Trainer-side curriculum filtering (the trainer enforcing
  one-mode-per-run). That's a separate session — the env
  supports both, the UI shows both, but the trainer's
  curriculum-selection logic is owned by `agents/`.
- Backfilling pragmatic-mode runs with retroactive
  fill-mode field if the JSONL was written before
  Session 03. Old rows can be `null` or `"unknown"` and
  that's fine — the UI should render them as a separate
  greyed badge ("Mode unknown — pre-dual-mode run").
- StreamRecorder1 changes — that's Session 02.
- ai-betfair UI changes. Drop a knock-on note in
  `ai-betfair/incoming/` if you want it to surface there
  too, but don't implement it.

## Useful pointers

- **F7 root analysis:** `plans/rewrite/phase-minus-1-env-audit/audit_findings.md`
  finding F7.
- **Dual-mode env design (Session 03):**
  `plans/rewrite/phase-minus-1-env-audit/session_prompts/03_dual_mode_fill_env.md`.
- **Existing scalping_mode UI surfacing pattern** (use as a
  template for the badge plumbing):
  `grep -rn "scalping_mode" frontend/src/app/`.
- **Existing API service:**
  `frontend/src/app/services/api.service.ts`.
- **CLAUDE.md** — add the new section near "Order
  matching".

## Estimate

Single session, 5–6 hours.

- 30 min: read surrounding code + Session 03 write-up.
- 90 min: backend (API additions).
- 3 hours: frontend (5 surfaces).
- 60 min: tests (pytest + Playwright).
- 30 min: manual QA.
- 15 min: write up + CLAUDE.md note.

If you're heading toward 7+ hours, ship surfaces 1, 2, and
the global header (3 of 5) plus the API + tests, and
sequence surfaces 3 + 4 (replay + model-detail) into a
follow-on session. Don't half-ship a surface — partial
rendering is worse than not rendering at all.
