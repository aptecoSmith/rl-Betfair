# Master TODO — session-level breakdown

Each session has a self-contained prompt under
[`session_prompts/`](session_prompts/). Sessions land in order;
later sessions presume earlier sessions are merged. Cross-session
dependencies noted in the table.

| # | Session | Depends on | Estimated time | Deliverable |
|---|---|---|---|---|
| 01 | Predictor loader | none | 1 session, 4–6 hr | `predictors/loader.py` + `predictors/segment_router.py` + tests |
| 02 | Observation wiring | 01 | 1 session, 3–4 hr | `OBS_SCHEMA_VERSION` 7 → 8, RUNNER_KEYS extension, byte-identical regression test |
| 03 | Strategy-mode switch | 02 | 1 session, 4–6 hr | `training.strategy_mode` config, env honours it, trainer tags registry, three smoke tests pass |
| 04 | Each-way action surface | 02, 03 | 1 session, 3–4 hr | `each_way` action signal added in `value_each_way` mode; `bm.place_back/place_lay` accept `each_way` kwarg; non-EW races mask the action space; settlement reuses `plans/ew-settlement/` path verbatim |
| 05 | Value-win smoke cohort | 03 | 1 session, ~4 hr | 1-day, 4-agent cohort runs end-to-end in `value_win` mode |
| 06 | Value-each-way smoke cohort | 03, 04 | 1 session, ~4 hr | Same in `value_each_way` mode |
| 07 | Three-way comparison | 03, 04, 05, 06 | 1 session, ~1 day | findings.md with verdict |

## Per-session checklist (template)

Each session prompt under `session_prompts/` follows the
established convention from `plans/scalping-active-management/`:

```
## Goal
1-2 sentences. What this session ships.

## Context to read
- relevant CLAUDE.md sections
- relevant prior plan files
- specific source files

## Deliverables
- code (file paths)
- tests (file paths + test names)
- doc updates (CLAUDE.md additions if any)

## Hard constraints
References hard_constraints.md by number.

## Success bar
Specific test names that must pass; specific behaviours that
must hold.

## Out of scope for this session
What NOT to do.
```

## Blockers and decisions to flag at session boundaries

### After Session 01

- Decision: where does `betfair-predictors`' inference code live
  on the rl-betfair worker's `sys.path`? (Sibling repo import,
  installable package, or vendored copy?) Recommend: sibling
  repo import via direct path append for now; revisit if
  `ai-betfair` needs the same. Operator approval before Session
  02 starts.

### After Session 02

- Decision: does the per-tick direction-predictor call run by
  default in arb mode, or behind its own `use_direction_predictor`
  flag? Recommend: behind its own flag, default off. The
  per-tick cost is non-zero; turn on per-cohort. Operator approval.

### After Session 03

- Decision: for `value_win` and `value_each_way` mode, does the
  default population already include the new genes
  (`value_edge_threshold` etc.) at default values, or are they
  injected per-mode-cohort? Recommend: always present in the
  CohortGenes dataclass at default 0; per-mode cohorts override.
  Aligns with §"v2 stack consumes aux-head loss weights"
  Path A.

### After Session 04

- Decision: do we ingest place markets going forward only, or
  backfill from the last N months of database backups?
  Backfill is needed for cohort training; depending on backup
  state, this may be a multi-session sub-effort. Operator
  decides backfill scope.

### After Session 05

- Decision: does PPO's settle-only reward shape work for
  `value_win`, or does the value mode need a per-step shaped
  contribution to densify gradient? Recommend: only revisit
  if Session 05's smoke produces zero gradient signal. Default
  to "no shaping" per hard_constraints.md §3.

### After Session 06

- Decision: same for `value_each_way`. EW commission is the
  same as win-market commission (see `plans/ew-settlement/`
  per-leg formula); no separate verification needed. Decide
  whether lay-side EW is worth adding in a follow-on
  experiment.

### After Session 07

- Decision: which mode's findings justify a follow-on plan?
  Operator reads findings.md and decides. Possible follow-ons:
  - Mode-mixing unified policy.
  - Live-inference port to `ai-betfair`.
  - v3 conversation if "fail with signal".
  - Aux-head retirement (now that predictors carry the signal).

## Cross-session invariants

See [`hard_constraints.md`](hard_constraints.md). The invariants
that govern session boundaries:

1. Flag-off byte-identical to pre-plan (regression test ALWAYS
   passes).
2. No env / matcher / bet_manager mechanics changes.
3. No new shaped reward terms in value modes.
4. Predictor weights are FROZEN.
5. Predictor `experiment_id` captured in every cohort row.
6. Three modes trained separately, evaluated jointly.

## Session pacing

Sessions are operator-driven, not autonomous. Each session
prompt is short enough that one focused work block (a Claude
Code session) can land it. Do NOT batch sessions — verify the
deliverable lands cleanly before opening the next session.

The smoke cohorts (Sessions 05, 06) and the three-way
comparison (Session 07) involve GPU training time. Per memory:
always GPU for cohorts. Session 07 runs concurrently for the
three modes (one cohort per mode in parallel) to minimise
wall-clock time.

## Risk register

| Risk | Mitigation |
|---|---|
| Predictor inference cost bottlenecks per-tick mode | Profile in Session 03 before committing to per-tick calls in cohorts |
| Each-way action surface mis-routes (`bet.is_each_way` not flipped) | Session 04 unit tests verify the bet-flag flow; smoke (Session 06) verifies `is_each_way == True` on ≥50% of bets |
| Non-EW races over-represented in training window | Curate the window at Session 06 cohort start; document in findings.md |
| Value-mode reward signal too sparse for PPO | Session 05 smoke; if signal is zero, revisit reward shape (with operator approval — this would be a hard_constraints §3 violation flagged for review) |
| Predictor `experiment_id` mismatch breaks re-eval tooling | Registry guard refuses on mismatch; tooling tested in Session 03 |
| `betfair-predictors` repo gets re-crowned mid-flight | Pin via experiment_id capture at Session 01; document in findings.md if it happens |

## After plan exit

Once Session 07 lands:

1. `plans/predictor-integration/` is closed (status changes
   from `planning` to `closed`).
2. Findings.md is the operator's reference for follow-on
   decisions.
3. `plans/INDEX.md` gets a new row.
4. CLAUDE.md gets new sections for OBS_SCHEMA v8, predictor
   bundle, strategy-mode switch.
5. Lessons learnt is propagated.
