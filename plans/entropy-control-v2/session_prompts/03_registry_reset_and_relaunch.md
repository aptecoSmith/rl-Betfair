# Session 03 prompt — Registry reset + activation-plan redraft (operator-gated)

**IMPORTANT:** Per `../hard_constraints.md` §23, this
session is a **manual operator step**. The instructions
below are for the operator (or an agent invoked by the
operator with explicit permission to touch the registry).
Do NOT autonomously archive files or reset plan states.
The launch itself (clicking Launch in the UI) is a
follow-on action AFTER this session's commit lands and
ends this plan's automated scope.

## PREREQUISITE — read first

- [`../purpose.md`](../purpose.md) — the plan's motivation
  and success criteria (C1–C5 in § What success looks
  like).
- [`../hard_constraints.md`](../hard_constraints.md). §11
  (checkpoint backward-compat on reset), §23 (operator-
  gated), §25 (don't bundle the relaunch into this commit),
  §26 (archive artefacts).
- [`../master_todo.md`](../master_todo.md) — Session 03
  deliverables.
- [`../progress.md`](../progress.md) — Sessions 01 and 02
  validation entries (both must have landed before this
  session starts; if either failed, this session BLOCKS
  per hard_constraints §24).
- `plans/naked-clip-and-stability/progress.md` — Session 05
  of the predecessor plan, which used this same archive-
  and-redraft pattern. Session 03 of this plan mirrors it.

## Pre-conditions (operator verifies before starting)

1. `plans/entropy-control-v2/progress.md` has Session 01
   and Session 02 entries with commit hashes. Both
   committed to `master`.
2. `pytest tests/ -q` is green on the latest commit.
3. Frontend `ng test --watch=false` is green.
4. The admin UI (frontend + API) starts cleanly (operator
   tests this before resetting, so a regression in startup
   is caught before destroying registry state).
5. No training run is currently executing. The worker is
   idle.

## Archive + reset

Same pattern as the `naked-clip-and-stability` Session 05
commit (`853a60c`). Archive path:

```
registry/archive_<ISO8601_compact_UTC>/
  models.db
  weights/
  training_plans/
```

With ISO8601 compact UTC being e.g. `20260419T184512Z`.

### 1. Stop any running workers / dev server

Per the activity log, make sure no training is in
progress. Check:

```
# On the main working tree
gh api /rest/... # if there's a training API
# OR
ps aux | grep training
```

If a worker is running, stop it cleanly (don't kill with
force — let it flush episode rows). Per the user memory
`feedback_taskkill_timing.md`, if taskkill doesn't complete
in seconds, pivot rather than escalating.

### 2. Create the archive

```bash
ISODATE=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "registry/archive_$ISODATE"
mv registry/models.db "registry/archive_$ISODATE/"
mv registry/weights "registry/archive_$ISODATE/"
cp -r registry/training_plans "registry/archive_$ISODATE/"  # copy, not mv — plans get redrafted, not archived away
mv logs/training/episodes.jsonl \
   "logs/training/episodes.pre-entropy-control-v2-$ISODATE.jsonl"
```

Verify the archive is complete:

```bash
ls -la "registry/archive_$ISODATE/"
# Expect: models.db, weights/ (with 64 .pt files from the
# A-baseline run), training_plans/ (with 4 JSON files).
```

### 3. Initialise a fresh registry

Per `hard_constraints` §17 of `naked-clip-and-stability`
(the same rule applies here): use the application's own
init path. In Python:

```python
from registry.model_store import ModelStore
store = ModelStore()  # creates registry/models.db from _SCHEMA_SQL
```

OR if a reset script exists:

```bash
python scripts/reset_registry.py  # if scripts/reset_registry.py exists
```

Recreate the empty weights dir:

```bash
mkdir -p registry/weights
```

Truncate the episodes log:

```bash
: > logs/training/episodes.jsonl
```

Verify:

```bash
sqlite3 registry/models.db "select count(*) from models"
# Expect: 0
sqlite3 registry/models.db "select name from sqlite_master where type='table'"
# Expect: 5 core tables — models, evaluation_runs, evaluation_days,
# genetic_events, exploration_runs
wc -c logs/training/episodes.jsonl
# Expect: 0
```

### 4. Redraft the activation plans

Four plans in `registry/training_plans/`:
- `activation-A-baseline.json`
- `activation-B-001.json`
- `activation-B-010.json`
- `activation-B-100.json`

Edit each to reset runtime/status fields while preserving
configuration:

```python
import json
from pathlib import Path
for plan in ("A-baseline", "B-001", "B-010", "B-100"):
    path = Path(f"registry/training_plans/activation-{plan}.json")
    doc = json.loads(path.read_text())
    doc["status"] = "draft"
    doc["started_at"] = None
    doc["completed_at"] = None
    doc["current_generation"] = None
    doc["current_session"] = 0
    doc["outcomes"] = []
    path.write_text(json.dumps(doc, indent=2))
```

Preserve byte-identical:
- `population_size`
- `n_generations`
- `arch_mix`
- `hp_ranges`
- `reward_overrides`
- `seed`
- `name`
- `notes`
- `plan_id`

Verify via the admin portal's plan-listing endpoint, or by
direct `cat registry/training_plans/*.json | jq '.status,
.started_at'` — expect all `"draft"` and all `null`.

### 5. Pre-reset state snapshot (for the commit body)

Capture the state that was archived so the commit body has
it for audit trail:

```
# Before this session's edits, from the archived
# training_plans JSON files:
| Plan | Pre-reset status | current_generation | outcomes |
|---|---|---|---|
| activation-A-baseline | ? | ? | ? |
| activation-B-001 | ? | ? | ? |
| activation-B-010 | ? | ? | ? |
| activation-B-100 | ? | ? | ? |
```

Fill in from the archived JSON (`registry/archive_<ISODATE>/
training_plans/*.json`).

### 6. INDEX update

`plans/INDEX.md` gets a new entry for this plan's
completion. Follow the existing row format:

```markdown
| <row number> | entropy-control-v2 | Target-entropy
controller + smoke-gate slope assertion | Complete |
<date> | Supersedes naked-clip-and-stability Session 03's
fixed-coefficient approach. |
```

### 7. Commit

```
chore(registry): archive pre-entropy-control-v2 registry + redraft activation plans

Archived (in registry/archive_<ISODATE>/):
- models.db (64 agents from the 2026-04-19
  activation-A-baseline run)
- weights/ (64 .pt files)
- training_plans/ (plan state snapshot)
- logs/training/episodes.jsonl →
  logs/training/episodes.pre-entropy-control-v2-<ISODATE>.jsonl
  (960 full-run rows + 6 smoke-test rows)

Fresh registry:
- New models.db (0 models; 5 core tables).
- weights/ recreated empty.
- episodes.jsonl truncated.

Activation plans redrafted (status='draft',
started_at/completed_at=None, current_generation=None,
current_session=0, outcomes=[]). Configuration
(population_size, n_generations, arch_mix, hp_ranges,
reward_overrides, seed, name, notes, plan_id) preserved
byte-identical.

Pre-reset state (for audit trail):
<table from Step 5>

Session 01 (commit <hash>) wired in the target-entropy
controller. Session 02 (commit <hash>) upgraded the
smoke-gate entropy assertion to a slope check. This
session is the final commit before operator relaunch.

Per hard_constraints §25, the relaunch itself is NOT
bundled into this commit — it's a follow-on operator
action that writes back into progress.md as a Validation
entry.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

Note: `registry/training_plans/` is gitignored (same
pattern as `naked-clip-and-stability` Session 05), so the
JSON edits don't appear in the commit diff. The commit
primarily documents the archive and plan-reset action in
its body.

## Hand-off to operator (post-commit)

1. Start the admin UI + API.
2. Open the training-launch page for
   `activation-A-baseline`.
3. Confirm "Smoke test first" is checked (default per
   `naked-clip-and-stability` Session 04).
4. Click Launch.
5. Watch the probe run in the learning-curves panel
   (badged smoke-test rows). Confirm the new slope-based
   assertion shows up in the assertion list.
6. On probe pass: full population launches. Watch for the
   validation criteria in `master_todo.md` "After Session
   03":
   - Pop-avg entropy converging toward 112 (the target) by
     ep 10, within ±20% by ep 15.
   - No ep-1 `policy_loss > 100` across the population.
   - `arbs_closed > 0` on at least one agent AND
     `arbs_closed / max(1, arbs_naked) > 0.3` sustained
     across the last 5 episodes for that agent.
   - At least one agent with a positive reward-trend
     slope across eps 8–15.
7. Capture findings in `progress.md` under a "Validation"
   entry — commit hash, outcome, scoreboard highlights, and
   either the green light for the B sweep or the
   diagnostics for a follow-up plan.

Per `hard_constraints` §23, the relaunch itself is NOT
bundled into Session 03's commit.

## Cross-session rules

- If Session 01 or Session 02 failed to land cleanly, this
  session BLOCKS. The commit history is the audit trail —
  if either of those sessions' commits is missing, the
  fresh registry would run with a broken controller or a
  broken gate.
- Do not make code or test changes in this session.
  Session 03 is archive + reset + docs only. If a code
  bug is found during verification, roll back and fix it
  in Session 01 or Session 02, then redo Session 03.

## After Session 03

Session 03 is the final session in this plan's automated
scope. After the operator's launch and the Validation
progress.md entry, this plan is complete. Next steps
depend on validation outcome:

- **If all C1–C5 pass:** proceed to the
  `scalping-active-management` activation playbook (the
  B-sweep).
- **If C1–C4 pass but C5 (reward trend) fails:** open the
  queued `reward-densification` plan.
- **If C1–C3 fail:** open a follow-up within this plan's
  territory (controller tune, target adjustment, extended
  smoke window). Capture in `lessons_learnt.md` with the
  diagnosis before opening.
