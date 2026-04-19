# Session 03 prompt — Training-plan redraft + archive (operator-gated)

**IMPORTANT:** Per `../hard_constraints.md` §21, this
session is a **manual operator step**. The instructions
below are for the operator (or an agent invoked by the
operator with explicit permission to touch the registry).
Do NOT autonomously archive files or redraft plan states.
The launch itself (clicking Launch in the UI) is a
follow-on action AFTER this session's commit lands and
ends this plan's automated scope.

## PREREQUISITE — read first

- [`../purpose.md`](../purpose.md) — success criteria
  (C1–C5 in § What success looks like).
- [`../hard_constraints.md`](../hard_constraints.md). §21
  (operator-gated), §23 (don't bundle relaunch into this
  commit), §24 (archive artefacts).
- [`../master_todo.md`](../master_todo.md) — Session 03
  deliverables.
- [`../progress.md`](../progress.md) — Sessions 01 and 02
  validation entries (both must have landed before this
  session starts; if either failed, this session BLOCKS
  per hard_constraints §22).
- `plans/entropy-control-v2/session_prompts/03_registry_reset_and_relaunch.md`
  — the same archive-and-redraft pattern this session
  mirrors.
- `registry/training_plans/` — locate the `fill-prob-aux-
  probe` JSON (created 2026-04-19) as the template; most
  of its shape carries over.

## Pre-conditions (operator verifies before starting)

1. `plans/reward-densification/progress.md` has Session 01
   and Session 02 entries with commit hashes. Both
   committed to `master`.
2. `pytest tests/ -q` is green on the latest commit.
3. `cd frontend && ng test --watch=false` is green.
4. Admin UI + API both start cleanly. No training run in
   progress.

## Archive + reset

Same pattern as
`entropy-control-v2` Session 03. Archive path:

```
registry/archive_<ISO8601_compact_UTC>/
  models.db
  weights/
  training_plans/
```

### 1. Create the archive

```bash
ISODATE=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "registry/archive_$ISODATE"
mv registry/models.db "registry/archive_$ISODATE/"
mv registry/weights "registry/archive_$ISODATE/"
cp -r registry/training_plans "registry/archive_$ISODATE/"
mv logs/training/episodes.jsonl \
   "logs/training/episodes.pre-reward-densification-$ISODATE.jsonl"
```

Verify:

```bash
ls -la "registry/archive_$ISODATE/"
# Expect: models.db, weights/, training_plans/
```

### 2. Initialise a fresh registry

```python
from registry.model_store import ModelStore
store = ModelStore()
```

Recreate weights dir + truncate log:

```bash
mkdir -p registry/weights
: > logs/training/episodes.jsonl
```

Verify:

```python
import sqlite3
conn = sqlite3.connect("registry/models.db")
assert conn.execute("select count(*) from models").fetchone()[0] == 0
assert len(Path("registry/weights").iterdir()) == 0
assert Path("logs/training/episodes.jsonl").stat().st_size == 0
```

### 3. Redraft the probe training plan

New plan JSON based on
`registry/training_plans/8438534b-1993-4309-b76f-8ac7c242acd6.json`
(the `fill-prob-aux-probe`). Key differences:

- `name`: `reward-densification-probe`
- `plan_id`: fresh UUID
- `seed`: 421 (different from fill-prob-aux-probe's 137)
- `reward_overrides`:
  - Keep `fill_prob_loss_weight: 0.0` (aux head off — we
    want a clean mark-to-market signal).
  - Keep `risk_loss_weight: 0.0`.
  - Do NOT add `mark_to_market_weight` — it's picked up
    from the config.yaml default (0.05 per Session 02).
    Including it explicitly would be belt-and-braces but
    creates a duplicate source of truth. Leave it to the
    config.
- `notes`: describe the experiment purpose and success
  criteria cross-referencing `purpose.md`.
- All other fields: copy from `fill-prob-aux-probe`.

Create script:

```python
import json, uuid
from datetime import datetime, timezone
from pathlib import Path

template_path = Path(
    "registry/training_plans/"
    "8438534b-1993-4309-b76f-8ac7c242acd6.json"
)
template = json.loads(template_path.read_text())

new_plan_id = str(uuid.uuid4())
new_plan = dict(template)  # shallow copy
new_plan.update({
    "plan_id": new_plan_id,
    "name": "reward-densification-probe",
    "seed": 421,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "notes": (
        "First validation probe for the reward-densification "
        "plan. 9 agents x 1 generation with "
        "mark_to_market_weight=0.05 engaged via the config.yaml "
        "default. Tests whether per-step MTM shaping breaks the "
        "passive/bleeding bifurcation that emerged in A-baseline "
        "and fill-prob-aux-probe 2026-04-19. See "
        "plans/reward-densification/purpose.md 'What success "
        "looks like' for criteria."
    ),
    "status": "draft",
    "started_at": None,
    "completed_at": None,
    "current_generation": None,
    "current_session": 0,
    "outcomes": [],
})
new_path = Path(f"registry/training_plans/{new_plan_id}.json")
new_path.write_text(json.dumps(new_plan, indent=2))
```

Verify:

```bash
cat registry/training_plans/<new_plan_id>.json | \
  jq '.name, .seed, .population_size, .n_generations, .status'
# Expect: "reward-densification-probe"
#         421
#         9
#         1
#         "draft"
```

### 4. `plans/INDEX.md` update

Append a new row:

```markdown
| <row> | reward-densification | 2026-04-19 | Per-step mark-
to-market shaping replaces race-settle-only raw P&L as the
training signal. Closes out entropy-control-v2's diagnosis
that entropy isn't the lever; reward sparsity is. |
```

### 5. Pre-reset state snapshot (for the commit body)

Capture the pre-archive state so the commit body has it for
audit trail:

```bash
sqlite3 "registry/archive_$ISODATE/models.db" \
  "select count(*) from models"
wc -l "logs/training/episodes.pre-reward-densification-$ISODATE.jsonl"
ls "registry/archive_$ISODATE/weights/" | wc -l
```

Format for the commit body:

```
Archived (in registry/archive_<ISODATE>/):
- models.db: N models (from fill-prob-aux-probe, partial)
- weights/: M .pt files
- training_plans/: plan state snapshot

logs/training/episodes.jsonl ->
  logs/training/episodes.pre-reward-densification-<ISODATE>.jsonl
  (R rows)
```

### 6. Commit

```
chore(registry): archive pre-reward-densification registry + redraft probe plan

Archived (in registry/archive_<ISODATE>/):
- models.db (partial from fill-prob-aux-probe -- N agents)
- weights/ (M .pt files)
- training_plans/ (plan state snapshot; fill-prob-aux-probe
  JSON preserved for reference)
- logs/training/episodes.jsonl ->
  logs/training/episodes.pre-reward-densification-<ISODATE>.jsonl
  (R rows; the A-baseline gen-0 partial + the fill-prob-
  aux-probe's 7-agent-completed state)

Fresh registry:
- New models.db (0 models).
- weights/ recreated empty.
- episodes.jsonl truncated.

New training plan: ``reward-densification-probe``
- 9 agents (3 per arch -- same arch_mix as fill-prob-aux-
  probe for clean comparison).
- 1 generation, 3 epochs, auto_continue=false.
- No reward_overrides overriding mark_to_market_weight --
  the config.yaml default (0.05 per Session 02) applies.
- seed=421 (different from fill-prob-aux-probe's 137).
- status=draft, all runtime fields null.

Session 01 (commit <hash>) landed the mark-to-market
mechanism with weight=0 default (byte-identical migration).
Session 02 (commit <hash>) set the default to 0.05. This
session is the final commit before operator relaunch.

Per plans/reward-densification/hard_constraints.md s23, the
relaunch itself is NOT bundled into this commit -- it is a
follow-on operator action that writes back into
progress.md as a Validation entry.

registry/training_plans/, registry/weights/ and
registry/models.db are gitignored per the project's
standard pattern; only the plan docs (progress.md,
INDEX.md) show in the diff. The archive path is on disk
(registry/archive_<ISODATE>/) and is documented above for
post-mortem reference.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## Hand-off to operator (post-commit)

1. Start the admin UI + API.
2. Training-plans page: confirm
   `reward-densification-probe` is visible with status
   `draft`.
3. Select the plan, tick "Smoke test first".
4. Click Launch.
5. Smoke test runs. Expected: PASS. The tracking-error
   gate at target=150 doesn't depend on the reward shape,
   so the MTM shaping should not affect smoke outcomes.
   If smoke FAILS: capture the failure modal diagnostics
   in `lessons_learnt.md` — the MTM shaping may be
   interacting with the 3-episode entropy trajectory in
   an unexpected way.
6. On probe pass: full 9-agent population trains. Watch the
   learning-curves panel for the validation criteria
   (purpose.md "What success looks like"):
   - ≥ 50 % of agents remain active through ep15
     (bets > 0, arbs_naked > 0).
   - `policy_loss` stays O(1)+ through ep15 (not
     crashed to ~0 like A-baseline).
   - ≥ 1 agent reaches reward > −500 by ep15.
   - ≥ 1 agent's `arbs_closed / arbs_naked` ratio
     clears 10 %.
   - `raw + shaped ≈ total` holds episode-by-episode
     (verify by inspecting any episodes.jsonl row:
     `raw_pnl_reward + shaped_bonus ≈ total_reward`).
7. Capture findings in a new Validation entry on
   `progress.md`. Same shape as the Validation entries in
   `naked-clip-and-stability/progress.md` and
   `entropy-control-v2/progress.md`.

## Cross-session rules

- If Session 01 or Session 02 failed to land cleanly, this
  session BLOCKS.
- Do not make code or test changes in this session. It's
  archive + reset + plan JSON only. If a code bug is found
  during verification, roll back and fix it in Session 01
  or Session 02, then redo Session 03.

## After Session 03

Session 03 is the final session in this plan's automated
scope. After the operator's launch and the Validation
`progress.md` entry, this plan is complete. Next steps
depend on validation outcome — see `purpose.md §What
happens next` and `master_todo.md §Queued follow-ons`.
