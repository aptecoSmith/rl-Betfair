# Session 05 prompt — Registry reset + activation-plan redraft (operator-gated)

**This session is OPERATOR-gated.** An agent MAY walk through
the steps and prepare the archive + reset, but the final
"launch activation-A-baseline" is a human action. Do NOT
auto-launch from within this session.

## PREREQUISITE — read first

- [`../purpose.md`](../purpose.md) — success criteria and
  failure modes the post-reset run will be judged against.
- [`../hard_constraints.md`](../hard_constraints.md). §17
  (archive paths), §18 (activation-plan redraft pattern),
  §19 (no archive deletion), §26–§28 (operator-gated).
- [`../master_todo.md`](../master_todo.md) — Session 05
  deliverables and the "After Session 05" validation
  framework.
- `plans/scalping-naked-asymmetry/progress.md` — the
  previous registry-reset precedent (same JSON-edit pattern
  on activation plans).
- `plans/policy-startup-stability/progress.md` —
  second-precedent for the reset pattern, including the
  activation-plan redraft SQL/JSON.

## Locate the code + data

```
ls registry/models.db registry/weights/ logs/training/episodes.jsonl
ls scripts/ | grep -i "reset\|prune\|archive"
grep -rn "activation-A-baseline\|activation_a_baseline\|current_generation" plans/scalping-active-management/ 2>/dev/null
```

Confirm before archiving:
1. `registry/models.db` is NOT currently being written to
   (no training process active — the operator stopped the
   run per `purpose.md`).
2. Whether `scripts/reset_registry.py` (or equivalent)
   exists. If it does, use it. If not, the manual sequence
   below covers the same ground.
3. The activation-plan JSON structure — `plans/INDEX.md`
   or the listing endpoint should reveal the four plans
   (`activation-A-baseline`, `B-001`, `B-010`, `B-100`) and
   their current `status`.

## What to do

### 1. Archive current state

Use ISO-date-Z timestamp format matching the existing
`registry/archive_*` folders:

```
DATE=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "registry/archive_$DATE"
mv registry/models.db "registry/archive_$DATE/models.db"
mv registry/weights "registry/archive_$DATE/weights"

mv logs/training/episodes.jsonl \
   "logs/training/episodes.pre-naked-clip-stability-$DATE.jsonl"
```

Verify via `ls`:
- `registry/archive_<DATE>Z/models.db` exists.
- `registry/archive_<DATE>Z/weights/` exists and contains
  the prior weight files.
- `logs/training/episodes.pre-naked-clip-stability-*.jsonl`
  exists.

### 2. Initialise fresh registry

If `scripts/reset_registry.py` exists, run it. Otherwise
the fresh init is whatever path the application takes when
`models.db` doesn't exist — launching the admin UI or
hitting the models endpoint should create a fresh schema.
Confirm via:

```
sqlite3 registry/models.db "select count(*) from models;"
# Expected: 0 (or whatever 'fresh' means — matches what
# previous resets produced)
```

Do NOT hand-write schema. The application's own
initialisation path is the source of truth.

### 3. Recreate an empty episodes stream

```
: > logs/training/episodes.jsonl
```

(Or `Write` tool with empty content — same effect.)

### 4. Redraft activation plans

The pattern, applied to each of `activation-A-baseline`,
`activation-B-001`, `activation-B-010`, `activation-B-100`:

```
status              = 'draft'
started_at          = null
completed_at        = null
current_generation  = null
current_session     = 0
outcomes            = []
```

The redraft can be done via the listing endpoint's update
route, via direct JSON edit (plans are on-disk JSON under
`plans/scalping-active-management/...` — check for the
activation-plan storage location), or via a small Python
script patterned on previous resets. Whichever route,
verify afterward:

```
for plan in activation-A-baseline activation-B-001 \
            activation-B-010 activation-B-100; do
  # however the status is queried — listing API, JSON grep,
  # sqlite read — confirm status=draft, outcomes=[]
done
```

All four MUST report `status=draft` with empty `outcomes`.

### 5. Update plans/INDEX.md

Add the new plan to the index with its completion status.
Pattern from existing entries:

```
- [naked-clip-and-stability](naked-clip-and-stability/)
  — Naked-winner clip (95% in shaped) + close bonus +
  PPO stability (KL early-stop, ratio clamp) + entropy
  control (halved coef + reward centering) + smoke-test
  gate. Landed 2026-04-18.
```

### 6. Commit

One commit per `hard_constraints.md §26`. Template:

```
chore(registry): archive pre-naked-clip-stability state + reset

Archives the 2026-04-18 gen-2 training state prior to the
naked-clip-and-stability reset:

  registry/archive_<DATE>Z/models.db    (from registry/)
  registry/archive_<DATE>Z/weights/     (from registry/)
  logs/training/episodes.pre-naked-clip-stability-<DATE>.jsonl

Fresh registry initialised; activation plans redrafted
(activation-A-baseline, B-001/010/100 all status=draft).

Old weights learned a reward shape that this plan
invalidates (full naked cash in raw + shaped 95% winner
clip + close bonus + softener removed). Carrying them
forward would pollute the new training signal —
full reset is deliberate.

See plans/naked-clip-and-stability/.

Follow-up (operator): launch activation-A-baseline with
"Smoke test first" checked. Validation criteria in
plans/naked-clip-and-stability/master_todo.md "After
Session 05".

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

### 7. Hand-off to operator

Session 05 ends at the commit. The launch itself is the
operator's action:

1. Start the admin UI + API.
2. Open the training-launch page for
   `activation-A-baseline`.
3. Confirm "Smoke test first" is checked (default per
   Session 04).
4. Click Launch.
5. Watch the probe run in the learning-curves panel
   (badged smoke-test rows).
6. On probe pass: full population launches. Watch for
   the validation criteria in `master_todo.md` "After
   Session 05".
7. Capture findings in `progress.md` under a
   "Validation" entry — commit hash, outcome, next
   step.

## Cross-session rules

- No `rm`/`rm -rf` on archive folders (`hard_constraints.md
  §19`).
- No pruning of archived models.
- No code changes in this session — archival + reset +
  docs only.
- No auto-launch (`hard_constraints.md §26`). The operator
  clicks Launch.

## If something goes wrong

- **Fresh registry initialisation fails:** don't retry
  blindly. Roll back the archive move (move files back
  from `registry/archive_<DATE>Z/` to `registry/`) and
  investigate. Capture in `lessons_learnt.md`.
- **Activation-plan redraft fails on one of four plans:**
  fix the one that failed; do NOT leave the others in
  status=draft while that one is non-draft. Partial state
  is worse than full-old-state.
- **Smoke test fails on first launch:** the gate worked as
  designed. Do NOT click Launch Anyway. Capture the
  failing assertion in `lessons_learnt.md` and open a
  follow-up plan for whichever session's fix was
  insufficient. This is the point of the gate.
