# Session 06 prompt — Registry reset + training plan redraft (operator-gated)

**IMPORTANT:** Per `../hard_constraints.md` §35, this
session is a **manual operator step**. The instructions
below are for the operator (or an agent invoked by the
operator with explicit permission to touch the registry).
The launch itself is a follow-on action (Session 07)
AFTER this session's commit lands.

## PREREQUISITE — read first

- [`../purpose.md`](../purpose.md) — success criteria 1–5.
- [`../hard_constraints.md`](../hard_constraints.md). §35
  (operator-gated), §38 (don't bundle re-launch into
  this commit), §39 (archive gene-sweep artefacts).
- [`../master_todo.md`](../master_todo.md) — Session 06
  deliverables.
- [`../progress.md`](../progress.md) — Sessions 01–05
  entries. ALL must have landed before this session
  starts; if any failed, this session BLOCKS (§36, §37).
- `plans/reward-densification/session_prompts/03_validation_launch.md`
  — the same archive-and-redraft pattern this session
  mirrors.
- `registry/training_plans/` — the gene-sweep plan JSON
  (3c86f935) as the template for the probe.

## Pre-conditions (operator verifies before starting)

1. `plans/arb-curriculum/progress.md` has Session 01, 02,
   03, 04, 05 entries with commit hashes. All committed
   to `master`.
2. `pytest tests/ -q` is green on the latest commit
   (operator runs once here, NOT during active training).
3. No training run is active. `tasklist | grep python.exe`
   returns nothing related to the worker.
4. The 2026-04-19 `reward-densification-gene-sweep`
   Validation entry is written into
   `plans/reward-densification/progress.md`.

## Archive + reset

Same pattern as `reward-densification` Session 03 and
`entropy-control-v2` Session 03.

### 1. Create the archive

```bash
ISODATE=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "registry/archive_$ISODATE"
mv registry/models.db "registry/archive_$ISODATE/"
mv registry/weights "registry/archive_$ISODATE/"
cp -r registry/training_plans "registry/archive_$ISODATE/"
mv logs/training/episodes.jsonl \
   "logs/training/episodes.pre-arb-curriculum-$ISODATE.jsonl"
```

Verify:

```bash
ls -la "registry/archive_$ISODATE/"
# Expect: models.db, weights/, training_plans/
python -c "import sqlite3; c=sqlite3.connect(
  'registry/archive_$ISODATE/models.db');
  print('archived:', c.execute('select count(*) from models').fetchone())"
wc -l "logs/training/episodes.pre-arb-curriculum-$ISODATE.jsonl"
```

### 2. Initialise a fresh registry

```python
from registry.model_store import ModelStore
ModelStore()
```

```bash
mkdir -p registry/weights
: > logs/training/episodes.jsonl
```

Verify:

```python
import sqlite3
conn = sqlite3.connect("registry/models.db")
assert conn.execute("select count(*) from models").fetchone()[0] == 0
```

### 3. Redraft the probe training plan

New plan JSON based on the 2026-04-19
`reward-densification-gene-sweep` plan
(registry/training_plans/3c86f935-...). Key differences:

- `name: "arb-curriculum-probe"`
- fresh UUID `plan_id`
- `seed: 7919` (different from gene-sweep's 1337)
- `naked_loss_anneal: {"start_gen": 0, "end_gen": 2}` —
  naked-loss-scale anneals from gene value to 1.0 across
  the first two generations; full strength from gen 2.
- `reward_overrides`: empty (defaults apply; new reward
  knobs from Sessions 02/03 get their config.yaml values).
- `training_overrides`: set
  `curriculum_day_order: "density_desc"` — arb-rich days
  first, pairs with BC.
- `hp_ranges` — INCLUDE all gene-sweep ranges PLUS:
  - `matured_arb_bonus_weight` — range `[0.0, 2.0]`.
  - `naked_loss_scale` — range `[0.05, 1.0]`.
  - `bc_pretrain_steps` — range `[0, 1500]`.
  - `bc_learning_rate` — range `[1e-5, 1e-3]`.
  - `bc_target_entropy_warmup_eps` — range `[2, 15]`.
- Carry over gene-sweep's:
  - `mark_to_market_weight` — `[0.0, 0.5]`
  - `inactivity_penalty` — `[0.0, 1.0]`
  - `naked_penalty_weight` — `[0.0, 3.0]`
  - `early_lock_bonus_weight` — `[0.0, 3.0]`
  - `fill_prob_loss_weight` — `[0.0, 0.3]`
  - `risk_loss_weight` — `[0.0, 0.3]`
  - `entropy_coefficient` — `[0.005, 0.1]`
  - `entropy_floor` — `[0.1, 2.0]`
  - `reward_clip` — `[5.0, 100.0]`
  - `arb_spread_scale` — `[0.3, 3.0]`
  - `reward_spread_cost_weight` — `[0.0, 0.5]`
  - `architecture_name` / `market_type_filter` choices.

### 4. Plan-authoring script

```python
import json, uuid
from datetime import datetime, timezone
from pathlib import Path

template = json.loads(Path(
    "registry/training_plans/3c86f935-4621-4b46-8fa7-cce5f97a103a.json"
).read_text())

new_plan_id = str(uuid.uuid4())
new_plan = dict(template)

# Update genes
new_hp = dict(template["hp_ranges"])
new_hp.update({
    "matured_arb_bonus_weight": {"type": "float", "min": 0.0, "max": 2.0},
    "naked_loss_scale": {"type": "float", "min": 0.05, "max": 1.0},
    "bc_pretrain_steps": {"type": "int", "min": 0, "max": 1500},
    "bc_learning_rate": {"type": "float", "min": 1e-5, "max": 1e-3},
    "bc_target_entropy_warmup_eps": {"type": "int", "min": 2, "max": 15},
})

new_plan.update({
    "plan_id": new_plan_id,
    "name": "arb-curriculum-probe",
    "seed": 7919,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "hp_ranges": new_hp,
    "naked_loss_anneal": {"start_gen": 0, "end_gen": 2},
    "training_overrides": {
        "curriculum_day_order": "density_desc",
    },
    "reward_overrides": {},
    "notes": (
        "33-agent x 4-generation probe for arb-curriculum. "
        "Combines BC pretrain on the oracle cache + "
        "matured-arb shaped bonus + naked-loss annealing "
        "(anneal window 0..2 gens) + curriculum day "
        "ordering (arb-rich days first). "
        "Success criteria (purpose.md): "
        "(1) >=80%% active through ep15; "
        "(2) arbs_closed/arbs_naked > 15%% on >=3 agents; "
        "(3) policy_loss stays O(1)+ on >=50%% of agents; "
        "(4) >=3 agents reach total_reward > 0 by gen 3; "
        "(5) raw+shaped ~= total every episode. "
        "Pairs with the 2026-04-19 reward-densification-"
        "gene-sweep Validation entry as baseline comparator."
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
print("Wrote", new_path, "plan_id:", new_plan_id)
```

### 5. `plans/INDEX.md` update

Append a new row:

```markdown
| <row> | arb-curriculum | 2026-04-19 | BC pretrain on
arb oracle + matured-arb shaped bonus + naked-loss
annealing + density-ordered curriculum. Attacks the
2026-04-19 local-minimum diagnosis (policy finds
"arb less" before "arb better") at four points
simultaneously. |
```

### 6. Validate the plan loads

```python
from training.training_plan import PlanRegistry
pr = PlanRegistry("registry/training_plans")
probe = [p for p in pr.list() if p.name == "arb-curriculum-probe"][0]
assert probe.status == "draft"
assert probe.seed == 7919
assert probe.population_size == 33
assert probe.n_generations == 4
print("plan validates:", probe.plan_id)
```

### 7. Commit

```
chore(registry): archive pre-arb-curriculum registry + redraft probe plan

Archived (in registry/archive_<ISODATE>/):
- models.db from the reward-densification-gene-sweep
  (pop 33 x 4 gens if the sweep completed; partial
  otherwise).
- weights/ (N .pt files).
- training_plans/ snapshot.
- logs/training/episodes.jsonl ->
  logs/training/episodes.pre-arb-curriculum-<ISODATE>.jsonl
  (R rows).

Fresh registry:
- New models.db via ModelStore() (0 models).
- weights/ recreated empty.
- episodes.jsonl truncated.

New training plan: arb-curriculum-probe
- 33 agents (11 per arch, same arch_mix as gene-sweep).
- 4 generations, auto_continue=true.
- naked_loss_anneal: {start_gen: 0, end_gen: 2}.
- training_overrides.curriculum_day_order: density_desc.
- hp_ranges extends gene-sweep's with
  matured_arb_bonus_weight [0.0, 2.0], naked_loss_scale
  [0.05, 1.0], bc_pretrain_steps [0, 1500],
  bc_learning_rate [1e-5, 1e-3],
  bc_target_entropy_warmup_eps [2, 15].
- seed=7919 (different from gene-sweep's 1337).
- status=draft.

Sessions 01-05 (commits <hash>..<hash>) built the oracle,
matured-arb bonus, naked-loss annealing, BC pretrainer
and curriculum day ordering. This session is the final
commit before operator relaunch.

Per plans/arb-curriculum/hard_constraints.md s38, the
relaunch itself is NOT bundled here -- it is a follow-on
operator action (Session 07) writing back into
progress.md as a Validation entry.

registry/training_plans/, registry/weights/ and
registry/models.db are gitignored per project standard;
only plan docs (progress.md) appear in the diff. Archive
path is on disk at registry/archive_<ISODATE>/ and
documented here for post-mortem reference.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## Cross-session rules

- If Session 01, 02, 03, 04, OR 05 didn't land cleanly,
  this session BLOCKS.
- Do not make code or test changes in this session. It's
  archive + reset + plan JSON only. If a bug surfaces
  during plan validation, roll back and fix in the
  offending session, then redo Session 06.
- Before starting, make sure the oracle cache is fresh —
  run `python -m training.arb_oracle scan --dates ...`
  on the current training-date window so Session 05's
  density ordering has up-to-date data. (Header format
  should include a `created_at` timestamp for audit.)

## After Session 06

Hand off to operator for Session 07 (launch +
validation). That session's results write back into
`progress.md` as the Validation entry.
