# Policy Startup Stability — Session 02 prompt

CLAUDE.md update + activation-plan reset. Prose + small JSON
edits. No code, no tests.

This session is also designed to be unattended-runnable: clear
deliverables, no decision points, mechanical operator-style
work.

## PREREQUISITE — read first

- [`../purpose.md`](../purpose.md) — context.
- [`../hard_constraints.md`](../hard_constraints.md) — §3 (no
  schema bumps), §12 (this is NOT a reward-scale change in the
  operator-visible sense), §19–§20 (don't bundle re-run, don't
  prune).
- [`01_advantage_normalisation.md`](01_advantage_normalisation.md)
  — Session 01's commit hash will be needed for the cross-link.
- `CLAUDE.md` — read the existing "Reward function: raw vs
  shaped" section to find the right insertion point for the
  normalisation note.
- `registry/training_plans/` — the four activation plan JSON
  files. Confirm names match
  `activation-A-baseline / B-001 / B-010 / B-100`.

## What to do

### 1. CLAUDE.md update

Find the "Reward function: raw vs shaped" section. After its
existing content (don't disturb the historical 2026-04-15 and
2026-04-18 paragraphs), add a new sub-section:

```markdown
## PPO update stability — advantage normalisation

The PPO update normalises the per-mini-batch advantage tensor
to mean=0, std=1 before the surrogate-loss calculation:

    adv_mean = advantages.mean()
    adv_std  = advantages.std() + 1e-8
    advantages = (advantages - adv_mean) / adv_std

This is load-bearing for any training run with large-magnitude
rewards (every scalping run — typical episode rewards land in
the ±£500 range, which without normalisation produces
gradients large enough to saturate action-head outputs on the
first PPO update). Without it, fresh-init agents reliably
exploded with `policy_loss` in the 10⁴–10¹⁴ range on episode 1
and lost the ability to ever fire `close_signal` /
`requote_signal` again — see
`plans/policy-startup-stability/` (commit `<session 01 hash>`).

Reward magnitudes in `episodes.jsonl` and `info["raw_pnl_reward"]`
are UNCHANGED by normalisation — the fix is purely on the
gradient pathway. Scoreboard rows from before the fix are
directly comparable to scoreboard rows after.
```

Replace `<session 01 hash>` with the actual short hash from
Session 01's commit (`git log --oneline -5` to find it).

If the existing CLAUDE.md text mentions "PPO" anywhere outside
this section in a way that contradicts the new note (unlikely
but worth a `grep -n "PPO\|advantage\|normalis" CLAUDE.md`),
reconcile.

### 2. Reset all four activation plans

Same pattern used in `scalping-naked-asymmetry` Session 02:

```python
import json, pathlib

PLAN_NAMES = (
    "activation-A-baseline",
    "activation-B-001",
    "activation-B-010",
    "activation-B-100",
)

for path in pathlib.Path("registry/training_plans").iterdir():
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("name") not in PLAN_NAMES:
        continue
    before = {
        k: data.get(k)
        for k in [
            "status", "started_at", "completed_at",
            "current_generation", "current_session", "outcomes",
        ]
    }
    data["status"] = "draft"
    data["started_at"] = None
    data["completed_at"] = None
    data["current_generation"] = None
    data["current_session"] = 0
    data["outcomes"] = []
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"reset {data['name']}: {before}")
```

Verify post-edit:

```python
import json, pathlib
for path in sorted(pathlib.Path("registry/training_plans").iterdir()):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("name", "").startswith("activation-"):
        continue
    print(f"  {data['name']:24s}  status={data['status']:8s}  outcomes={len(data['outcomes'])}")
```

All four should show `status=draft`, `outcomes=0`.

### 3. Append Session 02 entry to this plan's `progress.md`

Following the convention in
`scalping-equal-profit-sizing/progress.md`:

```markdown
## Session 02 — CLAUDE.md + activation reset (2026-04-XX)

**Landed.** Commit `<hash>`. Docs + JSON edits only — no code,
no tests.

- CLAUDE.md gains a new "PPO update stability — advantage
  normalisation" sub-section under "Reward function: raw vs
  shaped". Cross-links to Session 01's commit (`<hash>`).
- All four activation plans
  (`activation-A-baseline`, `activation-B-001/010/100`) reset
  to draft state via direct JSON edit. Verified each has
  `status=draft`, `outcomes=0`.
- The plan folder is now closed. Next operator action: launch
  activation-A-baseline and watch the learning-curves panel
  for the success criteria documented in `purpose.md`.
```

### 4. (Optional) Prune orphan models

Per `hard_constraints.md §20`, pruning is OPTIONAL and
ORTHOGONAL to this session — but if the operator hasn't done
it yet since the 2026-04-18 morning aborted run, the registry
still has ~16 non-garaged orphan models that'll show up in the
scoreboard alongside the 3 garaged keepers. Running
`scripts/prune_non_garaged.py` (with `--apply` once the
operator's seen the dry-run) tidies them.

If the operator hasn't requested this, DON'T do it without
asking — pruning is destructive (uses a backup but still). If
they HAVE requested it, the script's safety rails (backup,
dry-run default, refuse-when-no-garaged-models) handle the
mechanics; just run the dry-run first, paste its output to the
operator, and apply on confirmation.

## Exit criteria

- `git diff` shows: CLAUDE.md changed, four
  `registry/training_plans/*.json` changed, this plan's
  `progress.md` changed. Nothing else.
- All four activation plans verified as `status=draft`.
- `pytest tests/ -q` green (untouched, but worth confirming
  the JSON edits didn't somehow break anything that reads
  plan files).

## Acceptance

A reader opening CLAUDE.md cold finds the new "PPO update
stability" sub-section and can understand:
- What was added (per-batch advantage normalisation).
- Why it matters (large-magnitude rewards collapse action
  heads without it).
- Where to read more (this plan folder).

And the four activation plans are visibly ready to launch
without any further reset or cleanup.

## Commit

One commit, type `docs`. Cross-references Session 01's commit:

```
docs(scalping): note advantage normalisation in CLAUDE.md +
                reset activation plans

Adds a "PPO update stability — advantage normalisation"
sub-section to CLAUDE.md's "Reward function: raw vs shaped"
section. Documents the per-mini-batch normalisation that
landed in commit <session 01 hash> and explains why it's
load-bearing for any scalping training run.

Resets the four activation plans
(activation-A-baseline, activation-B-001/010/100) to draft
state — they were ready to launch but the previous run was
aborted on 2026-04-18 morning and their JSON state needed
clearing. All four now show status=draft, outcomes=[].

Plan folder plans/policy-startup-stability/ is now closed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## After Session 02

The operator launches activation-A-baseline. Watch the
learning-curves panel for the success criteria from
`purpose.md`:

- `policy_loss` series shows NO ep-1 spikes >100 across the
  population.
- `arbs_closed > 0` on multiple agents across multiple
  episodes.
- `best_fitness` per generation moves (not frozen).

Capture findings in `progress.md` under a "Validation" entry.

If the validation fails the same way (frozen fitness,
collapsed close), the next layer is action-head initialisation
— opens a fresh plan folder. (Don't fold init changes into
this plan; they're a different fix with their own
constraints.)
