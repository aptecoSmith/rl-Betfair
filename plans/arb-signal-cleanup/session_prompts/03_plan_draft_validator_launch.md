# Session 03 prompt — Plan draft + validator + launch (operator-gated)

## PREREQUISITE — read first

- [`../purpose.md`](../purpose.md) — the 3-cohort
  ablation design and success criteria.
- [`../hard_constraints.md`](../hard_constraints.md) —
  §24 (plan file), §25 (cohort split), §26 (fallback),
  §27 (validator), §29 (telemetry fields), §36–§41
  (cross-session + commit discipline).
- [`../master_todo.md`](../master_todo.md) — Session 03
  deliverables + exit criteria.
- [`../progress.md`](../progress.md) — Sessions 01 and
  02 entries. Before drafting the plan, verify the new
  config knobs behave as expected on a 1-agent × 1-day
  smoke run (spin up BetfairEnv with
  `force_close_before_off_seconds=30` and
  `shaped_penalty_warmup_eps=10`, step through a real
  data day, confirm the two new telemetry sets appear
  on the JSONL rows).
- `plans/arb-curriculum/progress.md` Session 06 — the
  registry-reset + plan-redraft template we're
  mirroring. Especially the structure of the plan JSON
  and the checklist of fields.
- `plans/arb-curriculum/session_prompts/07_validation_launch.md`
  — the operator launch sequence. This plan's launch
  follows the same pattern.
- `scripts/check_arb_curriculum_prereqs.py` and
  `scripts/validate_arb_curriculum.py` — the two
  scripts we're forking for this plan. Read them
  first to understand the contract.
- `training/training_plan.py` (and
  `registry/training_plans/277bbf49-…json` for a
  concrete example) — the plan data model. Critical:
  check whether cohort-level hyperparameter overrides
  are supported, or whether we fall back to three
  plan files.

## Why this is necessary

Sessions 01 and 02 landed the three mechanisms. This
session wires them into a probe plan, writes the
prereq + validator scripts, and hands off to the
operator for launch. Nothing runs until the operator
ticks "Smoke test first" in the admin UI and clicks
Launch.

The 3-cohort ablation (per `hard_constraints.md` §25)
is the load-bearing design choice. Without the
cohorts, a passing probe doesn't tell us which of the
three mechanisms did the work — and a failing probe
can't be tuned because we don't know which lever to
adjust.

## What to do

### 1. Cohort-split capability check (FIRST — 15 min)

Before writing any plan JSON, read
`training/training_plan.py` and confirm:

- Does `TrainingPlan.hp_ranges` support cohort-
  level overrides, or is it flat across the whole
  population?
- Does the GA mechanism support pinning specific
  genes for a subset of agents, or is every gene
  drawn from the same distribution for every agent?
- Does `arch_mix` support distinct gene ranges per
  architecture? (If so, we could co-opt that for
  cohorts at the cost of architecture uniformity
  per cohort — probably not acceptable, but worth
  knowing.)

**If cohort-level overrides exist:** use them. Write
ONE plan file with three sub-population blocks.
Document the approach in `progress.md`.

**If NOT:** fall back to THREE plan files
(`arb-signal-cleanup-probe-A.json`,
`-B.json`, `-C.json`). All three share the same
seed, same training dates, same test dates, same
n_generations, same architectures and arch_mix
(3 × 16 per file × 3 files = 48 total agents, vs
3 × 16 per cohort × 1 combined file = same totals
from the operator's perspective but three launches
instead of one).

**If the fallback is needed, note these operational
risks:**

- The three plans must run sequentially on the same
  registry snapshot. Operator must NOT launch B
  until A completes (or fails cleanly — with the
  Session 01 crash fix in place, this works now).
- Each plan's `seed` must differ (use 8101, 8102,
  8103). Same seed across plans would produce
  identical gene draws, which defeats the ablation
  purpose for genes that ARE drawn (not pinned).
- Cross-plan comparability: keep architectures,
  population size, n_generations, training dates,
  curriculum_day_order, reward-config defaults all
  identical across A / B / C. Only the three
  mechanism knobs + the `alpha_lr` gene treatment
  differ.

Document the chosen path in `progress.md` Session
03 entry.

### 2. Write the plan file(s)

Base the structure on
`registry/training_plans/277bbf49-…json`
(arb-curriculum-probe). Fields to CHANGE for this
plan:

```json
{
  "name": "arb-signal-cleanup-probe"
        OR "arb-signal-cleanup-probe-A" / "-B" / "-C",
  "plan_id": <new UUID per file>,
  "created_at": "2026-04-21T…",
  "population_size": 48 (single file) OR 16 (per file),
  "n_generations": 4,
  "generations_per_session": 1,
  "auto_continue": true,
  "seed": 8101 (single) OR 8101/8102/8103 (split),
  "status": "draft",
  "arch_mix": {
    "ppo_lstm_v1": 16,
    "ppo_time_lstm_v1": 16,
    "ppo_transformer_v1": 16
  },
  "hp_ranges": {
    // ALL existing arb-curriculum-probe genes unchanged:
    "arb_spread_scale", "bc_learning_rate",
    "bc_pretrain_steps", "bc_target_entropy_warmup_eps",
    "early_lock_bonus_weight", "entropy_coefficient",
    "entropy_floor", "fill_prob_loss_weight",
    "inactivity_penalty", "mark_to_market_weight",
    "matured_arb_bonus_weight", "naked_loss_scale",
    "naked_penalty_weight", "reward_clip",
    "reward_spread_cost_weight", "risk_loss_weight",
    // NEW:
    "alpha_lr": {"min": 1e-2, "max": 1e-1, "type": "float"},
    // PINNED (collapsed range) — see hard_constraints.md
    // s24 + lessons_learnt.md 2026-04-21 transformer
    // audit. Default 32 covers only ~13% of a typical
    // race; 128 covers ~54%; 256 (added by Session 01
    // per s14a-s14d) covers the full race for the
    // median case. The LSTM variants have full-day
    // memory and don't need this — pinning only
    // affects transformer agents. If the schema
    // rejects a zero-width range, use a narrow range
    // {min: 256, max: 256} or the single-value syntax
    // the plan data model supports.
    //
    // PREREQ: Session 01 must land before this plan
    // can validate — the 256 value is not accepted by
    // the codebase until the range-widening commit.
    "transformer_ctx_ticks": {"min": 256, "max": 256, "type": "int"}
  },
  "naked_loss_anneal": {"start_gen": 0, "end_gen": 2},
  "reward_overrides": null,
  // Session 02's knob is plan-level, not per-gene;
  // set via training.shaped_penalty_warmup_eps in the
  // plan's config-override block if the data model
  // supports it, else via config.yaml at launch time
  // (operator note).
}
```

Cohort-specific settings:

- **Cohort A:** `alpha_lr` drawn from gene range;
  `training.shaped_penalty_warmup_eps: 10`;
  `constraints.force_close_before_off_seconds: 30`.
- **Cohort B:** `alpha_lr` drawn from gene range;
  `training.shaped_penalty_warmup_eps: 0`;
  `constraints.force_close_before_off_seconds: 0`.
- **Cohort C:** `alpha_lr` pinned to `1e-2` (not
  drawn — the gene range on this cohort collapses to
  a point); `training.shaped_penalty_warmup_eps: 10`;
  `constraints.force_close_before_off_seconds: 30`.

If using a single plan with cohort blocks: confirm
the plan data model can express "pin gene X for
sub-population Y". If NOT, cohort C must be a
separate plan file (or the operator accepts that
cohort C's `alpha_lr` is drawn but at a pinned range
`[1e-2, 1e-2]`; check whether the GA tolerates a
zero-width range).

If using three plan files: each file's config-
override block simply sets the cohort's three knobs.

### 3. Cohort telemetry

Each episode's JSONL row needs a `cohort` field ("A"
/ "B" / "C"). If the plan data model carries a
"tag" or "label" per plan/sub-population, plumb that
through to the trainer and onto the JSONL row.

If no such plumbing exists: add a `plan_cohort`
field on `TrainingPlan` (default `None`) that's
serialised into the plan JSON and read by the
trainer at rollout time. Default `None` → JSONL
`cohort` is `"ungrouped"`.

This is a small data-model extension — if Session
03 ends up touching multiple files for it, that's
expected and in-scope per the telemetry contract
(§29).

### 4. `scripts/check_arb_signal_cleanup_prereqs.py`

Fork `scripts/check_arb_curriculum_prereqs.py`.
Change the checks to:

1. `config.yaml
   constraints.force_close_before_off_seconds == 0` at
   the config level (cohort overrides flip it on at
   plan level; config is the floor).
2. `config.yaml training.shaped_penalty_warmup_eps
   == 0` at config level (same reasoning).
3. Plan file(s) exist with `status == "draft"`;
   validate population sizes, n_generations,
   `alpha_lr` gene range, `naked_loss_anneal` window.
4. `data/oracle_cache/` has ≥ 1 date with
   `oracle_samples.npz` (BC prereq, inherited from
   arb-curriculum).
5. `registry/training_plans/277bbf49-…json` has
   `status == "failed"` — sanity check; the prior
   probe is abandoned, not a blocker.

Exit 0 on all pass.

### 5. `scripts/validate_arb_signal_cleanup.py`

Fork `scripts/validate_arb_curriculum.py`. Keep the
5 criteria identical (§27 — same semantics). Add:

- **Per-cohort pass/fail matrix** for C1 and C4. The
  other three criteria are population-wide only
  (C2 and C3 are existence checks; C5 is invariant).
- **Force-close diagnostic table** (per-cohort):
  mean `arbs_force_closed / race`, mean
  `arbs_naked / race`, mean
  `scalping_force_closed_pnl / race`.
- **BC diagnostics table** (per-cohort) — same as
  arb-curriculum validator, grouped by cohort.

Output format: plain text to stdout, similar to
arb-curriculum's output. Exit codes unchanged: 0 =
all pass, 1 = some fail, 2 = no data.

### 6. Sanity-check the validator against the prior probe

Run
`python scripts/validate_arb_signal_cleanup.py --log
logs/training/episodes.pre-arb-curriculum-<ISO>.jsonl`
(whatever archive name the arb-curriculum probe's
logs got). The validator should produce the SAME 3/5
result recorded in
`plans/arb-curriculum/progress.md` Validation entry.

Any divergence means the criteria semantics drifted;
fix before accepting the validator as ready.

The `cohort` field will be "ungrouped" for the prior
probe's rows; the per-cohort tables in the new
validator should degrade gracefully to "N/A" or "no
cohort data" instead of crashing.

### 7. `plans/INDEX.md`

Append a new row at the bottom:

```markdown
| 20 | [arb-signal-cleanup](arb-signal-cleanup/) | 2026-04-21 | **(latest)** Force-close at T−30s + per-agent `alpha_lr` gene + shaped-penalty warmup — attacks the 2026-04-21 arb-curriculum-probe Validation's 3/5 result (C1 entropy collapse + C4 reward-shape penalty) with a three-cohort ablation |
```

Update the prior row's "(latest)" marker in the
arb-curriculum row to plain bold-text, per the
`INDEX.md` convention.

### 8. Progress entry

Append a Session 03 entry to
[`../progress.md`](../progress.md). Fields:

- **What landed:** plan file(s), prereq checker,
  validator, INDEX entry, any data-model extensions.
- **Not changed:** no env / trainer / reward changes
  in this session.
- **Gotchas:** which cohort-split path (single plan
  vs three files), any data-model extensions needed
  for cohort telemetry.
- **Test suite:** no new pytest tests; sanity check
  consisted of validator run against prior probe's
  logs matching 3/5.
- **Next:** Operator launch. Link to the launch
  sequence below.

### 9. Launch sequence (for the operator, not this
session)

Document in the progress entry:

```
Launch sequence (operator):

1. Verify prereqs:
   python scripts/check_arb_signal_cleanup_prereqs.py
   (must exit 0)

2. Ensure training worker is running. The
   2026-04-21 _check_dead_thread race fix must be
   committed (check the worker log for "Training
   Worker started" with a recent PID).

3. Launch via admin UI:
   - Training Plans → select arb-signal-cleanup-probe
     (OR launch -A, wait for completion, then -B,
     then -C).
   - Tick "Smoke test first".
   - Click Launch.
   - Monitor episodes.jsonl for the cohort field to
     confirm telemetry plumbing.

4. After run completes:
   python scripts/validate_arb_signal_cleanup.py

5. Fill in the Validation template in progress.md
   with the validator's output.

6. Decision tree (per purpose.md):
   - All 5 pass → scale-run plan.
   - C1 pass, C4 fail → observation-space-audit.
   - C4 pass, C1 fail → controller-arch plan.
   - 1–4 all fail → observation-space-audit.
   - C5 fail → rollback, do NOT ship.
```

### 10. Commit

```
chore(registry): arb-signal-cleanup-probe plan + prereq/validator scripts

Plan file(s) for the three-cohort probe that validates
the Session 01/02 mechanisms (force-close at T-30s +
alpha_lr gene + shaped-penalty warmup) with ablation
attribution.

Cohort A (all three mechanisms), B (entropy velocity
only), C (warmup + force-close only). If the plan
data model doesn't support cohort-level overrides,
falls back to three plan files run serially — chosen
path recorded in progress.md.

- registry/training_plans/arb-signal-cleanup-probe*.json
  (one or three files depending on data-model
  capability).
- scripts/check_arb_signal_cleanup_prereqs.py — 5
  pre-launch checks.
- scripts/validate_arb_signal_cleanup.py — 5
  criteria + per-cohort pass/fail + force-close
  diagnostic + BC diagnostic.
- plans/INDEX.md — new row.
- plans/arb-signal-cleanup/progress.md — Session 03
  entry + launch sequence documentation.

Validator sanity-checked against prior probe's logs
(arb-curriculum-probe 2026-04-21); matches the
recorded 3/5 result.

No env / trainer / reward changes in this session
(Sessions 01 and 02 landed those).

Per plans/arb-signal-cleanup/hard_constraints.md
s24-s27.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## Do NOT

- Do NOT launch the run in this session. Per
  hard_constraints.md §39, the launch is an operator
  action that writes back into progress.md as a
  Validation entry. Session 03 is plan draft +
  scripts + documentation ONLY.
- Do NOT archive the registry or delete the
  277bbf49 plan. The failed plan stays in the
  registry as historical record.
- Do NOT skip the validator sanity check. If it
  doesn't reproduce the prior 3/5 result, the
  criteria semantics drifted and the new probe's
  results won't be interpretable against the
  baseline.
- Do NOT add genes to the schema that aren't
  explicitly called out in hard_constraints.md §24.
  Scope creep at this stage makes the ablation
  uninterpretable.
- Do NOT adjust the success criteria "to match" the
  new mechanism set. Same 5 criteria, same
  thresholds. The probe's job is to pass THESE
  criteria, not criteria redefined for its
  convenience.
- Do NOT run the full pytest suite during active
  training.

## After Session 03

Hand off to operator for launch. Operator fills the
Validation template in `progress.md` post-run.
