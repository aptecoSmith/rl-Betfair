# Session prompt — phase-5-restore-genes Session 03: validation + writeup

Use this prompt to open a new session in a fresh context. The prompt
is self-contained — it briefs you on the task, the design decisions
already locked from Sessions 01 and 02, and the constraints. Do not
require any context from prior sessions.

---

## The task

Sessions 01 and 02 wired 11 new genes into `CohortGenes` and
plumbed per-agent gene values through to the env / trainer
via the `--enable-gene` CLI switch. **Session 03 validates
the end-to-end gene flow and writes up the plan's verdict.**

End-of-session bar:

1. **Validation cohort runs.** A small dry-run cohort
   (12 agents × 1 generation × 1 day) with all 11 new genes
   enabled. Wall envelope: ~5-10 minutes (1 day = ~12k ticks
   × 12 agents sequentially).
2. **Per-agent gene values reflected in scoreboard.** Each
   row's `hyperparameters` dict contains the agent's draws
   for all 18 fields. Inspection confirms gene values vary
   across agents (uniform/log-uniform sampling visibly
   produces a spread).
3. **Behavioural attribution sanity check.** A quick read of
   the validation cohort's scoreboard confirms gene values
   correlate with expected env behaviour:
   - High `open_cost` → fewer / smaller bets.
   - High `matured_arb_bonus_weight` → positive shaped term
     reward bias.
   - High `stop_loss_pnl_threshold` → fewer stop-closes
     (looser threshold).
4. **`findings.md` populated** with:
   - Schema-extension table (all 18 genes + ranges + defaults).
   - Validation cohort summary.
   - CLI usage examples (incremental + everything-on).
   - Notes on dimension explosion: 18-gene GA needs more
     agents per generation OR more generations to explore
     meaningfully.
   - Per-gene plan-of-record references (which plan called
     each gene a gene).
   - Verdict: GREEN / AMBER / RED.
5. **Plan status updated** to GREEN in the
   `purpose.md` frontmatter (assuming verdict GREEN).
6. **Commit + push** the validation results and the
   findings.md.

## What you need to read first

1. `plans/rewrite/phase-5-restore-genes/purpose.md`.
2. `plans/rewrite/phase-5-restore-genes/session_prompts/
   01_gene_schema_and_breeding.md` and
   `02_cli_and_worker_plumbing.md` for what the prior
   sessions delivered.
3. `tools/peek_cohort.py` — the canonical live readout tool
   from cohort-visibility plan; you'll use it to inspect the
   validation cohort.
4. `data/processed_amber_v2_window/` — the locked AMBER v2
   data window. Validation uses this data dir.
5. The 4 most recent cohort dirs under `registry/` — what
   "good" cohort output looks like for comparison.

## What to do

### 1. Pre-flight (~10 min)

- Confirm Sessions 01 + 02 are committed and tests pass:
  ```
  pytest tests/test_v2_cohort_genes.py
          tests/test_v2_cohort_runner.py
          tests/test_v2_cohort_worker.py -v
  ```
- Confirm a no-flag cohort launch is byte-identical to
  pre-plan. (Optional sanity check — skip if Session 02's
  byte-identity test passed.)

### 2. Validation cohort launch (~10 min)

```
TS=$(date +%s)
OUT=registry/v2_phase5_validation_${TS}
mkdir -p "$OUT"
python -m training_v2.cohort.runner \
    --n-agents 12 --generations 1 --days 2 \
    --device cpu --seed 42 \
    --data-dir data/processed_amber_v2_window \
    --enable-gene open_cost \
    --enable-gene matured_arb_bonus_weight \
    --enable-gene mark_to_market_weight \
    --enable-gene naked_loss_scale \
    --enable-gene stop_loss_pnl_threshold \
    --enable-gene arb_spread_scale \
    --enable-gene fill_prob_loss_weight \
    --enable-gene mature_prob_loss_weight \
    --enable-gene risk_loss_weight \
    --enable-gene alpha_lr \
    --enable-gene reward_clip \
    --output-dir "$OUT" 2>&1 | tee "$OUT/cohort.log"
```

`--days 2` = 1 training day + 1 eval day. `--device cpu`
because this is a wiring-validation run, not a perf-validation
run; CPU is sufficient and avoids GPU resource contention if
something else is running.

Wall envelope: ~5-10 minutes. If past 15 minutes, kill — the
gene plumbing has a bug.

### 3. Inspect scoreboard (~15 min)

```
python -m tools.peek_cohort registry/v2_phase5_validation_<ts>
```

The peek tool surfaces per-agent macro metrics. For Phase 5
validation we care about:

- **All 12 agents complete.** No silent crashes from the
  plumbing.
- **Per-agent gene values vary.** Read the scoreboard.jsonl
  rows directly:

  ```
  python -c "
  import json
  for line in open('registry/v2_phase5_validation_<ts>/scoreboard.jsonl'):
      row = json.loads(line)
      h = row['hyperparameters']
      print(f\"{row['agent_id'][:8]} open_cost={h['open_cost']:.3f} \"
            f\"matured_bonus={h['matured_arb_bonus_weight']:.3f} \"
            f\"mtm={h['mark_to_market_weight']:.4f} \"
            f\"stop_loss={h['stop_loss_pnl_threshold']:.3f} \"
            f\"arb_scale={h['arb_spread_scale']:.3f} \"
            f\"alpha_lr={h['alpha_lr']:.4f}\")
  "
  ```

  Each agent's values should differ. The 12 draws should span
  a meaningful slice of each gene's range.

- **Behavioural sanity.** Compute per-agent metrics from the
  scoreboard:

  ```
  python -c "
  import json
  rows = [json.loads(l) for l in
          open('registry/v2_phase5_validation_<ts>/scoreboard.jsonl')]
  rows = [r for r in rows if 'agent_id' in r]
  # High open_cost should correlate with fewer bets.
  for r in sorted(rows,
                  key=lambda r: r['hyperparameters']['open_cost']):
      print(f\"open_cost={r['hyperparameters']['open_cost']:.3f} \"
            f\"bets={r['eval_bet_count']} \"
            f\"matured={r['eval_arbs_completed']}\")
  "
  ```

  Expect: agents at the high end of open_cost open fewer
  pairs. The correlation may not be perfect at n=12 over 1
  day of training — the agents haven't trained long enough
  for full gene effects to surface — but the trend should
  visible. If it's reversed (high open_cost → MORE bets),
  the plumbing is wrong.

### 4. Findings doc (~30 min)

Create `plans/rewrite/phase-5-restore-genes/findings.md`:

```markdown
---
plan: rewrite/phase-5-restore-genes
opened: 2026-05-03
status: GREEN
session_03_completed: <date>
---

# Phase 5 — restore-genes: cumulative findings

## What this phase delivered

CohortGenes grew 7 → 18 fields. Eleven new per-agent genes
were promoted from cohort-wide reward_overrides flags, with
documented ranges and an `--enable-gene NAME` CLI switch so
the operator decides which evolve per cohort.

| Gene | Range | Distribution | Default | Plan that called it a gene |
|---|---|---|---|---|
| open_cost | [0.0, 2.0] | uniform | 0.0 | selective-open-shaping master_todo §"Promote ..." |
| matured_arb_bonus_weight | [0.0, 5.0] | uniform | 0.0 | (designed-as-gene) |
| mark_to_market_weight | [0.0, 0.10] | uniform | 0.05 | reward-densification §"reward-densification-gene" |
| naked_loss_scale | [0.0, 1.0] | uniform | 1.0 | (designed-as-gene) |
| stop_loss_pnl_threshold | [0.0, 0.30] | uniform | 0.0 | force-close-architecture S02 |
| arb_spread_scale | [0.5, 2.0] | uniform | 1.0 | scalping mechanics gene |
| fill_prob_loss_weight | [0.0, 0.30] | uniform | 0.0 | scalping-active-management S02 |
| mature_prob_loss_weight | [0.0, 0.30] | uniform | 0.0 | per-runner-credit follow-on |
| risk_loss_weight | [0.0, 0.30] | uniform | 0.0 | scalping-active-management S03 |
| alpha_lr | [1e-2, 1e-1] | log-uniform | 1e-2 | arb-signal-cleanup §"alpha_lr as ..." |
| reward_clip | [1.0, 10.0] | uniform | 10.0 | (designed-as-gene) |

## Validation cohort

(Insert validation cohort summary table.)

## CLI usage

### Legacy (no Phase 5 genes evolve — byte-identical to pre-plan)

```
python -m training_v2.cohort.runner --n-agents 12 ...
```

### Incremental (one new gene per cohort)

```
python -m training_v2.cohort.runner \
    --n-agents 12 --generations 4 ... \
    --enable-gene open_cost
```

### Everything on (full 18-gene exploration)

```
python -m training_v2.cohort.runner \
    --n-agents 20 --generations 6 ... \
    --enable-gene open_cost \
    --enable-gene matured_arb_bonus_weight \
    --enable-gene mark_to_market_weight \
    ...
```

### Mutual exclusion

`--enable-gene open_cost --reward-overrides open_cost=1.0`
errors at startup. Operator picks one source of truth per
knob per run.

## Notes on dimension explosion

The GA's search space grew 7 → 18 dimensions. Existing 12-agent
cohort runs likely insufficient to explore the full space:

- 12 agents per generation across 18 dims = ~0.7 agents/dim
  per generation.
- Recommended for full 11-gene exploration: 20+ agents per
  generation, 6+ generations.
- For incremental (1-2 new genes enabled): 12 agents × 4
  generations is fine.

## Verdict

GREEN — all 11 new genes wired, switchable, validated. The
rewrite's per-agent search space is now feature-complete vs
the design intent.

## What's now possible (next plans)

The three follow-up directions in
`force-close-architecture/findings.md` are now testable as
gene-evolved cohorts:

1. **`open_cost` evolution** — multi-gen cohort with
   `--enable-gene open_cost` to find the working point.
2. **`matured_arb_bonus_weight`** as positive shaping
   complement.
3. **Multi-gen + everything-on** for the full v1-catch-up
   test.
```

Adjust as needed based on what the validation cohort
actually shows.

### 5. Update purpose.md status

In the frontmatter:

```yaml
---
plan: rewrite/phase-5-restore-genes
status: GREEN
opened: 2026-05-03
depends_on: ...
---
```

(Or AMBER / RED with rationale if validation surfaced issues.)

### 6. Commit + push

```
git add plans/rewrite/phase-5-restore-genes/findings.md \
        plans/rewrite/phase-5-restore-genes/purpose.md
git commit -m "docs(rewrite): phase-5-restore-genes ships GREEN"
git push origin master
```

## Stop conditions

- **Validation cohort fails to complete all 12 agents** →
  plumbing bug. Triage in Session 02's plumbing — likely the
  gene → reward_overrides path or the trainer-hyperparameter
  override.
- **All 12 agents have identical gene values** → the
  `enabled_set` isn't reaching `sample_genes`. Check the
  runner's call site.
- **Behavioural sanity check inverted** (e.g. high open_cost
  → MORE bets) → gene values reaching env reward_overrides
  with wrong semantics. Check `_build_per_agent_reward_
  overrides`.
- **Past 2 h excluding the validation cohort wall** → the
  task is small; if it's taking longer something's wrong.

## Hard constraints

Inherited from `purpose.md`. No env edits, no new mechanics,
no schema breaks.

## Out of scope

- New mechanic plans. Phase 5 is purely about per-agent
  search dimensions; no new env behaviour ships here.
- Production cohort runs. The validation here is plumbing
  validation only. Real verdict cohorts (testing whether
  Phase 5 + Phase 4 catches up to v1) live in their own
  follow-on plans, gated on Phase 5 + Phase 4 both shipping
  GREEN.
