---
session: phase-10-argmax-eval / S03
phase: rewrite/phase-10-argmax-eval
parent_purpose: ../purpose.md
depends_on: S01, S02
---

# S03 — validation cohort + plan close

## Context

S01 and S02 added the deterministic action path and wired
`--argmax-eval` through every eval code path. This session
validates that the change does what the purpose.md predicts
and writes the plan up.

Read `purpose.md` §"Success bar" items 4-6 first. Three gates
to pass:

1. **Reproducibility** — same agent + same day under argmax
   produces bit-identical pnl across two runs.
2. **Variance reduction** — the
   `658a7f72`/`e34193fd`/`9a20de9c` lineage's day_pnl spread
   under argmax is < £62 (1/3 of the £185 stochastic spread).
3. **Rank correlation** — Spearman ρ between
   3-day-stochastic-mean rank order and 1-day-argmax rank order
   over all 144 s06 agents is ≥ 0.7.

## Pre-reqs

- S01 and S02 done and merged.
- The s06 cohort is on disk at
  `registry/_phase7_s06_24agent_overnight_1777941123/` with
  144 agents, 144 saved weight files, and the original
  `scoreboard.jsonl`.
- The stochastic 3-day re-eval (`reeval_scoreboard_3day.jsonl`
  in the same dir) should be complete by Phase-10 launch
  time. If it's not, wait for it — Session 03's rank-
  correlation test compares against this file.
- ~3-9 GPU-hours free (single-day argmax-eval on 144 agents
  is ~3 hours; full 3-day argmax-eval is ~9 hours).

## What to do

### 1. Reproducibility test (Success-bar gate 1)

Pick the top-3 agents by composite_score from the s06
scoreboard. For each, run `tools/reevaluate_cohort.py
--argmax-eval` twice on the same single eval day. Assert the
two output rows have bit-identical numeric fields.

```bash
TOP3=$(python -c "
import json
from pathlib import Path
rows = [json.loads(l) for l in Path('registry/_phase7_s06_24agent_overnight_1777941123/scoreboard.jsonl').read_text().splitlines() if l.strip()]
rows.sort(key=lambda r: -r.get('composite_score', r.get('eval_total_reward', 0)))
print(' '.join(r['agent_id'][:8] for r in rows[:3]))
")
echo "Top-3: $TOP3"

# Run 1
python -m tools.reevaluate_cohort \
  --cohort-dir registry/_phase7_s06_24agent_overnight_1777941123 \
  --eval-days 2026-05-03 --argmax-eval --device cuda \
  --filter-agent-ids $TOP3 \
  --output reeval_argmax_repro_run1.jsonl

# Run 2 (identical command, different seed → must still produce
# identical results because argmax is deterministic).
python -m tools.reevaluate_cohort \
  --cohort-dir registry/_phase7_s06_24agent_overnight_1777941123 \
  --eval-days 2026-05-03 --argmax-eval --device cuda \
  --filter-agent-ids $TOP3 --seed 999 \
  --output reeval_argmax_repro_run2.jsonl
```

Compare the two output JSONL files; assert per-agent
`reeval_day_pnl`, `reeval_locked_pnl`, `reeval_naked_pnl`
identical.

If reproducibility fails: STOP. Open the env-side RNG paths.
Common culprits:
- Passive-fill ordering in `ExchangeMatcher`.
- Random RNG draws inside the env's settle path.
- A torch operation that's non-deterministic on CUDA (less
  likely to affect cash totals but can shift action selection
  if logits are tied).

Document the finding in `findings.md` even if it blocks the
gate. Reproducibility failure under argmax is a real
discovery — surface it.

### 2. Run the full argmax re-eval (1-day → 3-day for ranking)

```bash
python -m tools.reevaluate_cohort \
  --cohort-dir registry/_phase7_s06_24agent_overnight_1777941123 \
  --eval-days 2026-05-02 2026-05-03 2026-05-04 \
  --argmax-eval --device cuda \
  --output reeval_scoreboard_3day_argmax.jsonl
```

ETA ~9 hours wall on 144 agents × 3 days × ~75s/day.

If GPU budget is tight, run only the single-day argmax for the
rank-correlation gate (gate 3). Single-day argmax on 144
agents takes ~3 hours wall:

```bash
python -m tools.reevaluate_cohort \
  --cohort-dir registry/_phase7_s06_24agent_overnight_1777941123 \
  --eval-days 2026-05-03 \
  --argmax-eval --device cuda \
  --output reeval_scoreboard_1day_argmax.jsonl
```

The variance-reduction gate (item 2) only needs the 3 lineage
agents — those can be filtered for and run in <5 min:

```bash
python -m tools.reevaluate_cohort \
  --cohort-dir registry/_phase7_s06_24agent_overnight_1777941123 \
  --eval-days 2026-05-02 2026-05-03 2026-05-04 \
  --argmax-eval --device cuda \
  --filter-agent-ids 658a7f72 e34193fd 9a20de9c \
  --output reeval_lineage_argmax.jsonl
```

### 3. Variance-reduction test (Success-bar gate 2)

Read `reeval_lineage_argmax.jsonl`. Compute the day_pnl spread
(max - min) across the three lineage agents. Compare to the
stochastic baseline:

- Stochastic single-day baseline (from original s06 scoreboard):
  agents had day_pnl +£178, -£7, +£28 → spread = £185.
- Argmax single-day target: spread < £62.

Compute the argmax spread on each of the 3 eval days
separately AND on the 3-day mean. Report all four numbers in
findings. The single-day spread is the cleanest comparison
because it removes day-to-day market variance.

If gate 2 fails: the lineage spread is driven by training
stochasticity (channel 2 in the conversation), not eval
sampling. Useful finding — argmax can't fix it. Document
clearly and recommend the next experiment (multi-seed training
runs).

### 4. Rank-correlation test (Success-bar gate 3)

Compute Spearman ρ between:
- The 3-day-stochastic-mean rank order from
  `reeval_scoreboard_3day.jsonl` (sort all 144 agents by
  `reeval_day_pnl` descending).
- The 1-day-argmax rank order from
  `reeval_scoreboard_1day_argmax.jsonl` (same sort key).

Use scipy:
```python
from scipy.stats import spearmanr
rho, p = spearmanr(stoch_3day_pnls, argmax_1day_pnls)
print(f"Spearman rho = {rho:.3f}, p = {p:.3g}")
```

Gate: ρ ≥ 0.7.

If gate 3 fails: argmax-best agents are systematically
different from stochastic-multi-day-mean-best agents. The
policies are too uncertain at the argmax decision points; the
sampling reveals real structure that argmax hides. **Document
and don't ship as the default.** Argmax can stay as an
opt-in tool but it's not a clean replacement for sampling-based
eval.

### 5. Action-distribution diagnostic (informational)

For a few representative agents, log the action histogram
under both modes and compute the KL divergence between
`stochastic_action_dist` and `argmax_action_dist`. If KL is
near zero everywhere, the policy is essentially deterministic
already and argmax is free. If KL is large, the policy is
exploring under sampling and argmax is genuinely a different
distribution. Useful for understanding whether the GA is
selecting for high-confidence policies.

### 6. Write `findings.md`

Use this structure:

```markdown
---
plan: rewrite/phase-10-argmax-eval
session: S03
opened: 2026-05-XX
status: <GREEN / AMBER / RED>
---

# Phase 10 Session 03 — validation findings

## Verdict

<one-paragraph: did argmax-eval do what we hoped?>

## Cohort design

- Source: existing s06 cohort
  (`_phase7_s06_24agent_overnight_1777941123`), 144 trained agents.
- Re-eval days: 2026-05-02, 2026-05-03, 2026-05-04.
- Three test rollouts: stochastic baseline (already on disk),
  argmax 1-day, argmax 3-day.

## Gate 1 — reproducibility

- Top-3 agents argmax run 1 vs run 2: <PASS / FAIL with details>

## Gate 2 — lineage variance reduction

- Stochastic baseline (single-day, 2026-05-03): spread £185
  (+£178, -£7, +£28 across the three identical-gene siblings)
- Argmax single-day (2026-05-02): spread £<X>
- Argmax single-day (2026-05-03): spread £<X>
- Argmax single-day (2026-05-04): spread £<X>
- Argmax 3-day mean: spread £<X>
- Gate (< £62): <PASS / FAIL>

## Gate 3 — rank correlation

- Spearman ρ(3-day-stochastic-mean rank, 1-day-argmax rank) = <X>
- Gate (≥ 0.7): <PASS / FAIL>

## Sanity / informational

- Action-distribution KL(stochastic || argmax): mean / max
  across sampled agents.
- How many of the original "profitable" agents (+pnl in
  stochastic eval) remain profitable under argmax?
- Top-10 by argmax-pnl vs top-10 by stochastic-3-day-mean: how
  many overlap?

## Recommendation

<one of:
- "Adopt argmax-eval as the default. Variance is dominated by
  action sampling and argmax is the right primitive."
- "Adopt as opt-in. Argmax is reproducible and useful for
  development iteration but stochastic-multi-day-mean stays
  the headline measurement for production verdicts."
- "Don't ship as default — gate 3 failed and the rank-flip
  reveals the policies are genuinely exploring meaningful
  structure under sampling."
>

## What's locked

Phase 10 ships <GREEN / AMBER / RED>:
- The deterministic primitive lives in `RolloutCollector`.
- The `--argmax-eval` flag plumbs through every eval code path.
- The validation cohort showed <X / Y / Z>.
```

### 7. Update CLAUDE.md

Add a section under `## Reward function` (or as a new top-
level section) titled `## Eval modes: stochastic and argmax
(2026-05-XX)`. Document:

- The two action-selection paths and what they're for.
- The `--argmax-eval` flag's defaults (off everywhere).
- The `RolloutCollector.collect_episode(deterministic=...)`
  primitive.
- That training rollouts ALWAYS use sampling regardless.
- Cross-reference `plans/rewrite/phase-10-argmax-eval/`.

Keep terse — three short paragraphs, similar voice to the
surrounding sections.

### 8. Update `plans/rewrite/README.md`

Add a navigation line:

```
- `phase-10-argmax-eval/purpose.md` + `findings.md` (<verdict>,
  2026-05-XX) — deterministic argmax action selection at eval
  time; suppresses per-tick action-sampling noise that was
  swinging same-policy day_pnl by ±£200 on identical eval days.
```

Insert after the existing phase-7 line.

## Done when

- `findings.md` written.
- CLAUDE.md updated.
- `plans/rewrite/README.md` updated.
- Three gate outcomes recorded with numbers.
- Commit: `docs(rewrite): close phase-10 <GREEN/AMBER> —
  argmax-eval reproducible, lineage variance <reduced/unchanged>,
  rank ρ=<X>`.

## Stop conditions

- Gate 1 (reproducibility) failure: write findings + STOP. Do
  not run gates 2-3 until reproducibility is debugged. Env-side
  RNG audit goes into its own follow-on plan.
- Gate 3 (rank correlation) failure: ship as opt-in, not
  default. Document clearly why.
- If the full 3-day argmax re-eval would burn budget needed for
  another experiment, run just the lineage test (gate 2) and
  the single-day cohort (gate 3). The 3-day argmax re-eval is
  for completeness, not gating.

## Out of scope

- Re-running training under argmax (training stays stochastic
  by hard-constraint).
- Adding new architectures to compare under argmax-eval. Phase
  X is the consumer; this phase just provides the cleaner
  measurement primitive.
- Updating any frontend / scoreboard UI to show `eval_mode`
  badges. Field lands in JSONL; UI work is its own follow-on.
- Tuning multi-seed-per-agent training to also reduce channel-2
  noise. That's a separate (larger) lever; would benefit
  Phase X comparisons but is not required for Phase 10 to ship.
