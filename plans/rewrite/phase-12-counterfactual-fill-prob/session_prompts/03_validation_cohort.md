---
session: phase-12-counterfactual-fill-prob / S03
phase: rewrite/phase-12-counterfactual-fill-prob
parent_purpose: ../purpose.md
depends_on: S02
---

# S03 — V1 validation cohort

## Context

S01 ships offline labels. S02 wires them to the per-side widened
`fill_prob_head` and removes the agent-rollout label path.
S03 answers the question that motivated the entire plan:

> **Does the actor's natural-fill rate move above the 0.17 – 0.21
> ceiling once it has counterfactual fill information at decision
> time?**

If yes (and force-close rate drops, and composite_score doesn't
collapse) — V1 ships, becomes default in main GA cohorts. If
no — V2 (queue-aware fill simulation) follows.

Read `purpose.md` and `hard_constraints.md` first. The ceiling
is documented in
[phase-8-oracle-bc-pretrain/findings.md](../../phase-8-oracle-bc-pretrain/findings.md).

## Probe design — split cohort, 5 generations

Two arms, same shape as Phase 8's split overnight (which is the
proven template):

| Arm | fill_prob_loss_weight | All else |
|-----|----------------------|----------|
| A control | 0.0 (head trained but no loss → benign 0.5 column on actor_input) | per-transition-credit ON, no BC |
| B fill-prob ON | 0.10 | per-transition-credit ON, no BC |

Both arms identical otherwise:
- 12 agents per arm
- 5 generations (NOT 3 — Phase 8's overnight showed BC's gen-1
  effect changed across gens; V1 fill-prob may have similar
  evolution-dependent dynamics)
- 4 train + 3 eval days (50/50 default)
- env-overrides matching Phase 8 overnight:
  `target_pnl_pair_sizing_enabled=true`,
  `force_close_before_off_seconds=60`,
  `min_seconds_before_off=60`, `open_cost=1.0`
- `--enable-gene mature_prob_loss_weight` (per the overnight
  pattern; gives the actor's mature_prob head a per-agent spread
  for ρ analysis as a secondary signal)
- `--maturation-bonus-weight 10.0` so the GA selects on
  composite_score (otherwise selection is total_reward only,
  and the lower-spread / higher-fill regime would lose the
  selection signal)
- Same `--seed 42` across both arms — initial gene draws line up
  agent-by-agent at gen 0 for direct comparison

Output dirs:

```
registry/_phase12_s03_A_fillprob_off_{ts}/
registry/_phase12_s03_B_fillprob_on_{ts}/
```

## Pre-reqs

- S01 + S02 shipped and tests green.
- Fill-label cache populated for the cohort's training window:

```bash
python -m training_v2.fill_label_cli scan \
    --dates {auto-detect from --days 7 + data_dir} \
    --arb-spread-ticks 20 \
    --force-close-before-off-seconds 60
```

- 1-agent smoke run with `fill_prob_loss_weight=0.10` on the
  cohort's first training day completes without error before
  launching the full cohort. Catches missing-cache errors before
  burning ~16 hours.

## Wall budget

Per-agent per gen ≈ 396s (4 train + 3 eval days at the rates
observed in Phase 8 overnight).

- 12 agents × 5 gens × 396s = 23,760s ≈ **6.6 h per arm**
- Two arms back-to-back: **13.2 h**
- + ~10 min for label cache scan if not yet populated
- + ~5 min smoke run
- Total **~13.5 h**

Comfortably inside an 18 h overnight window.

## Runner commands

Two-arm shell pipeline mirroring `run_phase8_overnight.sh`:

```bash
TS=$(date +%s)
# Stage 0 — populate label cache (skip if already populated)
python -m training_v2.fill_label_cli scan \
    --dates {detected days} \
    --arb-spread-ticks 20 \
    --force-close-before-off-seconds 60

# Arm A — control
python -m training_v2.cohort.runner \
    --n-agents 12 --generations 5 --days 7 \
    --device cuda --seed 42 --data-dir data/processed \
    --per-transition-credit \
    --enable-gene mature_prob_loss_weight \
    --reward-overrides target_pnl_pair_sizing_enabled=true \
    --reward-overrides force_close_before_off_seconds=60 \
    --reward-overrides min_seconds_before_off=60 \
    --reward-overrides open_cost=1.0 \
    --reward-overrides fill_prob_loss_weight=0.0 \
    --maturation-bonus-weight 10.0 \
    --output-dir registry/_phase12_s03_A_fillprob_off_${TS}

# Arm B — fill_prob ON
python -m training_v2.cohort.runner \
    --n-agents 12 --generations 5 --days 7 \
    --device cuda --seed 42 --data-dir data/processed \
    --per-transition-credit \
    --enable-gene mature_prob_loss_weight \
    --reward-overrides target_pnl_pair_sizing_enabled=true \
    --reward-overrides force_close_before_off_seconds=60 \
    --reward-overrides min_seconds_before_off=60 \
    --reward-overrides open_cost=1.0 \
    --reward-overrides fill_prob_loss_weight=0.10 \
    --maturation-bonus-weight 10.0 \
    --output-dir registry/_phase12_s03_B_fillprob_on_${TS}
```

Use `set -e` and `tee` per arm log; halt-on-first-error.

## Metrics to compute

Per-agent eval row (already in scoreboard.jsonl):
- `eval_arbs_completed` (matured naturally)
- `eval_arbs_closed` (agent-closed)
- `eval_arbs_force_closed`
- `eval_arbs_naked`
- `eval_pairs_opened`
- `eval_total_reward`
- `eval_day_pnl`
- `eval_locked_pnl`
- `hyperparameters["mature_prob_loss_weight"]` (gene value, for ρ)

Derived per-agent:
- `natural_fill_rate = eval_arbs_completed / eval_pairs_opened`
- `force_close_rate = eval_arbs_force_closed / eval_pairs_opened`
- `mr = (eval_arbs_completed + eval_arbs_closed) / eval_pairs_opened`

Cross-arm aggregates (per gen):
- mean `natural_fill_rate`
- mean `force_close_rate`
- mean `mr`
- mean `composite_score`
- mean `eval_day_pnl`
- ρ(mature_prob_loss_weight, natural_fill_rate) — secondary

## Success gates

### Gate 1 — natural fill rate lifts (PRIMARY)

Compute mean `natural_fill_rate` across all 12 agents in Arm B at
each generation 0..4. The gate is on **gen 4** (final gen):

| Arm B gen-4 mean natural_fill_rate | Verdict |
|---|---|
| ≥ 0.30 | **GREEN** — ceiling broken; V1 ships |
| 0.25 – 0.30 | **AMBER** — directional, may need V2 queue-aware label |
| < 0.25 | **RED** — V1 conservative label insufficient |

The 0.25 threshold is a 1.2× relative lift over the 0.21
ceiling; 0.30 is 1.5×. The gen-4 (not gen-0) cutoff acknowledges
that the actor needs PPO updates to learn to USE the fill_prob
signal.

Also tracked: gen-0 vs gen-4 trajectory in Arm B. If gen-0
already shows the lift, the head is calibrated immediately; if
the lift only appears by gen-4, PPO is integrating the new
input over time.

### Gate 2 — force-close rate drops

| Arm B gen-4 force_close_rate ≤ Arm A gen-4 force_close_rate − 5 pp | Verdict |
|---|---|
| Yes | Selectivity working as designed |
| No | Fill prob is calibrated but actor isn't using it for selection |

Phase 8 overnight observed `fc_rate` at 0.72 – 0.76. A 5 pp drop
to 0.67 – 0.71 would be a substantial selectivity improvement.

### Gate 3 — composite score doesn't collapse

| Arm B gen-4 mean composite_score ≥ Arm A gen-4 mean composite_score − 50 | Verdict |
|---|---|
| Yes | Selectivity gain isn't paid for by collapse |
| No | Actor over-selective, opens too few pairs to matter |

The £50/day buffer accounts for run-to-run variance; tighter
than that risks calling a noisy result a fail.

### Gate 4 — gen-0 sanity (head is calibrated)

At gen 0 (before any PPO update), Arm B's `fill_prob_head`
predictions on a held-out batch should correlate with the
offline labels at ρ ≥ +0.3. This is a label-integrity check —
if the head can't predict its own training labels, S02's wiring
is broken. Compute by:

1. After gen-0 training, snapshot the policy.
2. Run forward pass on 1000 sampled (transition, runner, side)
   triples from the eval day.
3. Compute Pearson ρ between head prediction and offline label.
4. Both back and lay sides; both must hit ≥ +0.3.

This is a programmatic check, not a manual review.

## Verdict criteria

| Gate 1 | Gate 2 | Gate 3 | Gate 4 | Verdict |
|--------|--------|--------|--------|---------|
| ≥ 0.30 | ≤ -5pp | ≥ -£50 | ≥ +0.3 | **GREEN** — V1 ships |
| 0.25-0.30 | any | ≥ -£50 | ≥ +0.3 | **AMBER** — proceed to V2 |
| < 0.25 | any | any | < +0.3 | **RED** — investigate |
| < 0.25 | any | any | ≥ +0.3 | **RED-mechanism** — head calibrated but actor can't use signal; investigate input / capacity |

Distinguishing the two RED cases matters for what comes next.
RED-label (Gate 4 fails) → fix the label generator. RED-mechanism
(Gate 4 passes, Gate 1 fails) → the actor has the info but
can't use it; investigate obs schema, head capacity, or the
gradient pathway.

## Analysis template

Write to `plans/rewrite/phase-12-counterfactual-fill-prob/findings.md`:

```markdown
## V1 validation cohort

### Probe design
Arm A: fill_prob_loss_weight=0.0 (control)
Arm B: fill_prob_loss_weight=0.10
12 agents × 5 gens × 4 train + 3 eval. seed=42 both arms.

### Gate 1 — natural fill rate
| Gen | A nat_fr | B nat_fr | Δ |
|---|---|---|---|
| 0 | X | Y | Δ |
| 1 | X | Y | Δ |
| 2 | X | Y | Δ |
| 3 | X | Y | Δ |
| 4 | X | Y | Δ |

Gate (B gen-4 ≥ 0.30): PASS / FAIL

### Gate 2 — force-close rate
| Gen | A fc_rate | B fc_rate | Δ |
|---|---|---|---|
[gen-0 + gen-4]

Gate (B gen-4 ≤ A gen-4 - 5pp): PASS / FAIL

### Gate 3 — composite score
| Gen | A composite | B composite | Δ |
|---|---|---|---|
[gen-0 + gen-4]

Gate (Δ ≥ -50): PASS / FAIL

### Gate 4 — head calibration
ρ(fill_prob_back_pred, label_back) on 1000 held-out samples = X
ρ(fill_prob_lay_pred,  label_lay) on 1000 held-out samples = Y

Gate (both ≥ +0.3): PASS / FAIL

### Predicted vs realised positive rate
S01 reported back-side positive density of X on training days.
Arm B realised gen-4 natural_fill_rate of Y.
Calibration gap [discuss — tighter than predicted? Looser?]

### secondary signal: ρ(mature_prob_loss_weight, natural_fill_rate)
Arm A: ρ = X
Arm B: ρ = Y
[Phase 9's per-transition credit was supposed to make ρ positive.
The Phase 8 overnight showed it stayed noisy. Did adding
fill_prob change that?]

### Verdict
GREEN / AMBER / RED + paragraph.

### Recommended follow-on
[V2 work? Phase 13 (architectural)? Ship V1?]
```

## Done when

- Both arms complete without error.
- `findings.md` written with all 4 gate evaluations and verdict.
- `lessons_learnt.md` updated with:
  - The verdict
  - Whether gen-0 vs gen-4 trajectory in Arm B was step-function
    (head calibration was enough) or gradual (PPO integration
    matters)
  - Any unexpected behaviour (e.g. naked count rose,
    composite_score crashed)
- If GREEN: open a follow-on for making `fill_prob_loss_weight`
  default-on in main GA cohorts at a tuned default (start with
  0.10).
- Commit: `docs(rewrite): phase-12 S03 V1 validation [{verdict}]`.
