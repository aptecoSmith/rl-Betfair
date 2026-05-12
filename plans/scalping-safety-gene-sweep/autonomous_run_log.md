# Scalping safety-gene sweep — autonomous run log

Per-iteration progress through the plan. Newest entries at the
bottom.

## 2026-05-11 09:50 — Plan opened, cohort launching

**State entering iteration:** Predecessor cohort
`_predictor_SCALPING_overnight_1778458751` shipped with all
safety/shaping genes pinned to defaults. Plan scaffolded;
launch command being prepared.

**Work done:**
- Created `README.md`, `hard_constraints.md`, `master_todo.md`.
- Verified gene infrastructure: `training_v2/cohort/genes.py`
  Phase 5 has all six target genes registered with
  `--enable-gene NAME` activation. No code changes needed.

**Next iteration:** Launch the cohort, then schedule a
~30-min wakeup to confirm the first generation is rolling
cleanly (no KL blowup / GA collapse).

## 2026-05-11 17:58 — Cohort launched

**Tag:** `_predictor_SCALPING_safety_1778518690`
**Log:** `registry/_predictor_SCALPING_safety_1778518690.log`

**Configuration:**
- 12 agents × 8 generations, 3 training days, 3 eval days
  (eval=2026-05-04..2026-05-06).
- CUDA, seed 42, mutation_rate 0.2 (raised from predecessor's
  default 0.1 — wider gene exploration on a new gene pool).
- All 6 target genes active (confirmed in log):
  `fill_prob_loss_weight, mature_prob_loss_weight,
  matured_arb_bonus_weight, naked_loss_scale, open_cost,
  stop_loss_pnl_threshold`.
- Predictor bundle: same three production champions
  (`1c15250ee90d1b65`, `b23018bf5c8bcc70`,
  `conv1d_k3_s1_9659e9e9c3fb`).
- Lean obs (`--predictor-lean-obs`), scalping mode.

**Gene draw spot-check (Generation 1):**
- agent 1: open_cost=1.78, naked_loss_scale=0.42,
  stop_loss=0.009, matured_arb_bonus=0.43, fpw=0.066,
  mpw=3.02. **High open_cost, mid naked_scale, low stop_loss.**
- agent 4: open_cost=0.46, naked_loss_scale=0.08,
  stop_loss=0.07, matured_arb_bonus=1.45, fpw=0.030,
  mpw=2.11. **Heavy naked-anneal, light open-cost.**
- agent 12: open_cost=1.06, naked_loss_scale=0.96,
  stop_loss=0.28, matured_arb_bonus=3.03, fpw=0.227,
  mpw=3.76. **Aggressive everything — interesting test case.**

**Anomaly noted:** With `--days 6` the runner produced 3
train + 3 eval (predecessor `--days 6` produced 5 train + 1
eval). The GA will now select on the same 3-day window I was
treating as held-out. The final verdict will need a reeval
against a fresh window outside this 6-day span. Will resolve
when the cohort completes — note it but don't restart, the
gene exploration is what matters.

**Next iteration:** Schedule ~30 min wakeup to confirm
Generation 1 is progressing (no KL blowup / GA crash).
After that, ~2h heartbeats until completion (~12h ETA).

## 2026-05-12 06:53 — Cohort + held-out reeval complete

**State entering iteration:** Cohort `_predictor_SCALPING_safety_1778518690`
hit 96/96 rows at 06:33. Watcher auto-fired held-out reeval on
2026-04-28/29/30 (days fully outside training+eval window).
Reeval completed 06:53 in 19.2 min.

### In-sample generation trajectory (12 agents/gen)

| gen | mean | median | best | profitable | mean naked |
|---:|---:|---:|---:|---:|---:|
| 0 | −£160 | −£145 | +£237 | 3/12 | −£291 |
| 1 | −£114 | −£139 | +£245 | 2/12 | −£232 |
| 2 | −£94 | −£145 | +£314 | 4/12 | −£201 |
| 3 | −£93 | – | +£235 | 4/12 | −£201 |
| 4 | −£119 | – | +£96 | 3/12 | −£233 |
| 5 | −£144 | – | +£29 | 1/12 | −£276 |
| 6 | −£120 | – | +£417 | 3/12 | −£272 |
| 7 | **−£61** | – | +£210 | 3/12 | −£231 |

GA found a +£99 improvement Gen 0 → Gen 7 in-sample. Apparent
peak in Gen 2-3, then plateau-and-recover.

### Held-out reeval (top-5 by in-sample composite, 3-day fresh window)

| agent | gen | in-sample | held-out | locked | naked | mr |
|---|---:|---:|---:|---:|---:|---:|
| e72e9796 | 4 | +£29 | **+£97** | +£202 | −£63 | 0.209 |
| 3f7b641b | 6 | +£114 | +£15 | +£155 | −£95 | 0.254 |
| ccd8b87f | 7 | +£173 | −£69 | +£170 | −£199 | 0.232 |
| 78b36bee | 6 | +£417 | −£91 | +£154 | −£209 | 0.237 |
| e96dcc62 | 5 | −£27 | −£297 | +£155 | −£412 | 0.241 |

**Mean held-out: −£69. Median: −£69. Profitable: 2/5.**

### Comparison to predecessor cohort

| | predecessor | safety-gene | delta |
|---|---:|---:|---:|
| Mean held-out | −£28 | −£69 | **−£41 (worse)** |
| Profitable | 1/5 | 2/5 | +1 |
| Mean locked | +£190 | +£167 | −£23 |
| Mean naked | −£334 | −£196 | +£138 (better) |

### Verdict against success bar

`master_todo.md` set the bar at **≥3 of top-5 profitable on
held-out**. Actual: 2/5. **Bar not met.** The safety genes
genuinely shrank naked variance (+£138 mean improvement on
the naked term) but didn't lift enough agents into the
break-even zone.

### Why the bar wasn't met (probable causes)

1. **Agents aren't using champion p_win obs meaningfully.**
   Earlier A/B audit (commit `cc61cf1`) confirmed on
   acf9084a +/-£2 with champion ON vs OFF; on 56acc8e8
   champion ON makes things worse by £27. The predictor's
   known +28.9%/+19.9% ROI edge sits in obs but the policy
   doesn't route through it.

2. **In-sample top-1 (78b36bee +£417) was overfit.** Same
   pattern as predecessor: stochastic-mode +£417 →
   held-out −£91. Suggests "lucky stochastic action on a
   lucky day" rather than learned policy strength.

3. **GA plateaued at Gen 2-3.** The activated safety genes
   have a narrow useful range (stop_loss 0.07–0.30,
   open_cost 0.5–2.0, etc.) and the GA found those quickly.
   Gen 4-7 didn't break new ground.

### What WORKED

The locked floor (+£155-£202 across every single held-out
agent) is the load-bearing signal across both cohorts.
That's pure arb-pair mechanic, doesn't depend on predictor
obs. Roughly +£167 mean held-out per 3-day window is
real and consistent.

### Recommendation for next cohort

Hard action-mask gate on `champion_p_win` — option (3) from
the earlier triage. Force the env to refuse OPEN_BACK on
runners with `p_win < threshold_low` and OPEN_LAY on runners
with `p_win > threshold_high`. That would structurally
align the agent's directional opens with the predictor's
known edge instead of leaving it to the policy to learn
(which it doesn't).

### Loop status

Stopping. The safety-gene plan's success bar wasn't met;
the diagnosis is solid; the next plan needs an action-mask
gate (which is a code change to the action shim, not a
re-cohort).
