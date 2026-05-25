# Findings — non-scalping-directional-probe

## TL;DR

Both probes FAIL. Scalping remains the only mode where the
project's pwin signal has tradeable edge on held-out days.
Per the pre-registered decision table this closes the chapter
on directional value betting at this predictor calibration.

## Setup

- 5 seeds × 3 held-out days (2026-04-28 / 04-29 / 04-30) per probe
- env `strategy_mode = "value_win"`, `scalping_mode = False`
- gate `value_edge_threshold = 0.05`
- uniform-random policy (no GA, no BC, fresh init); the gate is
  the binding constraint
- predictor bundle: same three production manifests as
  `scalping-lay-quality-gate`
- per-bet logs: `registry/probe_{A_back,B_lay}/bets_<day>_seed<n>.jsonl`
- per-probe analysis JSON: `registry/probe_*/[_analysis.json]`

| Probe | Side filter | Sizing | Extra |
|---|---|---|---|
| A | back-only | flat £10 stake | — |
| B | lay-only  | fixed £20 liability | price ∈ [2, 20] (composes with gate) |

## Probe A (back-only)

| Metric | Result | Criterion | Verdict |
|---|---|---|---|
| Per-bet EV (mean of final_pnl) | **−£2.09** | PASS: >+0.50 / FAIL: <+0.10 | **FAIL** |
| Per-bet σ | £20.15 | — | — |
| Per-bet Sharpe | **−0.10** | PASS: >0.10 / FAIL: <0.05 | **FAIL** |
| Days profitable / 3 | **0/3** | PASS: ≥2 / FAIL: 0 | **FAIL** |
| Bets / day (mean) | 1385 | PASS: 20-300 / FAIL: <10 or >600 | **FAIL** (too loose) |
| Cumulative P&L (5 agents × 3 days) | **−£8,675** | — | — |
| Win rate | 22.8% | — | — |

**Calibration table (BACK, predicted = pwin):**

| Pwin decile | n | predicted | realised | delta |
|---|---|---|---|---|
| 0.019-0.086 | 415 | 0.060 | 0.075 | +0.015 |
| 0.086-0.129 | 415 | 0.114 | 0.012 | -0.102 |
| 0.129-0.177 | 415 | 0.150 | 0.041 | -0.109 |
| 0.177-0.200 | 415 | 0.191 | 0.065 | -0.126 |
| 0.200-0.223 | 415 | 0.210 | 0.157 | -0.053 |
| 0.223-0.417 | 415 | 0.297 | 0.328 | +0.031 |
| 0.417-0.430 | 415 | 0.424 | 0.436 | +0.012 |
| 0.430-0.534 | 415 | 0.477 | 0.410 | -0.067 |
| **0.534-0.681** | 415 | **0.593** | **0.255** | **-0.338** |
| **0.681-0.825** | 415 | **0.720** | **0.487** | **-0.233** |
| 0.825-0.825 | 6 | 0.825 | 1.000 | +0.176 |

**Verdict A: FAIL — solid.** The predictor is significantly
over-confident in the 0.50-0.75 admitted band (predicted 59-72%,
realised 26-49%). The value-edge gate at threshold 0.05 admits
runners in this miscalibrated zone and they lose money in bulk.

## Probe B (lay-only)

| Metric | Result | Criterion | Verdict |
|---|---|---|---|
| Per-bet EV (mean of final_pnl) | **−£0.69** | PASS: >+0.50 / FAIL: <+0.10 | **FAIL** |
| Per-bet σ | £8.05 | — | — |
| Per-bet Sharpe | **−0.09** | PASS: >0.10 / FAIL: <0.05 | **FAIL** |
| Days profitable / 3 | **0/3** | PASS: ≥2 / FAIL: 0 | **FAIL** |
| Bets / day (mean) | 518 | PASS: 20-300 / FAIL: <10 or >600 | borderline |
| Cumulative P&L (5 agents × 3 days) | **−£1,080** | — | — |
| Win rate (lay-win = runner loses) | 71.2% | — | — |

**Calibration table (LAY, predicted = 1 − pwin):**

| 1−Pwin decile | n | predicted | realised | delta |
|---|---|---|---|---|
| 0.225-0.785 | 155 | 0.500 | 0.361 | -0.139 |
| 0.785-0.845 | 155 | 0.828 | 0.748 | -0.079 |
| **0.845-0.888** | 155 | **0.868** | **0.587** | **-0.281** |
| 0.888-0.909 | 155 | 0.896 | 1.000 | +0.104 |
| 0.909-0.923 | 155 | 0.916 | 0.748 | -0.168 |
| 0.923-0.926 | 155 | 0.924 | 0.858 | -0.066 |
| **0.926-0.935** | 155 | **0.931** | **0.632** | **-0.299** |
| **0.935-0.960** | 155 | **0.948** | **0.665** | **-0.284** |
| 0.960-0.972 | 155 | 0.967 | 0.871 | -0.096 |
| **0.972-0.988** | 155 | **0.981** | **0.639** | **-0.342** |
| 0.989-0.989 | 5 | 0.989 | 1.000 | +0.011 |

**Verdict B: FAIL — small magnitude but clear sign.** Loss is
~£0.50–£0.78/bet across all 15 (seed, day) runs; not a single
sub-cohort recovered. The calibration table shows the predictor
is over-confident on lay-win probability in the 0.87–0.98
admitted band (predicted 87–98%, realised 59–87%) — almost
the entire admitted set sits in the miscalibrated zone. The
+0.05 edge floor was supposed to filter this out but didn't.

## Combined verdict + decision

Per the pre-registered decision table in `README.md::What
"success" looks like`:

- [x] **Both probes FAIL** → "scalping is the only mode where
  the project's pwin signal has tradeable edge." Update
  `memory/feedback_reliability_over_upside.md`; close the
  chapter on directional value betting at this predictor's
  current calibration.

## Why the probes failed (one-line diagnosis)

The predictor's win-probability output is well-calibrated at
the extreme deciles (very-low pwin and very-high pwin) but
**systematically over-confident in the middle deciles**, which
is exactly the region the value-edge gate admits — the gate
admits runners that LOOK +EV under the predictor's stated
probability but are actually closer to fair odds. Scalping
sidesteps the issue by capturing a bounded spread per pair
regardless of predictor calibration in the middle; directional
betting cannot.

## What this rules out and what it doesn't

**Rules out** (at this predictor calibration):

- Pure-directional back betting on any pwin gate of
  `pwin × P × (1−c) − 1 ≥ threshold` family
- Pure-directional lay betting on `(1−pwin) × (1−c) − pwin × (P−1) ≥ threshold`
  family at threshold 0.05 (including the lay-quality-gate
  proven price bucket)

**Does NOT rule out:**

- A re-calibrated predictor (Platt scaling / isotonic regression
  on held-out days) might shift the admitted set out of the
  miscalibrated middle deciles. The data exists for this — every
  probe bet log records `runner_champion_p_win` and
  `final_outcome`, so calibration curves are trivially derivable.
- Higher value-edge thresholds (0.10, 0.20, 0.50) might admit
  only the well-calibrated extreme deciles. Sample size shrinks
  fast — at threshold 0.50 you'd see maybe 5-15 bets/day. Not
  obviously a worse bet, but a different probe.
- Scalping-side improvements that EXPLOIT the calibration gap
  (e.g. tighten the scalping-side pwin gate to the proven
  extreme-decile bands).
- A different predictor architecture / training set.

## What's still useful

- `env/scalping_math.py::value_bet_edge` is correct and unit-
  tested; reusable for any future calibrated-predictor work.
- `env/betfair_env.py` value-edge gate + sizing override
  defaults to disabled = byte-identical on arb-mode cohorts.
  No need to revert.
- `tools/probe_directional.py` + `tools/analyse_directional_probe.py`
  are reusable for any future cohort-of-one-shot-bet variant
  (re-run with a re-calibrated predictor, or sweep thresholds).
- The per-bet JSONL logs at `registry/probe_{A_back,B_lay}/`
  are the calibration-curve raw data for any follow-on.

## Wall-clock summary

| Phase | Wall-clock |
|---|---|
| 0 Scaffold | ~15 min |
| 1 Sanity smoke | ~3 min |
| 2 Helper + env wiring + tests | ~1.5 h |
| 3 Pre-flight smoke | ~3 min |
| 4 Probe A (back, 5×3) | ~12 min |
| 5 Probe B (lay, 5×3) | ~12 min |
| 6 Analysis + findings | ~30 min |

**Total ~2.5 h** from scaffold to verdict, within the ~5 h
plan budget.

## References

- Plan: `plans/non-scalping-directional-probe/{README,master_todo,hard_constraints}.md`
- Probe A raw bets: `registry/probe_A_back/bets_*.jsonl`
- Probe B raw bets: `registry/probe_B_lay/bets_*.jsonl`
- Analysis JSON: `registry/probe_{A_back,B_lay}/_analysis.json`
- Predecessor (scalping side, same days, +EV proven):
  `plans/scalping-lay-quality-gate/findings.md`
