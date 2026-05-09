---
plan: rewrite/phase-16-ensemble-market-state
session: S04
deliverable: Validation cohort with full S01+S02+S03 stack
depends_on: S01, S02, S03 all landed and individually smoke-validated
---

# S04 — Validation cohort

## Goal

Run a multi-day cohort with the FULL stack (ensemble + market-
state features + cross-runner features) and validate that the
mature rate exceeds break-even consistently across multiple
eval days.

## Cohort shape

- 8 agents × 2 generations = 16 agent-runs
- 5 train days + 3 eval days
- Gate threshold gene-evolved [0.5, 0.95]
- BC pretrain 2000 steps with K=5 ensemble heads
- All other phase-15 settings preserved (BC + freeze + pos_weight=true)
- Wall budget: ~6-7 hours

Mirror `scripts/phase15_big_run.sh` exactly with the new code
in place. The script needs no changes — the ensemble + features
are built into the policy / env directly.

## Pre-launch checks

1. Oracle caches at v8 (or whatever post-S02/S03 OBS_SCHEMA_VERSION)
   for ALL train days (04-29 to 05-03, plus eval days for the
   feature engineer to compute cross-runner features at eval time).
2. Direction labels at the new schema version.
3. All 14 phase-16 tests pass (S01: 5, S02: 4, S03: 5).
4. Quick smoke (2 agents × 1 gen × 3 train + 1 eval) confirms no
   crashes with the full stack.

## Headline metrics (mat-rate-first)

Per the lesson from phase-14/15: **eval pnl on a single day is
noise**. Headline metrics for S04:

1. **Mat rate aggregated across 3 eval days, per agent**:
   `(matured + closed) / pairs_opened` summed across the 3 days,
   then divided. Expected per-agent variance is moderate; cohort
   distribution is informative.

2. **Force-close rate per agent**: `force_closed / pairs_opened`.
   Lower is better (means selectivity is working).

3. **Bets per agent per day**: too high = gate too loose; zero =
   gate too tight; sweet spot is 30-100/day for a calibrated
   selective agent (matches v8 single-day pattern).

4. **Ensemble agreement rate**: fraction of (tick, runner) where
   all K=5 predictors agree the runner crosses threshold. New
   diagnostic — adds visibility into how often consensus actually
   fires.

Eval pnl reported but treated as informative-not-load-bearing.

## Success bar

Compared to phase-15 big run (1/12 above 35% break-even):

- **Primary**: at least 4/16 agents above 35% mat rate (4× the
  phase-15 hit rate)
- **Secondary**: cohort mean mat rate ≥ 30% (vs phase-15 ~22%)
- **Tertiary**: best agent's mat rate ≥ 50% on 3-day eval (vs
  phase-15 best of 40%)

## What to watch

1. **Ensemble agreement rate trajectory**: starts low (untrained
   heads disagree) and climbs through training. If it stays
   uniformly low, ensemble isn't converging. If it stays
   uniformly high, the ensemble is collapsing to a single
   consensus point (bad — defeats the purpose). Healthy
   trajectory: 5-30% agreement at gate threshold by gen 2.

2. **Mat rate variance ACROSS agents at the same gate threshold**:
   phase-15 big run had T=0.75 → 40% (g0) and 9% (g1). If
   phase-16 narrows this spread, that's evidence the ensemble +
   features are reducing day-to-day variance.

3. **Cross-runner features' effect**: harder to test directly,
   but on race days with high HHI (concentrated money), low-
   share runners should NOT trigger gate fires. Look at the
   per-day eval bet count distribution.

## Failure modes and what they mean

- All agents NOOP: gate threshold range is too high for the
  ensemble's stricter min-of-K outputs. Lower the GA range to
  [0.4, 0.85] in a v2 run.
- All agents fire many low-precision bets: ensemble disagreement
  isn't filtering noise. Check ensemble agreement rate — should
  be ~5-30% at gate, NOT 80%+.
- Single agent dominates with high mat rate, all others fail:
  variance reduction didn't work; ensemble was too small or
  features weren't enough. Consider K=10 or escalating to phase
  17 (rolling calibration tracker).

## Done definition

- Cohort runs to completion (16 agents × 5 train + 3 eval days)
- Scoreboard analysis written up in `findings.md`
- `purpose.md` status field updated:
  - `SUCCESS` (4+/16 above 35% across 3 eval days)
  - `MARGINAL` (1-3/16 above; pivot decisions documented)
  - `NULL` (0/16 above; escalate to phase 17 with calibration
    tracker, OR re-investigate label spec)
- Single commit: `docs(rewrite): phase-16 S04 - validation cohort findings`
