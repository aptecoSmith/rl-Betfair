# Tensor-env feasibility — results

**STATUS:** EMPTY — fill as S0–S5 complete. Verdict at the bottom.

## S0 — multiprocess baseline (the wall to beat)
- Hardware: _cores / RAM-bw / GPU_ = …
- Anchor (from `tt_tick_002`): ~37.5 s/agent-day, 32-agent tranche ≈ 12 001 s,
  full 3-tranche ascent ≈ 36h. Controlled micro-run confirms: … s/agent-day.

## S1 — per-agent-day phase breakdown (%)
| phase | % of agent-day | batchable? |
|---|--:|---|
| policy forward | | yes (3A) |
| predictor/scorer obs | | amortize (S4a) |
| market/obs build | | amortize (S4a) / batch (3B) |
| env.step (match+settle+mask) | | hybrid (3C) |
| per-agent sampling | | NO (floor) |
| per-tick Python | | NO (floor) |
| PPO update | | partial |

## S2 — irreducible floor
- sampling + per-agent env-branch + per-tick Python = … s/agent-day
  ⇒ single-process ceiling ≈ …×.

## S3 — common vs rare matching fraction (representative slice)
- common-case (vectorizable): …% of (agent,tick) events
- rare: junk-edge …% · hard-cap …% · force-close …% · walk …% · unpriceable …%
- distribution note (fallback-heavy race types): …

## S4 — multipliers multiprocess can't capture
- (a) `f_ind` (agent-independent share, predictors INCLUDED): …%
      ⇒ amortization upper bound at N=64 ≈ …%
- (b) batched common-case matcher prototype: …× vs N canonical calls
      (routing/fallback overhead: … s)

## S5 — projected cohort wall + sensitivity
| common-case % \\ f_ind | 13% | 25% | 40% |
|---|--:|--:|--:|
| 60% | | | |
| 80% | | | |
| 95% | | | |
- projected full-ascent wall: … (vs 36h baseline) ⇒ …× over multiprocess

## VERDICT
**GO / MARGINAL / NO-GO** — _one paragraph: the number, where break-even sits,
and the recommended next action._
