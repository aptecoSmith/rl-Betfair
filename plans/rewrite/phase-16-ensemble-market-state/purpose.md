---
plan: rewrite/phase-16-ensemble-market-state
status: DRAFT
opened: 2026-05-09
parent: plans/rewrite/phase-15-direction-head-feature-slice
trigger: phase-15 big run revealed high day-by-day variance — 1/12
         agents above the 35% break-even mat rate; the rest spread
         either NOOP or low-precision. v8's positive single-day pnl
         did not generalise to a 3-day eval window.
---

# Phase 16 — Ensemble + market-state features

## Why this plan exists

Phase 15 validated **the predictor-as-feature architecture**:
multi-day BC produces a calibrated direction predictor; freeze
preserves the calibration through PPO; the gate at the upper
tail of predictions delivers selective trading.

But the big run exposed a real limit: **the upper tail isn't
stable across days.** One agent at T=0.75 hit 40% mature rate
across 3 eval days (above the 34.8% break-even). Another agent
at the same threshold hit 9%. Day 1 (05-04) was bet-active but
loss-making; day 3 (05-06) was NOOP for most agents.

The strategic mental model is correct: most of the time the
"box" stays dark; when it lights up rarely, you bet. But the
box's **reliability shifts day-to-day**, and the predictor
alone can't tell which kind of day it's on. That's what phase
16 fixes.

## What we're building

Two complementary mechanisms that attack the variance from
different angles:

### Mechanism 1 — Ensemble (Sessions S01)

Train **K = 5 independent direction predictors** during BC,
each with a different random seed. At inference, the gate
fires only when **all K predictors agree** the runner crosses
threshold (`min(P_back_k) >= T` for back, similarly for lay).

The intuition: when the input pattern matches what the
predictors were trained on, all K converge on similar outputs
and they agree. When the input is unfamiliar or noisy, the K
predictors' independent random seeds produce different outputs
and they disagree → don't bet.

This is **implicit uncertainty** via consensus. No new loss
function, no calibration tracking — just K parallel heads with
an AND-gate. Cost: 5× BC compute per agent (~5 minutes added
per agent), 5× direction-head parameters (still tiny vs the
LSTM).

### Mechanism 2 — Market-state features (S02) and cross-runner features (S03)

Phase-15's predictor reads only the runner's own feature slice
(125-dim per runner). It can't see:

- **What today's market is like overall** (volatile? thin
  book? unusual volume?). A human scalper feels this and trades
  more cautiously on weird days. The predictor has no sense of
  it.
- **What other runners are doing** (cross-runner money flow).
  A human scalper might think "horse A will shorten" pre-race,
  then mid-race notice "horse C is actually getting all the
  money — I should be on C." The current per-runner predictor
  for horse A keeps voting on horse A and never notices horse C
  is the action.

S02 adds market-state features (broadcast across runners):
- Rolling LTP volatility across the field (last N ticks)
- Rolling traded volume vs day average
- Mean ladder depth at top of book
- Mean spread width

S03 adds cross-runner features (per-runner, but RELATIVE to the
field):
- `volume_rank_in_field`: this runner's rank (1..N) by recent
  traded volume
- `volume_share_in_field`: fraction of total race volume going
  to this runner in last N ticks
- `ltp_velocity_zscore`: how fast this runner is moving relative
  to the field's mean velocity
- `field_concentration`: HHI of money distribution across the
  field (high = action concentrated on 1-2 horses; low =
  spread)

These cross-runner features force every per-runner prediction
to incorporate "where is the action actually happening?"
The horse-A predictor sees that horse C has volume rank 1 and
volume share 60% → it lowers its confidence on A and the gate
correctly stays dark for A even if A's own features look
modestly favourable.

## Sessions

| Session | Deliverable | Wall budget |
|---|---|---|
| S01 | Ensemble of K=5 direction predictors; gate uses min-of-K agreement | code: ~2h; smoke: ~30 min |
| S02 | Market-state features added to obs (broadcast across runners) | code: ~2h; smoke: ~30 min |
| S03 | Cross-runner features added per runner | code: ~2h; smoke: ~30 min |
| S04 | Validation cohort (8 agents × 2 gens × 5 train + 3 eval) | wall: ~6h |

Total: ~12 hours. Each session ships independently; intermediate
smokes report mature rate (NOT single-day pnl, per phase-14
lesson) on 3 held-out eval days as the success metric.

## Hard constraints

See [hard_constraints.md](hard_constraints.md). Highlights:

- **Headline metric is mature rate aggregated across multiple
  eval days, not single-day pnl.** Phase-14 lesson: per-day pnl
  variance is ±£600; single-day signals fool you.
- Each session ships in isolation (S01 alone, S02 alone, etc.)
  so failures are diagnosable. Combine in S04.
- Ensemble's K is fixed at 5 in S01; can be tuned later as a
  gene if results merit.
- Market-state and cross-runner features are pure feature-
  engineering additions — RUNNER_KEYS expands, OBS_SCHEMA_VERSION
  bumps, oracle/direction caches need re-scan.

## What this is NOT

- **Not a new training algorithm.** PPO + BC + freeze stack from
  phase-15 stays. Phase 16 adds K predictors and richer features.
- **Not a label-spec change.** Direction labels stay binary
  threshold-crossing at horizon=60.
- **Not a magnitude target.** That's deferred (was hypothesis
  #3 in phase-13).
- **Not the rolling-calibration tracker (option 2 in the
  scalping-box discussion).** That mechanism — track recent
  prediction accuracy intra-day, downscale gate when calibration
  is bad — is held in reserve. It's harder (60-tick lag on
  labels), more complex, and might not be needed if S01-S03
  deliver. See `lessons_learnt.md` "Deferred: rolling
  calibration tracker".

## Success bar (per session)

- **S01 (ensemble alone):** smoke run with K=5 ensemble shows
  fewer false-fire bets than K=1 baseline, AND mat rate at the
  fired bets is at or above the K=1 mat rate. Validates that
  consensus filters noise without sacrificing precision.

- **S02 (market-state features alone):** smoke shows the
  predictor's BCE on held-out days improves over baseline by
  any margin. (Even small lift is meaningful at this scale.)

- **S03 (cross-runner features alone):** as S02; specifically
  test on race days with concentrated money flow on a single
  runner — predictor should NOT light up for the wrong runners
  on those days.

- **S04 (full stack):** validation cohort with 8 agents × 2
  gens × 5 train + 3 eval days. Headline:
  - Mat rate aggregated across 3 eval days
  - At least 25% of agents above the 35% break-even
  - Mean mat rate across surviving agents above 35%

## Connection to operator's mental model

S01 (ensemble) maps to: "the box only lights when I'm sure."
Multiple internal predictors all agreeing = high confidence.

S02 + S03 (richer features) map to: "I read the market
texture, not just the horse in isolation." Just like a human
scalper notices "the action is on horse C, not horse A" — the
model now sees the same field-level information.

(S04 just tests that all of it works at cohort scale.)
