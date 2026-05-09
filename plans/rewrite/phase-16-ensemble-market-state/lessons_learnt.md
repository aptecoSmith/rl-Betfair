---
plan: rewrite/phase-16-ensemble-market-state
parent_purpose: ./purpose.md
---

# Lessons learnt

Append-only journal. Inherited lessons land first; per-session
entries land as work completes.

## Inherited from phase-15

- Read `plans/rewrite/phase-15-direction-head-feature-slice/
  lessons_learnt.md` end-to-end before starting any session.
- The BC + freeze + multi-day pipeline IS validated. Don't
  unpick it. Phase-16 builds ON TOP.
- LayerNorm on the head's input is mandatory (raw obs scales
  saturate kaiming-init weights without it).
- Detach direction_prob from actor_input is mandatory (prevents
  PPO from un-calibrating the frozen head's read-path through
  actor surrogate).
- BC pos_weight=true (default) calibrates the head to a
  rebalanced distribution that aligns with gate T~0.85; vanilla
  BCE calibrates to natural distribution and needs T~0.55.
  Phase-16 inherits pos_weight=true.

## Inherited from phase-14

- Per-day pnl variance is ±£600. Single-day eval pnl is NOT a
  reliable signal. **Headline must be mature rate, not pnl.**
- Phase-14 break-even mat rate (per-pair £3.37 mat / £1.80
  force-closed): **34.8%**.

## Phase-15 big run finding (2026-05-09)

Cohort `_phase15_big_1778282572` (8 agents × 2 gens × 5 train
+ 3 eval, gate evolved [0.5, 0.95]). 12/16 agents completed
before PC restart.

**Mat rates by gate threshold:**

| Threshold | Behaviour | Best mat% |
|---|---|---|
| ≥ 0.81 | NOOP entirely | — |
| 0.74-0.75 | 16-29 bets, low matur | 40% (1/3 agents) |
| 0.57-0.60 | 400-500 bets | 28-33% |

Only 1/12 agents above the 35% break-even. Same-threshold
variance is huge: T=0.75 produced 40% (g0) and 9% (g1) mat
rates. Single calibrated predictor isn't enough — day-by-day
variance dominates.

**Conclusion that drives phase-16:** the predictor needs
either (a) consensus across multiple predictors so noise
filters out, (b) richer features so it can disambiguate
"good day" from "noisy day", or (c) both. Phase-16 does both.

## Operator's mental model (load-bearing for plan design)

The operator articulated the right mental model for the
predictor: a box that mostly stays dark; rarely lights up;
when it lights up, you should bet. Phase-15 v8 demonstrated
this works on a single eval day.

The operator's intuition about cross-runner regime shifts is
the load-bearing motivation for S03:

> "You will be looking at horse a thinking 'this is going to
> shorten' But then you notice it's horse c actually that is
> getting all the money. You watch horse c shorten and think
> - I should jump on that train, it's clearly coming in and
> will come in more. Your guess based on data prerace was that
> horse a would shorten, but your observation in the race
> showed you should have been looking at horse c."

The current per-runner predictor for horse A reads only horse
A's features. It cannot notice "horse C is the action today."
Cross-runner features (S03) directly fix this:
volume_rank_in_field, volume_share_in_field, ltp_velocity_zscore,
field_concentration. With these in horse A's input slice, the
predictor knows "C has 60% of the volume; my prediction on A
should be lower confidence."

## Deferred: rolling intra-day calibration tracker

Documented but NOT in scope for phase-16. The mechanism:

- At tick T, the predictor outputs a confidence score
- 60 ticks later, we know if the prediction was right
- Rolling tracker computes recent-window prediction accuracy
- If accuracy drops below a bar → raise gate threshold for the
  rest of the day (or just stop betting)

Why deferred:
1. Requires plumbing labels back into the live agent at
   inference time (60-tick lag is workable but trickier).
2. The 60-tick lag means it can't react to a sudden regime
   shift, only a slow drift.
3. Needs the predictor to OUTPUT a confidence (not just a
   binary fire/no-fire), which is downstream of phase-16's
   ensemble choice.
4. Lower priority than the two mechanisms phase-16 attacks
   first (ensemble + features). If those don't close the
   day-to-day variance, calibration tracking goes next.

Operator agreed: "this does actually sound useful. Let's keep
it on ice for now but not forget the idea." Memorialised here.

## Methodological lesson — the metrics that matter

For any phase-16 session report, lead with:

1. **Mature rate aggregated across all eval days** (top-line).
2. **Force-close rate** (proxy for selectivity quality).
3. **Bets per agent per day** (informs whether selectivity is
   working).

Eval pnl reported but with caveat. Direction BCE on cached
labels is a sanity check (validates the head still calibrates),
not a strategic metric.
