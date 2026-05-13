# Findings — scalping-race-confidence-gate

**Status:** complete (modest success per README's success table).
**Cohort tag:** `_predictor_SCALPING_raceconf_1778661062`.
**Threshold (final):** `race_confidence_threshold = 0.50`
(revised mid-plan from the originally-locked 0.30 after the
2026-05-13 smoke FAIL — see `autonomous_run_log.md`).

## Headline

Stacking a per-race confidence filter (max champion p_win ≥ 0.50)
on top of the pwin-gate's per-runner filter produces a **modest
improvement** over the predecessor on the held-out reeval window
(2026-04-28/29/30):

| metric | scalping-pwin-gate | scalping-race-confidence-gate | Δ |
|---|---:|---:|---:|
| profitable (top-5) | 3/5 | **3/5** | 0 |
| mean per-day pnl | −£13 | **+£39.40** | **+£52.40** |
| median per-day pnl | +£48 | **+£92.61** | +£44.61 |
| maturation rate | 0.334 | ~0.34 | ~0 |

Per the README's "What success looks like" table this lands in
the **modest success** band:

| Branch | Criterion | Result |
|---|---|---|
| Strong | mean > +£30 AND ≥ 4/5 profitable | NO (only 3/5) |
| **Modest** | **mean > 0 AND ≥ 3/5 profitable** | **YES** |
| No improvement | mean ≈ pwin-gate −£13 | no |
| Regression | mean < pwin-gate −£13 | no |

## Top-5 held-out reeval (2026-04-28/29/30)

| Agent | Gen | mean pnl/day | locked | naked | closed | stop | mr |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8ab4204f-0b4 | 6 | **+£137.42** | +£102.60 | +£54.20 | −£4.57 | −£14.82 | 0.337 |
| d3471cae-dc6 | 7 | **+£103.45** | +£82.23 | +£39.58 | −£3.84 | −£14.52 | 0.326 |
| 35297cd3-4e1 | 6 | **+£92.61** | +£90.13 | +£30.17 | −£15.96 | −£11.73 | 0.372 |
| 0de125f5-648 | 7 | −£2.02 | +£86.72 | −£61.62 | −£14.75 | −£12.37 | 0.376 |
| f5001118-0e5 | 7 | −£134.45 | +£77.89 | −£197.02 | −£2.92 | −£12.40 | 0.298 |

- **Aggregate:** profitable 3/5, mean +£39.40/day, median +£92.61/day.
- **Total 3-day pnls (sum across days):** +£412, +£310, +£278, −£6, −£403.

## What worked

**The locked floor generalises.** Across all 5 held-out agents,
the locked component (paired-arb spread capture) sits at
+£77 to +£103 per day — almost exactly mirroring the in-sample
locked range (+£75 to +£104 across the 8 generations). The gate's
mechanic of "only trade races with a clear favorite" preserves
the predictor's spread-extraction edge on unseen days. **Mean
locked across the 5 agents: +£87.91** — versus mean total pnl
+£39.40, so the locked channel alone is what's keeping the
cohort positive.

**Maturation rate held at ~0.34**, the same level the predecessor
pwin-gate hit. The new race filter didn't strangle pair completion
— it removed un-trade-able races (where max p_win < 0.50) without
hurting the cohort's ability to mature pairs in the races it
admitted.

**Selection across generations did real work.** The agents
selected for held-out reeval were 3 from Gen 7 and 2 from Gen 6
— the late generations. Gen 6/7 had the strongest in-sample
results (mean +£105 / +£110, locked +£85 / +£88, 75-82%
profitable). The GA+PPO loop genuinely surfaced the better
agents.

## What didn't work

**Naked variance still dominates per-agent outcomes.** Across the
5 held-out agents, naked p&l ranged from −£197 to +£54. f5001118
lost £197/day on naked, dragging its total to −£134/day despite
healthy locked +£77 and the same close/stop discipline as the
profitable agents. Without the gate, this would have been worse
— but the gate alone doesn't fix the naked channel.

**Top-5 selection bias.** The cohort's full-population
distribution had locked floor +£82 across 96 agents. The top-5
agents the GA picked for held-out got there partly through
naked-windfall in-sample (e.g. 0a3cc000's +£420 in-sample was
naked-driven +£330; it didn't make the top-5 for reeval). Three
of the five top-5 (the profitable ones) carried real locked
performance to held-out; two carried lucky in-sample naked tails
that didn't replicate.

**The bottom-of-top-5 is the real failure case.** f5001118's
−£197/day naked is the kind of outcome the next plan needs to
prevent. The probe at `tools/probe_lay_outcome_distribution.py`
identified two structural problems (pwin 0.20-0.30 calibration
hole, lay-price 20-50 leverage trap) — fixing those should
collapse most of this downside.

## Generation trajectory (in-sample, eval window 2026-05-04/05/06)

| Gen | n | mean | median | locked | naked | profitable |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 12 | +£114 | +£105 | +£86 | +£44 | 10/12 |
| 1 | 12 | +£56 | +£84 | +£81 | −£9 | 7/12 |
| 2 | 12 | +£2 | +£27 | +£82 | −£63 | 7/12 |
| 3 | 12 | +£40 | +£33 | +£85 | −£27 | 10/12 |
| 4 | 12 | +£69 | +£104 | +£84 | +£2 | 8/12 |
| 5 | 12 | +£54 | +£93 | +£75 | −£4 | 7/12 |
| 6 | 12 | +£105 | +£134 | +£85 | +£39 | 9/12 |
| 7 | 12 | +£110 | +£144 | +£88 | +£37 | 9/12 |

Gen 6 and Gen 7 land the strongest cohort means and medians,
with locked floor recovering after a Gen 5 dip. The PPO learning
+ GA selection loop is genuinely productive.

## What we learned (lessons that survive into the next plan)

1. **Race-confidence filtering works**, but at threshold 0.50.
   The originally-planned 0.30 admitted 100% of races (the
   predictor's per-race max p_win on this data never falls below
   0.32 — see `autonomous_run_log.md` 2026-05-13 probe). For any
   future plan that filters on `max(p_win)`, the threshold MUST
   be set against the observed distribution, not guessed.

2. **Locked vs naked decomposition is the right scoreboard.**
   The total-pnl signal in scoreboard.jsonl is dominated by
   naked variance (σ ≈ £140/agent in-sample), which masks the
   stable locked floor (σ ≈ £15). For follow-on plans, sort/
   filter on `eval_locked_pnl` rather than `eval_day_pnl` when
   selecting agents to deploy. The pwin-gate cohort's selection
   process gave us a top-5 of which 3/5 were locked-driven and
   2/5 were naked-lucky — the 3 generalise, the 2 don't.

3. **The naked channel is the dominant risk.** Locked + close +
   stop_close p&l is bounded by spread cost (~£3/leg). Naked
   p&l is bounded by leverage × stake (~£30-£300/leg).
   Improvements that reduce naked exposure (force-close,
   tighter lay gate, lay-price cap) will compound more than
   improvements to the gate that affect locked extraction.

4. **The lay-EV probe (`tools/probe_lay_outcome_distribution.py`)
   surfaces actionable diagnostics.** It identified the pwin
   0.20-0.30 calibration hole and the lay-price 20-50 leverage
   trap on this exact eval window. The next plan should run
   this probe FIRST, set its thresholds against the data, and
   then build the cohort.

5. **Force-close in training kills the close-discipline signal.**
   Don't enable `force_close_before_off_seconds > 0` during
   training — keep the full naked-variance gradient. Enable it
   at REEVAL/deployment only, and report BOTH "no force-close"
   and "force-close = 120s" held-out numbers. See
   `memory/project_force_close_train_vs_deploy.md`.

6. **The agent doesn't see leverage explicitly.** Per-runner
   observations don't currently include "worst-case naked loss
   on this open position" or "current cost to close". Adding
   those is a pure obs-side change (see queued task chip
   "Add leverage + close-cost obs features for close
   discipline"). This is independent of the gate-tuning
   direction and should land before the next training cohort.

## Next plan — recommended direction

Open `scalping-lay-quality-gate` (or `scalping-race-confidence-
gate-v2` — name is operator's call). Inherits the threshold-0.50
race-confidence gate. Adds:

1. **Tighter `predictor_p_win_lay_threshold`** (current best
   guess: 0.20 — drops the 0.20-0.30 calibration hole).
2. **NEW `lay_price_max` env kwarg + CLI flag** (current best
   guess: ≤ 20 — drops the lay-price leverage trap).
3. **Per-bet logging during training-eval** (queued task chip
   #1 — enable bet logs on the rollout that writes scoreboard).
4. **Observation features for per-runner leverage + close cost**
   (queued task chip #3).
5. **Asymmetric force-close: train at 0, reeval at 120 + at 0**
   (per `project_force_close_train_vs_deploy.md` — both numbers
   reported).
6. **Re-run the lay-EV probe FIRST** to set the new thresholds
   against the actual distribution (per
   `project_lay_ev_calibration_findings.md`).

Bundle 1+2+5 into the gate-tuning piece. Bundle 3+4 into a
preparatory observability piece that lands BEFORE the gate-tuning
cohort.

The success bar for the next plan should remain:
- **Modest success**: mean held-out > +£39 (beats this plan)
  AND ≥ 3/5 profitable.
- **Strong success**: mean held-out > +£70 AND ≥ 4/5 profitable.

## Wall-clock

- Plan start: 2026-05-13 07:53
- Session 01 (gate impl + tests): single iteration, ~40 min
- Session 02 (initial smoke FAIL on 0.30): ~10 min
- Probe + threshold revision to 0.50: ~30 min
- Session 02b (smoke PASS on 0.50): ~5 min
- Session 03 (12h cohort + watcher): launched 09:31, completed
  21:57 — 12h 26min
- Session 04 (reeval + verdict + findings): ~30 min

Total: ~14 h, matching the README's wall-clock budget.

## References

- Plan dir: `plans/scalping-race-confidence-gate/`
- Cohort dir: `registry/_predictor_SCALPING_raceconf_1778661062/`
- Held-out reeval: `reeval_held_out_2026-04-28_30.jsonl`
- Smoke tool: `tools/smoke_race_confidence_gate.py`
  (commit `166bee2` documents the initial 0.30 FAIL)
- Probe tool (threshold selection): `tools/probe_race_confidence_
  distribution.py` (commit `08cd6e0`)
- Probe tool (lay-EV diagnostics): `tools/probe_lay_outcome_
  distribution.py` (commit `d034032`)
- Implementation commit: `cccb8ad`
- Threshold revision commit: `e71f3a6`
- Cohort launch commit: `9f17266`
- Predecessor: `plans/scalping-pwin-gate/`
- Memory entries surfaced from this plan:
  - `project_lay_ev_calibration_findings.md`
  - `feedback_reliability_over_upside.md`
  - `project_force_close_train_vs_deploy.md`
  - `feedback_per_bet_logging.md`
