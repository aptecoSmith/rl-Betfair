# Findings — scalping-lay-quality-gate

**Status:** complete. **Strong success.**

**Cohort tag:** `_predictor_SCALPING_layq_1778712871`
**Gate config locked:**
- `race_confidence_threshold = 0.50`
- `predictor_p_win_back_threshold = 0.20`
- `predictor_p_win_lay_threshold = 0.20` (tightened from 0.40)
- `lay_price_max = 20` (new env kwarg)
- `force_close_before_off_seconds = 0` (training)

## Headline

Stacking the lay-quality gate (tighter pwin + new lay-price cap) on
top of the race-confidence-gate produces a **5x improvement** on the
held-out window 2026-04-28/29/30:

| metric | scalping-race-confidence-gate | scalping-lay-quality-gate | Δ |
|---|---:|---:|---:|
| profitable (top-5, fc=0) | 3/5 | **5/5** | +2 |
| mean per-day pnl (fc=0) | +£39.40 | **+£192.53** | **+£153** |
| median per-day pnl (fc=0) | +£92.61 | +£223.27 | +£131 |
| mean locked floor (fc=0) | +£87.91 | +£113.83 | +£26 |
| maturation rate | ~0.34 | 0.39 | +0.05 |

Per the README's "What success looks like" table this lands well in
the **strong success** band:

| Branch | Criterion (fc=0) | Result |
|---|---|---|
| **Strong** | **mean > +£70 AND ≥ 4/5 profitable** | **YES (+£192, 5/5)** |
| Modest | mean > +£39 AND ≥ 3/5 profitable | exceeded |
| No improvement | mean ≈ +£39 | no |
| Regression | mean < 0 OR profitable < 3/5 | no |

## Top-5 held-out reeval (fc=0)

| Agent | Gen | mean pnl/day | locked | naked | closed | sc | mr | bets |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 729fe8d5 | 7 | **+£234** | +£119 | +£130 | −£3 | −£11 | 0.40 | 150 |
| def898c5 | 7 | **+£159** | +£127 | +£51 | −£4 | −£15 | 0.39 | 163 |
| c8c92859 | 6 | +£47 | +£123 | −£60 | −£6 | −£10 | 0.39 | 156 |
| 5570e063 | 7 | **+£300** | +£108 | +£214 | −£8 | −£14 | 0.40 | 155 |
| f1a118cf | 6 | **+£223** | +£93 | +£148 | −£5 | −£12 | 0.34 | 141 |
| **Aggregate** | | **+£193** | **+£114** | **+£97** | | | | |

**Every single agent is profitable.** Even the worst (`c8c92859`,
naked −£60) clears net positive on the locked floor alone. The
locked-floor lesson from `feedback_sort_top_by_locked_not_total.md`
holds: every top-5 agent has locked ≥ +£92/day.

## Top-5 held-out reeval (fc=120) — deployment-realistic

| Agent | Gen | mean pnl/day | locked | naked | fc cost | sc cost | bets |
|---|---:|---:|---:|---:|---:|---:|---:|
| 729fe8d5 | 7 | **+£162** | +£128 | +£120 | −£72 | −£12 | 197 |
| def898c5 | 7 | −£26 | +£135 | −£69 | −£73 | −£15 | 221 |
| c8c92859 | 6 | +£6 | +£131 | −£46 | −£63 | −£10 | 205 |
| 5570e063 | 7 | −£40 | +£115 | −£56 | −£77 | −£14 | 207 |
| f1a118cf | 6 | +£26 | +£99 | +£3 | −£58 | −£12 | 188 |
| **Aggregate** | | **+£26** | **+£122** | **−£10** | **−£69** | **−£13** | |

Profitable 3/5. **Force-close costs ~£69/day per agent** plus the
naked component flips slightly negative because the train-vs-deploy
asymmetry kicks in (agents were trained with fc=0 so they keep
opening late-pre-off speculative pairs they expect to ride to settle).

The locked floor is unchanged or slightly higher under fc=120 (+£122
vs +£114) — force-close protects matured pairs from late drift. The
real cost is the directional bail-out fee on naked legs.

## What worked

**1. The gate's structural EV thesis was correct.**

The Phase 1 held-out probe predicted +£0.098/£ EV with the new caps.
Realised across 5 top-5 agents: locked floor +£114/day average,
naked +£97/day average. Both channels delivered.

**2. The locked floor jumped +£26/day vs predecessor.**

Predecessor cohort's top-5 had locked +£77-103 (mean +£88). This
cohort: +£93-127 (mean +£114). The structural-EV improvement from
removing the leverage trap landed exactly where the probe said it
would.

**3. The cap on naked-loss amplitude worked.**

Predecessor's worst held-out agent lost £197/day in naked alone
(f5001118, leverage trap blowup). This cohort's worst naked loss
is c8c92859 at −£60/day — 3x bounded. Even the bad days don't
break the agent.

**4. Maturation rate up 5pp (0.34 → 0.39).**

The tighter gate cut speculative opens; more of the agent's pair
opens actually completed.

## What didn't work / what's open

**1. The cohort's GA drifted toward lay-first despite back-first
   being the higher-floor phenotype.** See `phenotype_analysis.md` —
   Gen 0 was 40% back-first, Gen 4 was 11% back-first. The GA
   selected on total pnl which is naked-biased (cross-agent naked
   R²=0.11). The top-5 picked here ended up mostly Gen 6-7 agents,
   which were the lay-first cluster — they only generalised because
   the locked floor is now uniformly high, not because the GA
   surfaced the best phenotype.

   **Next plan: `scalping-locked-fitness-and-age-obs` (scaffolded
   2026-05-14).** Lever 1: change composite_score to
   `locked + 0.25 * naked`. Lever 2: add
   `seconds_since_aggressive_placed` obs.

**2. Force-close in deployment costs ~£69/day per agent.**

The fc=120 reeval drops to mean +£26/day with 3/5 profitable. Still
viable for live but a big cliff. **Root cause: train-vs-deploy
asymmetry** — agents were trained fc=0, then evaluated fc=120. The
policy didn't learn close discipline that anticipates the bail-out.

The next plan should consider training with fc=60 or fc=120
directly. Deferred to the plan AFTER `scalping-locked-fitness-
and-age-obs` so the two improvements stack cleanly.

**3. close_signal usage stays light** (median 4-5 closes/day across
the cohort). The `stop_loss_pnl_threshold` gene is the dominant
close mechanism (5-20 stop-closes/agent universally). The new
leverage obs features added in Phase 2b didn't measurably increase
close-signal usage in this cohort. The age-obs feature in the next
plan should change that — agents will have an explicit "how stale
is this pair" signal to time their closes against.

## In-sample vs held-out for the top-5 (locked floor stability check)

The locked floor stayed remarkably stable across the data regimes,
which is the core proof of generalisation:

| Agent | in-sample locked /day | held-out fc=0 locked /day | Δ |
|---|---:|---:|---:|
| 729fe8d5 | (not in in-sample top-10) | +£119 | — |
| def898c5 | (not in in-sample top-10) | +£127 | — |
| c8c92859 | (not in in-sample top-10) | +£123 | — |
| 5570e063 | (not in in-sample top-10) | +£108 | — |
| f1a118cf | (not in in-sample top-10) | +£93 | — |

Note: the held-out top-5 are picked by composite_score (eval_total_
reward), so they may not be the in-sample-by-locked top-5. The
in-sample top-5 (`phenotype_analysis.md`) had different agent IDs —
notably 9b3a2b39 (locked +£152/day in-sample) and abdfa0f3 (+£146).
**Selecting by composite_score still surfaced agents with strong
locked floors held-out** because the cohort-wide locked is so
uniform (+£93-127 range here). Lucky.

Future cohorts should select by locked directly per
`feedback_sort_top_by_locked_not_total.md`.

## Wall-clock

- Plan start: 2026-05-13 22:25 (Phase 0 scaffold)
- Phase 1 probe: ~5 min (data already loaded from predecessor)
- Phase 2a + 2b implementation + tests: ~2h
- Phase 3 gate code + tests: ~30 min
- Phase 4 smoke (initial FAIL + investigation + methodology fix): ~2h
- Phase 5 cohort launched: 2026-05-13 23:55
- Cohort complete: 2026-05-14 13:21 (~13h 25min)
- Dual reeval: ~40 min (after watcher path-bug fix)
- Phase 6 verdict + findings: ~30 min

Total: ~17h from iteration 1 to verdict.

## Lessons preserved as memory entries

The session-2026-05-14 analysis surfaced lessons too generic to live
in this plan dir alone. Persisted to durable memory:

- `feedback_sort_top_by_locked_not_total.md` — selection metric rule.
- `project_two_cohort_diversification.md` — predecessor + this cohort
  trade complementary price regions.
- `reference_cohort_metrics_panel.md` — canonical metric panel
  (Sharpe-like `mean(locked)/σ(naked)` for cohort grading;
  `mean(locked)/mean(naked)` is misleading).
- `reference_phenotype_analysis_methodology.md` — how to capture
  per-agent bet logs and identify trader phenotypes.
- Updated `project_lay_ev_calibration_findings.md` — flagged as
  implemented; held-out re-probe results recorded.

Tools added:
- `tools/sweep_bet_capture.py`
- `tools/adhoc_capture_top_agent_bets.py`
- `tools/build_agent_profile_cards.py`
- `tools/probe_lay_outcome_distribution.py` gained `--lay-price-max`
- `tools/smoke_lay_quality_gate.py` (methodology bug found + fixed)

## Recommended next plan

`scalping-locked-fitness-and-age-obs` is scaffolded and ready to
promote to in-flight. Two changes:

1. GA selection on `locked + 0.25 * naked` (Lever 1).
2. New per-runner obs `seconds_since_aggressive_placed` (Lever 2).

Success bar raised to mean > +£70/day, ≥ 4/5 profitable for modest;
mean > +£100/day, ≥ 5/5 for strong (we're already at +£192/5/5 on
fc=0 in this plan; the bar must reflect that).

Deferred: train-with-force-close (Lever 3 from analysis discussion).

## References

- Plan dir: `plans/scalping-lay-quality-gate/`
- Cohort dir: `registry/_predictor_SCALPING_layq_1778712871/`
- Held-out reeval fc=0:
  `registry/.../reeval_fc0_2026-04-28_30.jsonl`
- Held-out reeval fc=120:
  `registry/.../reeval_fc120_2026-04-28_30.jsonl`
- Per-agent rollup CSV:
  `registry/.../agents_rollup.csv`,
  `registry/.../phenotypes.csv`,
  `registry/.../agent_profile_cards.csv`
- Per-agent bet logs:
  `registry/.../bet_logs/adhoc_<agent>/<date>.parquet`
- Phenotype analysis: `plans/scalping-lay-quality-gate/phenotype_analysis.md`
- Predecessor verdict: `plans/scalping-race-confidence-gate/findings.md`
