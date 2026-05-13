---
plan: scalping-race-confidence-gate
status: open
opened: 2026-05-13
predecessor: scalping-pwin-gate
sibling-attempt: scalping-direction-gate (smoke-failed 2026-05-12)
execution: fully autonomous (no operator interaction)
---

# Scalping race-confidence gate (stacks on pwin-gate)

## Why this plan exists

The `scalping-pwin-gate` cohort (`_predictor_SCALPING_pwingate_
1778571007`) met its success bar (3/5 profitable held-out, mean
−£13, median +£48 on 2026-04-28/29/30). Maturation rate jumped
from 0.235 → 0.334. The per-runner pwin gate works.

The sibling `scalping-direction-gate` plan tried to stack the
direction predictor as a per-(tick, runner) drift filter on top.
Its pre-flight smoke (commit `14ba2c7`) FAILED on
`drift_fire_rate` (2.38% vs ≥5% threshold). A follow-up
deterministic "lay whenever both gates permit" probe
(`tools/run_gated_lay_strategy.py`, 2026-05-13) confirmed that
even on the small admitted set, win rate is only 60% and 3-day
PnL is −£513 — the direction signal doesn't generalize to
race-outcome lay edge. Direction is a price-movement signal,
not a race-outcome signal.

**Race-confidence gate is the alternative second filter.** Instead
of per-tick "is this runner about to move?", it asks per-race "is
there ANY runner the champion predictor has high confidence
about?" If the answer is no (max `p_win` across the race is below
threshold), the entire race is skipped — no opens, no closes
(closes can still resolve naturally — see hard_constraints §11).

This mirrors the deterministic baseline that produced +28.9% /
+19.9% ROI on test set days 2/3 (`tools/run_predictor_strategy.py`)
— which required BOTH `edge > 0.05` AND `segment_strong_flag`,
i.e. a per-race "is this a race worth betting on" filter.

## Per-race confidence gate (design — locked)

For each race at construction time:

```
race_max_pwin = max(champion_p_win[sid] for sid in race.runner_metadata)
race_is_confident = race_max_pwin >= race_confidence_threshold
```

Then in compute_mask, for every (tick, slot) in a non-confident
race:

| Action | Rule |
|---|---|
| NOOP | always legal (existing rule) |
| OPEN_BACK | masked (race not confident — predictor has no view) |
| OPEN_LAY | masked (same) |
| CLOSE | masked (race-confident filter is INDEPENDENT of in-flight pair closes — see §11) |

**Wait — CLOSE handling.** The straightforward rule "mask
everything when race not confident" creates a problem: if the
agent opened a pair in a race that LATER becomes non-confident
(it shouldn't, p_win is computed once per race at construction
— so race confidence is constant for the race), they could be
trapped with an open pair they can't close.

Since `champion_p_win` is computed ONCE per race and CACHED, the
confidence flag is constant for the entire race. If a race is
non-confident, the agent never opens a pair there in the first
place. So masking CLOSE is fine — there will be no pair to
close.

**The gate is purely additive** — it never SETS a mask bit, only
clears them. Composes cleanly with the existing pwin gate.

## Hypothesis

The pwin gate's success was about back/lay direction. The
race-confidence gate is about race selection. In an "open" race
(say 8 runners all at p_win=0.13), the predictor essentially has
no view — laying any of them is equivalent to laying random.
In a "favorite-led" race (one runner at p_win=0.50, others
spread across 0.05-0.15), the predictor has high conviction —
laying the longshots is the predictor's actual edge.

Skipping non-confident races should:

1. **Reduce naked variance** — fewer "I have no idea who's going
   to win" bets that resolve randomly.
2. **Preserve the cohort's locked floor** — same arb mechanic on
   the races we DO trade.
3. **Improve held-out generalization** — by trading only races
   the predictor is confident about, we avoid bets where the
   predictor's calibration matters most (low-conviction races
   are where overfit shows up hardest).

Success bar: **≥3 of top-5 profitable on the same held-out window
(2026-04-28/29/30) AND held-out mean > pwin-gate's −£13**. We
beat the predecessor on both metrics, or the gate isn't adding
value.

## Threshold choice — default 0.50 (revised 2026-05-13)

The deterministic baseline uses `edge > 0.05` (champion p_win
minus implied-from-price), which on most prices means champion
p_win must be > ~0.15-0.25 to trigger. For a race-level gate,
we want at least one runner the predictor strongly favors.

**Initial default `race_confidence_threshold = 0.30` was a guess
and failed the pre-flight smoke** — the per-race max-p_win
distribution probed across 2026-05-01/02/04/05/06 (434 races)
showed `min(max-p_win) = 0.32`, so the 0.30 cut admitted 100% of
races and the gate was structurally inert. See
`autonomous_run_log.md` 2026-05-13 entries for the FAIL diagnostic
and the distribution probe (`tools.probe_race_confidence_
distribution`).

**Revised default `race_confidence_threshold = 0.50`** — lands
at the median of the observed max-p_win distribution (p50 =
0.5338 across the 5-day probe). Skips ~40% of races (legal_ratio
~60%, comfortably under §3's 80% bar), keeping ~60% of races as
the agent's training surface.

Observed distribution (5-day, 434 races):

| Quantile | max-p_win | Threshold candidate |
|---:|---:|---|
| p10 | 0.39 | 0.40 — too permissive (~16% skip) |
| p25 | 0.43 | 0.45 — borderline (~30% skip) |
| p50 | 0.53 | **0.50 — chosen (~40% skip)** |
| p75 | 0.61 | 0.60 — too strict (~72% skip) |

Smoke verifies the default produces enough trade-able races
(`race_qualification_rate ≥ 30%`) AND the gate does material
work (`legal_ratio ≤ 80%`) before launching the cohort.

## Autonomous execution

Single autonomous-run loop driven by
`session_prompts/00_autonomous_full_run.md`. No operator
interaction needed. Same shape as the direction-gate plan:

1. Session 01: implement env kwarg + per-race confidence cache +
   compute_mask filter + unit tests.
2. Session 02: pre-flight smoke against 2026-05-04. PASS gates
   the 12h cohort.
3. Session 03: launch cohort, watcher auto-fires reeval at 96
   rows.
4. Session 04: read held-out reeval, write findings.md, stop loop.

## Hard constraints

1. **Default-off byte-identical.** `race_confidence_threshold=0.0`
   (the constructor default) reproduces pre-plan behaviour
   bit-for-bit. Regression test enforces.
2. **Loud-fail on incompatible flags.** Env init raises if
   `race_confidence_threshold > 0` but
   `use_race_outcome_predictor=False`. We cannot read p_win we
   never computed.
3. **Pre-flight smoke MUST pass before cohort launch.** Three
   thresholds (hard_constraints.md §3); all must PASS.
4. **Same configuration as predecessor pwin-gate cohort.** 12
   agents × 8 generations × 6 days, same predictor bundle, same
   pwin thresholds (back=0.20, lay=0.40), same 6 safety genes.
   Only addition: `--race-confidence-threshold 0.30`.
5. **Held-out reeval against 2026-04-28/29/30.** Same window as
   predecessors so A/B is clean.
6. **No new shaping, no new genes, no architecture changes.**
   Pure action-mask + per-race cache.
7. **CLOSE-action handling.** Race-confidence is constant per
   race; non-confident races never open pairs; CLOSE-mask
   parallel to OPEN is safe because there will be no pair to
   close in those races.

## Out of scope

- Promoting `race_confidence_threshold` to a GA gene.
- Combining with the direction gate.
- Changing the pwin gate thresholds.

## What "success" looks like

Held-out reeval lands at:

- **Strong success**: mean held-out > +£30, ≥4/5 profitable
- **Modest success**: mean held-out > 0, ≥3/5 profitable
- **No improvement**: mean held-out ~= pwin-gate cohort's −£13
- **Regression**: mean held-out < pwin-gate's −£13

The next plan is decided by which branch lands:

- **Strong**: ship — connect top agent to ai-betfair shadow
  trading.
- **Modest**: tune threshold (try 0.40 or 0.20).
- **No improvement**: race-confidence filter doesn't help the
  RL agent (its policy already implicitly selects confident
  races via the per-runner pwin gate). Move on.
- **Regression**: skipping races starves the agent of training
  signal; cohort can't learn. Stop and diagnose.

## Wall-clock budget

- Implement + tests: ~1h
- Pre-flight smoke: ~30 min
- Full cohort: ~12h
- Held-out reeval: ~20 min

**Total: ~14h** from iteration 1 to verdict.

## References

- Predecessor `scalping-pwin-gate`:
  - `_predictor_SCALPING_pwingate_1778571007` registry
  - implementation commit `8589c82`
  - held-out verdict: mean −£13, 3/5 profitable, mr 0.334
- Sibling attempt `scalping-direction-gate`:
  - implementation commit `4fb7758`
  - smoke FAIL: drift_fire_rate 2.38% < 5% threshold
  - deterministic probe `tools/run_gated_lay_strategy.py`:
    60% win rate, −£513 over 3 days, lay-price asymmetry kills
    it; direction signal isn't a race-outcome signal
- Champion deterministic baseline:
  - `tools/run_predictor_strategy.py`: +28.9% / +19.9% test
    days 2/3 with `edge > 0.05 AND segment_strong_flag`
- Memory: `project_race_confidence_gate.md`
