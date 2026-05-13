# Hard constraints

Cross-session invariants. The autonomous-run loop defaults to the
recommendation in `master_todo.md` for any operator-decision
point — but the constraints below are inviolable. If progress
would require violating any of them, the loop STOPS and writes a
stop-condition entry to `autonomous_run_log.md`.

## 1. Default-off byte-identical

When `race_confidence_threshold=0.0` (the constructor default),
`compute_mask` output must be **bit-for-bit identical** to
pre-plan behaviour. Regression test (Session 01) enforces with
a fixed seed + 100-tick rollout comparison.

## 2. Loud-fail on incompatible flags

`BetfairEnv.__init__` raises `ValueError` if
`race_confidence_threshold > 0` but
(`use_race_outcome_predictor=False` OR `predictor_bundle is None`).
We cannot gate on a signal the env isn't computing.

## 3. Pre-flight smoke MUST pass before cohort

The pre-flight smoke (Session 02) on 2026-05-04 writes a
diagnostic with three numbers. ALL must satisfy:

| Metric | Threshold |
|---|---|
| `race_qualification_rate` = races where `max(p_win) ≥ threshold` / total | **≥ 30%** |
| `legal_with_race_gate / legal_with_pwin_only` (lay+back slot-ticks) | **≤ 80%** (gate must do material work) |
| `bets_matched` (full day, est. from uniform-random rollout) | **≥ 50** |

If any fails, do NOT launch the 12h cohort. Write the diagnostic
to `autonomous_run_log.md` and stop the loop.

## 4. Same configuration as predecessor pwin-gate cohort

The new cohort uses EXACTLY the same settings except the new gate:

- 12 agents × 8 generations × 6 days
- seed 42, mutation_rate 0.2
- scalping mode, lean obs
- predictor bundle: same three production manifests
- 6 Phase 5 safety genes enabled (same set)
- pwin gate: back=0.20, lay=0.40
- (new) `--race-confidence-threshold 0.30`

No threshold tuning, no architecture changes, no new shaping.

## 5. Held-out reeval against 2026-04-28/29/30

Same window the predecessor cohorts use. A/B comparison with
pwin-gate held-out result is clean.

## 6. No new shaping, no new genes, no architecture changes

Pure action-mask + per-race cache.

## 7. No premature stop on mid-flight cohort results

The verdict is determined by the held-out reeval, which only runs
after all 96 rows complete. In-sample regression mid-flight does
NOT trigger a stop.

## 8. Watcher auto-fires reeval

When the cohort hits 96 rows, the background watcher launched in
Session 03 automatically runs the reeval against 2026-04-28/29/30.

## 9. Race-confidence semantics are LOCKED

The rule is:

```
race_max_pwin = max(p_win across all runners in race)
race_is_confident = race_max_pwin >= race_confidence_threshold
```

Do not "improve" mid-flight by:

- using `p_placed` instead of `p_win`
- using `segment_strong_flag` instead of (or in addition to)
  `p_win`
- using a per-runner average instead of max
- adjusting `race_confidence_threshold` mid-cohort

Any of those is a new plan, not a mid-flight tweak.

## 10. Loop ends only on these conditions

The autonomous run loop terminates on ANY of:

1. **Verdict written**: held-out reeval complete AND
   `findings.md` committed.
2. **Stop condition triggered**: pre-flight smoke fails, OR a
   constraint above is about to be violated, OR three
   consecutive iterations make no progress.
3. **Crash recovery needed**: cohort crashes mid-run.

## 11. CLOSE-action handling on non-confident races

The gate masks OPEN_BACK, OPEN_LAY, AND CLOSE on every (tick,
slot) of a non-confident race. This is safe because:

- `race_max_pwin` is computed ONCE per race (champion's
  `predict_race` is per-race, not per-tick).
- The confidence flag is constant for the entire race.
- A non-confident race is non-confident from tick 0 — no opens
  ever fire there — so there's never a pair to close.

The agent can't get "trapped" with an open pair in a race that
flipped to non-confident, because the flag never flips. The
masking is uniform per race.

## 12. Composition with pwin gate (additive)

The race-confidence gate runs BEFORE the per-runner pwin checks.
If race is non-confident, ALL open/close actions are masked
regardless of pwin. If race IS confident, the existing pwin
filter runs unchanged on each runner.

The composition is logical AND:

```
OPEN_LAY legal at (race, slot) iff
   race_is_confident AND
   pwin[slot] <= lay_threshold AND
   (existing per-slot legality checks)
```

The race-confidence gate never makes a previously-masked action
legal. Pure restriction.
