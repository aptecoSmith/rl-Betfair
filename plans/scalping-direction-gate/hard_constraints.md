# Hard constraints

These are cross-session invariants. The autonomous-run loop
defaults to the recommendation in `master_todo.md` "After Session
NN" sections for any operator-decision point — but the constraints
below are inviolable. If progress would require violating any of
them, the loop STOPS and writes a stop-condition entry to
`autonomous_run_log.md`.

## 1. Default-off byte-identical

When `direction_gate_enabled=False` (the constructor default), the
env's `compute_mask` output must be **bit-for-bit identical** to
pre-plan behaviour. Regression test (Session 01 deliverable)
enforces this with a fixed seed + 100-tick rollout comparison.

## 2. Loud-fail on incompatible flags

`BetfairEnv.__init__` raises `ValueError` if
`direction_gate_enabled=True` but
(`use_direction_predictor=False` OR `predictor_bundle is None`).
We cannot gate on a signal the env isn't computing.

## 3. Pre-flight smoke MUST pass before cohort

The 30-minute smoke (Session 02) writes a diagnostic with three
specific numbers. ALL THREE must satisfy their thresholds:

| Metric | Threshold |
|---|---|
| `drift_fire_rate` = drift_fires / total_(tick, runner) | **≥ 5%** |
| `lay_legal_after_both_gates` / `lay_legal_after_pwin_only` | **≤ 60%** (gate is doing material work) |
| `bets_per_day` (matched, not attempted) | **≥ 50** (agent isn't starved) |

If any fails, do NOT launch the 12h cohort. Write the diagnostic
to `autonomous_run_log.md` and stop the loop.

## 4. Same configuration as predecessor pwin-gate cohort

The new cohort uses EXACTLY the same settings as the
predecessor except for the new direction-gate flag:

- 12 agents × 8 generations × 6 days (--days 6)
- seed 42, mutation_rate 0.2
- scalping mode, lean obs
- predictor bundle: same three production manifests
- 6 Phase 5 safety genes enabled (same set)
- pwin gate: back=0.20, lay=0.40
- (new) `--direction-gate-enabled`

No threshold tuning, no architecture changes, no new shaping
terms.

## 5. Held-out reeval against 2026-04-28/29/30

Same window the pwin-gate cohort uses. Hard-locked so the A/B vs
pwin-gate verdict is clean.

## 6. No new shaping reward, no new genes, no architecture change

This plan is pure: action-mask logic + plumbing + cache
population. Reward function, gene set, policy network, trainer —
all unchanged.

## 7. No premature stop

The loop does NOT stop early because the cohort looks bad
mid-flight. The verdict is determined by the held-out reeval,
which only runs after all 96 rows complete. In-sample regression
does NOT trigger a stop.

## 8. Watcher auto-fires reeval

When the cohort hits 96 rows, a background watcher (armed in
Session 03) automatically launches the reeval against
2026-04-28/29/30. No operator action required.

## 9. Direction signal interpretation is locked

The asymmetric gate semantics in `README.md` are LOCKED:

- OPEN_LAY requires `dir_fire_drift` (price predicted to rise)
- OPEN_BACK is NOT direction-gated (shorten signal is broken;
  no reliable down-direction signal exists)

Do not "improve" the gate mid-flight by adding shorten-based
filters or by inverting the semantics. The empirical evidence
for this asymmetry is in `tools/direction_predictor_accuracy.py`
(commit `00ba9b2`).

## 10. Loop ends only on these conditions

The autonomous run loop terminates on ANY of:

1. **Verdict written**: held-out reeval complete AND
   `findings.md` committed.
2. **Stop condition triggered**: pre-flight smoke fails, OR a
   constraint above is about to be violated, OR three consecutive
   loop iterations make no progress.
3. **Crash recovery needed**: cohort crashes mid-run; surface
   to operator (this is one of the legitimate stop conditions
   where operator input is genuinely required).
