# Hard constraints

Cross-session invariants. The autonomous-run loop defaults to
the recommendation in `master_todo.md` for any operator-decision
point — but the constraints below are inviolable. If progress
would require violating any of them, the loop STOPS and writes
a stop-condition entry to `autonomous_run_log.md`.

## 1. Default-off byte-identical for every new knob

When all new kwargs are at their default values — i.e.

- `lay_price_max = 0.0` (disabled)
- per-bet logging not requested by the caller
- `SCALPING_POSITION_DIM` extension columns all zero
  (no open positions)

— `compute_mask` output, env stepping, and reward streams must
be **bit-for-bit identical** to pre-plan behaviour. Regression
tests in Phase 2 and Phase 3 enforce.

## 2. Loud-fail on incompatible flags

`BetfairEnv.__init__` raises `ValueError` if
`lay_price_max > 0` but
(`use_race_outcome_predictor = False` OR
`predictor_bundle is None`). `lay_price_max` composes with the
pwin lay threshold; both require the predictor signal path to
be active.

`lay_price_max` is validated to lie in `[0, 1000]`. Values
above 1000 are likely a typo (Betfair's hard cap on lay price
is 1000); values below 0 are nonsensical.

## 3. Pre-flight smoke MUST pass before cohort

The pre-flight smoke (Phase 4) on 2026-05-04 writes a
diagnostic with four numbers. ALL must satisfy:

| Metric | Threshold |
|---|---|
| `race_qualification_rate` = races where `max(p_win) ≥ 0.50` / total | **≥ 30%** |
| `legal_with_lay_gate / legal_with_race_gate_only` (lay+back slot-ticks) | **≤ 80%** (gate must do material work) |
| `expected_per_£_lay_EV` on admitted set | **≥ −£0.05** (NEW — the whole point of this plan) |
| `bets_matched` (full day, est. from uniform-random rollout) | **≥ 50** |

If any fails, do NOT launch the 12h cohort. Write the
diagnostic to `autonomous_run_log.md` and stop the loop. The EV
threshold is the load-bearing one: if the gate-tuned admitted
set isn't +EV (or close to it), the lay-quality-gate
hypothesis is wrong and this plan should be paused, not
re-tuned mid-flight.

## 4. Same configuration as predecessor race-confidence-gate cohort

The new cohort uses EXACTLY the same settings except the new
knobs:

- 12 agents × 8 generations × 6 days
- seed 42, mutation_rate 0.2
- scalping mode, lean obs
- predictor bundle: same three production manifests
- 6 Phase 5 safety genes enabled (same set)
- `predictor_p_win_back_threshold = 0.20`
- `race_confidence_threshold = 0.50` (inherited, locked)
- (new) `predictor_p_win_lay_threshold` from Phase 1 probe
- (new) `--lay-price-max` from Phase 1 probe

No mid-flight threshold tuning, no architecture changes beyond
the obs-dim bump in Phase 2b, no new shaping.

## 5. Held-out reeval against 2026-04-28/29/30

Same window as predecessors. Reeval is run TWICE per cohort:

- `force_close = 0` (apples-to-apples vs predecessor)
- `force_close = 120` (deployment-realistic)

BOTH results are reported in `findings.md`. The verdict is
decided on the `force_close = 120` reeval (the deployment
number); the `force_close = 0` reeval is the A/B comparison.

## 6. No new shaping, no new genes, no architecture changes beyond Phase 2b

- Phase 2b WILL break architecture-hash (new obs columns) —
  this is correct-by-default behaviour; pre-plan weights
  cannot cross-load.
- No new shaped reward terms.
- No new GA genes. `lay_price_max` is a cohort-wide CLI flag,
  NOT a gene.

## 7. No premature stop on mid-flight cohort results

The verdict is determined by the held-out reeval, which only
runs after all 96 rows complete. In-sample regression mid-
flight does NOT trigger a stop.

## 8. Two watchers, both auto-fire reeval at 96 rows

- `auto_reeval_layq_no_forceclose.sh` — runs
  `tools/reevaluate_cohort.py` with no force-close override
  (reads cohort default = 0).
- `auto_reeval_layq_forceclose120.sh` — runs with
  `--reward-overrides force_close_before_off_seconds=120`.

Both watchers share the same 96-row gate. Heartbeat once per
hour after launch.

## 9. Lay-quality gate semantics are LOCKED

The rule is:

```
OPEN_LAY legal at (race, slot, tick) iff
   race_is_confident AND
   pwin[slot] <= predictor_p_win_lay_threshold AND
   (lay_price_max == 0 OR current_LTP <= lay_price_max) AND
   (existing per-slot legality checks)
```

Do not "improve" mid-flight by:

- using best-back price instead of LTP
- using a per-runner running max instead of current LTP
- adjusting `lay_price_max` or `predictor_p_win_lay_threshold`
  mid-cohort

Any of those is a new plan, not a mid-flight tweak.

## 10. Loop ends only on these conditions

1. **Verdict written**: BOTH held-out reevals complete AND
   `findings.md` committed.
2. **Stop condition triggered**: pre-flight smoke fails ANY
   §3 threshold, OR a constraint above is about to be
   violated, OR three consecutive iterations make no progress.
3. **Crash recovery needed**: cohort crashes mid-run.
4. **Phase 1 probe finds no positive lay bucket exists** —
   stop and surface; this is a new plan, not a knob retune.

## 11. Force-close train vs deploy

- Training: `force_close_before_off_seconds = 0` (cohort
  launch flag NOT set). Preserves naked-variance signal.
- Reeval: BOTH 0 and 120 reported. Per
  `memory/project_force_close_train_vs_deploy.md`.

## 12. Composition with existing gates (additive only)

The lay-price cap runs AFTER the race-confidence gate and AFTER
the pwin lay threshold. If any earlier gate masks OPEN_LAY, the
cap is irrelevant. The cap never makes a previously-masked
action legal. Pure restriction.

```
OPEN_LAY legal iff
   race_is_confident AND
   pwin[slot] <= lay_threshold AND
   (lay_price_max == 0 OR LTP <= lay_price_max) AND
   (existing per-slot legality checks)
```

## 13. Per-bet logging is required, not optional

The training-eval rollout that writes scoreboard.jsonl rows
MUST also write `registry/<TAG>/bet_logs/<agent_id>.jsonl` for
every agent. Schema per
`session_prompts/02_obs_and_logging.md`. The bet-log writer
MUST be byte-identical-safe when disabled (no path provided).

## 14. Phase commit hygiene

One commit per phase, plus 2a and 2b are committed SEPARATELY
(variables must be separable for analysis):

- Phase 0: `plan(scalping-lay-quality-gate): scaffold next stack-on plan`
- Phase 1: `findings(scalping-lay-quality-gate): Phase 1 probe`
- Phase 2a: `feat(scalping-lay-quality-gate): per-bet logging on training-eval`
- Phase 2b: `feat(scalping-lay-quality-gate): per-runner leverage/close-cost obs`
- Phase 3: `feat(scalping-lay-quality-gate): lay_price_max env kwarg + plumbing`
- Phase 4: `findings(scalping-lay-quality-gate): pre-flight smoke verdict`
- Phase 5: `plan(scalping-lay-quality-gate): cohort launched + watchers armed`
- Phase 6: `findings(scalping-lay-quality-gate): held-out verdict`

Do NOT push to origin; commit locally only.
