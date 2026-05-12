# Session 02 — Pre-flight smoke

Before committing 12h to a cohort, run a 30-minute smoke against
ONE day with the gate enabled. Verify three numbers against the
hard_constraints §3 thresholds. Pass → proceed to Session 03.
Fail → STOP the loop with a diagnostic.

This session exists ENTIRELY because of the operator's concern:
"I don't want to burn 12 hours to find out we're not using it."

## Deliverable

`tools/smoke_direction_gate.py` — a focused script that:

1. Loads ONE day (`2026-05-04`) + the production predictor bundle
2. Builds an env with `direction_gate_enabled=True`,
   `use_race_outcome_predictor=True`, `use_direction_predictor=True`,
   `predictor_lean_obs=True`, scalping mode, same pwin thresholds
   (back=0.20, lay=0.40)
3. Runs a uniform-random policy through the env to measure
   what the gate refuses (no PPO, no actual training — just
   counting refusals against a random sampler)
4. Builds a SECOND env with `direction_gate_enabled=False`
   (everything else identical) for a control comparison
5. Counts and prints a diagnostic table

## Diagnostic table format

```
DIRECTION-GATE SMOKE — 2026-05-04
==================================================================

POPULATION (regardless of policy):
  total (tick, runner) pairs ............ N
  drift fired ........................... N (X.X%)

LEGAL ACTIONS (post-mask) by gate config:
  baseline (pwin only):
    OPEN_BACK legal-tick-slot-count ..... N
    OPEN_LAY  legal-tick-slot-count ..... N
  with direction-gate:
    OPEN_BACK legal-tick-slot-count ..... N (delta: +N or -N)
    OPEN_LAY  legal-tick-slot-count ..... N (delta: -N)
    lay-legal ratio (with-gate / no-gate) X.X%

POLICY ROLLOUT (uniform-random over legal actions, 1 race
sample of 100 ticks):
  attempted opens BACK / LAY ............ N / N
  matched bets ........................... N
  refused-by-mask LAY (gate)............. N
  refused-by-matcher (book/cap) ......... N
==================================================================

VERDICT vs hard_constraints §3:
  drift_fire_rate ≥ 5%        ........... PASS / FAIL (actual X%)
  lay_legal_with_gate / no_gate ≤ 60%  ... PASS / FAIL (actual X%)
  bets_matched ≥ 50 (full day, est.) .... PASS / FAIL (estimate N)
```

The last block decides next action:
- ALL PASS → proceed to Session 03
- ANY FAIL → STOP loop

## Implementation outline

The smoke script's structure:

```python
def main(argv):
    args = parse(argv)  # --day, --device, manifests
    bundle = PredictorBundle.from_manifests(...)
    day = load_day(args.day, ...)

    cfg = scalping_train_config()
    cfg["training"]["strategy_mode"] = "arb"

    # Build TWO envs: with-gate and without-gate
    env_with = BetfairEnv(day, cfg, predictor_bundle=bundle,
                          use_race_outcome_predictor=True,
                          use_direction_predictor=True,
                          predictor_lean_obs=True,
                          predictor_p_win_back_threshold=0.20,
                          predictor_p_win_lay_threshold=0.40,
                          direction_gate_enabled=True)
    env_no = BetfairEnv(day, cfg, predictor_bundle=bundle,
                        use_race_outcome_predictor=True,
                        use_direction_predictor=True,
                        predictor_lean_obs=True,
                        predictor_p_win_back_threshold=0.20,
                        predictor_p_win_lay_threshold=0.40,
                        direction_gate_enabled=False)

    # Count drift fires in the population
    total_pairs = 0
    drift_fires = 0
    for race_idx in range(env_with._total_races):
        for (ti, sid), fired in env_with._tick_drift_fires_by_race[race_idx].items():
            total_pairs += 1
            if fired:
                drift_fires += 1

    # Walk legal-action counts across all ticks (no policy yet)
    legal_back_with = legal_lay_with = 0
    legal_back_no = legal_lay_no = 0
    env_with.reset()
    env_no.reset()
    # ... iterate ticks across all races; call compute_mask;
    # sum legal counts.

    # Run a short policy rollout on one race to measure
    # matched-bet count and refusal counts.
    # (use a uniform-random policy with no training)

    # Print diagnostic table.
    # Apply VERDICT logic against thresholds.
    # Exit 0 if PASS, 1 if FAIL.
```

The exact code is up to the implementer — what matters is the
diagnostic table format and the verdict logic.

## Wall-time budget

- Implementation: 30-45 min
- Smoke run: 5-10 min (one day of data, no PPO)
- Verdict: instant

If implementation drags past 1.5h, simplify the script — drop the
policy rollout, keep just the population/legal-action counts.
Those alone test the gate without needing the matcher.

## Acceptance

The smoke produces a complete diagnostic table on stdout. The
VERDICT line either says PASS (proceed to Session 03) or FAIL
(stop loop, write diagnostic to autonomous_run_log.md).

## On failure

The most likely failure modes:

1. **drift_fire_rate < 5%** — drift signal is much rarer than
   expected. The lay gate is over-tight. Consider relaxing —
   though this is a hard_constraint stop. Surface to operator.

2. **lay_legal ratio > 60%** — gate isn't doing material work.
   Most lay-legal ticks under pwin are still lay-legal under
   both gates. The drift signal correlates strongly with
   already-pwin-allowed lays. Either drift is too lenient or
   the two gates overlap perfectly. Surface for design review.

3. **bets_matched < 50** — agent is starved. Either too few
   legal opens OR matcher refusing what's allowed. Surface.

In any failure case, write a paragraph to autonomous_run_log.md
explaining what was measured vs expected, and stop the loop.
The operator decides whether to relax constraints or revisit the
design.

## Commit message template (on success)

```
tools(smoke_direction_gate): pre-flight diagnostic for direction gate

Counts drift fires, legal-action shifts vs no-gate baseline, and a
short policy rollout's matched-bet count. Verifies the gate
satisfies hard_constraints §3 before committing to a 12h cohort.

Smoke result on 2026-05-04:
  drift_fire_rate: X%
  lay_legal_ratio: X%
  bets_matched: N

VERDICT: PASS (proceeding to Session 03).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```
