# Session 02 — Pre-flight smoke

Before committing 12h to the cohort, verify the gate doesn't
starve the agent. Three numbers, three thresholds. Binary
PASS/FAIL.

This is the same discipline that saved 12h on the direction-gate
plan (smoke caught drift_fire_rate at 2.38% vs 5%; loop stopped
before launching).

## Deliverable

`tools/smoke_race_confidence_gate.py` — counts race
qualification, legal-action shift vs no-gate baseline, and a
uniform-random rollout's matched-bet count.

## Diagnostic table format

```
RACE-CONFIDENCE-GATE SMOKE — 2026-05-04
==================================================================

POPULATION (regardless of policy):
  total races ........................... N
  races confident (max p_win >= 0.30) .... N (X.X%)
  races skipped ......................... N

LEGAL ACTIONS (post-mask) by gate config:
  baseline (pwin only):
    OPEN_BACK legal-slot-tick count ..... N
    OPEN_LAY  legal-slot-tick count ..... N
  with race-confidence gate:
    OPEN_BACK legal-slot-tick count ..... N (delta: -N)
    OPEN_LAY  legal-slot-tick count ..... N (delta: -N)
    legal-tick ratio (with/no race gate)  X.X%

POLICY ROLLOUT (uniform-random over legal, 1 day):
  attempted opens BACK / LAY ............ N / N
  matched bets .......................... N
  refused-by-mask (race-gate) ........... N
  refused-by-matcher (book/cap) ......... N
==================================================================

VERDICT vs hard_constraints §3:
  race_qualification_rate >= 30%  ....... PASS / FAIL (actual X%)
  legal_ratio <= 80% (material work)  ... PASS / FAIL (actual X%)
  bets_matched >= 50 (full day est.)  ... PASS / FAIL (estimate N)
```

ALL PASS → proceed to Session 03.
ANY FAIL → STOP loop.

## Implementation outline

```python
def main(argv):
    args = parse(argv)  # --day, --threshold, --device, manifests
    bundle = PredictorBundle.from_manifests(...)
    day = load_day(args.day, ...)
    cfg = scalping_train_config()
    cfg["training"]["strategy_mode"] = "arb"

    # Two envs: with-gate and without-gate
    env_with = BetfairEnv(day, cfg, predictor_bundle=bundle,
                          use_race_outcome_predictor=True,
                          predictor_lean_obs=True,
                          predictor_p_win_back_threshold=0.20,
                          predictor_p_win_lay_threshold=0.40,
                          race_confidence_threshold=0.30)
    env_no = BetfairEnv(day, cfg, predictor_bundle=bundle,
                        use_race_outcome_predictor=True,
                        predictor_lean_obs=True,
                        predictor_p_win_back_threshold=0.20,
                        predictor_p_win_lay_threshold=0.40,
                        race_confidence_threshold=0.0)

    # Race qualification
    total_races = len(env_with.day.races)
    confident_races = sum(env_with._race_is_confident_by_race)

    # Walk legal counts across all ticks for both envs
    legal_back_with = legal_lay_with = 0
    legal_back_no = legal_lay_no = 0
    # ... per-tick compute_mask, sum

    # Uniform-random rollout on env_with to measure matched bets
    # ... rollout 1 day, count matches

    # Print + verdict
```

Use the same shape as `tools/smoke_direction_gate.py` if it
exists (Session 02 of the direction-gate plan should have written
it). Save typing.

## Wall-time budget

- Implementation: 30 min (clone direction-gate smoke + adjust)
- Smoke run: 5-10 min
- Verdict: instant

## On failure

Most likely failure modes:

1. **race_qualification_rate < 30%** — threshold 0.30 too tight
   on this day's race distribution. Surface, STOP. Do NOT lower
   threshold mid-session — that's a constraint violation
   (§9 locks the threshold).
2. **legal_ratio > 80%** — gate isn't doing material work; most
   races qualify. Less alarming than (1) but suggests the
   threshold isn't selective enough.
3. **bets_matched < 50** — agent starved. Probably driven by
   (1).

Write paragraph to autonomous_run_log.md and stop.

## Commit message template (on success)

```
tools(smoke_race_confidence_gate): pre-flight diagnostic + verdict

Counts race qualification, legal-action shift vs no-gate, and a
short rollout's matched-bet count. Gates the 12h cohort behind
hard_constraints.md §3 thresholds.

Smoke result on 2026-05-04:
  race_qualification_rate: X%
  legal_ratio: X%
  bets_matched: N

VERDICT: PASS (proceeding to Session 03).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```
