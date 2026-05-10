# Session 06 — Value-each-way smoke cohort

## Goal

Mirror Session 05 in `value_each_way` mode. Verify that the
each-way action surface (Session 04) supports cohort training
and that the policy can produce non-degenerate behaviour on
EW betting using champion `p_placed` as the prior.

## Context to read

- `plans/predictor-integration/strategy_modes.md` §"value_each_way".
- `plans/predictor-integration/comparison_protocol.md`.
- `plans/ew-settlement/purpose.md` — the EW settlement worked
  examples; useful for sanity-checking smoke P&L.
- Session 05's findings (`05_findings.md`) — apply lessons.
- Session 04's deliverable — confirm `bet.is_each_way`
  flips correctly for sample bets in the cohort logs.

## Deliverables

Same shape as Session 05:

| File | Touch |
|---|---|
| `plans/predictor-integration/session_prompts/06_findings.md` | NEW |
| Cohort logs/registry | GENERATE |

## Cohort spec

```yaml
strategy_mode: value_each_way
observations:
  use_race_outcome_predictor: true
  use_direction_predictor: false
  
training:
  cohort_size: 12
  generations: 6
  training_window: 4 days with at least N EW races each (filter
    out non-EW-race-only days at curate-window time)
  device: cuda
  
genes:
  predictor_feature_gain: [0.0, 1.0]
  each_way_edge_threshold: [0.02, 0.10]
  each_way_kelly_fraction: [0.0, 0.5]
  
  # Same defaults as Session 05
  fill_prob_loss_weight: 0.0
  mature_prob_loss_weight: 0.0
  risk_loss_weight: 0.0
  open_cost: 0.0
  ...
```

## Smoke success bar

Same shape as Session 05, plus EW-specific:

1. End-to-end 6 generations complete.
2. Episode JSONL well-formed.
3. At least 1 of 12 agents emits `bet_count > 0`.
4. **At least 50% of bets have `is_each_way == True`** for the
   active agents — confirms the policy is using the EW
   capability rather than reverting to straight-win.
5. `approx_kl` stable.

## Naive strawman

Run flat-£10 EW on champion's argmax(`p_placed`) for the
training and eval windows. The predictor's val/test
`pick_placed_rate` is 55–60%; on EW, the breakdown is:

- If runner WINS: full win odds + place fraction (very
  positive).
- If runner PLACES (didn't win): half-stake at place fraction
  + lose half-stake on win leg (mildly positive on long-odds,
  mildly negative on short-odds).
- If runner UNPLACED: lose full stake (negative).

Per `plans/ew-settlement/purpose.md` worked examples, the EW
math is well-defined. Compute the naive ROI for the eval
window using the EW formula; record both numbers in
`06_findings.md`.

## EW-specific risks

| Risk | Mitigation |
|---|---|
| Policy converges to straight-win bets in EW mode (`is_each_way == False` on most bets) | Smoke success bar §4 catches this; if it triggers, the policy decided EW wasn't worth the half-stake split. Surface in findings.md but don't auto-fail — it might be the right answer for some races |
| Non-EW races are too prevalent in the training window | Curate the window to EW-rich days (e.g. UK/IE flat handicaps, where EW is the norm); document the curation in findings.md |
| EW divisor variance across days/courses | Default 1/4 is most common; smaller divisors (1/3, 1/2) appear on short fields; the policy SEES `each_way_divisor` as a feature implicitly through `bet.effective_place_odds` derivations and learns |
| Lay-side EW behaviour | Per Session 04 operator decision: smoke is back-only. Lay EW is a follow-on experiment if back-only proves the mechanism |

## Hard constraints

Same as Session 05.

## Out of scope for this session

- Three-way comparison (Session 07).
- Lay-side EW betting (back-only smoke).
- Live-inference port.

## Findings template

Same as `05_findings.md` adapted for EW:

- Per-agent metrics including `is_each_way_bet_fraction`.
- Naive EW strawman comparison.
- Distribution of `effective_place_odds` across cohort bets
  (sanity check: should match the EW formula
  `(win_price - 1) / divisor + 1`).

## Operator decision after Session 06

If both smokes (05 + 06) pass: proceed to Session 07. If only
Session 05 passed: still proceed to Session 07 with two-way
comparison (arb baseline + value_win), and note `value_each_way`
mode as deferred or failed-with-signal. If Session 06 reveals
EW-specific bugs (e.g. `effective_place_odds` mis-computed),
hold Session 07 until fixes land.
