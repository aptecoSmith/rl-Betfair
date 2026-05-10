# Session 05 — Value-win smoke cohort

## Goal

Run the first real cohort in `value_win` mode. 4-day training
window, 12-agent population, 6 generations on GPU. Verify that
the policy can produce non-degenerate behaviour with predictor
signals enabled and the value-win reward shape (settle-only).

## Context to read

- `plans/predictor-integration/strategy_modes.md` §"value_win".
- `plans/predictor-integration/comparison_protocol.md` §"Real
  cohorts" + §"Acceptance gates".
- `plans/predictor-integration/hard_constraints.md` §3, §8.
- CLAUDE.md §"BC pretrain" — decide whether to use BC for value
  modes.
- CLAUDE.md §"Entropy control" + §"alpha_lr as per-agent gene"
  — entropy targeting for the smaller (4-dim/runner) action
  surface.

## Deliverables

| File | Touch |
|---|---|
| `plans/predictor-integration/session_prompts/05_findings.md` | NEW — operator-readable summary of the cohort outcome |
| Cohort logs/registry | GENERATE |
| Optional: gene range adjustments based on smoke evidence | MODIFY — propose changes to `training_v2/cohort/genes.py` if smoke data shows current ranges are wrong |

## Cohort spec

```yaml
strategy_mode: value_win
observations:
  use_race_outcome_predictor: true
  use_direction_predictor: false  # not needed for value-win; per-tick cost not justified
  
training:
  cohort_size: 12
  generations: 6
  training_window: 4 days disjoint from eval window (per data-dir-dependent select_days memory)
  device: cuda  # always GPU per memory
  
genes:
  # Standard ranges
  predictor_feature_gain: [0.0, 1.0]
  value_edge_threshold: [0.02, 0.10]
  value_kelly_fraction: [0.0, 0.5]  # capped at 0.5 for smoke; bigger Kelly genes for real run
  
  # Aux head weights default to 0 in this cohort — predictors carry the discrimination
  fill_prob_loss_weight: 0.0
  mature_prob_loss_weight: 0.0
  risk_loss_weight: 0.0
  
  # No scalping reward shaping
  open_cost: 0.0
  mature_arb_bonus_weight: 0.0
  naked_loss_scale: 1.0
  force_close_before_off_seconds: 0
  mark_to_market_weight: 0.0
  
  # Entropy controller — value-win action surface is 4-dim/runner; lower target than arb's 150
  target_entropy: 80  # tentative; adjust if controller saturates
```

## Smoke success bar

The cohort run is "smoke-success" if:

1. End-to-end completes 6 generations without crash.
2. Episode JSONL is well-formed (no NaN, no Inf, all expected
   fields present including new `strategy_mode` and predictor
   `experiment_id` columns).
3. At least 1 of 12 agents emits `bet_count > 0` averaged
   across the training window. (Zero-bet collapse is the
   symptom of a starved gradient signal.)
4. `approx_kl` stays within a reasonable range
   (< 1.0 per mini-batch on the median update).

## Naive-strawman comparison

After the cohort completes, run the predictor's flat-£10
argmax(`p_win`) on the same training window AND the same eval
window. Per the manifest: ~29% hit rate, +18.6% ROI on test.
Record both numbers in `05_findings.md` so the eventual
three-way comparison (Session 07) has the strawman ready.

## Findings template

```markdown
# Session 05 findings — value-win smoke cohort

## Cohort identity
- registry tag: ...
- predictor_champion_experiment_id: 1c15250ee90d1b65
- predictor_ranker_experiment_id: b23018bf5c8bcc70
- training window: ...
- eval window: ...

## Smoke success bar
- [ ] 6 generations completed
- [ ] episode JSONL well-formed
- [ ] ≥ 1 agent with bet_count > 0
- [ ] approx_kl stable

## Per-agent metrics
| agent | bet_count | raw_pnl_reward | hit_rate | edge_realised |
|...|

## Naive strawman
- training-window flat-£10 argmax(p_win) ROI: ...%
- eval-window flat-£10 argmax(p_win) ROI: ...%
- best agent training ROI: ...%
- best agent eval ROI: ...%

## Verdict
- Smoke: pass / fail
- Path forward: continue to Session 06 / iterate cohort spec / escalate
```

## Hard constraints

- §3 (no new shaped rewards): if signal too sparse to learn,
  flag for operator review BEFORE adding shaping.
- §8 (three modes separately): no mixing in this cohort.

## Out of scope for this session

- Real comparison against arb baseline (Session 07).
- Hyperparameter tuning beyond the smoke ranges.
- Each-way mode work (Session 06).

## Operator decision after Session 05

If smoke fails (zero `bet_count` across all 12 agents): pause
Session 06, decide:

1. Reward shape too sparse → revisit hard_constraints §3.
2. Action-surface initialisation too cold (no exploration) →
   revisit entropy controller settings.
3. Predictor signal not reaching policy correctly → revisit
   Session 02 wiring.

If smoke passes: continue to Session 06.
