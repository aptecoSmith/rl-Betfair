---
plan: predictor-integration
status: planning
opened: 2026-05-10
type: feature integration — wire betfair-predictors production models into the rl-betfair observation space and enable training across three strategy modes (arb / value / place)
---

# Predictor integration — wire `betfair-predictors` champions into the v2 RL stack

## What this plan does

Wire the three production champions from `betfair-predictors` into
`rl-betfair`'s observation space and add the trainer plumbing needed
to specialise an agent for one of three strategies:

| Strategy | What it does | Predictor signals it leans on | Action surface |
|---|---|---|---|
| **Arb** | Open back+lay pair on a runner; close on second-leg fill or T−N force-close. Already the active code path. | Direction predictor (`q10/q50/q90` per horizon, `fire_direction`). | `scalping_mode = True` (pair-based; current default). |
| **Value-win** | Single back/lay on a runner whose calibrated `p_win` exceeds market-implied probability by a margin. Hold to settle. | Race-outcome champion `p_win`. Ranker `top1_high_confidence_flag` + `softmax_share` for argmax variant. Champion's `segment_performance.json` for bucket routing. | `scalping_mode = False` + new `value_mode = True` (single-shot, hold to settle). |
| **Value-each-way** (operator-framed: "placers") | Single each-way bet on a runner whose calibrated `p_placed` exceeds the implied probability of placing. Hold to settle. EW settles half-stake on win + half-stake on place leg at fractional odds. | Champion `p_placed`. Champion `p_win` for the win-leg sizing. | Same single-shot surface as value-win, but `bet.is_each_way = True` is set at placement. Settlement uses existing `plans/ew-settlement/` infrastructure (already complete). |

The three modes share one observation schema (predictor outputs are
always present when the flag is on; modes simply differ in which
columns the policy weights). They share the env, matcher, bet
manager, registry, and frontend — only the action interpretation,
the reward gate, and (for place) the market-selection layer differ.

## What this plan does NOT do

- **Does not start a v3 repo.** The integration is structurally
  small: 4 race-level scalar columns per runner (champion `p_win` +
  `p_placed`; ranker `softmax_share` + `top1_high_confidence_flag`)
  plus existing direction-predictor outputs as live model calls.
  v2's `actor_input` concat pattern (CLAUDE.md §"fill_prob feeds
  actor_head", §"mature_prob_head feeds actor_head") accommodates
  per-runner external scalars by design. See [diagnosis.md](diagnosis.md)
  for the full v3-vs-integrate analysis.
- **Does not bundle the action-space rewrite.** `plans/rewrite/`
  is in flight at Phase 7 (AMBER) and is on a separate axis
  (action-space architecture, not observation content). The two
  plans compose: this plan's predictor outputs land in the same
  observation tensor v2's discrete-action policy already reads.
- **Does not change the env / matcher / bet_manager / data
  extractor for arb mode.** They are load-bearing and correct;
  `plans/rewrite/README.md` constraint §1 applies here too.
- **Does not redesign the supervised pipeline in
  `plans/rewrite/phase-0-supervised-scorer/`.** That's a separate
  internal scorer. The production champions REPLACE it as the
  per-runner discrimination signal feeding the policy; the
  internal scorer can stay or be retired in a follow-on plan,
  not bundled here.

## The load-bearing why (one paragraph)

The v2 stack has been training a pair-trade ("arbing") agent and
hitting a maturation-rate ceiling. The aux-head saga
(`fill_prob_head` → `mature_prob_head`) tried to give the policy
internal per-runner discrimination from RL gradients; Phase 7's
S03 leverage test showed the signal at weight 0.5 doesn't move
maturation_rate (mean delta −0.38 pp). Meanwhile in the
`betfair-predictors` repo a calibrated `p_win` model lands at
4.6% calibration gap and a ranker model picks the actual winner
in 69% of sealed-test markets. The discrimination signal v2 has
been trying to learn from sparse RL gradients exists, in
production, trained on dense supervised labels. **Wiring those
outputs into the observation is the cheapest test of the
"missing signal" hypothesis** — and the same wiring opens the
door to value-betting and place-betting strategies that v2's
arb-only training cannot reach.

## Recommended option (from [options_compared.md](options_compared.md))

**B+: Targeted integration into the existing repo, with a
strategy-mode switch added to the env and trainer.** Predictor
outputs become opt-in observation features behind
`observations.use_race_outcome_predictor: false`. A new
`training.strategy_mode: arb | value_win | value_each_way` config
key gates which reward shape and action interpretation the env
applies. Default is `arb`, byte-identical to current behaviour.

(Mode 3 is the operator-framed "placers" strategy. On Betfair
the place-betting capability is delivered via the **each-way
market mechanic** — half-stake-on-win + half-stake-on-place at
fractional odds derived from the same win-market price ladder.
The repo already has complete EW infrastructure
(`plans/ew-settlement/`, `plans/ew-metadata-pipeline/`); the
mode wires the agent's action to flip `bet.is_each_way = True`
on placement, and the existing settle path does the rest. **No
new place-market data pipeline is needed.**)

## Acceptance gate (from [success_criteria.md](success_criteria.md))

Per strategy mode, a smoke cohort plus a real cohort that
hits the strategy's defined "≥ random by margin M" bar
(see success_criteria.md). The integration is "done" when:

1. Flag-off path is byte-identical to pre-plan (regression test
   guard).
2. Each of the three modes has produced at least one cohort run
   end-to-end without crashing.
3. **At least one mode** beats v2's current arb-only baseline on
   a documented metric. Value-win is the most likely first hit;
   arb's predictor-augmented variant is the second; value-each-way
   is the third. (All three are now data-ready — EW settlement is
   already complete, so no mode is gated on a new data pipeline.)

If the integration runs but no mode beats baseline, the diagnosis
goes deeper than "missing feature" and the v3 conversation
re-opens with concrete signal (see [comparison_protocol.md](comparison_protocol.md)).

## Layout

| File | What |
|---|---|
| [`README.md`](README.md) | This file. The deliverable. |
| [`diagnosis.md`](diagnosis.md) | v2-already-in-progress; v3-vs-integrate verdict; load-bearing-vs-vestigial audit. |
| [`predictor_contracts.md`](predictor_contracts.md) | Exact field names + segment_performance routing per the three production manifests. Reference for the wiring code. |
| [`strategy_modes.md`](strategy_modes.md) | The three modes — action surface, reward shape, predictor weighting per mode. |
| [`integration_contract.md`](integration_contract.md) | Where in code the integration touches; the obs-schema-v8 spec; the flag plumbing; the byte-identical guarantee. |
| [`carry_forward.md`](carry_forward.md) | What survives from v2 / `plans/rewrite/`. |
| [`comparison_protocol.md`](comparison_protocol.md) | How to evaluate each mode against v2's current arb-only baseline. |
| [`success_criteria.md`](success_criteria.md) | When each mode is "done". |
| [`hard_constraints.md`](hard_constraints.md) | Cross-session invariants. |
| [`master_todo.md`](master_todo.md) | Session-level breakdown. |
| [`lessons_learnt.md`](lessons_learnt.md) | Empty starter; populated as sessions land. |
| [`session_prompts/`](session_prompts/) | Per-session prompts for the implementation phase. |

## Phasing (from `master_todo.md`)

| # | Session | Deliverable | Estimate |
|---|---|---|---|
| 01 | Predictor loader | `predictors/loader.py` exposes per-race callables for all three models, reads manifests + segment_performance.json at startup. | 1 session, 4–6 hr |
| 02 | Observation wiring | RUNNER_KEYS extension + `OBS_SCHEMA_VERSION` 7 → 8, flag plumbing, byte-identical guard test. | 1 session, 3–4 hr |
| 03 | Strategy-mode switch | New `training.strategy_mode` config; env honours it; trainer reward gate honours it; smoke test for each mode runs end-to-end with random weights. | 1 session, 4–6 hr |
| 04 | Each-way action surface | New `each_way` action signal; env wires it through `bm.place_back/place_lay(..., each_way=True)`; `value_each_way` mode skips non-EW races. **No data-pipeline work** — EW settlement is already complete in `plans/ew-settlement/`. | 1 session, 3–4 hr |
| 05 | Value-win smoke cohort | First real cohort in `value_win` mode; verify reward signal is non-degenerate. | 1 session, 4 hr training + analysis |
| 06 | Value-each-way smoke cohort | Same for `value_each_way` mode. | 1 session |
| 07 | Three-way comparison | Cohort runs across arb / value-win / value-each-way with shared seed-day set; report.md against the success_criteria.md bar. | 1 session, ~1 day |

**Total: ~5–7 sessions of focused work.** Sessions 01–03 unblock
arb-mode predictor augmentation and value_win mode; Session 04
adds the each-way action surface; Sessions 05–06 are smoke
cohorts; Session 07 is the verdict.

## Hard constraints (apply to ALL sessions)

See [`hard_constraints.md`](hard_constraints.md). Highlights:

1. Flag-off byte-identical to pre-plan. Regression test guards.
2. Don't touch env / matcher / bet_manager mechanics. Add new
   code; don't refactor old code.
3. Don't re-derive EW settlement. `plans/ew-settlement/` is
   complete; reuse `bm.settle_race(each_way_divisor=...,
   number_of_places=...)` verbatim.
4. Don't add new shaped-reward terms. Predictors enter the obs;
   reward shape changes ONLY where the strategy mode genuinely
   needs it (single-shot bets settle to win/lose, no scalp pair).
5. Don't share predictor-loaded weights across the GA population
   in a way that collapses diversity. The predictors are FROZEN
   at startup; their outputs are static features, identical across
   all agents per runner per race. Diversity comes from the policy,
   not the predictor.

## Cross-references

- `plans/rewrite/README.md` — the action-space rewrite this plan
  composes with, not replaces.
- `plans/rewrite/phase-7-port-aux-heads/findings.md` — the
  empirical evidence that internal aux heads can't carry the
  per-runner discrimination signal alone.
- `betfair-predictors/docs/intended_consumer.md` — the documented
  RL-side integration shape, including the
  `observations.use_race_outcome_predictor` flag name.
- `betfair-predictors/production/race-outcome/manifest.json` —
  the calibrated champion contract.
- `betfair-predictors/production/race-outcome-ranker/manifest.json` —
  the ranker companion contract.
- `betfair-predictors/production/direction-predictor/manifest.json` —
  the price-mover champion already partially integrated as
  engineered features at OBS_SCHEMA v7 (Phase 14 S02).

## Open questions surfaced during planning

1. **Direction-predictor model output vs engineered features (v7).**
   OBS_SCHEMA v7 added 10 engineered correlates of the
   direction-predictor's input features. The predictor itself is
   not yet called per-tick. Decide: do we keep the engineered
   features and add a per-tick model call, or replace the
   engineered features with the model outputs? Recommend: ADD,
   don't replace. The engineered features have already been
   trained against; ripping them costs a registry reset for no
   strategy gain. The per-tick model adds a 3 (horizons) × 3
   (quantiles) = 9-dim block per runner.
2. **Place markets ARE each-way markets in this codebase.**
   On Betfair the place-betting capability is delivered via
   each-way (EW) bets, not via separate TO-BE-PLACED market
   ingestion. `plans/ew-settlement/` (4 sessions, complete
   2026-04-11) already gives `BetManager.settle_race` correct
   doubled-stake + place-fraction handling; `Race` has
   `each_way_divisor` and `number_of_each_way_places`; `Bet`
   has `is_each_way`. Place odds are derived from win odds
   via `(win_odds - 1) / divisor + 1`. **No data pipeline
   work is needed.** Session 04 only adds the env-side
   action-surface switch that flips `is_each_way` on placement.
3. **Champion's segment_performance routing inside the env.**
   The contract says weak/insufficient buckets should be skipped
   by the consumer. For RL: do we mask predictor outputs to a
   "no-signal" sentinel in weak buckets, or feed them through
   anyway and let the policy learn the bucket-conditional? Recommend:
   mask to NaN-equivalent sentinel + a binary
   `segment_strong_flag` per runner; let the policy condition.

These three are answered concretely in
[`integration_contract.md`](integration_contract.md).
