# Hard constraints — apply to ALL sessions in this plan

Cross-session invariants that govern every commit under
`plans/predictor-integration/`. If a session feels like it
needs to violate one, stop and escalate — don't bundle the
violation.

## 1. Flag-off byte-identical to pre-plan

`observations.use_race_outcome_predictor: false` AND
`observations.use_direction_predictor: false` AND
`training.strategy_mode: arb` MUST produce numerically
identical output to the pre-plan baseline on a 1-day cohort.

The regression test
`tests/test_predictor_integration.py::test_flag_off_is_byte_identical_to_pre_plan`
is the load-bearing guard. It runs in CI on every commit.

## 2. No env / matcher / bet_manager mechanics changes

The simulator is the moat (CLAUDE.md is 700+ lines of
correctness facts). Adding `strategy_mode` as a kwarg and
honouring it for action-surface / reward-gate is allowed.
Changing matching rules, force-close logic, equal-profit
sizing, or settlement is NOT allowed in this plan. If a
predictor integration appears to require it, that's a
follow-on plan.

## 3. No new shaped reward terms

Value modes get realised P&L only at settle (no scalping
shaping). Arb mode keeps its current shaping unchanged. Don't
add per-step shaping for the value modes "to densify the
signal" — if PPO can't learn from settle-only rewards on the
value modes, the diagnosis goes to the policy/trainer side
(value modes are SIMPLER than arb; the architecture should
work).

## 4. Predictor weights are FROZEN

Predictors are loaded once at trainer startup, held read-only
across the cohort, identical across all worker processes.
Nothing in this plan trains predictor weights or shifts them
via gradient. If the upstream `betfair-predictors` repo
re-crowns a model, this plan's cohorts pin to the
`experiment_id` of the predictor at training start (recorded
in the registry).

## 5. Don't share BC-pretrained or predictor-conditioned
weights across the GA population

Standard rule from `plans/arb-improvements/lessons_learnt.md`:
sharing pretrained weights collapses GA diversity. The
predictor outputs are static features (identical per runner
per race across the population); diversity comes from the
policy weights and the genes. Don't accidentally start the
population from a single pre-trained checkpoint that read
predictor features once.

## 6. Don't re-derive EW settlement

`plans/ew-settlement/` is complete (Sessions 01–04, finished
2026-04-11). `BetManager.settle_race` correctly handles
each-way settlement when `bet.is_each_way = True` (doubled
stake → half-stake on win leg + half-stake on place leg at
fractional odds derived from win odds). `Race` carries
`each_way_divisor` and `number_of_each_way_places`. `Bet`
carries `is_each_way`, `each_way_divisor`, `number_of_places`,
`settlement_type`, `effective_place_odds`. Session 04 of THIS
plan reuses the path verbatim; the only addition is the
action-surface flag that triggers `is_each_way = True` at
placement time. **Don't touch the settle path.**

## 7. Capture predictor `experiment_id` in every cohort row

Two cohort runs with different predictor `experiment_id`s
are NOT cross-comparable. The registry record carries the
champion / ranker / direction `experiment_id` per cohort.
Re-eval tooling refuses on mismatch. This is the same
discipline the rewrite plan applies to OBS_SCHEMA_VERSION.

## 8. Three modes, trained separately, evaluated jointly

Don't merge the three modes into a unified action space in
this plan. Train each mode in its own cohort. Evaluate on a
shared held-out window (Session 07). Mode-mixing is a
post-this-plan question.

## 9. Predictor outputs are observations, not authority

The policy SEES predictor outputs and decides. The env does
NOT hard-gate bets on predictor outputs (e.g. "block bets
where champion `p_win` < threshold"). The reward stays tied
to actual P&L. Trust-but-verify: the policy must learn the
gating rule from gradients, not from the env enforcing it.
This is the same discipline the predictor manifests describe
for the live-inference consumer; the RL consumer gets the
same contract.

## 10. Loader robustness

The predictor loader must handle:

- Missing manifest → raise loudly.
- Schema mismatch → raise loudly.
- Cold-start categorical (unseen course / jockey / trainer)
  → use `<UNKNOWN>` token per F2/F5 contract.
- Insufficient-data segment → emit `segment_strong_flag = 0`.

Silent fallback is forbidden. The flag-off path is the only
zero-cost no-op path.

## 11. Don't refactor `agents_v2/discrete_policy.py` in this plan

The policy class is unchanged in shape. The new dims appear
inside the existing per-runner obs slice; the actor_input
contract from `plans/fill-prob-in-actor` and
`plans/per-runner-credit` is preserved verbatim. If a session
appears to need a policy change, escalate — that's a separate
plan.

## 12. Don't retire the v2 internal scorer or aux heads in
this plan

`plans/rewrite/phase-0-supervised-scorer/`, `fill_prob_head`,
`mature_prob_head`, `risk_head` all stay wired. Their gene
weights may default to 0 in cohorts where the predictors are
the primary discrimination signal, but the heads themselves
are not removed. Retirement is a follow-on plan once Session
07 evidence supports it.

## 13. Don't expand the scope mid-plan

Out of scope:

- Live inference (ai-betfair side; cross-repo, separate effort).
- Each-way settlement adjustments (separate plan exists).
- New market types (jumps vs flat, IE vs UK regional differences).
- Streaming-recorder integration (StreamRecorder1 backups).
- Frontend redesign for predictor visualisation.

If a session uncovers cross-cutting needs, file in `incoming/`
of the appropriate target repo per the cross-repo postbox
discipline; don't bundle.
