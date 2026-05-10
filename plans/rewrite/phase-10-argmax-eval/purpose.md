---
plan: rewrite/phase-10-argmax-eval
status: design-locked
opened: 2026-05-05
depends_on: rewrite/phase-7-port-aux-heads
---

# Phase 10 — argmax-eval: deterministic action selection at eval time

## Purpose

The 2026-05-05 multi-eval-day investigation (commit `b0a5f44` +
`reevaluate_cohort.py` smoke test) showed that **action-sampling
stochasticity at eval time produces ~£200+ of day-pnl swing on the
exact same trained policy on the exact same eval day**.

Concrete evidence: agent `658a7f72` from the
`_phase7_s06_24agent_overnight_1777941123` cohort produced a
day_pnl of **+£178.32** on its original eval rollout against
2026-05-03. A re-evaluation of the *same weights* against the *same
day* with a different action-sampling seed produced **+£55.43** —
a £123 swing from a single different RNG roll. Agent `81c80d76`
swung from **-£0.19** to **-£336** (a £336 swing on the exact same
weights + day). The naked_pnl channel carries most of the
variance: 658a7f72's naked went from +£187 to -£56 between the two
seeds.

This is not a small effect. It dominates any cross-agent or
cross-architecture P&L comparison the operator might want to make.
Three causes, in order of magnitude:

1. **Per-tick categorical sampling.** `out.action_dist.sample()` in
   `training_v2/discrete_ppo/rollout.py:464` rolls a weighted dice
   each tick. Different RNG → different action selection → different
   open/close decisions → different naked exposure carried into race
   settle.
2. **Per-tick Beta sampling for stake size.** `stake_dist.sample()`
   at `:482` rolls another dice for the £-stake on opens. Different
   stakes magnify or shrink the cash impact of each decision.
3. **Naked-pnl amplification.** Even small differences in which
   pairs end up naked-into-the-off get multiplied by race outcomes,
   so the per-day naked total is the loudest noise channel.

The Phase 7 work and the composite_score / multi-eval-day work
(commits `c25470b`, `b0a5f44`) made the GA's selection signal
correct, but the cash signal it ranks on is still drowning in
action-sampling noise. The user's intuition was right: even agents
that look profitable are mostly riding lucky dice.

## What this phase does

Add an opt-in **deterministic / argmax** action-selection path that:

1. Replaces `out.action_dist.sample()` with `argmax(logits)` —
   always picks the most-likely action.
2. Replaces `stake_dist.sample()` with `stake_dist.mean` —
   uses the Beta distribution's expected value (`alpha / (alpha +
   beta)`) instead of a random draw.
3. Is gated by a single boolean flag carried through
   `RolloutCollector.collect_episode(deterministic=False)` so the
   training rollouts (which MUST stay stochastic for PPO to be
   PPO) and the eval rollouts can pick independently.
4. Defaults to **stochastic** so existing call paths are
   byte-identical to pre-plan behaviour at the policy / collector
   level. The eval-side default is also kept stochastic until the
   validation cohort proves the change does what we expect; an
   `--argmax-eval` CLI flag on the cohort runner / standalone train
   / `tools/reevaluate_cohort.py` opts the operator in.

Three deliverables:

1. **Policy-level deterministic action path** in
   `agents_v2/discrete_policy.py` and `RolloutCollector` —
   `deterministic=True` produces the argmax action + the Beta-mean
   stake. The action's `log_prob` is computed against the
   deterministic action so the rollout buffer's invariants
   (transition.log_prob = dist.log_prob(transition.action)) hold
   under both modes.
2. **Eval-path wiring** in:
   - `training_v2/cohort/worker.py::train_one_agent` — eval
     rollout uses `deterministic=argmax_eval`.
   - `training_v2/cohort/batched_worker.py::train_cluster_batched`
     — same.
   - `training_v2/discrete_ppo/train.py` standalone CLI — same.
   - `tools/reevaluate_cohort.py` — same (currently always
     stochastic).
3. **Validation cohort + cross-eval**:
   - Re-evaluate the existing
     `_phase7_s06_24agent_overnight_1777941123` cohort with
     `--argmax-eval` against three days. Compare against the
     stochastic 3-day re-eval (`reeval_scoreboard_3day.jsonl`).
   - Show argmax-eval is **bitwise reproducible** on the same day
     (two argmax runs on day X produce identical day_pnl).
   - Show argmax-eval **rank-correlates well with stochastic
     multi-day mean** (the agents argmax says are good are
     largely the same agents the stochastic mean says are good)
     but at much lower per-rollout cost — single-day argmax is a
     usable fast-path signal for development iteration.

## Why this is its own phase

`composite_score` (commit `c25470b`) and `multi-eval-day`
(commit `b0a5f44`) addressed the SELECTION signal — what scalar
the GA sorts on. Phase 10 addresses the MEASUREMENT signal — what
scalar each individual eval rollout produces. These are
orthogonal: composite_score still works, multi-eval-day still
works, and argmax stacks on top.

Bundling argmax into either prior phase would have widened them
beyond their stated purpose:
- composite_score is a sort-key change (1 file, ~30 lines core);
  argmax is a forward-pass / collector change (3 files, gradient-
  pathway aware). Different blast radius.
- multi-eval-day reduces noise via averaging; argmax reduces noise
  by removing it at source. Both work; choosing one without
  testing the other was the right small-step.

After Phase 10 ships:
- Operator gets a single-day-fast eval signal for development
  iteration (no need for 3-day means just to compare two
  architectures).
- Multi-eval-day means become a robustness check rather than a
  noise-reduction necessity.
- The GA selection signal can stack: composite_score over
  multi-day-mean over argmax-eval = three orthogonal noise
  reductions on the same scalar.

## What's locked

### The deterministic action choice is `argmax(logits)`, not `argmax(probs)`

```python
if deterministic:
    action = out.action_dist.logits.argmax(dim=-1)
else:
    action = out.action_dist.sample()
```

`argmax(logits)` and `argmax(probs)` are mathematically equivalent
under softmax (softmax is monotone), so the choice is a clarity
preference, not a correctness one. Logits are what the policy
already exposes on `Categorical(logits=masked_logits)` so we
avoid a redundant softmax pass. Action-mask handling carries over
unchanged because masked-out actions have `logit = -inf`,
guaranteed to lose the argmax contest.

### The deterministic stake choice is `Beta.mean`, not `Beta.mode`

```python
if deterministic:
    stake_unit_t = out.stake_alpha / (out.stake_alpha + out.stake_beta)
else:
    stake_unit_t = Beta(out.stake_alpha, out.stake_beta).sample()
```

Two design choices were considered:

- **`Beta.mean = α / (α + β)`** — the expected value. Always
  defined for any α, β > 0. This is what we use.
- **`Beta.mode`** — the peak of the density. Defined only when
  α > 1 and β > 1; outside that region the density is monotone
  (no interior mode). Would require a fallback branch.

Mean is simpler, always defined, and matches the "expected /
typical" policy intent. The trade-off: when α and β are very
close to 1 (uniform-ish posterior), the mean ≈ 0.5 stake regardless
of asymmetry that might exist; the policy's true preference is
weakly expressed. This is fine — that's exactly the situation
where the policy is uncertain and any deterministic choice is
arbitrary.

### `log_prob` under deterministic mode

The rollout buffer's invariant is
`transition.log_prob = transition.action_dist.log_prob(transition.action)`
— the PPO update relies on this. Under deterministic mode the
collector still computes `log_prob` against the chosen action:

```python
log_prob_action_t = out.action_dist.log_prob(action).detach().squeeze()
```

This works regardless of whether `action` came from `.sample()`
or `argmax`. Eval rollouts don't drive PPO updates (they're
rollout-only) so the log_prob value is informational, not
load-bearing — but keeping the same code path means the rollout
buffer schema is unchanged and any future "train under argmax"
experiment doesn't need a parallel path.

### Default mode is stochastic everywhere

`RolloutCollector.collect_episode(deterministic=False)` and
`train_one_agent(argmax_eval=False, ...)` defaults preserve
byte-identical behaviour. Operators opt in via
`--argmax-eval` on the cohort runner, standalone train, and
`tools/reevaluate_cohort.py`. The deterministic mode is never
used during training rollouts — those stay stochastic by
construction (a deterministic-trained PPO is a different
algorithm with different convergence properties; out of scope).

### Validation gate is "lower variance, similar ranking"

Pass conditions for the validation cohort:

1. **Reproducibility.** Two argmax-eval rollouts of the same
   trained policy on the same eval day produce
   *bit-identical* day_pnl, locked_pnl, and naked_pnl values.
   Single integer-comparison test.
2. **Variance reduction.** On the existing s06 cohort: the
   spread of single-day argmax-eval day_pnl across the three
   identical-gene lineage agents
   (`658a7f72` / `e34193fd` / `9a20de9c`) is at least **3×
   smaller** than the spread under stochastic eval (which was
   £185 day_pnl across the three).
3. **Rank correlation.** Spearman ρ between
   3-day-stochastic-mean ranking and 1-day-argmax ranking, over
   all 144 cohort agents, is **≥ 0.7**. We're not asking argmax
   to be the *same* signal as multi-day-mean — just that it's a
   strongly correlated cheaper proxy.

Failure modes worth surfacing:

- If reproducibility (1) fails: there's another stochastic
  source we missed (env-side passive matching, e.g.). Stop and
  audit the env's own RNG paths under `deterministic=True`.
- If variance reduction (2) fails: argmax doesn't actually
  flatten the lineage spread → likely the lineage spread is
  driven by training stochasticity (channel 2 in the
  conversation), not eval sampling. Useful finding either way.
- If rank correlation (3) is poor: argmax-best agents are
  systematically different from sampled-best agents → policies
  are too uncertain at the argmax decision points; the noise
  they make under sampling reveals real structure that argmax
  hides. Document and don't ship as the default.

### No env edits

The argmax change lives entirely in the policy / collector
layer. The env (`BetfairEnv`, `ExchangeMatcher`, `BetManager`)
sees an action int + stake float exactly as before, regardless
of whether those came from sampling or argmax. The env may
have its own stochastic paths (passive-fill ordering, etc.)
which are out of scope; those become the validation
reproducibility test in §"What's locked" item 1.

### CUDA↔CUDA self-parity holds

Same load-bearing guard inherited from prior phases. Two
argmax-eval runs at fixed seed and identical args produce
bit-identical per-agent results.

### Compatibility with Phase 8 and Phase 9

Both Phase 8 (oracle BC pretrain) and Phase 9 (per-transition
credit) are open / not yet shipped at the time this plan is
written. Phase 10 is designed to land cleanly alongside either
or both:

- **Phase 8 — orthogonal.** Phase 8 adds an oracle-driven BC
  pretrain step BEFORE the first PPO rollout. It's a training-
  side change; Phase 10 is an eval-side change. They share no
  files: Phase 8 touches `DiscretePPOTrainer.__init__` /
  pre-rollout BC loop and a new `training_v2/oracle/` module;
  Phase 10 touches `RolloutCollector.collect_episode` and the
  worker eval-rollout call. Either can ship first.

- **Phase 9 — same file, different statements.** Phase 9
  inserts collector-side tracking around the `env.step()` call
  inside `_collect_rollout` (snapshot `len(bm.bets)` before and
  after each step). Phase 10 gates `Categorical.sample()` and
  `Beta.sample()` calls in the same function. The two edits land
  on different statements and should merge cleanly in either
  order. **No semantic conflict** because Phase 10's hard
  constraint that training stays stochastic means Phase 9's
  per-transition labels are always derived from stochastic
  rollouts — the deterministic mode never runs during training.
  At eval time Phase 9's labels aren't used (BCE loss is
  training-only), so the two mechanisms operate on disjoint
  rollouts.

If both 8 and 9 ship before Phase 10, the interaction at
landing time is purely textual (resolve any merge conflicts in
`rollout.py` / `worker.py`) — no design rework needed. If
Phase 10 ships first, the gating is forward-compatible: when
Phase 9 lands its `_collect_rollout` tracking it'll see the
`deterministic` kwarg already in the function signature and
nothing breaks.

### Schema growth, not break

`scoreboard.jsonl` rows gain an optional `eval_mode` field
(`"stochastic"` | `"argmax"`) that defaults to `"stochastic"`
when absent, so pre-plan rows stay readable. No changes to
existing eval_* field names.

## Success bar

The plan ships GREEN iff:

1. **`RolloutCollector.collect_episode(deterministic=True)`
   returns a batch where every transition's `action` is
   `logits.argmax()` and every stake is `Beta.mean`.** Forward-
   path test on a fixture with rigged logits.
2. **`train_one_agent(argmax_eval=True, ...)` runs the eval
   rollout under deterministic mode.** Integration test that
   spies on the collector to confirm the flag flows through.
3. **`tools/reevaluate_cohort.py --argmax-eval` produces
   bit-identical day_pnl across two consecutive runs on the
   same agent + same day.** Reproducibility test.
4. **Validation gate (item 1 in "What's locked"): same agent +
   same day under argmax = identical pnl.** Run
   `reevaluate_cohort.py --argmax-eval` twice on the s06 top-3
   agents; assert all three numeric fields identical.
5. **Variance reduction gate (item 2): the
   `658a7f72`/`e34193fd`/`9a20de9c` lineage's day_pnl spread
   under argmax-eval is < £62 (1/3 of the £185 stochastic
   spread).**
6. **Rank correlation gate (item 3): Spearman ρ between
   3-day-stochastic-mean ranking and 1-day-argmax ranking on
   all 144 s06 agents is ≥ 0.7.**
7. **CUDA↔CUDA self-parity at fixed seed** continues to hold
   for argmax-eval rollouts (existing Phase 3 guard extended).

## Sessions

### Session 01 — deterministic action path in collector + policy

Pure mechanical work — gate the per-tick `Categorical.sample()`
and `Beta.sample()` behind a `deterministic: bool` flag.

Files:
- `training_v2/discrete_ppo/rollout.py::RolloutCollector.collect_episode`
  — accept `deterministic: bool = False` kwarg; gate action
  sampling and stake sampling.
- `agents_v2/discrete_policy.py` — no change needed (the policy
  already exposes `action_dist.logits` and the Beta `alpha`/`beta`
  on `PolicyOutput`).

Tests in `tests/test_v2_argmax_eval.py` (new file):

1. `test_collector_deterministic_action_is_argmax_of_logits` —
   rig the policy's masked_logits to known values via a stub;
   assert collected actions equal `logits.argmax()` per tick.
2. `test_collector_deterministic_stake_is_beta_mean` — rig
   `stake_alpha=2.0, stake_beta=1.0` (Beta.mean = 2/3); assert
   collected stake_unit ≈ 0.6667 within fp epsilon on every
   tick where the action uses stake.
3. `test_collector_default_is_stochastic_byte_identical` —
   `collect_episode()` (no kwarg) under fixed RNG produces
   bit-identical transitions to a pre-plan reference fixture.
   Byte-identical regression guard.
4. `test_collector_log_prob_invariant_holds_under_deterministic`
   — `transition.log_prob = action_dist.log_prob(action)` for
   every transition under `deterministic=True`. The PPO
   buffer's invariant must not break (even though eval doesn't
   call PPO).
5. `test_collector_action_mask_respected_under_deterministic` —
   if some actions are masked out (logit=-inf), the argmax
   never picks them. Edge case the existing mask logic
   already guarantees, but worth a regression guard.

Session prompt: `session_prompts/01_add_deterministic_action_path.md`.

### Session 02 — wire `argmax_eval` flag through eval code paths

Trainer-side and tooling-side wiring. No new collector logic;
just routing the bool from the operator-facing flag down to the
collector.

Files:
- `training_v2/cohort/worker.py::train_one_agent` — accept
  `argmax_eval: bool = False`; pass to the eval-rollout
  collector call. Per-day eval loop already exists from
  Phase 8 multi-day work.
- `training_v2/cohort/batched_worker.py::train_cluster_batched`
  — same.
- `training_v2/cohort/runner.py::run_cohort` — accept
  `argmax_eval`, pass through.
- `training_v2/cohort/runner.py::main` — add `--argmax-eval`
  CLI flag (default False, byte-identical to pre-plan).
- `training_v2/discrete_ppo/train.py` standalone CLI — same
  `--argmax-eval` flag, passed to its eval rollout call.
- `tools/reevaluate_cohort.py` — same flag (default False
  preserves the current behaviour; operator opts in for the
  validation runs).
- `_agent_result_to_scoreboard_row` — add `eval_mode` field
  (`"stochastic"` | `"argmax"`).

Tests added to `tests/test_v2_argmax_eval.py`:

6. `test_train_one_agent_argmax_eval_flag_reaches_collector` —
   spy on `RolloutCollector.collect_episode` calls; assert
   `deterministic=True` is passed when `argmax_eval=True` is
   set, and `False` otherwise.
7. `test_run_cohort_argmax_eval_flag_plumbs_through` — end-to-end
   integration: launch a tiny 2-agent cohort with
   `argmax_eval=True`; assert the scoreboard rows carry
   `eval_mode == "argmax"` and the per-agent `eval_*` fields
   are reproducible across two cohort runs at the same seed.
8. `test_reevaluate_cohort_argmax_eval_reproducible` —
   reuse a real saved model from the s06 cohort; assert two
   `--argmax-eval` runs produce bit-identical day_pnl on the
   same eval day.

Session prompt: `session_prompts/02_wire_into_eval_codepaths.md`.

### Session 03 — validation cohort + plan close

Re-evaluate the existing s06 cohort under `--argmax-eval` and
compare to the stochastic 3-day re-eval already on disk.

Steps:

1. Run `tools/reevaluate_cohort.py --argmax-eval` against the
   s06 cohort (144 agents) on the three eval days
   `2026-05-02 / 2026-05-03 / 2026-05-04`. Output:
   `reeval_scoreboard_3day_argmax.jsonl`.
2. Verify reproducibility (Success-bar item 4): pick the top-3
   agents by composite_score, run the argmax reeval on each
   for the same day twice, assert identical pnl.
3. Compute variance-reduction (Success-bar item 5): isolate the
   lineage agents (`658a7f72` / `e34193fd` / `9a20de9c` —
   identical genes) and compare their argmax-eval day_pnl
   spread vs the £185 stochastic spread.
4. Compute rank-correlation (Success-bar item 6): Spearman ρ
   between the 3-day-stochastic-mean rank order
   (`reeval_scoreboard_3day.jsonl`) and 1-day-argmax rank order
   (any single day from `reeval_scoreboard_3day_argmax.jsonl`)
   over all 144 agents.
5. Write `findings.md` with the three gate outcomes + the
   action-distribution histograms (does argmax-eval produce
   different action mixes than stochastic? — useful diagnostic
   for whether the policy is "near-deterministic anyway" vs
   "loses real exploration under argmax").

Update CLAUDE.md to add a section under
`## Reward function` (or a new top-level section) noting that
the cohort runner now exposes `--argmax-eval` for deterministic
eval rollouts.

Session prompt: `session_prompts/03_validation_and_writeup.md`.

## Hard constraints

Inherited from rewrite plan + phase-7:

1. **No env edits.** All work in `agents_v2/`, `training_v2/`,
   and `tools/`.
2. **Default behaviour is byte-identical** to pre-plan when no
   `--argmax-eval` flag is set. Existing tests must continue
   to pass without modification.
3. **Training rollouts stay stochastic.** PPO is a stochastic-
   policy algorithm; deterministic-trained PPO is a different
   algorithm with different convergence properties (DPG, etc.)
   and is out of scope.
4. **The collector's deterministic mode applies to BOTH the
   action sample AND the stake sample.** Mixed mode (argmax
   action + sampled stake) would still have £-stake variance
   and would not be reproducible. Don't half-do this.
5. **`log_prob` invariant must hold.** Even under
   deterministic mode, `transition.log_prob` must equal
   `action_dist.log_prob(transition.action)`. The rollout
   buffer schema is fixed.
6. **Schema is forward-only.** New optional `eval_mode` field
   on scoreboard rows; default `"stochastic"` when absent.
7. **Validation cohort uses the existing s06 cohort.** No new
   training runs needed for Phase 10 validation — just
   re-evaluate the agents already on disk under the new mode.
8. **Same `--seed 42`** for any cross-cohort comparison. CUDA↔
   CUDA self-parity holds.

## Out of scope

- Training under deterministic action selection. PPO needs
  exploration. A "deterministic PPO" experiment is its own
  research direction (DPG / DDPG variants) and is not in this
  plan's scope.
- Replacing `Beta.mean` with a more sophisticated stake
  policy (e.g. fractional Kelly, contract-based sizing).
  Current scope is "what does the trained policy think the
  expected stake is" — that's `Beta.mean`. Smarter stake
  selection is a follow-on plan.
- Architecture comparison cohorts using argmax-eval. That's
  the *consumer* of this plan, not the plan itself. Phase X
  (architectures) gets to use the cleaner signal once Phase
  10 ships.
- A "best-of-N argmax" mode (e.g. run argmax 3 times with
  different action-mask perturbations and pick the median).
  Argmax is bit-identical at fixed seed by construction —
  no aggregation needed.
- Frontend / scoreboard display changes for the new
  `eval_mode` field. The field lands in JSONL; UI can read or
  ignore it.

## Useful pointers

- Action sampling: [`training_v2/discrete_ppo/rollout.py:464`](../../../training_v2/discrete_ppo/rollout.py)
  (Categorical.sample) and `:482` (Beta.sample).
- Policy forward exposes the right primitives:
  [`agents_v2/discrete_policy.py`](../../../agents_v2/discrete_policy.py)
  — `PolicyOutput` carries `action_dist`, `stake_alpha`, `stake_beta`.
- Multi-eval-day plumbing already in place from Phase 8:
  [`training_v2/cohort/worker.py::train_one_agent`](../../../training_v2/cohort/worker.py)
  — the per-day eval loop just needs the deterministic flag
  added; the loop structure is unchanged.
- Re-eval tool to extend:
  [`tools/reevaluate_cohort.py`](../../../tools/reevaluate_cohort.py).
- Empirical evidence that motivated this plan:
  - Smoke-test conversation log 2026-05-05 19:36
  - `registry/_phase7_s06_24agent_overnight_1777941123/scoreboard.jsonl`
    (original eval under stochastic) vs the smoke-test reeval
    output (also stochastic, different seed).
  - `reeval_scoreboard_3day.jsonl` (stochastic 3-day mean,
    in-flight at time of this plan being written).

## Estimate

- Session 01 (collector flag + 5 tests): **~1.5 hours**.
- Session 02 (eval-path wiring + 3 integration tests): **~2
  hours**. Each of the four call sites
  (`worker.train_one_agent`, `batched_worker`, `train.py` CLI,
  `reevaluate_cohort.py`) is a small kwarg-plumbing edit.
- Session 03 (re-eval cohort run + analysis + writeup):
  **~1 hour wall + ~3-5 hours GPU**. Re-evaluating 144 agents
  × 3 days under argmax should take roughly the same wall as
  the stochastic re-eval (~9 hours) — argmax doesn't speed up
  the rollout itself, just removes its variance. If the
  validation only needs single-day-argmax for the
  rank-correlation test, the budget drops to ~3 hours.

Total: ~5 h human + 5-9 h GPU.

If past 3 h on Session 02 excluding tests, stop and check
scope — the kwarg plumbing should be fast. A long session
means the eval rollout has more state interleaving than this
plan accounts for.

## When to do this

After the in-flight stochastic 3-day re-eval finishes
(~04:30 on 2026-05-06). The stochastic re-eval is the
baseline that Session 03 compares argmax-eval against —
running the two side-by-side gives the cleanest before/after.

If the operator wants to start Phase 10 *during* the
stochastic re-eval, Session 01 (collector + tests) is fully
parallel — no GPU contention with the running re-eval since
it's pure code + cpu tests. Session 02 wiring is similar.
Only Session 03's validation re-eval needs GPU and so should
wait for the in-flight job to free it up.
