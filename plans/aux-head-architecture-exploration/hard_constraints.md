# Hard constraints — aux-head architecture exploration

## §1. Backward compatibility break is acceptable

Any architecture change to aux heads breaks the `actor_head` input
dim and the head weight shapes. `load_state_dict(strict=True)` will
refuse pre-fix weights. **This is correct** — the architecture
genuinely changed; old weights can't safely be reinterpreted under
the new layout. Each candidate variant gets its own
architecture-hash so the registry recognises them as distinct.

CLAUDE.md sections "fill_prob feeds actor_head" and "mature_prob_head
feeds actor_head" already document this precedent. Same pattern
applies.

## §2. Each candidate must preserve the actor_head input contract

The per-runner column-concat that the actor_head reads
(`concat([runner_emb_i, lstm_last, fill_prob_i, mature_prob_i,
direction_back_prob_i, direction_lay_prob_i]`)) MUST stay in place.
Aux heads' role of "feed a per-runner calibrated probability into
the actor's action selection" doesn't change. Only HOW the head
computes that probability is up for redesign.

## §3. The supervised BCE auxiliary loss stays attached

Every candidate must still receive BCE supervision from
`direction_prob_loss_weight × BCE(direction_back/lay_prob,
direction_back/lay_label)`. The PPO surrogate loss flows back
through the same head (i.e. no `detach()` on the actor's read of
the head's output unless we explicitly want to break that flow as
a variant under test).

## §4. Test on a single training day first

Before committing GPU to a multi-day, multi-agent probe cohort,
each candidate must demonstrate:

a. Forward pass returns expected shapes (no architecture bugs)
b. Backward pass produces non-None gradients on all head weights
c. After ~30 minutes of single-day training, the head's BCE has
   descended materially below the pos-weighted random floor.

Skip any candidate that fails (a) or (b). Move to the multi-day
probe cohort only after (c) clears.

## §5. Probe cohort scale

5 agents × 1 generation × 5 training days × 3 eval days per
candidate. ~1.5 hours per candidate on the cohort's hardware. Five
candidates → ~7.5 hours total — fits in a single overnight session.

## §6. Comparison fixture

All probe cohorts share:

* Same `--enable-gene` set as the Phase-15 launch (7 evolving
  genes per CLAUDE.md cohort log line)
* `--use-direction-predictor` ON
* `--predictor-lean-obs` ON
* `direction_prob_loss_weight` pinned at a fixed value (NOT a
  gene) so the variable being measured is the architecture, not
  the loss weight. Suggested pin: `1.0` — the middle of the
  current gene range.
* Same 5 training days (a subset of the Phase-15 set). Suggested:
  2026-04-08, 04-11, 04-15, 04-19, 04-22.
* Same 3 eval days for like-for-like BCE comparison. Suggested:
  2026-04-10, 04-17, 04-23.

## §7. Acceptance metric for "winning" candidate

A candidate is the winner if it satisfies ALL of:

a. `train_mean_direction_back_bce` and
   `train_mean_direction_lay_bce` on the eval days descend by
   ≥ 5 % relative below the baseline (single-Linear-on-lstm_last)
   architecture's BCE on the same eval days.
b. `train_mean_fill_prob_bce`, `train_mean_mature_prob_bce`,
   `train_mean_risk_nll` do NOT regress by > 3 % relative
   compared to the baseline.
c. `eval_total_reward` does NOT regress by > 10 % relative compared
   to the baseline cohort.

(c) is the policy-output sanity check — adding more head capacity
to direction_prob_head shouldn't tank the agent's action-selection
quality just because the actor_head now sees a different aux-head
distribution.

## §8. Regression-test guards

The winning candidate must land with regression tests at
`tests/test_v2_aux_head_architecture.py`. Required tests:

a. Forward pass on a synthetic obs returns the documented shapes.
b. Backward pass through BCE loss produces non-None gradients on
   all head weights.
c. `load_state_dict(strict=True)` refuses weights with the OLD head
   shape (architecture-hash break).
d. `load_state_dict(strict=True)` ACCEPTS weights with the new
   head shape.
e. A no-op rollout with `direction_prob_loss_weight=0` produces
   byte-identical action distributions to a baseline (i.e. the new
   head's contribution to `actor_head`'s input is well-defined and
   stable when not supervised).

## §9. Documentation

Each candidate gets a short writeup in this plan dir:
`candidate_<name>_results.md`. Include:

* The architecture diff vs baseline (one-liner per layer change)
* Probe-cohort BCE numbers (back/lay/fill/mat/risk, baseline +
  candidate, side by side)
* Probe-cohort eval_total_reward, side by side
* Verdict against §7

Winner gets a final summary in this plan's `findings.md`.
