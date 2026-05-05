---
plan: rewrite/phase-9-per-transition-credit
status: OPEN
opened: 2026-05-05
depends_on: phase-7-port-aux-heads (AMBER)
---

# Phase 9 — Per-transition credit for aux-head BCE labels

## Why this plan exists

The v2 aux-head BCE labels (fill_prob, mature_prob) are computed once per
rollout and broadcast identically to EVERY transition's mini-batch entry:

```python
# training_v2/discrete_ppo/aux_labels.py::compute_per_runner_aux_labels
#
# Aggregates per-pair outcomes into per-slot labels.
# "Across-race aggregation. The slot index is the per-race runner_map
#  position, which carries different selection_ids across races."
```

On a typical 11k-transition rollout spanning 80–107 races:
- A "slot" (e.g. slot 3) corresponds to a DIFFERENT physical runner in
  each race — different horse, different price history, different arb outcome.
- The per-slot label aggregates all those different runners' outcomes into a
  single scalar (e.g. `mature_label[3] = max(...)` across every race).
- That scalar is then broadcast to all 11,000 transitions, regardless of
  which race and tick each transition came from.

The signal is diluted by a factor of roughly `n_races` (~90). A pair that
matured on race 7 produces a positive mature_label that is also fed to every
transition from races 1–6 and 8–107, even though those transitions have
nothing to do with race 7's outcome.

Phase 7 Session 03 validated this is the binding constraint: at weight 0.5 the
BCE gradient moves action KL by a median 0.0069 (gate is 0.1). Raising
mature_prob_loss_weight to 3.4–4.0 in S04–S06 improved the gradient magnitude
slightly but still couldn't move maturation_rate — the dilution is structural,
not a magnitude problem.

## The fix

**Assign each pair's label to the single transition where that pair was
OPENED, not to all transitions.**

At the moment a bet is placed (step T in the rollout), we know:
- The `pair_id` of the newly opened pair.
- The current step index T.

At end of rollout, when pair outcomes are known (matured / force-closed /
naked), we have `pair_id → outcome`. We can then assign:
- `mature_label = 1.0` to the single transition at step T where the
  pair's open leg was placed.
- `mature_label = 0.0` to that same transition if the pair force-closed
  or went naked.

Transitions where no bet was opened get no BCE gradient at all
(masked out) — the label is only defined where a decision was made.

The expected gradient SNR improvement: instead of one signal spread across
11,000 transitions, the signal lands on the ~200–500 opening transitions in
a typical rollout. That's a 20–50× concentration of credit assignment.

## Design

### What needs to change

**1. Track open-step index at bet placement (`RolloutCollector`).**

The rollout collector at `training_v2/discrete_ppo/rollout.py` captures
transitions step-by-step. When `env.step()` places a bet, the resulting
`Bet` object needs to be associated with the current step index. Options:

- **Option A (info-dict):** `BetfairEnv.step()` returns `info["bets_opened"]`
  — a list of `(pair_id, step_within_episode)` for every new matched bet on
  that step. The rollout collector appends these to a `pair_open_steps`
  dict. No change to Bet or Transition.

- **Option B (Bet attribute):** Add `step_index: int | None` to `Bet`
  (in `env/bet_manager.py`). Set it at placement time if the env is running
  under a rollout collector (pass a mutable counter). The collector reads
  `env.all_settled_bets` at end-of-rollout and uses `b.step_index` directly.

- **Option C (collector-side diff):** Before each step the collector
  snapshots `len(env.bet_manager.bets)`. After the step, the difference
  reveals newly placed bets; the collector records `(pair_id, step_T)`.
  No env change needed.

Option C is the least invasive — it doesn't touch `BetfairEnv` or `Bet`
and keeps the tracking entirely inside the rollout collector.

**2. New label-assignment function (`aux_labels.py`).**

Replace (or supplement) `compute_per_runner_aux_labels` with
`assign_per_transition_labels(pair_open_steps, pair_outcomes, n_steps)`.

Returns a pair of arrays of shape `(n_steps,)`:
- `per_step_mature_label: np.ndarray[float32]` — 0/1/NaN per step.
- `per_step_mature_mask: np.ndarray[bool]` — True only at open steps.

The trainer applies the BCE loss only where `per_step_mature_mask` is True.

**3. Trainer BCE computation.**

`DiscretePPOTrainer._compute_aux_losses` currently reads per-slot labels
from a `PerRunnerAuxLabels` struct and broadcasts them. The new path:

- In the mini-batch loop, slice `per_step_mature_label` and
  `per_step_mature_mask` by `mb_idx` (same as other per-step arrays).
- Apply `F.binary_cross_entropy(sigmoid(mature_logit[step, slot]),
  per_step_mature_label[step])` only where `per_step_mature_mask[step]`
  is True.
- `mature_logit[step, slot]` is the `mature_prob_head` output for the
  runner that was opened at step `step`.

The tricky part: at step T the agent opened a bet on runner `slot_k`.
The mature_prob_head's output is a vector over all runners. We need to
pick `mature_logit[T, slot_k]` — i.e. the logit for the specific
runner that was opened, not all runners. This requires storing
`open_runner_slot` alongside `open_step` in the collector.

**4. What happens to fill_prob BCE?**

fill_prob's label is broken (Phase 8 + per-runner-credit analysis). Per-
transition credit doesn't fix the label content, only the assignment
granularity. Since the GA already votes fill_prob_loss_weight → 0,
per-transition credit for fill_prob is unlikely to change that verdict.

Recommendation: drop fill_prob BCE from the per-transition credit work
and apply per-transition only to mature_prob. fill_prob continues at
weight 0; its head still runs (contributing a nearly-constant 0.5 column
to actor_head), but receives no BCE gradient. A future plan can decide
whether to remove the head entirely or give it a corrected label.

**5. risk_head NLL.**

The NLL loss for `risk_head` already uses a `risk_mask` (True only for
completed pairs). It has the same slot-broadcast problem but the impact is
secondary — NLL shapes the backbone rather than the actor directly. Leave
risk unchanged in this phase. Note in the plan if NLL should be upgraded
in a follow-on.

## Hard constraints

1. **The per-slot path must stay available as a fallback.** Use a
   `training.per_transition_credit: bool` config flag (default `false`).
   When `false`, behaviour is byte-identical to pre-plan. When `true`, the
   per-transition path runs. This lets us do a clean A/B probe.
2. **No env changes.** Use Option C (collector-side diff) for open-step
   tracking. Do not add `step_index` to `Bet` or modify `BetfairEnv.step()`.
3. **Do not change `env/bet_manager.py` or the reward path.** This is
   purely a label-assignment change; raw P&L and shaped reward are unaffected.
4. **fill_prob BCE stays at weight 0** and is not retrofitted with
   per-transition credit in this phase.
5. **Per-episode JSONL gains `per_transition_credit_active` field**
   (True/False) so scoreboard comparisons can be filtered correctly.
   Pre-plan rows are missing → treated as False.
6. **Regression guard**: `test_per_slot_path_is_byte_identical_when_disabled`
   — with the config flag off, training output is identical to Phase 7 run.

## Success bar

S01 (collector-side tracking + label function):
- Integration test: for a 2-race synthetic rollout where race 1 has a
  matured pair on slot 2 at step 40 and race 2 has a force-closed pair on
  slot 0 at step 110:
  - `per_step_mature_label[40] == 1.0`, mask True at 40.
  - `per_step_mature_label[110] == 0.0`, mask True at 110.
  - All other steps: mask False, label ignored.
- Byte-identity test: config flag = False → output identical to Phase 7.

S02 (trainer integration):
- BCE loss with per-transition credit fires on ~200–500 transitions per
  rollout (the real open-step count), not 11k.
- `n_bce_targets` logged per update (confirms the count is in expected range).

S03 (validation cohort):
- 12 agents × 2 gens × 4 days: half with `per_transition_credit=true`,
  half without.
- Gate: with-credit agents show higher correlation between
  `mature_prob_loss_weight` and maturation_rate than without-credit agents.
  Concretely: ρ(mature_prob_loss_weight, maturation_rate) should be positive
  in the with-credit cohort. Pre-plan Phase 7 showed essentially ρ ≈ 0.
- Secondary gate: action-distribution KL (per-runner-credit || no-credit)
  ≥ median 0.1 in the with-credit cohort (matching the Phase 7 S03 gate that
  per-slot credit failed).

## Session structure (rough)

| Session | Deliverable |
|---|---|
| S01 | Collector-side open-step tracking; `assign_per_transition_labels`; unit tests |
| S02 | Wire per-transition labels into `DiscretePPOTrainer`; `n_bce_targets` logging |
| S03 | Validation cohort; confirm KL gate passes; ρ positive |

## Relationship to Phase 8 (Oracle BC)

Phase 9 and Phase 8 are designed to work in sequence on the same actor
decision. Phase 9 should be implemented and validated first. Phase 8's
S03 validation then runs with `per_transition_credit=true` active in
BOTH arms — testing BC's marginal contribution on top of clean label
credit, not against the noisy per-slot baseline.

The intended compound effect:
- Phase 9 alone: actor gets clean gradient at open decisions — learns
  WHICH opens tend to mature.
- Phase 8 alone: actor warm-starts toward oracle arb ticks — learns
  WHERE profitable arbs exist. Effect is limited if per-transition
  credit is off (the subsequent BCE signal is too noisy to refine).
- Both together: BC primes WHERE, per-transition BCE refines WHICH
  within those locations. Expected to be additive.

Phase 9 S03 uses no BC pretrain (clean isolated validation). Phase 8
S03 uses per-transition credit in both arms (tests BC's marginal
contribution in the clean-label environment).

## What's NOT in scope

- Fixing fill_prob's broken label (the label content is wrong regardless of
  assignment granularity; addressed by Phase 8's oracle approach).
- Per-transition credit for risk NLL (secondary effect; defer to follow-on).
- Changing env, reward path, or any v1 code.
- Full GA tuning run — S03 is a 2-gen diagnostic probe, not a production run.
