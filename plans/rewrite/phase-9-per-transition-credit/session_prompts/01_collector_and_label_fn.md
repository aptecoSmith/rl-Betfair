---
session: phase-9-per-transition-credit / S01
phase: rewrite/phase-9-per-transition-credit
parent_purpose: ../purpose.md
---

# S01 — collector-side open-step tracking + `assign_per_transition_labels`

## Context

Read `purpose.md` and `hard_constraints.md` first. The core problem:
`compute_per_runner_aux_labels` in `training_v2/discrete_ppo/aux_labels.py`
broadcasts one per-slot label to all ~11k transitions in a rollout.
The same slot carries a different physical runner in each of the ~90
races, so the label is a noisy aggregate from ~90 independent
outcomes. This session adds the infrastructure to assign mature_prob
labels to the single transition where each pair was opened.

No trainer changes in this session. The deliverable is the tracking
logic and a new label-assignment function, fully unit-tested.

## Pre-reqs

- Read [`training_v2/discrete_ppo/rollout.py`](../../../../training_v2/discrete_ppo/rollout.py)
  — full file. Understand how `_collect_rollout` (or equivalent)
  loops over `env.step()` calls and appends `Transition` objects.
  Find the exact line where `env.step()` is called. The
  collector-side diff wraps this call.
- Read [`training_v2/discrete_ppo/transition.py`](../../../../training_v2/discrete_ppo/transition.py)
  — the `Transition` dataclass. Confirm it doesn't already carry
  per-pair label fields. Note whether it's frozen/slots or mutable.
- Read [`training_v2/discrete_ppo/aux_labels.py`](../../../../training_v2/discrete_ppo/aux_labels.py)
  — full file. Understand `PerRunnerAuxLabels` and
  `compute_per_runner_aux_labels`. The new function lives alongside
  these and takes different inputs.
- Read [`env/bet_manager.py`](../../../../env/bet_manager.py) —
  confirm `BetManager.bets` is a list (ordered, `len()` reliable)
  and that each `Bet` has `pair_id`, `side` (BACK/LAY),
  `force_close` (bool, may be absent on old Bet objects — guard
  with `getattr(b, "force_close", False)`), and `market_id` /
  `selection_id` for runner-slot lookup.
- Read [`training_v2/cohort/worker.py`](../../../../training_v2/cohort/worker.py)
  — find where `env.all_settled_bets` is read at end of rollout
  (used by the current per-slot path). The new path also reads
  this, but only at end-of-rollout for outcome classification.

## Deliverables

### 1. `PairOpenRecord` dataclass (`training_v2/discrete_ppo/aux_labels.py`)

Add alongside existing code:

```python
@dataclass(frozen=True)
class PairOpenRecord:
    pair_id: str
    step_index: int      # 0-based transition index in this rollout
    runner_slot: int     # env slot at time of open
```

### 2. Collector-side open-step tracking

In the rollout collector (wherever `env.step()` is called in
`training_v2/discrete_ppo/rollout.py`):

```python
bets_before = len(env.bet_manager.bets)
obs_next, reward, done, info = env.step(action)
new_bets = env.bet_manager.bets[bets_before:]

for bet in new_bets:
    if bet.pair_id is None or getattr(bet, "close_leg", False):
        continue   # skip close legs — we label the OPEN decision only
    slot = _slot_for_bet(bet, market_to_runner_map, max_runners)
    if slot is not None:
        pair_open_records.append(PairOpenRecord(
            pair_id=str(bet.pair_id),
            step_index=current_step,
            runner_slot=slot,
        ))
```

`pair_open_records: list[PairOpenRecord]` is accumulated across the
full rollout and returned from `_collect_rollout` alongside the
existing `Transition` list.

The `market_to_runner_map` and `max_runners` are already available
in the collector (used for per-runner reward attribution). Reuse
the existing reference — do not recompute.

### 3. `assign_per_transition_labels` (`training_v2/discrete_ppo/aux_labels.py`)

New function:

```python
def assign_per_transition_labels(
    pair_open_records: list[PairOpenRecord],
    all_bets: Iterable[Bet],
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign mature_prob BCE labels to open-step transitions only.

    Returns
    -------
    mature_label : np.ndarray, shape (n_steps,), float32
        1.0 at the step where a cleanly-matured pair was opened;
        0.0 at the step where a force-closed or naked pair was opened;
        0.0 everywhere else (but mask is False there — don't use).
    mature_mask  : np.ndarray, shape (n_steps,), bool
        True only at steps where a pair was opened (label is defined).
        False everywhere else — trainer skips BCE on unmasked steps.
    """
```

Outcome classification per pair (same logic as `compute_per_runner_aux_labels`):

- `count = len(pair_legs)`:
  - `count < 2`: naked → `label = 0.0`
  - `count >= 2` AND any leg has `force_close=True`: force-closed → `label = 0.0`
  - `count >= 2` AND no force-close: cleanly matured or agent-closed → `label = 1.0`

When multiple pairs opened on the SAME step (can happen when two
runners are signalled at the same tick), assign the maximum label
at that step. If any one of them matured cleanly, the step gets
`label = 1.0`.

### 4. Tests (`tests/test_v2_per_transition_credit.py`)

Seven tests:

1. `test_matured_pair_label_1_at_open_step` — two-race rollout;
   race 1 opens pair on runner 0 at step 40, pair matures.
   Assert `mature_label[40] == 1.0`, `mature_mask[40] == True`.
2. `test_force_closed_pair_label_0_at_open_step` — pair opens at
   step 40, force-closed (second leg has `force_close=True`).
   Assert `mature_label[40] == 0.0`, `mature_mask[40] == True`.
3. `test_naked_pair_label_0_at_open_step` — pair opens at step 40,
   only one leg ever matches (naked). Assert same as force-closed.
4. `test_non_open_steps_have_mask_false` — 200-step rollout with
   one pair opened at step 40. Assert
   `mature_mask.sum() == 1` and `mature_mask[40] == True`.
5. `test_multiple_pairs_same_step_max_label` — two pairs open at
   step 40; one matures (label 1.0), one goes naked (label 0.0).
   Assert `mature_label[40] == 1.0`.
6. `test_close_legs_not_tracked` — confirm that a bet with
   `close_leg=True` is not added to `pair_open_records`; the
   close tick has `mature_mask == False`.
7. `test_empty_rollout_no_pairs` — rollout with no bets placed;
   `mature_label` is all 0.0, `mature_mask` is all False, no
   crash.

## Stop conditions

- **Stop and ask** if `env.bet_manager.bets` is replaced (not
  appended to) mid-rollout when a race ends. If it resets per
  race, the `bets[bets_before:]` diff will miss bets from
  previous races. Check whether `all_settled_bets` is the right
  source instead. The key invariant: each new bet's `pair_id` must
  be traceable to an index in the rollout's `Transition` list.
- **Stop and ask** if `Transition` is frozen (e.g. `@dataclass(frozen=True)`)
  AND adding a `pair_open_records` return value from the collector
  requires a non-trivial refactor of the caller. Document the
  blocker; don't force it.
- **Stop and ask** if two pairs with the same `pair_id` appear in
  different races (pair_id collision). If pair_ids are not
  globally unique per rollout, the outcome-classification lookup
  will misassign labels.

## Done when

- All 7 tests in `tests/test_v2_per_transition_credit.py` pass.
- Existing tests unchanged.
- `_collect_rollout` returns `pair_open_records` alongside
  `transitions` (whatever the existing return value structure is).
- No trainer changes yet — `pair_open_records` is collected but
  not yet consumed. That's S02.
- Commit: `feat(rewrite): phase-9 S01 - per-transition open-step
  tracking + assign_per_transition_labels`.
