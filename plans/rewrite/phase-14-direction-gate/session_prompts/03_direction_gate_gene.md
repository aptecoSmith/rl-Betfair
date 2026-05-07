---
session: phase-14-direction-gate / S03
phase: rewrite/phase-14-direction-gate
parent_purpose: ../purpose.md
---

# S03 — `direction_gate_threshold` per-agent gene + policy mask

## Context

Read `purpose.md`, `hard_constraints.md` (especially §10–§16),
and S01's deliverable.

The probes showed that even with the per-runner head architecture
(S01) and augmented features (S02), the cohort's PPO trajectory may
not learn to ACT on the direction signal — phase-13's NULL
suggested the credit-assignment path is too noisy at this scale.

S03 adds the belt-and-braces: a per-agent gene
`direction_gate_threshold` whose effect is **mechanical**, not
learned. When enabled, the policy masks `OPEN_BACK_i` /
`OPEN_LAY_i` actions whose runner's max(P_back, P_lay) sits below
the threshold. The agent gets no choice — even if PPO learned
nothing useful from the head, the env enforces selectivity.

The gate gene is range-clamped to **[0.5, 0.95]** (operator-
agreed, with the upper-cap pushback I made — at 0.99+ an agent
opens 0 pairs and starves PPO of training signal).

A separate `direction_gate_enabled: bool` flag gates the
mechanism on/off. Default `False` = byte-identical to S01+S02
without S03.

## Pre-reqs

- Read `agents_v2/discrete_policy.py::DiscreteLSTMPolicy.forward`
  and `_apply_mask` after S01 lands — the new mask plumbing
  reuses the legality-mask hook.
- Read `agents_v2/action_space.py` — confirm the categorical
  layout (NOOP, OPEN_BACK_0..R-1, OPEN_LAY_0..R-1, CLOSE_0..R-1).
  The gate touches OB and OL slots only.
- Read `training_v2/cohort/genes.py` — Phase 5 gene scaffolding
  pattern. The gate gene mirrors `open_cost` / `naked_loss_scale`
  shape.

## Design decisions resolved here

### D1. Gate lives in the policy's forward pass

`DiscreteLSTMPolicy.__init__` accepts new optional kwargs
`direction_gate_enabled: bool = False` and
`direction_gate_threshold: float = 0.5`. Stored on `self`. The
threshold value is clamped to [0.5, 0.95] at construction (defensive
guard against callers passing values outside the gene range).

In `forward()`, AFTER the existing legality mask is applied to
`logits` and BEFORE the `Categorical` is constructed:

```python
if self.direction_gate_enabled and self.direction_gate_threshold > 0.5:
    # max P over the two sides per runner.
    direction_max = torch.maximum(
        direction_back_prob, direction_lay_prob,
    )  # (batch, R)
    gate_pass = direction_max >= self.direction_gate_threshold
    # OPEN_BACK indices: 1..R; OPEN_LAY indices: R+1..2R; CLOSE: 2R+1..3R.
    # NOOP at index 0 — never gated.
    # Build a per-action gate mask matching the categorical layout.
    n_actions = logits.shape[-1]
    gate_mask = torch.ones(
        batch, n_actions, dtype=torch.bool, device=logits.device,
    )
    R = self.max_runners
    # OPEN_BACK_i lives at action_idx = encode(OPEN_BACK, i) = 1 + i.
    # OPEN_LAY_i  lives at action_idx = encode(OPEN_LAY,  i) = 1 + R + i.
    # We mask these where gate_pass[:, i] is False.
    open_back_idx = torch.arange(R, device=logits.device) + 1
    open_lay_idx = torch.arange(R, device=logits.device) + 1 + R
    # gate_pass: (batch, R). Broadcast into the action layout.
    gate_mask[:, open_back_idx] = gate_pass
    gate_mask[:, open_lay_idx] = gate_pass
    # Apply: turn masked OPEN actions into -inf, leaving NOOP and
    # CLOSE_* untouched.
    masked_logits = masked_logits.masked_fill(~gate_mask, float("-inf"))
```

### D2. Why apply BEFORE the Categorical

`Categorical(logits=...)`'s softmax handles `-inf` cleanly — the
masked positions get probability 0 and remaining positions
re-normalise. Sampling never returns a masked index. PPO's
`new_log_prob = dist.log_prob(stored_action)` returns `-inf` only
if a stored action is itself in the masked set — which can't
happen because the mask was applied at rollout time too.

The gate mask AND-s with the legality mask (both must pass), so
no double-mask conflicts.

### D3. NOOP is NEVER gated; CLOSE is NEVER gated

Per `hard_constraints.md §14, §15`. The gate is purely an
OPEN-side filter. An agent at threshold 0.95 with no high-
confidence runners simply emits NOOP — that's the selectivity
working. CLOSE actions stay legal so the agent can always exit
existing positions.

### D4. Gene wiring

`CohortGenes` gains:

```python
direction_gate_enabled: bool = False
direction_gate_threshold: float = 0.5
```

Both default to "disabled". `direction_gate_enabled` is a
**plan-level boolean** (operator passes `--reward-overrides
direction_gate_enabled=true`); `direction_gate_threshold` is a
**Phase 5 gene** evolved per-agent via `--enable-gene
direction_gate_threshold` like the other Phase 5 genes.

Range / sampling per `genes.py`:

```python
DIRECTION_GATE_THRESHOLD_RANGE: tuple[float, float] = (0.5, 0.95)
```

Sampled uniform on [0.5, 0.95]. Default-when-disabled = 0.5
(the gate's no-op floor).

Add to `_PHASE13_TRAINER_HP_KEYS` in `worker.py`:
- `direction_gate_enabled` (bool passthrough)
- `direction_gate_threshold` (float gene value)

The worker passes both into the policy's constructor via
`policy_factory(direction_gate_enabled=..., direction_gate_threshold=...)`.

### D5. Plumbing through the cohort runner

The policy is constructed in two places:
- `training_v2/cohort/worker.py::_train_one_agent` (cohort path).
- `training_v2/discrete_ppo/train.py::main` (single-agent path).

Both must pass the gate config to `DiscreteLSTMPolicy.__init__`.
Use the existing `hp` dict pattern.

### D6. Diagnostics on the scoreboard

`EpisodeStats` gains `direction_gate_enabled_active: bool` and
`direction_gate_threshold_active: float` so the JSONL row makes
the gate visible in post-hoc analysis.

`UpdateLog` doesn't gain new fields — the gate doesn't change
PPO update behaviour, only which actions PPO sees.

The `episode_complete` event-log line gains a `gate=on@T=0.85`
suffix when active so live monitoring shows it.

### D7. NOT a learned threshold

The threshold is FIXED for an agent's lifetime (sampled at gene
draw, never mutated within an agent). Per
`hard_constraints.md §16` precedent — like `alpha_lr`, the gate
gene is set once and stable. The GA evolves it across
generations, not within an agent's training.

## Deliverables

### 1. Policy changes

- `agents_v2/discrete_policy.py::DiscreteLSTMPolicy.__init__`
  accepts `direction_gate_enabled`, `direction_gate_threshold`.
- `forward()` applies the gate mask per D1 / D2.
- The mask AND-s with the legality mask (no override).

### 2. Gene + worker plumbing

- `training_v2/cohort/genes.py`: add the two fields, range
  constant, defaults, sample / mutate / crossover paths.
- `training_v2/cohort/worker.py`: pass the gate config from `hp`
  into `DiscreteLSTMPolicy(...)` at construction.
- `_PHASE13_TRAINER_HP_KEYS` (or rename to `_PHASE14_*` if
  cleaner) extended.

### 3. Diagnostics

- `EpisodeStats`: 2 new fields.
- Operator log line + scoreboard row include the gate state.
- `train_per_day` row: include `gate_threshold_active`.

### 4. Tests — `tests/test_v2_direction_gate.py`

NEW file. Tests:

1. `test_gate_disabled_is_byte_identical_to_phase13` — same seed,
   same obs, gate disabled produces identical policy outputs to a
   policy without the gate plumbing.

2. `test_gate_masks_open_back_when_p_back_below_threshold` —
   construct a synthetic policy whose `direction_back_prob` is
   forced to a known low value; assert OPEN_BACK_i logit is
   `-inf` after the mask.

3. `test_gate_masks_open_lay_when_p_lay_below_threshold` —
   symmetric.

4. `test_gate_does_not_mask_when_max_above_threshold` — at least
   one of P_back, P_lay above threshold → both
   OPEN_BACK_i and OPEN_LAY_i remain finite (the gate uses
   `max(P_back, P_lay)`, not per-side; we want the agent free to
   pick the side it prefers).

5. `test_gate_does_not_touch_noop_or_close` — NOOP logit stays
   finite at any threshold; CLOSE_i logit stays finite at any
   threshold.

6. `test_gate_ands_with_legality_mask` — when both legality and
   direction-gate would mask a position, the result is masked.
   When legality masks but direction doesn't, the position is
   masked. When direction masks but legality doesn't, masked.

7. `test_gate_threshold_clamped_to_range` — constructing the
   policy with `direction_gate_threshold=0.99` clamps to 0.95;
   with 0.4 clamps to 0.5.

8. `test_gate_gene_evolves_per_agent` — sample two agents with
   `enabled_set={direction_gate_threshold}`; assert the two
   threshold values are independent samples.

### 5. Smoke-run validation

Before declaring done:

```bash
python -X utf8 -m training_v2.cohort.runner \
  --n-agents 2 --generations 1 --days 4 --n-eval-days 1 \
  --output-dir registry/_phase14_s03_smoke_$(date +%s) \
  --seed 42 --device cuda \
  --reward-overrides direction_prob_loss_weight=0.1 \
  --reward-overrides direction_gate_enabled=true \
  --enable-gene direction_gate_threshold
```

Confirm:
- The cohort runs to completion without errors.
- Each agent's eval rollout has `eval_pairs_opened > 50`.
- The scoreboard row carries
  `direction_gate_threshold_active` and
  `direction_gate_enabled_active`.

If `eval_pairs_opened ≤ 50` on either agent, the threshold range
[0.5, 0.95] may be too aggressive at the cohort's distribution
of direction probabilities — investigate before launching S04.

## Stop conditions

- **Stop and ask** if the gate-disabled byte-identity test fails.
  Some other code path is reading the new fields when they should
  be inert.

- **Stop and ask** if the smoke run's eval rollouts produce
  `pairs_opened ≤ 50`. Either:
  - The fresh-init head outputs P > 0.5 too rarely (tuning needed).
  - The threshold clamp is wrong direction.
  - PPO collapsed to NOOP-only because of the gate.

- **Stop and ask** if PPO `approx_kl` blows up after the gate is
  active. The `Categorical(logits=-inf, ...)` path can produce
  numerical instability if ALL non-NOOP logits go to `-inf`
  (shouldn't happen as long as NOOP stays finite, but worth
  watching).

## Done when

- `DiscreteLSTMPolicy` consumes `direction_gate_enabled` /
  `direction_gate_threshold`.
- 8 tests in `tests/test_v2_direction_gate.py` pass.
- CohortGenes + worker pass the config through to the policy.
- Smoke cohort run shows the gate working (eval row has the
  field, agent opens > 50 pairs).
- Existing tests pass (`pytest tests/test_v2_*.py -q` green).
- Lessons-learnt entry documents:
  - Observed direction_max distribution on a real day (so
    operators have a reference for sane threshold ranges).
  - Smoke-run pairs_opened at min/max threshold — confirms the
    [0.5, 0.95] clamp is the right range.
- Commit: `feat(rewrite): phase-14 S03 - direction_gate_threshold
  gene + policy-side action mask`.
