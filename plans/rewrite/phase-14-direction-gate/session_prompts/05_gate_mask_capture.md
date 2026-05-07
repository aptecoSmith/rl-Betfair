---
session: phase-14-direction-gate / S05
phase: rewrite/phase-14-direction-gate
parent_purpose: ../purpose.md
---

# S05 — capture gate mask at rollout, reuse at PPO update

## Context

Read `purpose.md`, `findings.md` (smoke result), and
`lessons_learnt.md` first.

The smoke (`registry/_phase14_smoke_1778185382/`) surfaced a
structural bug in S03's gate: `DiscreteLSTMPolicy._apply_direction_gate`
recomputes the mask from the head's CURRENT outputs at every
forward pass. At rollout time, head outputs P_back=0.85 → action
is legal → agent samples it. At PPO update time, weights have
drifted, P_back=0.79 → same action is now masked at -inf →
`log_pi_new = -inf` → `approx_kl = inf` → per-mini-batch KL
early-stop fires after 1 mini-batch. PPO runs ~1/600th of the
budgeted updates.

15 of 39 per-day update logs in the smoke show `approx_kl=inf`.
The bug is structural; this session is the fix.

## Pre-reqs

- Read `agents_v2/discrete_policy.py` end-to-end after S03 — the
  `_apply_direction_gate` helper and the forward-pass call site.
- Read `training_v2/discrete_ppo/transition.py` —
  `Transition` and `RolloutBatch`. The new `gate_mask` field
  follows the existing `mask` (legality) precedent.
- Read `training_v2/discrete_ppo/rollout.py::_collect` — where
  the legality mask is captured per tick.
- Read `training_v2/discrete_ppo/trainer.py::_ppo_update` — where
  `mb_mask = masks[mb_idx]` is sliced per mini-batch and passed
  into `policy.forward(mb_obs, mask=mb_mask)`.

## Design decisions

### D1. New field: `RolloutBatch.gate_mask` (and `Transition.gate_mask`)

`(n_steps, action_space.n)` bool, captured at rollout time when
`policy.direction_gate_enabled=True`. `None` (or all-True
sentinel) when the gate is disabled.

When the gate is active, the rollout-time effective mask is
`legality AND gate`. Capturing the COMBINED mask (vs storing
gate-only and re-AND-ing) is simpler and avoids any chance of
the legality and gate getting re-applied in different order at
update time.

### D2. Capture pattern in `RolloutCollector._collect`

After each `policy(obs, mask=legality_mask)` call:

```python
# Effective mask = positions where masked_logits is finite.
# Already finite at rollout time (we sample from these); -inf at
# the rollout-time-blocked positions. Cast to bool: True where
# legal AND gate-pass, False elsewhere.
if policy.direction_gate_enabled:
    gate_mask_t = torch.isfinite(out.masked_logits).cpu().numpy()
    gate_mask_arr[n_steps] = gate_mask_t.squeeze(0)
else:
    gate_mask_arr[n_steps] = True  # all-True sentinel; trainer skips
```

### D3. Use pattern in `DiscretePPOTrainer._ppo_update`

When `batch.gate_mask` is not None (and not the all-True
sentinel), AND it with the legality mask before passing to
`policy.forward`:

```python
mb_legality_mask = masks[mb_idx]
if batch.gate_mask is not None and batch.gate_mask.any() and not batch.gate_mask.all():
    mb_gate_mask = gate_mask_t[mb_idx]
    mb_effective_mask = mb_legality_mask & mb_gate_mask
else:
    mb_effective_mask = mb_legality_mask
out = self.policy(mb_obs, hidden_state=mb_hidden, mask=mb_effective_mask)
```

### D4. Disable in-forward gate when `mask` is supplied

`DiscreteLSTMPolicy.forward` accepts `mask: torch.Tensor | None`.
When supplied, the mask is used as-is — the gate's in-forward
recompute is SKIPPED. Existing semantics preserved when no mask
is passed; gate-on rollouts get the in-forward mask.

```python
masked_logits = self._apply_mask(logits, mask)
if self.direction_gate_enabled and mask is None:
    # Only apply the gate's in-forward recompute when no
    # explicit mask was provided. Rollouts pass mask=None
    # (legality-only); update-time passes the captured
    # rollout-time effective mask.
    masked_logits = self._apply_direction_gate(
        masked_logits, direction_back_prob, direction_lay_prob,
    )
```

Wait — this is wrong. The rollout DOES pass a legality mask. We
need a different signal. Use a new explicit flag instead:

```python
def forward(self, obs, hidden_state=None, mask=None,
            apply_direction_gate: bool | None = None):
    ...
    masked_logits = self._apply_mask(logits, mask)
    # Gate semantics:
    #   apply_direction_gate=None (default): use the in-forward
    #     recompute when self.direction_gate_enabled is True.
    #     This is the rollout-time path.
    #   apply_direction_gate=False: skip the in-forward recompute.
    #     The trainer passes this explicitly when feeding a
    #     rollout-captured mask via `mask`.
    if apply_direction_gate is None:
        apply_direction_gate = self.direction_gate_enabled
    if apply_direction_gate:
        masked_logits = self._apply_direction_gate(...)
```

The trainer passes `apply_direction_gate=False` at update time;
the rollout collector calls without the kwarg (default applies
gate per `direction_gate_enabled`).

### D5. Backwards compatibility

Existing callers of `policy.forward(obs)` and
`policy.forward(obs, mask=...)` keep working — the new kwarg is
optional with sensible default. Tests in
`tests/test_v2_direction_gate.py` continue to pass.

## Deliverables

1. **`Transition` + `RolloutBatch`** (transition.py):
   - New optional field `gate_mask: np.ndarray | None`. Default
     None.
   - `transitions_to_rollout_batch` carries it through.

2. **`RolloutCollector._collect`** (rollout.py):
   - Pre-allocate `gate_mask_arr: (n_steps_estimate, action_n)
     bool` if `policy.direction_gate_enabled`.
   - After each forward pass, capture
     `torch.isfinite(out.masked_logits)` into the buffer.
   - On batch construction, attach the gate_mask array.

3. **`DiscreteLSTMPolicy.forward`** (discrete_policy.py):
   - Accept `apply_direction_gate: bool | None = None` kwarg.
   - When None, fall back to `self.direction_gate_enabled`.
   - When False, SKIP the in-forward gate (caller has supplied a
     pre-computed mask via `mask`).

4. **`DiscretePPOTrainer._ppo_update`** (trainer.py):
   - When `batch.gate_mask` is not None, AND it with the
     legality mask. Pass `apply_direction_gate=False` to the
     policy.

5. **Tests** — `tests/test_v2_direction_gate.py`:
   - `test_gate_mask_captured_at_rollout` — synthetic rollout
     under gate-on; assert `batch.gate_mask` carries the right
     shape + bool values matching the rollout-time
     `masked_logits`.
   - `test_gate_mask_reused_at_update_keeps_kl_finite` — run
     `_ppo_update` with a captured gate mask; assert all
     `approx_kl` values are finite. Mock the head's weights to
     drift between rollout and update so the in-forward gate
     WOULD have masked stored actions; with the captured mask in
     place, KL stays finite.

## Stop conditions

- **Stop and ask** if the captured gate mask diverges from the
  rollout-time effective mask in any tick. The capture is
  load-bearing for everything downstream.

- **Stop and ask** if the trainer's `approx_kl` still goes to inf
  with the captured-mask path. The fix isn't sufficient and the
  bug is somewhere else.

## Done when

- `RolloutBatch.gate_mask` populated when gate is active.
- `_ppo_update` passes the stored mask + `apply_direction_gate=False`.
- Smoke re-run: `approx_kl` is finite across all updates with
  gate active.
- Tests in `tests/test_v2_direction_gate.py` pass (new + existing).
- Commit: `fix(rewrite): phase-14 S05 - capture gate mask at
  rollout, reuse at PPO update`.
