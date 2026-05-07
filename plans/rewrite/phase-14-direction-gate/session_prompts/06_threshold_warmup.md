---
session: phase-14-direction-gate / S06
phase: rewrite/phase-14-direction-gate
parent_purpose: ../purpose.md
---

# S06 — Threshold cold-start warmup

## Context

Read `purpose.md`, `findings.md` (smoke result), and S05 (the
gate-mask capture fix).

The smoke surfaced a second issue: at gate threshold ≥0.88 on a
fresh-init head, 3 of 4 agents emitted ZERO bets across the
5-day window. Fresh-init head sigmoid output sits near 0.5;
threshold 0.88+ blocks essentially every runner. PPO has no
reward gradient → the agent never learns to act → stays
NOOP-only forever.

This is exactly the regime the threshold gene range [0.5, 0.95]
was designed to explore (the strict-gate region is where the
OOS probe found profitability). Cold-starting agents in that
regime starves them; we need warm-up.

## Pre-reqs

- Read `agents_v2/discrete_policy.py` after S05 (with
  apply_direction_gate kwarg + captured-mask path).
- Read `training_v2/discrete_ppo/trainer.py` —
  `_bc_warmup_eps` is the precedent pattern (same shape:
  per-agent eps counter, linear interpolation toward gene value).
- Read CLAUDE.md "BC-pretrain warmup handshake (2026-04-19)".

## Design decisions

### D1. Anneal threshold from 0.5 → gene value across N PPO updates

Linear interpolation:

```
def effective_threshold(self, eps_since_start: int) -> float:
    if not self.direction_gate_enabled:
        return self.direction_gate_threshold
    if self._gate_warmup_eps <= 0:
        return self.direction_gate_threshold
    if eps_since_start >= self._gate_warmup_eps:
        return self.direction_gate_threshold
    frac = float(eps_since_start) / float(self._gate_warmup_eps)
    return 0.5 + frac * (self.direction_gate_threshold - 0.5)
```

At eps=0 the effective threshold is 0.5 (the no-op floor — gate
filters very few rows). By eps=N it's the agent's gene value.

### D2. New gene: `direction_gate_warmup_eps: int = 5`

Operator-controlled, default 5 generations. Mirrors
`bc_target_entropy_warmup_eps` (also default 5).

NOT a Phase 5 GA-evolved gene — it's a cohort-wide knob the
operator tunes once. Same precedent as `bc_pretrain_steps`.

### D3. The eps counter

`DiscreteLSTMPolicy` doesn't currently know what episode/gen
it's on. The trainer DOES — `self._eps_since_bc` is the existing
counter. New analogue: `self._eps_since_gate_start: int`,
incremented per `_ppo_update`.

The trainer reads `effective_threshold(eps_since_gate_start)`
and writes it into the policy via a setter
`policy.set_effective_gate_threshold(value)` before each
forward pass. Or — cleaner — the policy reads
`self._eps_since_gate_start` directly via a method the trainer
sets:

```python
# In trainer
def set_eps_since_gate_start(self, eps: int):
    self._eps_since_gate_start = int(eps)

# Trainer increments after each rollout/update.
self._eps_since_gate_start += 1
```

Pass to policy via a property or method call before forward.

Actually the cleanest is just to have the policy expose
`effective_direction_gate_threshold` as a property reading from
a value the trainer pokes in:

```python
# In DiscreteLSTMPolicy
self._effective_gate_threshold: float | None = None

def set_effective_gate_threshold(self, value: float) -> None:
    self._effective_gate_threshold = float(value)

# In _apply_direction_gate
threshold = (
    self._effective_gate_threshold
    if self._effective_gate_threshold is not None
    else self.direction_gate_threshold
)
```

The trainer computes the effective threshold per-update and
calls `policy.set_effective_gate_threshold(value)` before
rolling out the next episode.

### D4. Worker plumbing

`worker.py::_train_one_agent`:
- Read `direction_gate_warmup_eps` from `hp` (default 5).
- Pass to trainer constructor or set on trainer post-construction.
- Trainer increments `_eps_since_gate_start` on each rollout.

Mirror the existing `bc_target_entropy_warmup_eps` handling.

### D5. Diagnostics

`EpisodeStats` gains `effective_gate_threshold_active: float`
so per-episode JSONL logs the warm-up trajectory.

## Deliverables

1. **`DiscreteLSTMPolicy`** (discrete_policy.py):
   - `_effective_gate_threshold` field.
   - `set_effective_gate_threshold(value)` setter.
   - `_apply_direction_gate` reads the effective value, falling
     back to `self.direction_gate_threshold`.

2. **`DiscretePPOTrainer`** (trainer.py):
   - `_eps_since_gate_start: int = 0`.
   - `_gate_warmup_eps: int` from `hp.get("direction_gate_warmup_eps", 5)`.
   - Per `_ppo_update`: compute `effective = 0.5 + frac × (gene - 0.5)`,
     call `policy.set_effective_gate_threshold(effective)`, increment
     counter.
   - `EpisodeStats.effective_gate_threshold_active` populated.

3. **`CohortGenes`** (genes.py):
   - New field `direction_gate_warmup_eps: int = 5`.
   - `to_dict` extended.
   - Sample / mutate paths pin to default (operator-controlled).

4. **Worker** (worker.py):
   - `_PHASE14_TRAINER_HP_KEYS` += {"direction_gate_warmup_eps"}.
   - `_build_trainer_hp` passes it through.

5. **Tests**:
   - `test_warmup_starts_at_floor` — at eps=0 the effective
     threshold is 0.5.
   - `test_warmup_reaches_gene_value` — at eps=N the effective
     threshold equals the gene value.
   - `test_warmup_inactive_when_eps_zero` —
     `direction_gate_warmup_eps=0` immediately uses gene value
     (no warmup).
   - `test_warmup_inactive_when_gate_disabled` — disabled gate
     ignores warmup config.

## Stop conditions

- **Stop and ask** if the warmup trajectory introduces NaN /
  inf in `approx_kl`. The S05 fix should have made the trainer
  robust to threshold changes mid-training; if not, S05 needs
  re-checking.

## Done when

- Warmup linear-interp from 0.5 → gene value over N eps.
- `direction_gate_warmup_eps` flows through gene → worker →
  trainer → policy.
- Tests pass.
- Smoke re-run: agents at strict thresholds (0.88+) emit
  ≥50 bets/day by gen 2 (warmup at default 5 eps still active
  in gen 1, fully strict by gen 5).
- Commit: `feat(rewrite): phase-14 S06 - direction_gate_threshold
  warmup`.
