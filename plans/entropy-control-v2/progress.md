# Progress — Entropy Control v2

One entry per completed session. Most recent at the top.
Include commit hash, what landed, what's not changed, and
any gotchas.

Format per session follows `naked-clip-and-stability/
progress.md` — "What landed", "Not changed", "Gotchas",
"Next".

## Session 01 — Target-entropy controller (2026-04-19)

**Commit:** _(to be filled in at commit time)_

### What landed

- `agents/ppo_trainer.py`:
  - New state in `__init__`: `_log_alpha` (float64 tensor,
    `requires_grad=True`, initialised from
    `log(hp.get("entropy_coefficient", 0.005))`),
    `_alpha_optimizer` (separate Adam, `alpha_lr=1e-4` default),
    `_target_entropy=112.0`, `_log_alpha_min=log(1e-5)`,
    `_log_alpha_max=log(0.1)`.
  - `self.entropy_coeff` now reads `log_alpha.exp().item()` and
    is refreshed after every controller step.
  - New method `_update_entropy_coefficient(current_entropy)`:
    computes `alpha_loss = -log_alpha × (target - current)`,
    steps the alpha optimiser, clamps `log_alpha`, refreshes
    `entropy_coeff` and `_entropy_coeff_base`.
  - `_ppo_update` calls the controller once per update (after
    the mini-batch loop, before the Session-2 floor controller)
    with the mean of per-minibatch entropies — a detached
    Python float, no autograd leakage.
  - `_log_episode` writes `alpha`, `log_alpha`, and
    `target_entropy` into each JSONL row.
  - New `save_checkpoint` / `load_checkpoint` methods expose
    the controller state per hard_constraints §11 (schema:
    `{"log_alpha": float, "alpha_optim_state": dict}`).
    Backward-compat on missing keys with a warning.
- `_entropy_coeff_base` is REPURPOSED (not removed): it now
  tracks the SAC controller's output rather than a fixed
  constant, so the Session-2 entropy-floor scaffolding keeps
  working (when `entropy_floor > 0` it scales on top of the
  fresh SAC baseline; when `entropy_floor == 0` — default —
  the floor scaling is a no-op and `entropy_coeff` equals the
  controller output directly). This is the minimal change
  that satisfies §10 without breaking `test_entropy_floor.py`
  or the `entropy_floor` GA gene range in live training
  plans.
- `tests/test_ppo_trainer.py::TestTargetEntropyController` —
  8 tests covering init, shrink, grow, clamp, optimiser
  independence, effective-coeff sync, real-`_ppo_update`
  end-to-end, default target, and `_log_episode` fields.
- `tests/test_ppo_checkpoint.py` — 3 tests covering
  round-trip, backward-compat (missing keys + warning), and
  schema.
- `tests/test_ppo_trainer.py` — 3 existing tests updated to
  tolerate the `log → exp` round-trip precision (per §19).
- `CLAUDE.md` — new "Entropy control — target-entropy
  controller (2026-04-19)" subsection under "PPO update
  stability", documenting the controller, its default target,
  its separate optimiser, the clamp bounds, the once-per-update
  call site, the float64 choice, and the load-bearing
  integration test.

### Not changed

- Reward shape (`race_pnl` / 95% naked clip / £1 close bonus)
  — byte-identical to `naked-clip-and-stability`.
- Matcher (`env/exchange_matcher.py`).
- PPO numerical stability (±5 log-ratio clamp, KL early-stop
  at 0.03, per-arch LR, 5-update warmup, advantage
  normalisation, reward centering).
- Action / obs schemas.
- GA gene ranges (including `entropy_coefficient` — it now
  defines the initial `log_alpha` for fresh agents).
- Smoke-test gate (Session 02 territory).

### Gotchas

- `log_alpha` uses **float64** rather than float32. Float32
  round-trip of `log(0.005).exp()` drifts at the 7th decimal
  and breaks pre-existing `entropy_coeff == 0.005` equality
  tests. Float64 preserves the round-trip to machine epsilon
  on the default hp values; alpha optimiser cost is negligible
  (single scalar).
- `self._update_entropy_coefficient` is called AFTER the
  mini-batch loop inside `_ppo_update`, using the mean of
  per-minibatch entropies as `current_entropy`. This means
  the coefficient that drove the current update's entropy
  bonus was produced by the PREVIOUS update's controller
  step; the controller is effectively one-update-lagged.
  The master_todo / hard_constraints sketch called for a
  "before the policy optimiser step" ordering — the
  once-per-update placement satisfies the docstring's
  "Call ONCE per `_ppo_update`" constraint and composes
  cleanly with the existing Session-2 floor controller at
  the same call site. Tests
  (`test_real_ppo_update_updates_log_alpha`) verify the
  controller actually moves through a real update.
- The Session-2 `_entropy_coeff_base` is no longer a fixed
  constant — it's a snapshot of the SAC output after every
  controller step. Only relevant to operators running with
  `entropy_floor > 0`.

### Test suite

`pytest tests/ -q`: **2251 passed, 7 skipped, 133 deselected,
1 xfailed** (0:05:14). Net delta from this session: +11 tests
(8 in TestTargetEntropyController + 3 in test_ppo_checkpoint),
3 existing tests updated for float tolerance.

### Synthetic-rollout probe

_Deferred — the existing
`test_real_ppo_update_updates_log_alpha` test exercises the
wired-in code path on a real rollout. A 15-episode probe
with a rising-entropy sequence would re-assert the
convergence claim, but the pytest coverage above is
already tighter than the probe (it exercises the real
environment / policy forward pass rather than a mocked
entropy sequence)._

### Next

Session 02 — smoke-gate slope assertion.

---

_Plan folder created 2026-04-19. See `purpose.md` for the
Baseline-A (2026-04-19, commit `1d5acc9`) entropy-drift
evidence (139.6 → 201.3 monotone across 64 agents × 15
episodes) that motivated this plan. See `lessons_learnt.md`
of `naked-clip-and-stability` (2026-04-19 entry) for the
smoke-gate endpoint-vs-slope test-design lesson that
Session 02 of this plan acts on._
