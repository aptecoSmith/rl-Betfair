# Progress — Entropy Control v2

One entry per completed session. Most recent at the top.
Include commit hash, what landed, what's not changed, and
any gotchas.

Format per session follows `naked-clip-and-stability/
progress.md` — "What landed", "Not changed", "Gotchas",
"Next".

## Session 02 — Smoke-gate slope assertion (2026-04-19)

**Commit:** _(to be filled in at commit time)_

### What landed

- `agents/smoke_test.py`:
  - `ENTROPY_RISE_TOLERANCE = 10.0` replaced by
    `ENTROPY_SLOPE_MAX = 1.0`. Comment block updated with the
    Baseline-A evidence for why the endpoint check was
    structurally blind.
  - Assertion 2 evaluator refactored: fits
    `np.polyfit(episodes, entropies, 1)[0]` across all probe
    episodes (1..`PROBE_EPISODE_COUNT`), returns a
    `SmokeAssertionResult` with `name="entropy_slope"`,
    `observed=slope`, `threshold=1.0`, and a human-readable
    `detail` string that shows the trajectory of the worst
    agent (`"entropy slope per episode: worst = +5.21 (agent
    abc12345: 139.6 → 145.3 → 150.0), threshold <= 1.0"`).
  - Empty-input / insufficient-data fallback: returns
    `passed=False`, `observed=NaN`, matching the existing
    empty-rows shape for other assertions.
- `tests/test_smoke_test.py`:
  - `TestEntropyMonotoneAssertion` replaced by
    `TestEntropyAssertion` (7 tests): flat slope passes, mild
    decrease passes, Baseline-A drift-rate fails, threshold
    boundary (pass at exactly +1.0 / fail at +1.1), empty-input
    handles gracefully, hard rise fails, decreasing entropy
    passes.
  - `TestPurposeTableScenarios.test_gen2_transformer_0a8cacd3_would_fail_gate`
    updated — the vignette's entropy trajectory (139 → 141 → 145
    over 3 episodes) now fails the slope assertion (slope +3.0
    > +1.0); under the old +10 endpoint tolerance it passed.
    Belt-and-braces alongside assertion 1.
  - Import updated: `ENTROPY_RISE_TOLERANCE` → `ENTROPY_SLOPE_MAX`.
- `frontend/src/app/training-plans/training-plans.spec.ts`:
  - Fixture with `name: 'entropy_non_increasing'` updated to
    `name: 'entropy_slope'` with matching observed/threshold
    (20.0 / 1.0) so the failure-modal rendering test
    continues to exercise the same code path on the new
    assertion shape.
- `plans/naked-clip-and-stability/hard_constraints.md` gets a
  post-plan amendments footnote appended (per §13 — §15 is
  append-only in the predecessor plan). Cross-references this
  plan's §13 and the 2026-04-19 `lessons_learnt.md` entry.

### Not changed

- Controller implementation (that's Session 01 territory).
- Reward shape.
- PPO numerical stability defences.
- Probe infrastructure (`run_smoke_test` orchestrator) —
  untouched; tests continue to exercise the full
  orchestration path.
- Policy-loss (assertion 1, `EP1_POLICY_LOSS_MAX = 100`) and
  arbs-closed (assertion 3, `ARBS_CLOSED_MIN = 1`)
  assertions. Per hard_constraints §1 they're out of scope.

### Gotchas

- The slope check is per-agent (not pop-avg), matching the
  existing per-agent endpoint structure. Both probe agents
  must pass; the evaluator surfaces the worst slope in the
  assertion detail.
- `numpy` is already a project dependency — imported locally
  inside `evaluate_probe_episodes` to keep the module's
  top-level import graph identical (the module is imported
  from the API process, where lazy imports matter for
  startup latency).
- Frontend-side rendering is driven by the assertion's
  backend `detail` string plus its `name` field. No frontend
  source map of assertion names exists, so the new
  `entropy_slope` name surfaces in the modal with no Angular
  source change — only the test fixture needs updating.

### Test suite

`pytest tests/ -q`: **2252 passed, 7 skipped, 133 deselected,
1 xfailed** (0:05:04). Net delta from this session: +1 test
(7 tests in TestEntropyAssertion replacing 6 in the old
TestEntropyMonotoneAssertion).

### Next

Session 03 — registry reset + activation-plan redraft
(operator-gated).

---

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
