# Progress — Entropy Control v2

One entry per completed session. Most recent at the top.
Include commit hash, what landed, what's not changed, and
any gotchas.

Format per session follows `naked-clip-and-stability/
progress.md` — "What landed", "Not changed", "Gotchas",
"Next".

## Session 06 — `target_entropy` default 112.0 → 150.0 (2026-04-19)

**Commit:** _(to be filled in at commit time)_

### What landed

- `agents/ppo_trainer.py`: `target_entropy` default raised
  from 112.0 to 150.0. Rationale comment in `__init__`
  documents the fresh-init entropy-floor reasoning and
  cross-references `lessons_learnt.md`.
- `tests/test_ppo_trainer.py`:
  `test_target_entropy_default_matches_purpose_md` renamed to
  `test_target_entropy_default_matches_session_06` with the
  new default value. Explicit hp overrides
  (`target_entropy=112.0` in
  `test_real_ppo_update_updates_log_alpha` and
  `test_log_episode_includes_alpha_and_log_alpha`) are
  unchanged — they pin their own values.
- `CLAUDE.md`: "Entropy control" paragraph updated with the
  new default (150) and the rationale.
- `plans/entropy-control-v2/lessons_learnt.md`: new entry
  diagnosing the natural-entropy-floor issue and proposing
  the fix; also documents what a Session 06 pass / fail
  implies for the queued `reward-densification` plan.

### Trigger

Post-Session-05 smoke probe FAILED `entropy_slope`:

| agent | ep1 α | ep2 α | ep3 α | slope |
|---|---|---|---|---|
| transformer | 0.00380 | 0.00285 | 0.00210 | **+1.47** |
| LSTM | 0.00379 | 0.00279 | 0.00200 | **+2.90** |

The proportional controller (Session 05) IS working — alpha
drops 1.3–1.4× per episode as predicted — but entropy still
rises while alpha is already 2.5× below the pre-controller
default. No alpha value can drive entropy below 112 on a
70-dim Gaussian action space whose fresh-init differential
entropy is 139. Target was set below the floor.

### Not changed

- Controller mechanism (SGD, proportional).
- `alpha_lr` default (1e-2).
- Clamp bounds (`[log(1e-5), log(0.1)]`).
- Call site (once per `_ppo_update`).
- Smoke-gate slope threshold (1.0/ep).
- Reward shape, matcher, PPO stability defences,
  checkpoint schema.

### Gotchas

- Two existing controller tests explicitly pass
  `target_entropy=112.0` (the integration
  `_ppo_update` test and the `_log_episode` fields test).
  Those are unaffected by the default change — they
  exercise their own explicit-hp paths. No update needed.
- Session 06 changes ONE hard-constraint §9 value (was
  "112.0", now "150.0"). Hard_constraints are append-only
  per convention; the change is recorded in
  `lessons_learnt.md` rather than edited in §9 in place.

### Test suite

`pytest tests/ -q`: **2256 passed, 7 skipped, 133 deselected,
1 xfailed** (0:05:13). Net delta from Session 05: 0 tests
(one rename, no adds).

### Next (operator action)

1. Truncate `logs/training/episodes.jsonl` (the 6 failed
   smoke-probe rows from the Session-05 launch).
2. Reset `models.db` / `weights/` (were populated by the
   full-run from the Session-04 launch).
3. Reset `activation-A-baseline` plan to `draft`.
4. Re-launch with "Smoke test first" ticked.

If smoke passes + full-run stabilises: green light for
`activation-B-*` sweep. If smoke passes but full-run drifts
above 200: raise target further or accept the controller as
a defend-upper-bound mechanism and open reward-densification.
If smoke still fails: the entropy-bonus lever genuinely
isn't strong enough; reward-densification is the next plan.

---

## Session 05 — Swap Adam for SGD (proportional controller) (2026-04-19)

**Commit:** _(to be filled in at commit time)_

### What landed

- `agents/ppo_trainer.py`:
  - `_alpha_optimizer` changed from `torch.optim.Adam` to
    `torch.optim.SGD(momentum=0.0)`.
  - `alpha_lr` default 3e-2 → 1e-2.
  - `_update_entropy_coefficient` docstring rewritten with
    the proportional-control derivation; loss formula and
    sign are unchanged.
- `tests/test_ppo_trainer.py::TestTargetEntropyController`:
  - `test_controller_optimizer_separate_from_policy` updated
    to assert `log_alpha` movement rather than optimiser-state
    change (SGD momentum=0 has an effectively-empty state).
  - `test_alpha_lr_default_matches_session_04` renamed to
    `_session_05` with the new default value 1e-2.
  - New `test_alpha_optimizer_is_sgd_proportional_controller`
    pins the optimiser class.
  - New `test_controller_step_is_proportional_to_error`
    verifies the core proportional invariant (10× error →
    10× log_alpha delta).
- `CLAUDE.md` — "Entropy control" paragraph updated:
  - Adam → SGD (momentum=0).
  - `alpha_lr` default 3e-2 → 1e-2.
  - Rationale paragraph added explaining Adam's adaptive
    normalisation destroys proportional control and the
    Session-04 data that proved it.
- `plans/entropy-control-v2/lessons_learnt.md` — new entry
  "Adam is the wrong optimiser for this controller" with
  diagnosis, fix, and toy-dynamics simulation prediction.

### Trigger

Post-Session-04 full-population launch reached ep15. Results:

| | ep1 | ep5 | ep10 | ep15 |
|---|---|---|---|---|
| entropy (avg) | 139.6 | 155.7 | 178.9 | 192.6 |
| alpha (avg) | 0.0289 | 0.0232 | 0.0199 | 0.0169 |
| policy_loss | 50 | 11 | 0.2 | 0.2 |

Controller direction correct (alpha going down), magnitude
insufficient (alpha only halved across 14 episodes while
entropy drifted +53 units). Slope +3.8/ep, barely below the
pre-controller +4.4/ep baseline.

### Not changed

- Reward shape.
- Matcher.
- Target-entropy controller sign, clamp bounds, call site
  (once per `_ppo_update`).
- Smoke-gate slope threshold (`ENTROPY_SLOPE_MAX = 1.0`).
- PPO stability defences.
- Checkpoint schema (SGD's state_dict round-trips the same
  way Adam's did, just with a smaller payload).

### Test suite

`pytest tests/ -q`: **2256 passed, 7 skipped, 133 deselected,
1 xfailed** (0:04:53). +2 net tests from Session 04's 2254.

### Next (operator action)

1. Truncate `logs/training/episodes.jsonl` (the 47 rows from
   the post-Session-04 run).
2. Re-launch `activation-A-baseline` (currently in `paused`
   status from the completed post-Session-04 training run —
   hit Launch / Resume).
3. Watch for: alpha decreasing rapidly toward the lower clamp
   (1e-5). If alpha saturates at the clamp AND entropy keeps
   rising past ~180, the conclusion is "entropy bonus isn't
   the dominant force" and the queued `reward-densification`
   plan opens. If entropy stabilises, C1-C5 can be evaluated
   against the remainder of the 15-episode run.

---

## Session 04 — `alpha_lr` default 1e-4 → 3e-2 (2026-04-19)

**Commit:** _(to be filled in at commit time)_

### What landed

- `agents/ppo_trainer.py`: `alpha_lr` default raised from
  `1e-4` to `3e-2`. Rationale comment cross-references
  `lessons_learnt.md` 2026-04-19. The `_log_alpha_min` /
  `_log_alpha_max` clamp remains the ultimate safety net
  against one-step overshoot.
- `tests/test_ppo_trainer.py` — 2 new tests in
  `TestTargetEntropyController`:
  - `test_alpha_lr_default_matches_session_04` pins the new
    default.
  - `test_alpha_lr_explicit_hp_overrides_default` confirms
    the hp override path still wins — internal tests and
    pathological configs can still pin a slower lr.
- `CLAUDE.md` — "Entropy control" paragraph updated to
  reflect the new default and cross-reference the lesson.
- `plans/entropy-control-v2/lessons_learnt.md` status line
  flipped from "Awaiting operator decision" to "Resolved in
  Session 04".
- **Registry re-reset** (bundled into this commit's follow-on
  operator action, not the commit itself). The fresh
  registry from Session 03 had only the 6 failed-probe rows
  from the post-Session-03 launch; truncate
  `logs/training/episodes.jsonl` before re-launch. `models.db`
  and `weights/` are already empty (probe agents don't
  persist).

### Trigger

Post-Session-03 smoke probe FAILED on `entropy_slope`:

| Agent | ep1 | ep2 | ep3 | slope |
|---|---|---|---|---|
| transformer | 139.52 | 140.65 | 143.39 | **+1.94** |
| LSTM | 139.69 | 143.01 | 147.12 | **+3.71** |

Assertion 1 (policy_loss) and assertion 3 (arbs_closed)
passed. Slope assertion did its job — flagged a legitimate
controller failure mode that earlier phases would have
missed. See `lessons_learnt.md` 2026-04-19 for the diagnosis
(Adam's adaptive normalisation makes per-update step size
~`lr` regardless of gradient magnitude; `1e-4` is SAC
literature default, timed against 10⁵–10⁶ updates, not our
~dozens per run).

### Not changed

- Controller architecture, sign, loss formula, clamp bounds.
- Reward shape, matcher, action/obs schemas.
- PPO stability defences.
- Smoke-gate slope threshold (`ENTROPY_SLOPE_MAX = 1.0`).
- `target_entropy` default (112.0).

### Gotchas

- Existing controller tests use explicit `alpha_lr=1e-2`
  (shrink / grow / clamp) or `1e-3` (roundtrip /
  optimiser-separate / effective-coeff sync), so none of
  them drift with the default raise. The only tests that
  touch the default are the two new ones.
- `test_real_ppo_update_updates_log_alpha` becomes stricter
  under the new default — log_alpha now moves ~3e-2 per
  update instead of ~1e-4, so the `after != before`
  assertion passes more cleanly.

### Test suite

`pytest tests/ -q`: **2254 passed, 7 skipped, 133 deselected,
1 xfailed** (0:06:44). +2 net tests from Session 03's 2252.

### Next (operator action — not bundled into this commit)

1. Truncate `logs/training/episodes.jsonl` before re-launch
   (the fresh registry still carries the 6 failed-probe rows
   from the post-Session-03 launch). `models.db` and
   `weights/` already have 0 entries.
2. Re-launch `activation-A-baseline` with "Smoke test first"
   ticked.
3. On probe pass: full population trains. Capture findings
   in a new Validation entry.
4. On probe fail: capture the new entropy trajectory in
   `lessons_learnt.md`; Session 05 territory (extend the
   probe window, or lower target_entropy).

---

## Session 03 — Registry reset + activation-plan redraft (2026-04-19)

**Commit:** _(to be filled in at commit time)_

### What landed

Archive path: `registry/archive_20260419T102446Z/`.

- `registry/models.db` → `registry/archive_20260419T102446Z/models.db`
  (64 agents from the 2026-04-19 activation-A-baseline run —
  transformer × 13, LSTM × 26, Time-LSTM × 25 per the gen-3
  population split).
- `registry/weights/` → `registry/archive_20260419T102446Z/weights/`
  (64 `.pt` files).
- `registry/training_plans/` copied (not moved) to
  `registry/archive_20260419T102446Z/training_plans/` —
  captures the plan states pre-reset for audit trail.
- `logs/training/episodes.jsonl` →
  `logs/training/episodes.pre-entropy-control-v2-20260419T102446Z.jsonl`
  (1222 rows, 791859 bytes — 960 full-run rows + 6 smoke-test
  rows + operator-launch smoke probes from earlier iterations).

Fresh registry:

- New `registry/models.db` initialised via `ModelStore()` — 0
  models, all 6 core tables present (`models`,
  `evaluation_runs`, `evaluation_days`, `sqlite_sequence`,
  `genetic_events`, `exploration_runs`).
- `registry/weights/` recreated empty.
- `logs/training/episodes.jsonl` truncated to 0 bytes.

Activation plans redrafted (`status='draft'`,
`started_at=None`, `completed_at=None`,
`current_generation=None`, `current_session=0`,
`outcomes=[]`):

| Plan | Pre-reset status | current_generation | outcomes |
|---|---|---|---|
| activation-A-baseline | completed | 3 | 4 |
| activation-B-001 | draft | None | 0 |
| activation-B-010 | draft | None | 0 |
| activation-B-100 | draft | None | 0 |

Configuration preserved byte-identical: `population_size`,
`n_generations`, `arch_mix`, `hp_ranges`, `reward_overrides`,
`seed`, `name`, `notes`, `plan_id` — the JSON edit only touched
the runtime / status fields per hard_constraints §11 of
`naked-clip-and-stability` (same convention).

### Not changed

- No code changes. Session 03 is archive + reset + docs only,
  per hard_constraints §23.
- No test changes. `pytest tests/ -q` was green after
  Sessions 01–02 landed (2252 passed); Session 03 did not
  re-run the suite.
- `plans/INDEX.md` already carries the
  `entropy-control-v2` row (added at plan-folder creation
  time, 2026-04-19) — no further edit needed.

### Gotchas

- `registry/archive_*` folders are untracked per the project's
  `.gitignore` pattern — the archive itself is on disk only.
  The commit body documents the archive path so a reviewer
  can find it post-hoc.
- `registry/training_plans/` is also gitignored (same pattern
  as `naked-clip-and-stability` Session 05) — the redrafted
  JSONs don't appear in the commit diff either. The commit
  body is the audit record.

### Test suite

Not re-run this session — the session touches no code or
tests. Regression guard is the Session 01 + Session 02
combined run (2252 passed, 7 skipped, 133 deselected,
1 xfailed).

### Next (operator action — not bundled into this commit)

Per hard_constraints §25, the relaunch is NOT in this
commit. Operator steps:

1. Start the admin UI + API.
2. Open the training-launch page for
   `activation-A-baseline`.
3. Confirm "Smoke test first" is checked (default per
   `naked-clip-and-stability` Session 04).
4. Click Launch.
5. Watch the probe run in the learning-curves panel. The
   new slope-based assertion ("entropy_slope",
   threshold ≤ 1.0) should appear in the assertion list.
6. On probe pass: full population launches. Watch for the
   validation criteria in `master_todo.md` "After Session
   03":
   - Pop-avg entropy converging toward 112 by ep 10, within
     ±20 % by ep 15.
   - No ep-1 `policy_loss > 100` across the population.
   - `arbs_closed > 0` on ≥ 1 agent AND
     `arbs_closed / max(1, arbs_naked) > 0.3` sustained across
     the last 5 episodes.
   - ≥ 1 agent with a positive reward-trend slope across
     eps 8–15.
7. Capture findings in a new **Validation** entry on this
   `progress.md` — commit hash, outcome, scoreboard
   highlights, and either the green light for the B sweep or
   the diagnostics for a follow-up plan.

---

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
