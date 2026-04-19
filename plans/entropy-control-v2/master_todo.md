# Master TODO — Entropy Control v2

Three sessions, one commit per session, hard-constrained by
`hard_constraints.md`. Sessions 01–02 are automatable;
Session 03 is operator-gated.

---

## Session 01 — Target-entropy controller (learned log_alpha)

**Status:** pending

**Deliverables:**

- `agents/ppo_trainer.py`:
  - New `__init__` state:
    - `self._log_alpha: torch.Tensor` — initialised from
      `log(hp.get("entropy_coefficient", 0.005))`, set
      `requires_grad=True`.
    - `self._alpha_optimizer: torch.optim.Adam([self._log_alpha],
      lr=hp.get("alpha_lr", 1e-4))`.
    - `self._target_entropy: float = hp.get("target_entropy",
      112.0)`.
    - `self._log_alpha_min: float = math.log(1e-5)`.
    - `self._log_alpha_max: float = math.log(0.1)`.
  - New method `_update_entropy_coefficient(current_entropy:
    float) -> None`:
    - Computes `alpha_loss = -self._log_alpha *
      (self._target_entropy - current_entropy)`.
    - Calls `self._alpha_optimizer.zero_grad();
      alpha_loss.backward(); self._alpha_optimizer.step()`.
    - Clamps `self._log_alpha.data` to
      `[_log_alpha_min, _log_alpha_max]`.
    - Reassigns `self.entropy_coeff =
      self._log_alpha.exp().item()`.
  - `_ppo_update` calls `_update_entropy_coefficient(entropy.
    mean().item())` AFTER the policy's forward+loss but BEFORE
    the policy optimiser's `.step()`. The entropy value
    passed in is detached from the policy's autograd graph
    (no leak).
  - `_log_episode` writes `alpha` and `log_alpha` fields
    into the per-episode JSONL row so the learning-curves
    panel can plot them alongside entropy.
- Remove the `_entropy_coeff_base` scaffolding (per §10) —
  it was never used beyond storing a constant.
- Checkpoint format extended (per §11):
  - `save_checkpoint` writes `log_alpha` (as `float`) and
    `alpha_optim_state` (as `dict`) into the checkpoint dict.
  - `load_checkpoint` reads both; missing keys → fresh-init
    from default + warning log.
- `tests/test_ppo_trainer.py` — new class
  `TestTargetEntropyController`:
  1. `test_log_alpha_initialises_from_entropy_coefficient`.
  2. `test_controller_shrinks_alpha_when_entropy_above_target`.
  3. `test_controller_grows_alpha_when_entropy_below_target`.
  4. `test_log_alpha_clamped_within_bounds`.
  5. `test_controller_optimizer_separate_from_policy`.
  6. `test_effective_entropy_coeff_matches_log_alpha_exp`.
- `tests/test_ppo_checkpoint.py` (new file or extend existing
  checkpoint tests):
  1. `test_checkpoint_roundtrip_preserves_log_alpha`.
  2. `test_checkpoint_backward_compat_missing_log_alpha`.
- `CLAUDE.md`: new paragraph under "PPO update stability"
  documenting the controller, its target, its separate
  optimiser, and its clamp bounds.

**Exit criteria:**

- `pytest tests/ -q` green (expect ≈ +8 net tests).
- Pre-existing `test_invariant_raw_plus_shaped_equals_
  total_reward` still green — controller is training-
  dynamics-only, doesn't touch reward.
- Pre-existing `test_real_ppo_update_feeds_per_step_mean_
  to_baseline` still green — controller slots in alongside
  reward centering, doesn't replace it.
- Pre-existing PPO stability tests from Session 02 of
  `naked-clip-and-stability` still green.
- Synthetic-rollout probe (documented in `progress.md`, not
  a pytest): 15-episode synthetic rollout with the
  controller; entropy trajectory converges to within ±15%
  of 112 by ep 15. This is the equivalent of the Session 03
  qualitative probe in the predecessor plan.

**Acceptance:** every test in
`TestTargetEntropyController` passes; the 15-episode
synthetic-rollout probe shows entropy controlled around
target; at least one test exercises the real `_ppo_update`
code path (not just the controller method in isolation, per
the 2026-04-18 units-mismatch lesson).

**Commit:** one commit, type `feat(agents)`. First line:
`add target-entropy controller (learned log_alpha, SAC-
style)`. Body includes the A-baseline entropy drift
evidence (139.6 → 201.3 across 64 agents) and the
controller design in one paragraph. Trailer notes the
pytest delta.

**Session prompt:** [`session_prompts/01_target_entropy_
controller.md`](session_prompts/01_target_entropy_controller.md).

---

## Session 02 — Smoke-gate slope assertion

**Status:** pending

**Deliverables:**

- `agents/smoke_test.py`:
  - Replace the endpoint entropy assertion (currently
    `ep3.entropy <= ep1.entropy + 10.0`) with a slope
    assertion:
    ```python
    episodes = np.array([r["episode"] for r in probe_rows], dtype=float)
    entropies = np.array([r["entropy"] for r in probe_rows], dtype=float)
    slope = np.polyfit(episodes, entropies, 1)[0]
    entropy_slope_ok = slope <= 1.0
    ```
  - Assertion label/description updated to reflect the
    change: "entropy slope ≤ 1.0 per episode" instead of
    "entropy non-increasing".
  - The assertion runs per-agent; both probe agents must
    pass.
- `tests/test_smoke_test.py` — update existing entropy
  tests + add new:
  1. `test_slope_assertion_passes_on_flat_entropy`.
  2. `test_slope_assertion_passes_on_mild_decrease`.
  3. `test_slope_assertion_fails_on_a_baseline_drift_rate`
     — slope `+5`, the A-baseline observed rate.
  4. `test_slope_assertion_at_threshold_boundary` — slope
     exactly 1.0 passes; 1.01 fails.
  5. Existing entropy-assertion tests (pass / at-threshold /
     fail) updated to use the new slope-based inputs.
- Frontend: the smoke-failure modal already shows "observed
  / threshold / PASS|FAIL" per assertion — the text label
  update from §13 flows through automatically. Verify in
  Chrome DevTools against a fabricated-failure run.
- `hard_constraints.md` in `naked-clip-and-stability/` gets
  a cross-reference footnote in §15 pointing to this plan's
  §13 for the updated assertion. (The file itself is
  append-only per `naked-clip-and-stability` rules; add the
  note at the end, don't edit §15 in place.)

**Exit criteria:**

- `pytest tests/ -q` green (expect ≈ +4 net tests, plus
  some updates to existing ones).
- `cd frontend && ng test --watch=false` green.
- Manual browser verification: fabricate a failure via
  `ep1_policy_loss_threshold=0` config override (same
  technique as Session 04's acceptance pathway), confirm
  the modal shows the slope-based assertion label.

**Acceptance:** the Session 04 smoke gate from `naked-clip-
and-stability` is still functional — pass and fail paths
both exercised. The new assertion catches the A-baseline-
rate drift (slope ~4–5) where the old assertion passed it.

**Commit:** one commit, type `fix(smoke-test)`. First line:
`entropy slope check replaces endpoint-at-ep3 comparison`.
Body cross-references the 2026-04-19 `lessons_learnt.md`
entry explaining why the endpoint check was structurally
blind to drift.

**Session prompt:** [`session_prompts/02_smoke_gate_
slope_assertion.md`](session_prompts/02_smoke_gate_slope_assertion.md).

---

## Session 03 — Registry reset + activation-plan redraft (operator-gated)

**Status:** pending

**Deliverables:**

- Archive current registry + episode log:
  - `registry/models.db` →
    `registry/archive_<isodate>Z/models.db`.
  - `registry/weights/` →
    `registry/archive_<isodate>Z/weights/`.
  - `logs/training/episodes.jsonl` →
    `logs/training/episodes.pre-entropy-control-v2-<isodate>.jsonl`.
  - `registry/training_plans/` copied into the archive
    folder for audit-trail of the activation plans'
    configuration at the moment of archive.
- Fresh registry:
  - New `registry/models.db` initialised via `ModelStore()`.
  - `registry/weights/` recreated empty.
  - `logs/training/episodes.jsonl` truncated to 0 bytes.
- Activation plans redrafted via the JSON-edit pattern used
  in `naked-clip-and-stability` Session 05:
  `status='draft'`, `started_at=None`, `completed_at=None`,
  `current_generation=None`, `current_session=0`,
  `outcomes=[]` on all four of `activation-A-baseline`,
  `B-001/010/100`. Configuration (`population_size`,
  `n_generations`, `arch_mix`, `hp_ranges`,
  `reward_overrides`, `seed`, `name`, `notes`, `plan_id`)
  is preserved byte-identical.
- `plans/INDEX.md` entry appended for this plan.

**Exit criteria:**

- New `registry/models.db` has `select count(*) from models`
  → **0**.
- `episodes.jsonl` has 0 bytes.
- All four activation plans status=draft (verified via the
  admin portal's plan listing OR direct JSON read).
- `git status` clean except for the gitignored archive
  folders.
- The pre-archive run state (as was at plan start) is
  captured in the archive path so post-mortem comparisons
  remain possible.

**Acceptance:** operator can tick "Smoke test first" in the
UI, launch `activation-A-baseline`, and the probe runs
cleanly against the fresh registry with the Session 01
controller wired in.

**Commit:** one commit, type `chore(registry)`. First line:
`archive pre-entropy-control-v2 registry + redraft
activation plans`. Body cross-references Sessions 01–02
commit hashes and notes the archive location for
post-mortem.

**Session prompt:** [`session_prompts/03_registry_reset_
and_relaunch.md`](session_prompts/03_registry_reset_and_relaunch.md).

---

## After Session 03: launch + validate

Once Session 03 lands and the registry is reset:

1. **Operator launches `activation-A-baseline`** with the
   smoke-test-first checkbox ticked.
2. **Smoke test runs.** Expected: passes. The new
   slope-based assertion replaces the old endpoint one; a
   functioning controller produces slope ≈ 0, comfortably
   below the `+1.0` threshold.
3. **If probe fails:** capture the failure modal diagnostics
   in `lessons_learnt.md` and review — either the controller
   has a Session-01 bug (entropy goes wrong direction in the
   first 3 episodes, possibly because alpha-optimiser's
   first step is too aggressive) or the target is wrong.
4. **Full population trains** if the probe passes. Watch
   the learning-curves panel for:
   - Pop-avg entropy converging toward 112 (the target) by
     ep 10, staying within ±20% by ep 15.
   - No ep-1 `policy_loss > 100` across the population
     (regression guard from Session 02 of the predecessor
     plan).
   - `arbs_closed > 0` on at least one agent AND
     `arbs_closed / max(1, arbs_naked) > 0.3` sustained
     across the last 5 episodes for that agent.
   - At least one agent shows a positive reward-trend
     slope across eps 8–15.
5. **Capture findings** in `progress.md` under a
   "Validation" entry. Same shape as the 2026-04-19
   Validation entry in `naked-clip-and-stability/progress.md`.
6. **Green light for B sweep** if validation succeeds — the
   activation playbook's `activation-B-001/010/100` plans
   run. Otherwise open the queued `reward-densification`
   follow-up plan if the bottleneck has shifted to reward
   sparsity, or a targeted controller-tune follow-up if
   entropy control itself still needs work.

## Queued: `reward-densification` (next plan after this one, if needed)

Not in this plan's scope — queued for if entropy is
controlled but reward still doesn't trend up. That plan
would distribute the race-level reward across the steps
that built up to it, providing dense per-step signal so
the policy gradient doesn't collapse to near-zero on the
~4700 quiet steps per race.
