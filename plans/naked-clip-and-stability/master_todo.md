# Master TODO — Naked-Windfall Clip & Training Stability

Five sessions, one commit per session, hard-constrained by
`hard_constraints.md`. Sessions 01–04 are automatable; Session
05 is operator-gated.

---

## Session 01 — Reward shape: full cash in raw, 95% winner clip + close bonus in shaped

**Status:** pending

**Deliverables:**
- `env/betfair_env.py::_settle_current_race`: refactor the
  scalping reward branch.
  - Raw gains the full `sum(per_pair_naked_pnl)` (winners AND
    losers).
  - Shaped gains `−0.95 × sum(max(0, per_pair_naked_pnl))`
    (winner clip) and `+1.00 × n_close_signal_successes`
    (close bonus).
  - Remove the `0.5 *` softener on the existing
    `naked_loss_term` — losers now enter raw at full cash
    value.
- `CLAUDE.md`: new 2026-04-18 paragraph in "Reward function:
  raw vs shaped" documenting the new formula. Historical
  entries preserved per `hard_constraints.md §25`.
- `tests/test_forced_arbitrage.py`: new class
  `TestNakedWinnerClipAndCloseBonus`:
  1. Single naked winner (+£100) — raw=+100, shaped=−95, net=+5.
  2. Single naked loser (−£80) — raw=−80, shaped=0, net=−80.
  3. Mixed (one +£100, one −£80) — raw=+20, shaped=−95, net=−75.
  4. Scalp that used close_signal — raw=locked_pnl,
     shaped=+1, net=locked+1.
  5. Multiple close_signal successes in one race — shaped
     accumulates N × 1.0.
  6. `raw + shaped ≈ total_reward` invariant (delegates to the
     existing invariant test; this row exists so failure
     points here, not upstream).

**Exit criteria:**
- `pytest tests/ -q` green.
- Pre-existing
  `test_invariant_raw_plus_shaped_equals_total_reward` green.
- Manual trace of one race with two naked pairs (one win, one
  loss) through `_settle_current_race` produces the expected
  `raw_pnl_reward` and `shaped_bonus` values on `info`.

**Acceptance:** every row of `purpose.md`'s outcome table is
covered by a unit test and passes.

**Commit:** one commit, type `fix(env)`. First line names the
reward-scale change. Body includes the worked examples from
`purpose.md`. Trailer notes the pytest delta (+N tests).

**Session prompt:** `session_prompts/01_reward_shape.md`.

---

## Session 01b — Raw = race_pnl (loss-closed pairs correctly negative)

**Status:** pending

**Why this session exists:** review after Session 01 was
drafted caught that `raw = scalping_locked_pnl +
sum(per_pair_naked_pnl)` silently excludes
`scalping_closed_pnl`. A pair closed via `close_signal` at a
loss contributes `raw=0` (locked floor) + `+£1 (close bonus
in shaped)` = net `+£1` — rewarding the agent for a trade
that actually lost real cash. The refinement: set raw to the
whole-race cashflow (`race_pnl`), which correctly includes
close-leg losses. See `hard_constraints.md §4a` for the full
context and `purpose.md` outcome table (loss-closed row).

**Deliverables:**
- `env/betfair_env.py::_settle_current_race`: change the
  scalping raw-reward branch so `race_reward_pnl = race_pnl`
  directly. This replaces Session 01's
  `scalping_locked_pnl + naked_full_term` with the
  whole-race P&L.
- `CLAUDE.md`: update the "Scalping mode (2026-04-18 —
  `naked-clip-and-stability`)" paragraph Session 01 added.
  Replace the `scalping_locked_pnl + sum(per_pair_naked_pnl)`
  formula with `race_pnl`; keep the shaped terms unchanged;
  add one sentence noting that loss-closed pairs now
  contribute their actual loss to raw (previously excluded).
- `tests/test_forced_arbitrage.py`:
  `TestNakedWinnerClipAndCloseBonus` gains one new test
  `test_loss_closed_scalp_reports_full_loss_in_raw`:
  synthesise a close_signal that closes a pair at −£5 cash
  (locked_pnl floored to 0), assert `raw=−£5`, `shaped=+£1`,
  `net=−£4`. Update any existing Session 01 tests whose
  expected values assumed the `locked + naked_full` formula
  — specifically the mixed-outcome test (`raw=+£20` under
  Session 01's formula becomes `raw = +£20 +
  scalping_closed_pnl_for_that_fixture` under 01b).

**Exit criteria:**
- `pytest tests/ -q` green.
- Pre-existing
  `test_invariant_raw_plus_shaped_equals_total_reward`
  green.
- Manual trace of a loss-closed pair through
  `_settle_current_race` produces the expected
  `info["raw_pnl_reward"] = −£5` (for a £5-loss close).

**Acceptance:** the loss-closed row in `purpose.md`'s outcome
table is covered by a unit test and passes; the invariant
test from `scalping-naked-asymmetry` still passes.

**Commit:** one commit, type `fix(env)`. First line names the
refinement ("raw = race_pnl; loss-closed pairs now correctly
negative"). Body cross-references Session 01's commit hash
and explains the bug that prompted the refinement.

**Session prompt:** `session_prompts/01b_race_pnl_and_loss_closed.md`.

---

## Session 02 — PPO stability: KL early-stop + ratio clamp + per-arch LR

**Status:** pending

**Deliverables:**
- `agents/ppo_trainer.py`:
  - Add `log_ratio` clamp to `[−20, +20]` before `.exp()` in
    the surrogate-loss computation.
  - Add KL early-stop: compute per-epoch approx-KL
    (`(old_logp - new_logp).mean()`), break out of the epoch
    loop when it exceeds `0.03` (default, threshold
    configurable via hyperparameter `kl_early_stop_threshold`).
  - Verify the existing 5-update LR warmup
    (`agents/ppo_trainer.py:1114`) covers all three
    architectures. If the transformer code path bypasses it
    (audit via grep), fix the bypass.
- `agents/ppo_transformer_v1.py` (or wherever the transformer's
  default hyperparameters live): halve the default
  `learning_rate` relative to the LSTM default. Document the
  reason in a comment cross-linking this plan.
- `tests/test_ppo_trainer.py` (or a new file if the trainer
  test is overloaded):
  1. Synthetic high-KL rollout — early-stop fires after epoch
     1, subsequent epochs skipped. Assert via the epoch
     counter.
  2. Ratio clamp — a pathological log-ratio (e.g. 50) results
     in `.exp()` input of exactly 20, not 50. Assert no NaN
     / Inf in the surrogate loss.
  3. Warmup coverage — confirm the warmup factor is applied
     on the first call to `_ppo_update` for each
     architecture (parameterised test).

**Exit criteria:**
- `pytest tests/ -q` green.
- Synthetic PPO update with reward scale ±£500 no longer
  produces `policy_loss > 100` on the first update (unit
  test asserting this).

**Acceptance:** the unit test that triggered the original
agent `3e37822e-c9fa` collapse (from
`plans/policy-startup-stability/`) continues to pass with
the new stability code — regression guard.

**Commit:** one commit, type `fix(agents)`. First line names
the change. Body references the transformer `0a8cacd3`
episode-1 `policy_loss=1.04e17` evidence from
`purpose.md`.

**Session prompt:** `session_prompts/02_ppo_stability.md`.

---

## Session 03 — Entropy control: halved coefficient + reward centering

**Status:** pending

**Deliverables:**
- `agents/ppo_trainer.py`:
  - Halve default `entropy_coefficient` from `0.01` to
    `0.005`. Update the `hp.get("entropy_coefficient", 0.01)`
    default at line ~468.
  - Add a reward-centering pass before advantage
    computation: maintain a running mean of episode rewards
    (EMA, α=0.01), subtract from raw returns prior to the
    advantage calculation.
- `tests/test_ppo_trainer.py`:
  1. Reward centering preserves advantage ordering on a
     synthetic rollout (post-centering advantages differ
     from pre-centering by a constant only, up to
     floating-point tolerance).
  2. Running-mean update is monotonic on a monotonic reward
     sequence (sanity).
  3. Entropy coefficient default is `0.005` for fresh-init
     agents.

**Exit criteria:**
- `pytest tests/ -q` green.
- The existing advantage-normalisation test from
  `plans/policy-startup-stability/` still passes — centering
  slots in front of it, doesn't replace it.

**Acceptance:** combined Session 02 + Session 03 changes
produce a synthetic transformer rollout whose ep-1 entropy
is ≥ ep-3 entropy (anti-diffusion signal). This is a
qualitative test in the notebook attached to the session
prompt; not a pytest assertion.

**Commit:** one commit, type `fix(agents)`. First line names
the entropy change. Body explains the centering rationale
(no advantage-ordering change).

**Session prompt:** `session_prompts/03_entropy_and_centering.md`.

---

## Session 04 — Smoke-test gate (UI tickbox + assertion harness)

**Status:** pending

**Deliverables:**
- **Backend:**
  - `TrainingLaunchOptions` (or equivalent DTO on the launch
    endpoint): new `smoke_test_first: bool = True` field.
  - New `scripts/run_smoke_test.py` or `agents/smoke_test.py`
    orchestrating the 2-agent × 3-episode probe. Writes
    episodes to the same `episodes.jsonl` with a
    `smoke_test: true` field.
  - New assertion runner:
    `agents/smoke_test.py::evaluate_probe(episodes) -> SmokeResult`.
    Returns pass/fail + per-assertion detail.
  - Training launch flow: when `smoke_test_first=True`,
    probe runs first; on fail, return a structured response
    with failure details; on pass, proceed to the full
    population.
- **Frontend:**
  - `training-plans` (or `training-launch`) component: new
    checkbox "Smoke test first (recommended)", default
    checked.
  - Failure-response handler: display failure modal with
    per-assertion results, "Launch anyway" (with confirmation)
    and "Re-run smoke test" buttons.
  - `training-monitor` learning-curves panel: colour or badge
    smoke-test episodes distinctly so operators can see them
    without confusion.
- **Tests:**
  - `tests/test_smoke_test.py` (or equivalent): unit tests
    for each assertion. Parameterised — one test per
    assertion × pass/fail case.
  - `tests/test_training_launch.py`: integration test of the
    smoke-test gate. Mock a probe pass → full population
    starts; mock a probe fail → full population does NOT
    start.
  - Frontend `ng test --watch=false`: new spec for the
    checkbox + failure modal.

**Exit criteria:**
- `pytest tests/ -q` green.
- `cd frontend && ng test --watch=false` green.
- Manual e2e through the browser preview: tick box, launch,
  see probe run, verify UI updates — per CLAUDE.md's
  "verify in browser" rule (see user memory).

**Acceptance:** launching with smoke-test-first ticked runs
the probe; launching a run where the probe's ep-1
`policy_loss > 100` produces the failure modal in the UI and
the full population does not start.

**Commit:** one commit, type `feat(training)`. First line names
the gate. Body explains the three assertions and cross-links
to Sessions 01–03 (which provide the fixes the gate verifies).

**Session prompt:** `session_prompts/04_smoke_test_gate.md`.

---

## Session 05 — Registry reset + activation-plan redraft (operator-gated)

**Status:** pending

**Deliverables:**
- Archive current `registry/models.db` and `registry/weights/`
  to `registry/archive_<isodate>Z/`.
- Archive current `logs/training/episodes.jsonl` to
  `logs/training/episodes.pre-naked-clip-stability-<isodate>.jsonl`.
- Redraft all four activation plans
  (`activation-A-baseline`, `B-001/010/100`) via the same
  JSON-edit pattern used 2026-04-17 and 2026-04-18:
  `status='draft'`, `started_at=None`, `completed_at=None`,
  `current_generation=None`, `current_session=0`,
  `outcomes=[]`.
- Update `CLAUDE.md` if any cross-reference to the reset
  needs adding (most likely none — Session 01 handles the
  reward-shape CLAUDE.md update).
- Update `plans/INDEX.md` with this plan's completion entry.

**Exit criteria:**
- New `registry/models.db` is fresh (16 initial models or
  whatever the init script produces; verify via `sqlite3
  registry/models.db "select count(*) from models"`).
- `episodes.jsonl` is empty.
- All four activation plans status=draft (verify via the
  listing endpoint or direct JSON read).
- `git status` clean except for the gitignored archive
  folders.

**Acceptance:** the operator can tick "Smoke test first" in
the UI, launch `activation-A-baseline`, and the probe runs
cleanly against the fresh registry.

**Commit:** one commit, type `chore(registry)`. First line
names the reset. Body cross-references Sessions 01–04 commit
hashes and notes the archive location for post-mortem.

**Session prompt:** `session_prompts/05_registry_reset_and_launch.md`.

---

## After Session 05: launch + validate

Once Session 05 lands and the registry is reset:

1. **Operator launches `activation-A-baseline`** with the
   smoke-test-first checkbox ticked.
2. **Smoke test runs.** Expected: passes. If it fails,
   capture the failure modal's diagnostics in
   `lessons_learnt.md` and open a follow-up plan for
   whichever session's fix proved insufficient.
3. **Full population trains** if the probe passes. Watch the
   learning-curves panel for:
   - No ep-1 `policy_loss > 100` across the 16 agents.
   - Entropy trending downward across episodes for most
     agents.
   - `arbs_closed > 0` and `arbs_closed / arbs_naked > 0.3`
     on at least one agent.
4. **Capture findings** in `progress.md` under a
   "Validation" entry. Same shape as the baseline-comparison
   entries in `scalping-naked-asymmetry/progress.md`.
5. **Green light for B sweeps** if validation succeeds —
   the activation playbook's B-001/010/100 plans run.
   Otherwise open a follow-up plan per the failure-mode
   section of `purpose.md`.
