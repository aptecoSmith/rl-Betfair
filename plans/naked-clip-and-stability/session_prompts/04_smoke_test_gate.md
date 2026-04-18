# Session 04 prompt — Smoke-test gate (UI tickbox + assertion harness)

## PREREQUISITE — read first

- [`../purpose.md`](../purpose.md) — the three pathologies
  the smoke test exists to catch before burning 8 hours on a
  broken run.
- [`../hard_constraints.md`](../hard_constraints.md). §15
  (gate semantics — default ON, 3 assertions, fail =
  modal + override), §16 (smoke-test episodes tagged in
  `episodes.jsonl`), §20 (pytest green), §21 (ng test green
  for this session), §26–§28 (cross-session).
- [`../master_todo.md`](../master_todo.md) — Session 04
  deliverables (backend + frontend + tests).
- `frontend/src/app/training-plans/` — the existing
  training-launch UI. Entry point for the tickbox.
- `frontend/src/app/training-monitor/` — the live learning-
  curves panel that needs to colour/badge smoke-test
  episodes distinctly.
- `frontend/src/app/models/training-plan.model.ts` — the DTO
  the tickbox updates.
- The backend training-launch endpoint (grep for
  `training_plan` in the Python backend).

## Locate the code

```
grep -rn "training-plans\|TrainingPlan\|trainingPlan\|launch" frontend/src/app/training-plans/
grep -n "smoke\|probe\|SmokeTest" frontend/src/app/ agents/
grep -rn "class.*TrainingLaunch\|def.*launch_training\|POST.*training\|training_plan.*launch" .
```

Confirm before editing:
1. The launch request/response contract. If the backend has
   a strict DTO (pydantic, dataclass), add the field there.
2. Whether a "probe" or "dry-run" pattern already exists in
   the backend (maybe from training validation). If it does,
   reuse the harness shape.
3. The operator memory entry "Verify frontend in browser
   before done" — the exit criteria for this session include
   an in-browser verification via the preview workflow (see
   `CLAUDE.md`-adjacent user memory).

## What to do

### 1. Backend: SmokeTestRunner

New file `agents/smoke_test.py` (or wherever the training
orchestration lives):

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class SmokeAssertionResult:
    name: str
    passed: bool
    observed: float
    threshold: float
    detail: str

@dataclass(frozen=True)
class SmokeResult:
    passed: bool
    assertions: list[SmokeAssertionResult]
    probe_model_ids: list[str]

def run_smoke_test(
    training_plan,  # whatever DTO the launch endpoint receives
    base_config: dict,
    episode_writer,  # same writer the full run uses
) -> SmokeResult:
    """Run a 2-agent × 3-episode probe before the full GA
    launch. Returns pass/fail + per-assertion detail.

    Agents used:
      - 1 × ppo_transformer_v1 (default hp)
      - 1 × ppo_lstm_v1 (default hp)

    Episodes are written to the same episodes.jsonl stream
    with a `smoke_test: true` field so the live training
    panel can display them distinctly.

    Assertions (all must pass for gate to pass):
      1. ep1.policy_loss < 100 on both probe agents
      2. ep3.entropy <= ep1.entropy on both probe agents
      3. max(ep1..ep3.arbs_closed) >= 1 on AT LEAST ONE
         probe agent
    """
```

Implementation notes:
- The probe must reuse the full training harness so its
  behaviour matches production. Don't build a mini-PPO
  fork.
- 3 episodes × 2 agents at ~30 s/agent/day × 6 days per
  episode = ~18 minutes. Acceptable as a pre-flight.
- The probe must write to `episodes.jsonl` with
  `smoke_test: true` on every row. This is
  `hard_constraints.md §16`.
- The probe agents should NOT be persisted to the registry
  as evaluatable models. They're ephemeral — their
  weights live only as long as the probe process.

### 2. Backend: launch-endpoint integration

Update the training-launch path:

```python
def launch_training(request: TrainingLaunchRequest) -> TrainingLaunchResponse:
    if request.smoke_test_first:
        result = run_smoke_test(request, config, episode_writer)
        if not result.passed:
            return TrainingLaunchResponse(
                status="smoke_test_failed",
                smoke_result=result,
            )
    return _start_full_population(request, config)
```

The failing-probe response carries enough detail for the
frontend to display per-assertion results.

### 3. Frontend: checkbox

In `frontend/src/app/training-plans/training-plans.html`
(or wherever the launch form lives), add a checkbox above
the Launch button:

```html
<mat-checkbox [(ngModel)]="launchOptions.smokeTestFirst"
              color="primary">
  Smoke test first (recommended)
</mat-checkbox>
<p class="mat-caption">
  Runs a 2-agent × 3-episode probe before the full
  population to catch training instability before burning
  hours of compute. See
  <a href="...">plans/naked-clip-and-stability</a>.
</p>
```

Default `launchOptions.smokeTestFirst = true` in the
component initialiser.

### 4. Frontend: failure modal

When the launch response is `status=smoke_test_failed`,
open a modal dialog showing:

- **Headline:** "Smoke test failed — full population not
  launched."
- **Per-assertion table:** name, observed, threshold,
  pass/fail.
- **Actions:**
  - "Re-run smoke test" — re-submits with the same plan.
  - "Launch anyway" — opens a confirmation modal, then
    submits with `smokeTestFirst=false`.
  - "Cancel" — closes the modal, plan remains in draft.

Use the existing confirm-modal pattern from the rest of
the admin portal — grep for existing `MatDialog` usages.

### 5. Frontend: learning-curves colouring

In `training-monitor`, smoke-test episodes (rows with
`smoke_test: true`) render in a visually distinct treatment
— faded colour or dashed line. The exact design is an
operator-preference judgement call; pick something and
document in `progress.md`. The non-negotiable is: operators
must be able to tell smoke-test rows apart from real-run
rows at a glance (§16).

### 6. Tests (backend)

New file `tests/test_smoke_test.py`:

```python
class TestSmokeAssertions:
    def test_ep1_policy_loss_under_threshold_passes(self):
        """Fabricated episodes row with ep1 policy_loss=50 →
        passes."""
    def test_ep1_policy_loss_at_threshold_fails(self):
        """ep1 policy_loss=100 → fails (strict less-than)."""
    def test_entropy_non_increasing_passes(self):
        """ep1 entropy=150, ep3 entropy=140 → passes."""
    def test_entropy_rising_fails(self):
        """ep1 entropy=140, ep3 entropy=180 → fails."""
    def test_arbs_closed_one_agent_passes(self):
        """Two probe agents; one has arbs_closed=3 by ep3,
        other has 0 — passes (at-least-one rule)."""
    def test_arbs_closed_both_zero_fails(self):
        """Both probe agents arbs_closed=0 across all eps →
        fails."""

class TestSmokeTestEndpoint:
    def test_launch_with_smoke_pass_starts_full_run(self):
        """Mocked probe returns SmokeResult(passed=True) →
        launch response is status=started."""
    def test_launch_with_smoke_fail_does_not_start_full_run(self):
        """Mocked probe returns SmokeResult(passed=False) →
        launch response is status=smoke_test_failed,
        carries assertion detail."""
    def test_launch_without_smoke_flag_skips_probe(self):
        """request.smoke_test_first=False → probe not
        invoked, full run starts directly."""
```

### 7. Tests (frontend)

New spec in `frontend/src/app/training-plans/`:

```typescript
describe('Smoke test checkbox', () => {
  it('defaults to checked', ...);
  it('includes smokeTestFirst in launch payload when checked', ...);
  it('sends smokeTestFirst=false when unchecked', ...);
  it('opens failure modal on smoke_test_failed response', ...);
  it('re-run button re-submits with smokeTestFirst=true', ...);
  it('launch-anyway confirmation submits with smokeTestFirst=false', ...);
});
```

### 8. Browser verification

Per the user-memory note "Verify frontend in browser before
done" + "Full stack up for UI verify":

1. Start both backend API and frontend preview servers
   (`preview_start` with the appropriate ports from the
   user-memory "Port allocation" entry —
   rl-betfair=4202/8001/8002/9000).
2. Launch a plan with the checkbox ticked. Fabricate a
   scenario the probe will fail (e.g. temporarily dial
   `ep1_policy_loss_threshold` to 0 in a config override).
3. Verify the failure modal renders with the three
   assertion rows.
4. Click "Re-run smoke test" — verify it re-submits.
5. Click "Launch anyway" — verify the confirmation modal,
   confirm, verify the full run starts.
6. Uncheck the box and launch — verify the probe is
   skipped.
7. Capture a `preview_screenshot` of the failure modal for
   `progress.md`.

### 9. Full suite

```
pytest tests/ -q
cd frontend && ng test --watch=false
```

Both green.

### 10. Commit

```
feat(training): smoke-test gate — 2-agent probe before full GA launch

New UI checkbox (default ON) runs a 2-agent × 3-episode
probe before committing to the full 16-agent GA training
run. Probe uses one transformer, one LSTM, default
hyperparameters; episodes tag `smoke_test: true` in the
episodes.jsonl stream.

Three assertions gate the full launch:
  1. ep1.policy_loss < 100 on both probe agents
  2. ep3.entropy ≤ ep1.entropy on both probe agents
  3. arbs_closed ≥ 1 on at least one probe agent by ep 3

On fail: UI modal with per-assertion detail + re-run /
launch-anyway buttons. Full population does NOT start.

Motivation: gen-2 2026-04-18 training exposed three
pathologies (policy-loss blow-up, rising entropy,
arbs_closed collapse) that only became visible after 7
episodes of burn. Sessions 01–03 address the root causes;
this gate prevents future regressions from burning hours
before they surface.

Tests: N new in tests/test_smoke_test.py (backend), N new
in frontend spec. pytest tests/ -q: <delta>. ng test
--watch=false: green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## Cross-session rules

- Gate must not alter the full-run behaviour when the
  checkbox is unticked (§15 "When OFF").
- Probe agents MUST write to the same episodes.jsonl
  stream as the full run (§16).
- No weights persisted to the registry for probe agents.

## After Session 04

1. Append a `progress.md` entry: commit hash, the gate's
   wiring overview, browser-verification screenshot
   reference, test deltas.
2. Hand back for Session 05 (registry reset + launch —
   operator-gated, NOT automated).
