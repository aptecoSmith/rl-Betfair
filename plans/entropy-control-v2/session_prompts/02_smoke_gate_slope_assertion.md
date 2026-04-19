# Session 02 prompt — Smoke-gate slope assertion

## PREREQUISITE — read first

- [`../purpose.md`](../purpose.md) — why the endpoint check
  is structurally blind to drift (§ Diagnosis, subsection
  referencing the 2026-04-19 lesson).
- [`../hard_constraints.md`](../hard_constraints.md). §13
  (slope formula and threshold), §14 (per-agent not
  pop-avg), §15 (what happens if controller passes but
  slope check fails), §18–§20 (tests green + specific
  test names), §24 (Session 02 blocks if Session 01 didn't
  land).
- [`../master_todo.md`](../master_todo.md) — Session 02
  deliverables.
- `plans/naked-clip-and-stability/lessons_learnt.md`
  2026-04-19 entry — endpoint-vs-slope test-design
  analysis.
- `plans/naked-clip-and-stability/progress.md` 2026-04-19
  Validation entry — concrete evidence (ep1 entropy 139.6,
  ep3 entropy 145.3, +5.7 drift passed the old `≤ ep1 + 10`
  assertion).
- `agents/smoke_test.py` — the file being edited. Locate
  the current `ep3.entropy <= ep1.entropy + 10.0`
  assertion.
- `tests/test_smoke_test.py` — the file whose entropy
  tests are being updated.

## Locate the code

```
grep -n "entropy" agents/smoke_test.py
grep -n "EntropyAssertion\|entropy_threshold\|non_increasing" agents/smoke_test.py tests/test_smoke_test.py
```

Confirm before editing:

1. The current assertion lives in a single function/method
   — not duplicated across files.
2. The assertion produces a `SmokeAssertionResult` (or
   equivalent) with `name`, `observed`, `threshold`,
   `passed` fields — the frontend modal reads these.
3. The test file has parameterised pass / at-threshold /
   fail cases for the existing endpoint assertion.

## What to do

### 1. Replace the assertion

In `agents/smoke_test.py` — find the entropy assertion and
change it:

```python
# Before:
def _evaluate_entropy(probe_rows: list[dict]) -> SmokeAssertionResult:
    ep1 = probe_rows[0]["entropy"]
    ep3 = probe_rows[2]["entropy"]
    threshold = ep1 + 10.0  # tolerant endpoint check
    passed = ep3 <= threshold
    return SmokeAssertionResult(
        name="entropy_non_increasing",
        observed=ep3,
        threshold=threshold,
        passed=passed,
    )

# After:
def _evaluate_entropy(probe_rows: list[dict]) -> SmokeAssertionResult:
    import numpy as np
    episodes = np.array([r["episode"] for r in probe_rows], dtype=float)
    entropies = np.array([r["entropy"] for r in probe_rows], dtype=float)
    slope = float(np.polyfit(episodes, entropies, 1)[0])
    threshold = 1.0  # entropy may rise at most 1 per episode
    passed = slope <= threshold
    return SmokeAssertionResult(
        name="entropy_slope",
        observed=slope,
        threshold=threshold,
        passed=passed,
    )
```

Notes:

- The `name` field changes from `entropy_non_increasing` to
  `entropy_slope` — the frontend modal's row label updates
  automatically via the existing rendering code. If the
  frontend has any hardcoded string matching on the old
  name (search in the Angular component), update that too.
- `numpy` is already a project dependency (used throughout
  `env/` and `agents/`); no new import added to the
  top-level dependency graph.
- If `probe_rows` has fewer than 2 rows, `polyfit` raises.
  The existing empty-input behaviour returns a failure
  result with a sentinel observed value — match that shape
  for robustness. Add a guard:
  ```python
  if len(probe_rows) < 2:
      return SmokeAssertionResult(
          name="entropy_slope",
          observed=float("nan"),
          threshold=threshold,
          passed=False,
      )
  ```

### 2. Update tests

`tests/test_smoke_test.py` — rewrite the entropy-assertion
test class (`TestEntropyAssertion` or similar):

```python
class TestEntropyAssertion:
    def _rows(self, entropies: list[float]) -> list[dict]:
        return [
            {"episode": i + 1, "entropy": e}
            for i, e in enumerate(entropies)
        ]

    def test_slope_assertion_passes_on_flat_entropy(self):
        """[140, 140, 140] → slope 0 → passes (≤ 1.0)."""
        result = _evaluate_entropy(self._rows([140.0, 140.0, 140.0]))
        assert result.passed is True
        assert abs(result.observed) < 1e-6

    def test_slope_assertion_passes_on_mild_decrease(self):
        """[140, 139, 138] → slope −1 → passes."""
        result = _evaluate_entropy(self._rows([140.0, 139.0, 138.0]))
        assert result.passed is True
        assert result.observed == pytest.approx(-1.0, abs=1e-6)

    def test_slope_assertion_fails_on_a_baseline_drift_rate(self):
        """[139.6, 145.3, 150.0] → slope ≈ +5.2 → FAILS.
        This is the actual A-baseline 2026-04-19 drift rate
        that the old endpoint assertion let through."""
        result = _evaluate_entropy(self._rows([139.6, 145.3, 150.0]))
        assert result.passed is False
        assert result.observed > 1.0

    def test_slope_assertion_at_threshold_boundary(self):
        """[140, 141, 142] → slope exactly 1.0 → passes.
        [140, 141.1, 142.2] → slope > 1.0 → fails."""
        pass_result = _evaluate_entropy(self._rows([140.0, 141.0, 142.0]))
        assert pass_result.passed is True
        fail_result = _evaluate_entropy(self._rows([140.0, 141.1, 142.2]))
        assert fail_result.passed is False

    def test_slope_assertion_handles_empty_input(self):
        """Empty rows → passed=False, observed=NaN — same
        shape as other assertions' empty-input behaviour."""
        import math
        result = _evaluate_entropy([])
        assert result.passed is False
        assert math.isnan(result.observed)
```

Also update any aggregate / vignette test that pinned the
old `entropy_non_increasing` name — search for it:

```
grep -rn "entropy_non_increasing" tests/
```

Rename string matches. The vignette test that recreates the
gen-2 transformer `0a8cacd3` ep1..ep3 pathology (Session 04
deliverable in `naked-clip-and-stability`) should still
produce a FAILING slope assertion — that transformer's ep1→ep3
climb was faster than 1/ep so the same vignette fails both
assertions. Verify the vignette's entropy inputs
produce slope > 1.0.

### 3. Update the `naked-clip-and-stability` hard_constraints

`plans/naked-clip-and-stability/hard_constraints.md` §15
mentions the endpoint assertion explicitly. Per this plan's
§13, do NOT edit §15 in place — append a footnote at the
end of the file cross-referencing this plan's Session 02:

```markdown
---

## Post-plan amendments

**2026-04-19** (via `plans/entropy-control-v2/` Session 02,
commit `<hash>`): §15's entropy assertion is updated from
the endpoint check (`ep3.entropy <= ep1.entropy + 10.0`) to
a slope check (`np.polyfit(episodes, entropies, 1)[0] <=
1.0`). See `plans/entropy-control-v2/hard_constraints.md`
§13 for the new formula and `plans/naked-clip-and-
stability/lessons_learnt.md` 2026-04-19 for the rationale.
```

### 4. Frontend verification

The frontend modal reads `observed`, `threshold`, and
`passed` from the assertion result. Label rendering is
driven by the assertion's `name` field. The label text
either:
- comes from a frontend-side map (`{entropy_non_increasing:
  "Entropy non-increasing", ...}`) that needs updating, OR
- comes directly from the backend (in which case no
  frontend change needed).

Locate it:

```
grep -rn "entropy_non_increasing\|entropy_slope" frontend/src/
```

If a map exists, update it. If the label comes straight
from the backend, no frontend source change — the new
label ("entropy_slope") will surface as-is.

### 5. Browser verification

Per the user memory `feedback_verify_in_browser.md`: the
smoke gate's exercise requires a full stack. Fabricate a
failure via an override (the existing Session 04 technique
lets the probe fail deterministically by overriding
`ep1_policy_loss_threshold=0`). Launch through the UI,
confirm the modal shows the new assertion label and the
slope observed value.

If fabricating an entropy-slope failure needs a different
override, add one: the config schema already has
`ep1_policy_loss_threshold`; add `entropy_slope_threshold`
as a sibling so the operator can fabricate slope failures
for exercise purposes. Default to `1.0`; override to e.g.
`-0.1` to force the current flat-entropy probe to fail.

### 6. Full suite

```
pytest tests/ -q
```

Must be green. Regression guards:

- `tests/test_ppo_trainer.py::TestTargetEntropyController`
  — Session 01's tests. Untouched by Session 02.
- `tests/test_smoke_test.py` — the updated entropy-
  assertion tests above.
- `tests/test_api_training.py::TestSmokeTestGateDTO` — the
  5 DTO tests from `naked-clip-and-stability` Session 04.
  Unchanged by this session.

Frontend:

```
cd frontend && ng test --watch=false
```

Must be green. The modal-rendering tests may need updating
if they pin on the old assertion name.

### 7. Commit

```
fix(smoke-test): entropy slope check replaces endpoint-at-ep3 comparison

The old `ep3.entropy <= ep1.entropy + 10.0` assertion is
structurally blind to slow monotone drift — it only compares
two of the three probe episodes. The A-baseline 2026-04-19
validation (64 agents × 15 eps) had pop-avg entropy drift
139.6 → 201.3 across the full run; at ep3 the drift was
only +5.7, well inside the `+10` tolerance. The gate passed
the run; the pathology manifested past ep5.

Replace with a slope check: fit a line through all three
probe-episode entropies and require slope ≤ 1.0 per episode.
A controller holding entropy at target produces slope ≈ 0;
the A-baseline drift rate (~5/episode) now fails.

Tests: N new + M updated in tests/test_smoke_test.py.
Updated naked-clip-and-stability §15 via a post-plan
amendment footnote.

See plans/entropy-control-v2/ for the full plan context
and plans/naked-clip-and-stability/lessons_learnt.md
2026-04-19 for the test-design rationale.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## Cross-session rules

- No controller changes (that's Session 01). If the
  controller's test suite fails on legitimate Session 02
  changes, that's a bug — controller and gate are
  orthogonal.
- No reward-path changes.
- No PPO-stability changes.

## After Session 02

1. Append a `progress.md` entry: commit hash, the
   assertion change, test counts, frontend verification
   result.
2. Hand back for Session 03 (registry reset + relaunch).
   Session 03 is operator-gated per §23.
