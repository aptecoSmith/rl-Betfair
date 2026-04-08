# Integration Testing — Research-Driven

Slow tests, opt-in. Run at the end of a session that touched
training behaviour, or before merging a session that ships a new
policy. **Not** part of the fast feedback loop in
`initial_testing.md`.

Marked with `@pytest.mark.gpu` or `@pytest.mark.slow` so they are
skipped by default. The session prompt for any item that ships a
training change must explicitly call out which integration tests it
expects to run, and `progress.md` must record the result.

---

## Always-on (when running integration suite)

These predate this folder and continue to apply.

- A real PPO loop on the smallest viable fixture, asserting that
  the policy's loss curves are not NaN and that bet counts are
  non-zero by step N.
- Genome-plumbing tests that load every gene in the search schema
  and assert it reaches its consumer.
- One full training run on a 1-day eval window, asserting the
  invariants in `next_steps/integration_testing.md` continue to
  hold.

If any of those are red, the research-driven session has not
introduced its own integration tests yet — fix the existing ones
first.

---

## Per-proposal additions

### P1 — money-pressure features

- **Gradient-norm check on P1 columns (session 22 — shipped):**
  `scripts/session_22_p1d_compare.py --dry-run` builds both policies
  and validates obs-slicing. After ≥1 gradient step on a non-collapsed
  policy, `check_p1_gradient_norm()` asserts the L2 norm of
  ∂value/∂obs at OBI+microprice+traded_delta+mid_drift columns is
  non-zero. (Must be run on a fresh policy, not one that has already
  collapsed to 0 bets — collapsed policies produce near-zero gradients
  everywhere.)
- **Comparison run (session 22 — shipped, result: inconclusive):**
  Full run in `scripts/session_22_p1d_compare.py`. P1 obs vs pre-P1
  obs, raw P&L on held-out eval window, results in `progress.md`. Not
  a strict pass/fail. Single-seed result is dominated by training
  variance; future re-runs should use the evolutionary infrastructure
  (N≥10 agents per config) for a meaningful signal. See `progress.md`
  session 22 entry for diagnosis.

### P2 — spread-cost shaped reward

- A 1-day training run with the new shaping term, asserting that
  the shaped accumulator carries the new cost and that the
  `raw + shaped ≈ total` invariant holds across the whole run.
- Bet count per race in eval is lower than the pre-P2 baseline by
  a non-noise margin. (If it isn't, the term may be too small —
  surface in `lessons_learnt.md` and tune.)
- Raw P&L on the eval window is recorded for the Phase-1
  decision-gate comparison.

### P3 + P4 — passive orders, queue, cancel

- A 1-day training run *from a fresh init* (action-space change
  forces this) asserting that:
  - The aggression histogram across emitted actions is not
    collapsed to one mode by the end of training.
  - At least one race per eval day has a non-zero cancel count.
  - At least one race per eval day has a non-zero passive-fill
    count (i.e. the queue model actually fires).
- Aggressive-only baseline run for comparison — train P3+P4 with
  the aggression flag forced to "always cross" and check it
  reproduces the pre-P3 policy's behaviour. Sanity check that the
  new code paths are dormant when not invoked.
- Raw P&L on the eval window is recorded for the Phase-2
  decision-gate comparison.

### P5 — UI fill-side annotation

No integration tests. Pure UI work, covered by snapshot tests in
`initial_testing.md`.

---

## Cross-repo integration (`ai-betfair`)

Once any research-driven proposal lands a policy that goes to
`ai-betfair`, the live wrapper has its own integration tests that
must pass before deployment. Those live in `ai-betfair`'s plan
folder, **not here**. The audit of what they need to cover is in
`downstream_knockon.md` of this folder. Owe a cross-link in
`progress.md` whenever a new policy is handed across.

---

## What stays out

- Smoke tests fast enough to run every save → `initial_testing.md`.
- Anything that requires a human watching the UI →
  `manual_testing_plan.md`.

If the integration suite for this folder grows past 30 minutes
end-to-end, split it: keep the per-session "did this break the
fundamentals" tests, and move comparison runs into a nightly job
that posts results to `progress.md` rather than blocking sessions.
