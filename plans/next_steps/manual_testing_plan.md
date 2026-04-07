# Manual Testing Plan — Next Steps

**Scope:** human-in-the-loop verification. Things that can't
reasonably be automated, or where automation would miss the
thing we care about.

**Not in scope:** pytest-driven tests (fast or integration). If a
thing can be pytest-driven without more than 10 lines of
scaffolding, it belongs in `initial_testing.md` or
`integration_testing.md`, not here.

## Why manual tests exist

Most correctness in this codebase is automatable. A handful of
things are not:

- **Visual UI verification.** Does the plan editor actually
  render the new fields? Do range sliders clamp correctly in the
  browser? Do error toasts show up on invalid submits?
- **Subjective sanity checks on training curves.** "Does this
  look like it's learning something?" is a human judgement and
  no test can replace reading the plots.
- **Cross-session integration.** Does the full loop — edit a
  plan in the UI, launch a run from the UI, watch the dashboard
  update, inspect the coverage page afterwards — work end-to-end
  on a real browser against a real backend?
- **Launching the real multi-generation run.** The run in
  Session 11 is a one-off human-triggered action with a
  pre-flight checklist and a post-run review. Nothing about it
  is automated.

## Manual test protocol

Every manual test step follows the same structure:

1. **Preconditions** — what state the system needs to be in
   before the check makes sense.
2. **Action** — what the human does.
3. **Expected observation** — what the human should see. Be
   specific. "Looks right" is not an expected observation.
4. **Pass / fail / notes** — recorded in `progress.md` under
   that session's entry.

Manual steps that fail are either:
- **Automated next session** (if it turns out they could have
  been), **or**
- **Re-triggered** next time as an explicit manual step.

Never quietly drop a failing manual step.

## Standing manual checks

These apply whenever the relevant subsystem is touched. A session
that touches the subsystem must run the matching block and record
the result.

### M1 — UI smoke (touches: any config gene, any training-plan field)

**Preconditions:**
- Backend running: `./start-training.sh --no-gpu` or equivalent
  that just serves the API.
- Frontend running: `npm run dev` under `frontend/`.

**Actions:**
1. Visit the config / search-ranges page. Confirm every gene
   currently in `config.yaml → search_ranges` is rendered.
2. Visit the training-plan list page. Confirm historical plans
   load.
3. Open the plan editor. For every new gene introduced in the
   current session, confirm it has an editor widget and that
   attempting to save an invalid value (e.g. `max < min`) is
   rejected client-side AND server-side.
4. Visit the schema inspector view (if shipped in Session 8 /
   the housekeeping sweep). Confirm every gene listed there
   matches `config.yaml`.

**Expected observation:**
- No missing widgets. No console errors in the browser dev
  tools. Server-side validation errors render as inline field
  errors, not as an unhelpful modal.

### M2 — Training launch (touches: any code path from config to trainer)

**Preconditions:**
- Fresh `TrainingPlan` targeting a very small fixture day and
  population (pop=6, 1 generation, 2 days).

**Actions:**
1. Launch the plan via the UI.
2. Watch the dashboard until the first generation reports.
3. Inspect the resulting log file under `logs/training/`.

**Expected observation:**
- Dashboard updates in real time (tick counter, episode counter).
- Log file contains per-episode entries with
  `raw_pnl_reward + shaped_bonus ≈ total_reward` on every line.
- No NaNs, no exceptions.

### M3 — Post-run review (Session 11 only)

**Preconditions:**
- Session 11's real multi-generation run has completed.

**Actions:**
1. Open the post-run analysis notebook / script.
2. Walk through each of the five Session 9 invariants (from
   `arch-exploration/session_9_gpu_shakeout.md`) against the
   Session 11 data.
3. Cross-check the coverage page against the plan's intended
   architecture mix.
4. Pull a random sample of 5 agents across the population;
   inspect their reward curves and bet distributions for
   obvious pathologies (all-zero action, constant bet size,
   NaN spikes).

**Expected observation:**
- All five invariants hold, or failures are documented in
  `progress.md` and `lessons_learnt.md`.
- Coverage matches the plan.
- No agent shows obvious pathological behaviour — or if one
  does, its agent id is logged and filed as a follow-up.

## Recording results

Manual test runs get recorded in `progress.md` under the session
that triggered them. Format:

```
**Manual tests run:** M1, M2
  - M1: pass (2026-04-NN)
  - M2: pass with note — dashboard lag ~3s during initial rollout
```

Any failure gets a matching entry in `lessons_learnt.md`.
