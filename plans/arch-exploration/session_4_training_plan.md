# Session 4 — Training plan / Gen-0 coverage tracker

## Before you start — read these

- `plans/arch-exploration/purpose.md`
- `plans/arch-exploration/master_todo.md` (Session 4)
- `plans/arch-exploration/testing.md` — **CPU-only, no training.**
- `plans/arch-exploration/progress.md` — confirm Sessions 1-3 are done.
- `plans/arch-exploration/lessons_learnt.md`
- `plans/arch-exploration/ui_additions.md` — this session produces the
  biggest UI workload; make sure every new knob is appended there.
- `agents/population_manager.py` — you are going to hook into the
  Gen-0 generation code. Read it fully before designing the planner.

## Goal

Build a **training plan** layer that:

1. Records every Gen-0 population we generate — population size,
   architecture mix, hyperparam ranges, seed, timestamp, outcome
   summary.
2. Provides a pre-flight check that refuses to launch a Gen-0 with a
   population too small to give each architecture at least N agents
   (default N=5, configurable).
3. Can bias new Gen-0 sampling toward configurations that have not yet
   been well-covered in the historical record.
4. Is queryable by the UI so a user can see what's been tried without
   grepping the filesystem.

This session is the structural step that lets every subsequent
session actually deliver measurable progress.

## Scope

**In scope:**
- New module, e.g. `training/training_plan.py`, containing:
  - A `TrainingPlan` dataclass describing one Gen-0 configuration.
  - A `PlanRegistry` that persists plans to disk (JSON under
    `registry/training_plans/` — match the existing `registry/`
    convention, do not invent a new top-level folder).
  - Coverage math: given the history, compute a "coverage score" per
    architecture and per hyperparam bucket.
  - A pre-flight `validate(plan)` that raises or warns on
    insufficient population size.
  - A `bias_sampler(plan, history)` that returns a modified
    `hp_specs` weighting sampling toward under-covered regions. Keep
    the bias gentle — it should tilt, not override.
- Hook `population_manager.create_initial_population` (or whatever
  the exact entry point is called) to accept an optional
  `TrainingPlan` and honour it.
- Backend endpoints for the UI:
  - `GET /api/training-plans` — list
  - `GET /api/training-plans/{id}` — detail
  - `POST /api/training-plans` — create + validate (does NOT launch)
  - `GET /api/training-plans/coverage` — coverage stats
- **Outcome updates:** after a generation completes, the planner
  records a summary (best fitness, mean fitness, did any arch die
  out). Hook into the existing training loop's per-gen callback.

**Out of scope:**
- The UI itself. Session 8 builds the frontend. This session builds
  the backend + registry + a minimal JSON API the UI will consume.
- Actually running a Gen 0 under the new planner — that's Session 9.
- Transformer architecture (Session 6). For now the planner only
  knows about `ppo_lstm_v1` and `ppo_time_lstm_v1`. Session 6 will
  add the third as a schema extension, not a planner rewrite.

## Coverage math — keep it simple

Do not over-engineer this. A minimum-viable coverage model:

- **Architecture coverage:** count of agents per architecture in the
  historical record. A warning fires if any architecture has fewer
  than `min_arch_samples` (default 15) total.
- **Hyperparam coverage:** for each numeric gene, bucket the
  historical values into deciles of its configured range. A gene is
  "well-covered" if at least 60% of buckets have ≥1 sample. A gene
  is "poorly-covered" otherwise.
- **Bias:** the `bias_sampler` nudges sampling distributions for
  poorly-covered genes by widening their effective range or
  upweighting empty buckets. Gentle — e.g. 1.5× sampling probability
  for empty buckets relative to populated ones.

A more sophisticated scheme (Latin hypercube, Bayesian bandit, etc.)
is explicitly deferred. Document in `lessons_learnt.md` if the simple
scheme proves inadequate.

## Tests to add

Create `tests/arch_exploration/test_training_plan.py`:

1. **Plan round-trip.** Create a `TrainingPlan`, save to disk via
   `PlanRegistry`, reload, assert equality.

2. **Validate rejects undersized populations.** Plan with pop=6 and 3
   architectures → validator raises or warns.

3. **Coverage with empty history** returns zero coverage and flags
   everything as poorly-covered.

4. **Coverage with synthetic history.** Feed in 50 fake historical
   agents, assert the coverage breakdown matches what you'd compute
   by hand on a small example.

5. **Bias sampler nudges empty buckets.** Construct a history where
   `gamma` has samples only in [0.95, 0.96]. Call `bias_sampler` and
   assert the returned spec for `gamma` has increased sampling weight
   in the [0.96, 0.999] region. Exact mechanism is up to you; the
   test just has to verify the nudge happens.

6. **Outcome update round-trip.** Record a generation outcome against
   a saved plan, reload, assert the outcome persists.

7. **API endpoints** (thin tests with the existing test client):
   list, get, post-validate, coverage. 200 on happy paths, 422 on
   invalid plans.

All CPU, fast.

## Session exit criteria

- All tests pass.
- `progress.md` Session 4 entry.
- `lessons_learnt.md` updated with any real-world wrinkles of the
  coverage math (there will be some — this is new infrastructure).
- `ui_additions.md` — Session 4 is the biggest UI entry. Re-read the
  file, confirm every backend field is represented in the UI task
  list, append anything you added during implementation that wasn't
  previously listed.
- Commit.

## Do not

- Do not run an actual training generation from this session. No
  fitness calls, no GPU, no PPO rollouts. Everything is schema,
  persistence, and pure-function coverage math.
- Do not delete `config.yaml`-based launch as a working path. The
  planner is an additional layer, not a replacement — users must
  still be able to run `start_training.sh` with just config.yaml.
- Do not move the planner's storage location outside `registry/`
  unless you have a very good reason.
