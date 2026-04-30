---
plan: rewrite/phase-3-cohort
status: design-locked
opened: 2026-04-29
depends_on: phase-2-trainer (AMBER, 2026-04-29 — see findings.md §"Verdict")
---

# Phase 3 — GPU pathway, multi-day training, GA cohort, frontend

## Purpose

Take Phase 2's single-agent / single-day / CPU-only trainer and
turn it into a real cohort run that can be compared head-to-head
against v1's cohort-M.

Phase 2's success bar was "one agent, one day, loss curves sane."
Phase 3's success bar is "12-agent cohort, multi-day, beats v1 on
the three architectural metrics that motivated the rewrite"
(README §"Success bar (Phase 3)").

Four things gate Phase 3 success:

1. **GPU.** Single-CPU rollout took 113 s/episode in Session 03.
   12 agents × 30 days × 4 epochs is **~37 hours on CPU**.
   Unworkable. v1 had a working CUDA pathway with pinned memory +
   non-blocking transfers; v2 currently hardcodes `device="cpu"`.
2. **Multi-day training.** Phase 2 trained on one day to keep scope
   tight. Bar 2 (value loss descends) was AMBER specifically because
   per-episode reward variance dominates the signal at 5 episodes.
   Multi-day training resolves this either way.
3. **GA cohort scaffolding.** Worker pool, gene schema, breeding,
   mutation, registry integration. v1 has all of this in
   `training/run_training.py` + `training/worker.py`; v2 has none of
   it. The rewrite's hard constraint §3 (parallel tree, no v1
   imports) means we re-implement the patterns, not import them.
4. **Frontend wiring.** v1's UI consumes a websocket event stream
   (`websocket_events.py` schema). v2 needs to emit the same shape
   so the existing UI keeps working without UI changes (rewrite hard
   constraint §"Out of scope").

## Why one phase, not four

Each individual piece is small. The risk of bundling is "while
we're at it" creep. The risk of splitting is integration drift —
GPU + multi-day work that's never run through the cohort pipeline
will turn out to have wrong assumptions. Single phase, four
sessions, one integration test at the end.

## What's locked

### GPU contract

Phase 3 honours v1's GPU patterns verbatim where they're correctness
facts (pinned memory + non-blocking transfers), and re-derives v2-
specific shape contracts where v1's are wrong-by-default for the new
architecture (per-runner value tensors, hidden-state batching).

The GPU pathway must produce **numerically identical** training
trajectories to the CPU pathway given the same seed. "Identical
within 1e-5 tolerance for total_reward and value_loss" is the
acceptance bar — anything looser hides a real bug. v1's CUDA
pathway met this bar; we hold v2 to the same standard.

CLI: `--device {cpu,cuda,cuda:N}` flag on `train.py`. Default
`cpu` for backward compat with Phase 2 reproducibility. Phase 3's
cohort runner explicitly opts into `cuda`.

### Multi-day training

Per-day rollouts → per-day PPO updates → next day. The episode
boundary IS the day boundary. No cross-day GAE bootstrapping (the
env's `_settle_current_race` already terminates the episode at end
of day; Phase 1's findings §3 noted the env's natural episode
boundaries and the rewrite respects them).

Day ordering: deterministic shuffle per agent seed for Phase 3
Session 02. Curriculum (density-sorted, oracle-density, etc.)
deferred to a follow-on plan if multi-day flat ordering shows
order-dependence problems.

### Cohort scope (Phase 3 first run)

12 agents, 7 days of training data, 1 day held out for evaluation.
This is intentionally smaller than v1's typical cohort (66
agents × 30 days) — Phase 3's job is to prove the architecture
works, not to beat v1 on a maxed-out cohort. If 12 / 7 / 1 looks
healthy, scale-up is a follow-on plan.

### What does NOT change

- Env, matcher, bet_manager, force-close, equal-profit sizing —
  Phase 2 hard constraint §1 still applies.
- Data pipeline — Phase 2 hard constraint §2 still applies.
- v1 not imported — rewrite hard constraint §3.
- No new shaped rewards, no entropy controllers, no advantage
  normalisation, no LR warmup, no reward centering — rewrite hard
  constraints §5, §6.
- Locked Phase 2 hyperparameters carry forward unchanged.

## Success bar (Phase 3)

The phase ships iff **all** of:

1. **GPU pathway numerically matches CPU pathway.** Same seed,
   same day, same hyperparameters → `total_reward` and
   `value_loss_mean` match within 1e-5 across 5 episodes.
2. **GPU is meaningfully faster than CPU.** Wall time
   per-episode on a CUDA device < 50 % of the Phase 2 CPU
   baseline (113 s/episode → < 57 s/episode). On a real GPU we
   expect 5–10×; 2× is the loose bar to catch "GPU enabled but
   bottlenecked elsewhere."
3. **Multi-day training resolves Bar 2.** A single agent trained
   on 7 days × 4 episodes shows value-loss curves that descend
   monotonically (with episode-level noise) within each day's
   epochs, and overall trend across days. **AMBER from Phase 2
   becomes GREEN here, or this is a finding.**
4. **GA cohort runs end-to-end.** 4-agent dry-run cohort
   completes one full training pass + writes scoreboard rows in
   the existing format the registry consumes.
5. **Frontend renders v2 events without code changes.** The
   existing UI shows v2 cohort progress live — same scoreboard,
   same per-episode chart, same training curves panel.
6. **12-agent cohort beats v1 cohort-M on the three architectural
   metrics:**
   - Mean force-close rate **< 50 %** (vs v1 ~75 %).
   - ρ(open_cost-equivalent gene, fc_rate) **≤ −0.5** (vs v1 ~0).
   - **At least one agent positive on raw P&L** on the held-out
     test day (vs v1 0–7/66 historically).

If 1–5 PASS but 6 FAIL: Phase 3 ships AMBER, the architecture is
correctly wired but the rewrite's bet didn't pay. Go to
findings.md and decide whether to iterate inside v2 or step back
(rewrite README §"Success bar").

If 1–5 fail in any combination: Phase 3 doesn't ship. The
architecture has a bug, not a hyperparameter problem.

## Deliverables

A new directory `training_v2/cohort/` (parallel to v1's
`training/`) with:

- `training_v2/cohort/worker.py` — single-agent training driver
  (multi-day loop, per-day PPO updates, registry write, websocket
  emit). Replaces v1 `training/worker.py`.
- `training_v2/cohort/runner.py` — cohort orchestrator (worker
  pool, GA breeding, scoreboard aggregation). Replaces v1
  `training/run_training.py`.
- `training_v2/cohort/genes.py` — Phase 3 gene schema (~6–8
  genes per the rewrite README §"What survives" GA row).
- `training_v2/cohort/events.py` — websocket-event adapter
  (translates v2 trainer / worker events → v1 schema the UI
  expects).

Extensions to existing v2 files:

- `training_v2/discrete_ppo/train.py` — `--device` flag, multi-
  day loop, day curriculum hook.
- `training_v2/discrete_ppo/trainer.py` — pinned-memory + non-
  blocking transfers in `_ppo_update`. Pre-allocated GPU obs
  buffer reused across rollout steps.
- `training_v2/discrete_ppo/rollout.py` — pinned-memory transfers
  in `_collect`. GPU device awareness on hidden state.

Tests under `tests/`:

- `tests/test_v2_gpu_parity.py` — same seed, same day, CPU vs
  CUDA → matching rewards / value losses to 1e-5 tolerance.
  Marked `@pytest.mark.gpu` so the default suite skips it; the
  GPU runner picks it up.
- `tests/test_v2_multi_day_train.py` — synthetic 3-day dataset →
  trainer iterates correctly, episode boundaries align with day
  boundaries.
- `tests/test_v2_cohort_worker.py` — worker.py runs one agent
  end-to-end through 1 day, writes correct scoreboard row.
- `tests/test_v2_websocket_events.py` — events.py output shape
  matches v1's `websocket_events.py` schema (read v1 for
  reference, do not import).

A short writeup at `plans/rewrite/phase-3-cohort/findings.md`:
GPU vs CPU benchmarks, multi-day Bar-2 verdict, cohort comparison
table, success-bar verdict.

## Hard constraints

In addition to all rewrite hard constraints (README §"Hard
constraints"):

1. **GPU pathway must match CPU pathway numerically.** No "close
   enough." Hidden bugs that show up only at scale are the most
   expensive kind to fix.
2. **No v1 trainer / worker / runner imports.** Read v1 as
   reference; re-implement in `training_v2/cohort/`. Same rule as
   Phase 2.
3. **No frontend code changes.** v2 events go through an adapter
   layer that emits the v1 schema. UI stays as-is. If v2 wants to
   surface a metric the UI doesn't yet show, that's a follow-on
   plan, not Phase 3 scope.
4. **No new shaped rewards.** If multi-day training reveals a
   reward-shape problem, that's a finding, not a fix.
5. **No GA gene additions beyond the locked schema.** The Phase 3
   gene set is pinned in `02_genes_locked.md` (Session 03's
   first deliverable). Changing it mid-cohort invalidates breeding.
6. **GPU work uses pinned memory + non_blocking transfers.** v1's
   `agents/ppo_trainer.py:2131-2142` is the canonical reference.
   Don't reinvent.
7. **Cohort writes the SAME registry shape v1 writes.** The
   registry / scoreboard / weight-file layout is shared between
   v1 and v2 during the comparison window. New `arch_name` /
   `state_dict_shape` discriminate v2 weights from v1 weights;
   everything else is identical.

## Out of scope

- v1 deletion (after Phase 3 success per rewrite README).
- 66-agent cohort (Phase 3 caps at 12; scale-up is a follow-on).
- Curriculum day ordering beyond "deterministic shuffle"
  (follow-on plan if multi-day flat ordering shows problems).
- BC pretrain (rewrite removes BC; Phase 0's scorer replaces the
  discriminative half).
- Per-agent device pinning (Phase 3 runs all agents on a single
  GPU sequentially or threaded; multi-GPU GA is a follow-on).
- Performance optimisation beyond "GPU pathway works" (hyper-
  optimisation of throughput is a follow-on).

## Phase 2 hand-offs that constrain Phase 3

From `plans/rewrite/phase-2-trainer/findings.md` (Session 03,
AMBER, 2026-04-29):

1. **Phase 2 CPU baseline = 113 s/episode**, 11872 transitions, 744
   PPO mini-batch updates per episode. Phase 3 GPU bar is < 57 s
   on a CUDA device with the same input shape.
2. **`approx_kl` was 0.017–0.036 across 5 episodes** with KL
   threshold 0.15. KL early-stop never tripped. Phase 3's GPU
   pathway must preserve this — KL is the canonical
   "rollout-and-update saw the same hidden state" check; if KL
   spikes on GPU, the device handshake on `pack_hidden_states` is
   broken.
3. **`EpisodeStats` was extended with diagnostic fields**
   (`action_histogram`, `advantage_*`, `day_pnl`) in Session 03.
   Phase 3's worker / runner read these directly to populate the
   websocket events.
4. **`RolloutCollector.last_info`** was added so the trainer can
   read terminal `day_pnl` without touching the env. Phase 3's
   multi-day loop reads `last_info` between day-episode boundaries
   to populate the per-day scoreboard row.
5. **Hidden-state pack/slice is dim-1 batched for LSTM.** Phase 1's
   `pack_hidden_states` already implements this; Phase 3 just calls
   it. Phase 3 must pass packed states to the policy on the
   correct device (the current implementation moves them but
   doesn't pin them).

## Sessions

1. **`01_gpu_saturation.md`** — `--device` flag, pinned memory +
   non-blocking transfers in trainer + rollout, GPU baseline vs
   CPU baseline, parity test. End-of-session check: Bar 1 + Bar 2
   PASS.
2. **`02_multi_day_training.md`** — multi-day loop in `train.py`,
   day boundary handling, deterministic day-shuffle, run one
   agent on 7 days. End-of-session check: Bar 3 PASS (Phase 2's
   AMBER becomes GREEN).
3. **`03_cohort_scaffolding.md`** — `training_v2/cohort/{worker,
   runner,genes}.py`, GA breeding + mutation, registry integration,
   4-agent dry-run cohort. End-of-session check: Bar 4 PASS.
4. **`04_frontend_and_first_cohort.md`** — `training_v2/cohort/
   events.py` websocket adapter, run 12-agent / 7-day cohort,
   compare to v1 cohort-M, write findings.md. End-of-session
   check: Bar 5 + Bar 6 PASS (or AMBER with documented finding).

Each session is independently re-runnable. Session 02 imports
Session 01's GPU pathway verbatim; Session 03 imports Session 02's
multi-day train CLI; Session 04 imports Session 03's cohort
scaffolding. Same "if a later session finds a problem in an
earlier session, revisit the earlier session" rule as Phases 1
and 2.
