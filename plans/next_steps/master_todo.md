# Master TODO — Next Steps

Ordered session list. Tick boxes as sessions land. Each session has
its own prompt file under `sessions/session_NN_*.md`.

This file is the **execution order** — shorter and more decisive
than `next_steps.md`, which is the raw backlog. Items are promoted
from the backlog into this list as we decide to tackle them.
`not_doing.md` records backlog items we deliberately parked, with
reasons and promotion criteria.

When a session completes:
1. Tick its box here.
2. Add an entry to `progress.md`.
3. Append any learnings to `lessons_learnt.md`.
4. If the session introduced new configurable values, append them
   to `ui_additions.md`.

Numbering continues from the arch-exploration phase — its Session 9
was the final one — so the first session in this folder is
Session 10.

---

## Phase 1 — Clear debt before the real run

Small, cheap, high-confidence. Do this first so the Phase 2 real
run isn't contaminated by known stale state.

- [x] **Session 10 — Housekeeping sweep**
  (`sessions/session_10_housekeeping.md`)
  - Fix stale `obs_dim` assertion.
  - Audit retired names (`observation_window_ticks`, scalar
    `reward_early_pick_bonus`).
  - Confirm schema-inspector view shipped (or file as follow-up).
  - Triage TODO/FIXME debt.
  - Run the integration suite once at start and again at end to
    confirm no regression.
  - CPU-only tests.

## Phase 2 — First real multi-generation run

The whole point of the arch-exploration phase was to enable this.
Everything downstream depends on its results.

- [ ] **Session 11 — First real multi-generation exploration run**
  (`sessions/session_11_real_run.md`)
  - 5 generations × 30 agents × even arch mix, fixed seed.
  - Uses existing planning / training infrastructure.
  - GPU allowed — explicitly an integration session.
  - Post-run analysis script committed.
  - Reprioritisation pass at the end updates `next_steps.md` and
    `master_todo.md`.

## Phase 3 — Promote backlog items, in rough order of confidence

**Do not start Phase 3 sessions until Session 11 has landed and
its results have been read.** Each prompt is written to re-check
Session 11 evidence before proceeding. If the run doesn't
justify a session, leave it un-ticked and promote it later — or
retire it to `not_doing.md`.

Sessions in this phase are **independent of each other** unless a
prompt explicitly declares a dependency. Run them in whatever
order matches Session 11 findings. Rough default order:

- [ ] **Session 12 — Hold-cost reward term**
  (`sessions/session_12_hold_cost.md`)
  - Design pass first, then implement.
  - Uses the Option D (Session 7) template for zero-mean shaping.

- [ ] **Session 13 — Promote `mini_batch_size` and `ppo_epochs`**
  (`sessions/session_13_ppo_risky_knobs.md`)
  - Discrete `int_choice` genes with narrow safe ranges.
  - `max_grad_norm` conditional on Session 11 evidence.

- [ ] **Session 14 — Arch-specific ranges beyond `learning_rate`**
  (`sessions/session_14_arch_specific_ranges.md`)
  - Generalise `TrainingPlan.arch_lr_ranges`.
  - Structural-gene scoping (`lstm_*` only for LSTM archs, etc.).

- [ ] **Session 15 — `ppo_feedforward_v1` baseline architecture**
  (`sessions/session_15_feedforward_baseline.md`)
  - Non-recurrent baseline sharing existing encoders.
  - Answers "do we actually need recurrence?"

- [ ] **Session 16 — `ppo_hierarchical_v1` architecture**
  (`sessions/session_16_hierarchical_transformer.md`)
  - Per-tick attention across runners, then LSTM over ticks.
  - Replaces mean/max pooling with attention pooling.
  - Outer sequence model choice committed first.

- [ ] **Session 17 — Optimiser and LR schedule work**
  (`sessions/session_17_optimiser_warmup.md`)
  - Linear LR warmup (default 0 = no-op).
  - Weight decay + AdamW optional, Session-11-evidence-gated.
  - **Gated on Session 11 evidence** that any arch is
    under-training.

## Phase 4 — Parking lot (see `not_doing.md`)

Explicitly **not scheduled**. Each has a recorded reason and
concrete promotion criteria in `not_doing.md`. Do not schedule
speculatively.

- [ ] Coverage math upgrades (Latin hypercube, Bayesian bandit)
- [ ] LSTM `num_layers > 2`
- [ ] Market / runner encoder changes

If Session 11+ results ever justify promoting one of these, add a
new numbered session file and move its tick box out of this
parking lot.
