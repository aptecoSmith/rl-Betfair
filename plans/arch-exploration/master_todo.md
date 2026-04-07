# Master TODO — Architecture & Hyperparameter Exploration

Work is organised into sessions. Each session has its own `session_*.md`
prompt file. Sessions are ordered so that **each one delivers a usable
intermediate result** — do not skip ahead.

Tick items off as they land. When a session completes, update
`progress.md` with what shipped and record anything surprising in
`lessons_learnt.md`.

---

## Phase 1 — Fix what's broken, expand what's cheap

Goal: by end of phase 1, every gene in the schema is plumbed through,
and a few obvious new genes exist. Population still uses existing
architectures. This phase alone should measurably improve Gen-0 signal.

- [ ] **Session 1 — Plumb reward genes through to env**
  (`session_1_reward_plumbing.md`)
  - Fix the dead `reward_early_pick_bonus` / `reward_efficiency_penalty`
    / `reward_precision_bonus` path. Every agent must train with its
    own sampled reward shaping.
  - Retire or repurpose `observation_window_ticks` (currently unused).
  - Add CPU-only tests that assert per-agent reward plumbing is live.

- [ ] **Session 2 — Expand PPO hyperparameter schema**
  (`session_2_ppo_schema.md`)
  - Add `gamma`, `gae_lambda`, `value_loss_coeff` as mutable genes with
    sensible ranges. Trainer already reads these — this is schema-only.
  - Do NOT add `mini_batch_size`, `ppo_epochs`, `max_grad_norm` (risky).
  - Tests: sampling produces values in range; trainer picks them up.

- [ ] **Session 3 — Expand reward hyperparameter schema**
  (`session_3_reward_schema.md`)
  - Promote `early_pick_bonus_min`, `early_pick_bonus_max`,
    `early_pick_min_seconds`, and `terminal_bonus_weight` to mutable
    genes. The first three are currently hardcoded in `config.yaml`;
    the last is currently a locked coefficient of 1.0.
  - Constraint: `early_pick_bonus_max` must be ≥ `early_pick_bonus_min`
    after sampling/mutation. Enforce in the sampler or via a repair
    step.
  - Tests: sampled values flow through to env; clamping works.

## Phase 2 — Planning & coverage infrastructure

Goal: we stop losing track of what's been tried and we can deliberately
target arch coverage per Gen 0.

- [ ] **Session 4 — Training plan / Gen-0 coverage tracker**
  (`session_4_training_plan.md`)
  - New module that records every Gen-0 configuration run: architecture
    mix, hyperparameter ranges, population size, seed.
  - Pre-flight check before Gen 0: warn/error if population size is
    too small to guarantee each architecture gets at least N agents.
  - Ability to bias new Gen-0 populations toward configurations not yet
    well-covered.
  - Tests: coverage math, plan loading/saving, bias logic.

## Phase 3 — Architecture expansion

Goal: wider search space over model shape.

- [ ] **Session 5 — LSTM structural knobs**
  (`session_5_lstm_structural.md`)
  - Promote `lstm_num_layers` ({1, 2}), `lstm_dropout` ([0, 0.3]),
    `lstm_layer_norm` ({false, true}) to mutable genes.
  - Thread through both `PPOLSTMPolicy` and `PPOTimeLSTMPolicy`.
  - Tests: policy instantiates at each combination, CPU forward pass
    works, output shapes are correct. No GPU.

- [ ] **Session 6 — `ppo_transformer_v1` architecture**
  (`session_6_transformer_arch.md`)
  - New policy class: same market/runner encoders, transformer encoder
    over a tick-context window instead of LSTM, same actor/critic heads.
  - Register in architecture registry.
  - New genes: `transformer_heads` {2,4,8}, `transformer_depth` {1,2,3},
    `transformer_ctx_ticks` {32,64,128}.
  - Repurpose the retired `observation_window_ticks` slot into
    `transformer_ctx_ticks` cleanly (no silent migration of stale gene).
  - Arch-specific LR range (transformers usually want different LR
    distributions than LSTMs) — add to the coverage tracker from
    Session 4.
  - Architecture cooldown: an agent that just switched arch keeps its
    arch for one generation before it can switch again.
  - Tests: policy instantiates, CPU forward pass, shape checks,
    registry lookup.

## Phase 4 — Deferred reward work

- [ ] **Session 7 — Drawdown-aware shaping (DESIGN PASS FIRST)**
  (`session_7_drawdown_shaping.md`)
  - Design before implementation. The obvious formulation is not
    zero-mean and would drift the policy toward conservatism even for
    random agents.
  - Write the design into this file before touching code. Get sign-off.
  - Then implement with tests.

## Phase 5 — UI rework

- [ ] **Session 8 — UI wiring for new knobs**
  (`session_8_ui_additions.md`)
  - Every new config added in sessions 1–7 becomes editable in the UI.
  - See `ui_additions.md` for the running list of UI work. That file is
    appended to at the end of every earlier session — when a developer
    adds a new knob, they must add a UI task before the session is
    marked complete.

## Phase 6 — Final verification

- [x] **Session 9 — Full Gen-0 GPU training run**
  (`session_9_gpu_shakeout.md`)
  - This is the one session where GPU tests are allowed. All prior
    sessions use CPU-only tests for fast feedback (see `testing.md`).
  - Run a full Gen-0 with the new planner, verify coverage, inspect
    logged episode stats for sanity.
  - Update `lessons_learnt.md` with anything surprising.
  - **Done (2026-04-07):** 21-agent, single-generation shakeout on
    4 train + 2 test days completed in 2530.5 s on an RTX 3090 with
    zero errors. All five post-run invariants held (gene variance,
    reward-gene → bet-count correlation `r = -0.255`, raw+shaped
    discrepancy ≤ 1e-4, arch coverage 7/7/7 exact, no duplicate
    episode-1 fingerprints). Infrastructure ready for the actual
    multi-generation exploration run (new session).
