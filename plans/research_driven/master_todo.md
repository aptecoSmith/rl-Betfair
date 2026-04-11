# Master TODO — Research-Driven

Ordered session list. Tick boxes as sessions land. Each session has
its own prompt file under `sessions/session_NN_*.md`.

This file is the **execution order** — shorter and more decisive
than `proposals.md`, which is the unranked menu. Items are promoted
from `proposals.md` into this list only after the relevant
`open_questions.md` decisions are made.

When a session completes:
1. Tick its box here.
2. Add an entry to `progress.md`.
3. Append any learnings to `lessons_learnt.md`.
4. If the session changes the obs schema, action space, or matcher,
   update `downstream_knockon.md` to reflect what `ai-betfair` now
   needs to do.

Numbering continues from `next_steps/`. Pick the next free number
when promoting an item — do not start from 1.

---

## Phase 0 — Cross-repo prerequisite (deployment-gate only)

The phantom-fill bug in `ai-betfair` (`bugs.md` R-1, audited in
`downstream_knockon.md` §0) must be fixed before any policy trained
under research-driven changes is **deployed for live trading**.

It is **not** a prerequisite for *training-side* work in this repo.
Sessions in Phase 1 and Phase 2 may be merged to master and run in
training without waiting for the live wrapper fix — they only block
when handing a new policy across the repo boundary for live
deployment. This is a deliberate change from an earlier draft that
gated the whole programme on the cross-repo fix; see
`design_decisions.md` 2026-04-07 entry "Phantom-fill gate is on
deployment, not on training-side work" for the reasoning.

The two streams (training-side improvements here, live-wrapper fix
in `ai-betfair`) should run in parallel. They are part of the same
family of "the simulator/wrapper has been too optimistic about
fills"; sequencing them strictly was overreach.

- [ ] **Phantom-fill fix shipped in `ai-betfair`** (cross-repo,
      tracks the deployment gate — tick when confirmed in
      production logs).
- [x] **Session 18 — R-2 self-depletion fix**
      (`sessions/session_18_r2_self_depletion.md`)
  - `BetManager` gains `_matched_at_level` accumulator.
  - `ExchangeMatcher._match` gains optional
    `already_matched_at_top` parameter; stays stateless.
  - Must land before P3/P4 because passive orders stretch the
    depletion window.

---

## Phase 1 — Cheap, no action-space change

These can land without breaking the action space and without the
big `ai-betfair` work in `downstream_knockon.md` §3. They give the
fastest signal on whether the research material is even pointing at
something the agent can use.

P1 is split into four sessions (19–22) so each diff is small enough
to review cleanly and fit into a Sonnet context window.

- [x] **Session 19 — P1a: OBI feature + obs schema bump**
      (`sessions/session_19_p1a_obi.md`)
  - [x] Create `env/features.py` (dependency-free, vendorable).
  - [x] `obi_topN` feature; RUNNER_DIM 110 → 111.
  - [x] `OBS_SCHEMA_VERSION = 2`; `validate_obs_schema`; schema enforced in `model_store` + `population_manager`.
  - [x] `info["debug_features"]` per-runner dict.
  - [x] 13 tests (7 pure + 3 env + 3 schema).

- [x] **Session 20 — P1b: weighted microprice feature**
      (`sessions/session_20_p1b_microprice.md`)
  - Second static feature.
  - Bounded by best-back / best-lay (tested).

- [x] **Session 21 — P1c: windowed features (traded delta + mid
      drift)** (`sessions/session_21_p1c_windowed.md`)
  - Two features in one session because they share windowing.
  - First features in this folder with cross-tick state.
  - State lives on the env; `features.py` stays pure.

- [x] **Session 22 — P1d: re-train and Phase 1 decision gate**
      (`sessions/session_22_p1d_retrain.md`)
  - Train one policy on the new obs; compare vs baseline on the
    9-day eval window under the Q3 metric.
  - Gradient-norm sanity check on the new columns.
  - Recommendation recorded in `progress.md`.
  - **Gate result: INCONCLUSIVE — single-seed comparison dominated
    by training variance (P1 policy collapsed to 0 bets in this
    seed; baseline did not). Both policies + comparison script ship.
    Proceeding to P2 — features are correctly wired (gradient flows
    at non-collapsed init, confirmed in separate run). Do not rely on
    single-seed PPO for Phase 2 gate; use evolutionary infrastructure
    instead.**

- [x] **Session 23 — P2: spread-cost shaped reward**
      (`sessions/session_23_p2_spread_cost.md`)
  - Design pass committed separately before implementation.
  - `Bet.ltp_at_placement` stashed at placement time; no matcher API changes.
  - Formula: `matched_stake × |fill − ltp| / ltp` (direction-independent half-spread).
  - `efficiency_penalty` unchanged — the two terms are complementary.
  - Intentional asymmetry documented in code, lessons_learnt.md, and design pass.
  - Knock-on: deployment note for `ai-betfair` low-bet-count
    alerting (no code change). `downstream_knockon.md` §2.

- [x] **Session 24 — P5: UI fill-side annotation**
      (`sessions/session_24_p5_ui_fill_side.md`)
  - Tiny session; can land at any time after session 18.
  - Knock-on: parity annotation on the `ai-betfair` live dashboard
    (separate repo, separate session).

**Decision gate at end of Phase 1.** Owned by session 22. If P1
delivers clearly better policies on the Q3 metric, proceeding to
Phase 2 is still justifiable; if P1 does not improve things, P2
(spread cost) may still be worth running before deciding about
Phase 2, because P2 addresses a different dimension (friction, not
information). Ship P5 whenever convenient — it doesn't depend on
eval results.

---

## Phase 2 — Expensive, action-space changes

Only start once Phase 1 is done, the eval metric is settled, and
operator has answered Q1 (selection-only vs execution-aware) in
`open_questions.md` with **B** or **B-lite-was-not-enough**.

P4 is split into three small sessions (25–27) and P3 into three
small sessions (28–30). Each P-item alone is too big for a single
focused session — splitting lets each piece be reviewed in
isolation.

### P4 — Queue-position bookkeeping (sessions 25–27)

- [x] **Session 25 — P4a: queue-snapshot bookkeeping (state only)**
      (`sessions/session_25_p4a_queue_snapshot.md`)
  - [x] `PassiveOrder` dataclass + `PassiveOrderBook` class (option B).
  - [x] Snapshot `queue_ahead_at_placement`; accumulate
        `traded_volume_since_placement` via runner total_matched deltas.
  - [x] `on_tick` wired into `betfair_env.step()`; `info["passive_orders"]` exposed.
  - [x] 14 assertions across 8 test classes; all pass.
  - Matcher unchanged.

- [x] **Session 26 — P4b: passive-fill triggering + budget
      reservation** (`sessions/session_26_p4b_passive_fill.md`)
  - [x] Fill when traded volume exceeds queue-ahead + passive self-depletion.
  - [x] Budget reserved at placement (back: deduct stake; lay: reserve liability).
  - [x] No double-subtraction on fill — reservation converts in-place.
  - [x] Passive self-depletion accumulator (`_passive_matched_at_level`) distinct from aggressive accumulator.
  - [x] `info["passive_fills"]` emitted per tick.
  - [x] 19 assertions across 9 test classes; all pass.
  - [x] Junk filter blocks fill on drifted ticks; fill resumes when LTP returns.

- [x] **Session 27 — P4c: race-off cleanup**
      (`sessions/session_27_p4c_race_off_cleanup.md`)
  - [x] Unfilled passives cancelled at race end with zero P&L.
  - [x] Budget reservations released.
  - [x] Efficiency-penalty interaction decided and pinned (YES — cancelled passives count toward `bet_count` for efficiency penalty; API call friction is real).
  - [x] Hook point (A): top of `_settle_current_race`.
  - [x] 13 assertions across 9 test classes; all pass.
  - P4 as a whole is now complete (sessions 25–27).

### P3 — Passive orders + cancel action (sessions 28–30)

Sessions 28 and 29 **must ship as a pair**. Cancel without passive
placement is a no-op; passive placement without cancel is a trap.
See `hard_constraints.md` #9.

- [x] **Session 28 — P3a: aggression flag in action space**
      (`sessions/session_28_p3a_aggression_flag.md`)
  - Extend per-slot action with discrete passive/aggressive flag.
  - Action schema bump (`ACTION_SCHEMA_VERSION = 1`).
  - `actions.force_aggressive=true` config override reproduces
    pre-P3 policy exactly (regression backstop).
  - 16 tests in `test_p3a_aggression_flag.py`; all pass.

- [x] **Session 29 — P3b: cancel action**
      (`sessions/session_29_p3b_cancel.md`)
  - [x] Per-slot cancel flag; "cancel oldest open passive on this
    runner" semantics.
  - [x] Cancel + place in the same tick = atomic move.
  - [x] No `modify` action (parked, ND-1 in `not_doing.md`).
  - [x] `ACTION_SCHEMA_VERSION = 2`; `ACTIONS_PER_RUNNER = 4`.
  - [x] `cancel_oldest_for(selection_id, reason)` on `PassiveOrderBook`.
  - [x] 13 tests in `test_p3b_cancel.py`; all pass.

- [x] **Session 30 — P3c: Phase 2 re-train + diversity check +
      decision gate** (`sessions/session_30_p3c_retrain.md`)
  - Fresh init, same HP as session 22 where possible.
  - Diversity assertions: aggression histogram not collapsed,
    cancel rate and passive-fill rate non-trivial.
  - Regression sanity check before training (force-aggressive
    reproduces Phase 1 policy).
  - Recommendation filed: **keep-code-only** (P3+P4 policy
    collapsed to zero bets in eval; single-seed PPO uninformative
    as phase gate; code paths correct but deployment stays on
    Phase 1 policy).

**Phase 2 decision gate — CLOSED (2026-04-11).** Outcome:
**keep-code-only.** P3+P4 code stays in the simulator (correct,
tested, regression-checked) but deployment to `ai-betfair` stays
on the simpler Phase 1 policy. Single-seed PPO cannot meaningfully
discriminate — evolutionary infrastructure needed for a real gate.

---

## Phase 3 — P1 extensions

Additional obs features that don't change the action space. Cheap
and worth adding regardless of Phase 2 retrain results — more
information in the obs vector doesn't hurt.

- [x] **Session 31 — P1e: order-book churn rate (PARKED)**
      (`sessions/session_31_p1e_book_churn.md`)
  - Parked: tick cadence check failed (6–10s median). Original
    ≤2s threshold was too narrow — see session 31b.

- [x] **Session 31b — P1e: order-book churn rate (revised)**
      (`sessions/session_31b_p1e_book_churn.md`)
  - Per-tick normalised churn: how much the visible book rearranged
    between consecutive ticks. OBS_SCHEMA_VERSION 4 → 5.
  - 16 tests, all pass. RUNNER_DIM 114 → 115.

---

## Items NOT promoted

Anything in `proposals.md` that is not yet ticked above is
deliberately **not** in the execution queue. Either:

- Open questions in `open_questions.md` haven't been answered, or
- It depends on a Phase 2 item that hasn't landed.

When promoting an item, copy its acceptance criteria from
`proposals.md` into the session prompt file under `sessions/` —
don't rely on the proposal text alone, because `proposals.md` is
allowed to be revised after promotion.
