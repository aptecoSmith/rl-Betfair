---
plan: selective-open-shaping
status: draft
---

# Master todo — selective-open-shaping

Two-session shape:
- **Session 01** — implement the mechanism + regression tests.
- **Session 02** — gene-sweep smoke probe + decision to promote or
  discard.

## Pre-session gate

Do NOT start Session 01 until `post-kl-fix-reference` (plan
`dcb97886…`) has completed BOTH its generations. We need:

- [ ] Plan `status: completed` in
      `registry/training_plans/dcb97886-…json`.
- [ ] gen-0 + gen-1 force-close rate measured from
      `logs/training/episodes.jsonl`.
- [ ] Decision point: if post-KL-fix force-close rate is < 40 %
      (the KL fix on its own taught the agent to be selective),
      close THIS plan as "resolved upstream" and move on.

---

## Session 01 — implement + test

### Read first

- [ ] `plans/selective-open-shaping/purpose.md` + `hard_constraints.md`.
- [ ] CLAUDE.md sections:
  - "Reward function: raw vs shaped"
  - "Force-close at T−N (2026-04-21)"
  - "Equal-profit pair sizing (scalping)"
- [ ] `env/betfair_env.py`:
  - `_maybe_place_paired` — the open-time hook point.
  - `_settle_current_race` — the refund walk (anchored to the
    `pair_bets` dict introduced in `da05332`).
  - `_compute_scalping_reward_terms` — where the shaped-reward
    components are summed.
- [ ] `plans/ppo-kl-fix/lessons_learnt.md` Meta-lesson 2 —
      integration-level tests are load-bearing.

### Implement

- [ ] Add `open_cost: float = 0.0` as a field on `BetfairEnv`,
      read from `reward_overrides` / hyperparam dict at env
      construction time (same plumbing as
      `matured_arb_bonus_weight`).
- [ ] Extend `RaceRecord` with `pairs_opened: int` (count of
      successful `_maybe_place_paired` calls this race) and
      `open_cost_shaped_pnl: float` (net shaped contribution from
      this plan — charges minus refunds).
- [ ] In `_maybe_place_paired`, on successful pair placement
      (aggressive matched AND passive posted), increment a per-race
      counter and subtract `open_cost` from the shaped accumulator.
      Hard_constraints §3.
- [ ] In `_settle_current_race`'s `pair_bets` walk, for each pair
      classified as `matured` or `closed`, add `open_cost` back to
      the shaped accumulator. `force_closed` and `naked` do NOT
      refund. Hard_constraints §2.
- [ ] Thread `open_cost_shaped_pnl` into `_compute_scalping_reward_terms`'s
      shaped return value. Keep separate from existing shaped terms
      so it's individually inspectable. Hard_constraints §4.
- [ ] Surface `open_cost_active`, `open_cost_shaped_pnl`,
      `pairs_opened` on the info dict and
      `EpisodeStats` → episodes.jsonl. Hard_constraints §11.

### Register the gene

- [ ] Add `open_cost` to the allowable hyperparameter ranges in
      whatever registry / training-plan validator enforces hp
      schemas.
- [ ] Gene range: `{min: 0.0, max: 1.0, type: float}` in plan
      files. Hard-bound `[0.0, 2.0]` in the trainer.
- [ ] Expose `open_cost` on the `EpisodeStats` row as
      `open_cost_active` (for the learning-curves panel).

### Regression tests (integration-level — Hard_constraints §13)

- [ ] `test_open_cost_zero_is_byte_identical` — env with
      `open_cost=0.0` produces the same `race_pnl`,
      `scalping_closed_pnl`, `scalping_force_closed_pnl`,
      `naked_pnl`, and `shaped_bonus` as a pre-plan env build
      on the same synthetic day.
- [ ] `test_matured_pair_refunds_open_cost` — a day constructed
      to produce exactly one matured pair yields
      `open_cost_shaped_pnl == 0.0` (one charge, one refund) at
      settle.
- [ ] `test_closed_pair_refunds_open_cost` — same, for an agent-
      chosen close_signal pair.
- [ ] `test_force_closed_pair_does_not_refund` — force-close
      pair yields `open_cost_shaped_pnl == -open_cost` at settle.
- [ ] `test_naked_pair_does_not_refund` — naked pair yields
      `open_cost_shaped_pnl == -open_cost` at settle.
- [ ] `test_zero_mean_invariant_across_mature_only_race` — a race
      where every opened pair matures yields
      `open_cost_shaped_pnl == 0.0` (Hard_constraints §6).
- [ ] `test_mixed_race_sums_correctly` — race with all four
      outcomes yields the exact arithmetic sum of
      `−N_opened + (N_matured + N_closed) × open_cost`.
- [ ] `test_shaped_only_never_touches_raw_pnl` — verifies
      Hard_constraints §4 by asserting `raw_pnl_reward` matches
      the pre-plan value irrespective of `open_cost`.

### Verify existing guards still pass

- [ ] `tests/test_ppo_trainer.py` full suite (62 tests).
- [ ] `tests/test_forced_arbitrage.py` (pre-existing reward-
      invariant tests).
- [ ] `tests/test_mark_to_market.py::test_invariant_raw_plus_shaped_with_nonzero_weight`.
- [ ] `tests/test_ppo_trainer.py::TestRecurrentStateThroughPpoUpdate`
      (all 4).

### CLAUDE.md

- [ ] Add subsection "Selective-open shaping (2026-04-NN)" under
      "Reward function: raw vs shaped" per Hard_constraints §12.
      Style: match the existing subsections for per-step MTM,
      naked-loss annealing, and matured-arb bonus.

### Lessons learnt + progress

- [ ] Write `plans/selective-open-shaping/lessons_learnt.md`
      Session 01 entry. Cover:
  - What landed.
  - Any unexpected complications (e.g., a partial-fill-on-open
    case that forced a classification tweak).
  - Scoreboard comparability: pre-plan rows comparable on
    `raw_pnl_reward`; `shaped_bonus` magnitudes differ.
- [ ] Close Session 01. Plan status stays `draft` until Session 02
      probe is complete.

---

## Session 02 — gene-sweep smoke probe

### Probe design

- [ ] Create a new training plan JSON —
      `selective-open-shaping-probe`. Cohort label `O`. Seed
      distinct from all prior (K=8301, W=8201, A=8101, B=8102,
      C=8103) — suggest `8401`.
- [ ] Population 12, `arch_mix` 4/4/4 (same as
      post-kl-fix-reference).
- [ ] `n_generations: 1`, `auto_continue: false` — diagnostic run,
      not GA.
- [ ] `hp_ranges.open_cost` = `{min: 0.0, max: 1.0, type: float}`
      (full range). Other genes cloned from
      `post-kl-fix-reference` so the only independent variable is
      `open_cost`.
- [ ] Notes section must:
  - State the hypothesis (higher open_cost → lower force-close
    rate).
  - State the success threshold (e.g., a detectable negative
    correlation between `open_cost_active` and
    `arbs_force_closed` across the 12 agents, |ρ| > 0.5).
  - Name the decision criteria for Session 03.

### Run

- [ ] Launch via the operator's usual worker-start flow. Not
      auto-starting from this plan to keep the GPU-use decision
      with the operator.
- [ ] Monitor the first 2-3 episodes for signs of degenerate
      `bet_count = 0` collapse at high gene values
      (Hard_constraints §7, Purpose §Risks). Kill early if so.

### Analysis

- [ ] Compute Spearman ρ(open_cost_active, arbs_force_closed) and
      ρ(open_cost_active, mean arbs_completed ÷ arbs_opened) across
      the 12 agents.
- [ ] Bucket agents into [0.0-0.25], [0.25-0.5], [0.5-1.0] open_cost
      tiers. Compare median force-close rate per tier.
- [ ] Write `plans/selective-open-shaping/probe_outputs/
      session_02_analysis.md` with the numbers.

### Decision

- [ ] If the gene sweep shows a clean drop (force-close rate
      responds to open_cost and the agent doesn't collapse):
  - Promote `open_cost` to the main gene set. Add it to the next
    production training plan's `hp_ranges`.
  - Plan status: `complete`.
- [ ] If the gene sweep shows no effect OR the agent collapses
      under any active gene value:
  - Write up the negative result in `lessons_learnt.md`.
  - Plan status: `complete` with a note for the operator. Consider
    the fill-probability-head-conditioning angle (Purpose §Out
    of scope) as the next lever.
- [ ] If partial (some agents respond, some don't):
  - Narrow the gene range and run Session 03 with a tighter sweep.

---

## Close-out (whichever path)

- [ ] CLAUDE.md updated (done in Session 01; confirm entry is
      live).
- [ ] `lessons_learnt.md` has both session entries.
- [ ] If promoted: new `hp_ranges.open_cost` default in the
      operator's canonical training plan template.
- [ ] If discarded: the mechanism stays in the codebase with gene
      default 0.0 (byte-identical). No revert required.
