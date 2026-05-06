---
plan: rewrite/phase-13-directional-scalping
parent_purpose: ./purpose.md
---

# Lessons learnt

Append-only journal. One entry per session as it lands. Keep entries
short and concrete: what we observed vs what the plan assumed, what
changed in the plan as a result, and any constraint that needs
hardening.

Inherited lessons (read before any session):

- `plans/selective-open-shaping/lessons_learnt.md` — the per-tick
  vs per-race gradient delivery lesson (Session 02, 2026-04-25).
  Direction signal must arrive at the open decision tick, not at
  settle. The S03 wiring follows this: BCE loss is computed
  per-transition with the cached label for that tick, not at end-
  of-race.

- `plans/fill-prob-in-actor/lessons_learnt.md` and
  `plans/mature-prob-in-actor/lessons_learnt.md` — the
  architecture-hash break protocol and the gradient-through guard
  pattern. S03 reuses both verbatim.

- `plans/naked-clip-and-stability/lessons_learnt.md` — units-
  mismatch bug. Any new accounting term that lands in `race_pnl`
  must be in the same units as the existing terms (per-race cash
  £). Stop-loss closes (S04) cost real cash; their P&L lands in
  `race_pnl` directly, not in shaped.

- `plans/rewrite/phase-7-port-aux-heads/lessons_learnt.md` — the v2
  `hp` dict / `--reward-overrides` precedence trap. Any new gene
  added by this plan
  (`direction_prob_loss_weight`,
   `direction_horizon_ticks`,
   `direction_threshold_ticks`,
   `mtm_stop_loss_threshold`,
   `bc_direction_target_weight`)
  is read from `hp` ONLY in the v2 trainer; the worker pre-merges
  `--reward-overrides` into `hp` via `_build_trainer_hp`. Do NOT
  copy the v1 `hp.get(name, config["reward"][name])` precedence
  pattern — it has a silent-swallow failure under `CohortGenes.
  to_dict()` semantics.

- `plans/rewrite/phase-12-counterfactual-fill-prob/` — once that
  plan lands, copy any cache-format / determinism / density-print
  patterns it established. The two label generators are sister
  modules and should share conventions.

---

## S01 — Feature audit

Landed 2026-05-06. Confidence read = **strong signal already there**.
Categories a (static price), b (recent direction), c (order book
pressure), e (market structure), and g (own position) are densely
populated. Category d (trade flow) has the headline features
(`traded_delta`, `vol_delta_3/5/10`, `book_churn`); the captured-but-
unused `TradedVolumeLadder` is a known cheap follow-on (memory note
`traded_volume_ladder_unused`) but NOT a prerequisite. Category f
(cross-runner) has rank / gap features but no cross-runner trade-flow
tensor — sequel work.

Unexpected discovery: `mid_drift` is already a 1-tick directional
proxy in the obs vector. The new direction head's threshold horizon
must be longer than 1 tick (default 5 per `purpose.md`) so the head
adds information `mid_drift` does not. Recorded in `findings.md`
"Notes for S02".

No `RUNNER_KEYS` gaps where features are computed but dropped. No
narrow-PR opportunities. Proceeding directly to S02.

## S02 — Direction-label generator

(Append on completion.)

## S03 — Direction head wired into actor

(Append on completion.)

## S04 — MTM-loss stop-loss

(Append on completion.)

## S05 — Direction-targeted BC pretrain

(Append on completion.)

## S06 — Validation cohort

(Append on completion.)
