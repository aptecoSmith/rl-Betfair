# Reward-Densification — Session 01 prompt

Current session: **Session 01 — Mark-to-market scaffolding
(knob at 0 default, byte-identical migration)**.

Detailed brief:
[`session_prompts/01_mark_to_market_scaffolding.md`](session_prompts/01_mark_to_market_scaffolding.md).

Before starting, read:

- [`purpose.md`](purpose.md) — the diagnosis from
  entropy-control-v2 + fill-prob-aux-probe 2026-04-19,
  the mark-to-market design sketch, and success criteria.
- [`hard_constraints.md`](hard_constraints.md) — 24
  non-negotiables. §5–§9 (MTM semantics), §10–§11 (knob
  and default), §12–§14 (telemetry + invariant), §15–§18
  (testing), §19–§20 (reward-scale change protocol).
- [`master_todo.md`](master_todo.md) — three-session scope
  and per-session exit criteria.
- `CLAUDE.md` — "Reward function: raw vs shaped" section;
  this plan adds a new "Per-step mark-to-market shaping"
  paragraph downstream of that one.
- `env/betfair_env.py` — the file being edited. Locate the
  per-step reward-assembly path and the
  `_settle_current_race` method; the MTM computation plumbs
  into the per-step accumulator for `shaped_bonus`.
- `plans/entropy-control-v2/progress.md` — the 2026-04-19
  Validation entry (target-entropy controller works, entropy
  is not the lever) and the follow-on
  `fill-prob-aux-probe` evidence.
- `plans/naked-clip-and-stability/lessons_learnt.md` — the
  2026-04-18 "reward centering units mismatch" entry;
  parallel failure mode for this plan is "shaped
  accumulator is out by a factor of something" — the same
  caller-contract integration test (`test_real_ppo_update_
  feeds_per_step_mean_to_baseline`) guards the centering
  path; we need a comparable guard for the MTM telescope
  property (see `test_mtm_telescopes_to_zero_at_settle` in
  the Session 01 prompt).
