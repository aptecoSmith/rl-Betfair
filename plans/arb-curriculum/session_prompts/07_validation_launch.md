# Session 07 prompt — Validation launch (operator-gated)

**IMPORTANT:** Per `../hard_constraints.md` §35 and §38,
this session is the final operator-gated step. The
instructions below are for the operator. This session
writes back into `progress.md` but does NOT produce a
commit (§38).

## PREREQUISITE — read first

- [`../purpose.md`](../purpose.md) — "What success looks
  like" criteria 1–5.
- [`../hard_constraints.md`](../hard_constraints.md). §24
  (invariant correctness gate), §33 (reward-scale
  change), §35 (operator-gated), §38 (no commit for this
  session), §40 (progress entry format).
- [`../master_todo.md`](../master_todo.md) — Session 07
  deliverables.
- [`../progress.md`](../progress.md) — every session
  1–6's entries must exist.
- `plans/reward-densification/progress.md` Validation
  entry — the baseline comparator.

## Hand-off to operator

### 1. Launch

1. Start admin UI + API stack (normal dev stack).
2. Training Plans page: confirm `arb-curriculum-probe`
   is visible with status `draft`.
3. Select the plan, tick "Smoke test first".
4. Click Launch.

### 2. Smoke test — expected

- 3 agents × 3 episodes probe. BC pretrain runs before
  PPO on each. The tracking-error gate at
  `target_entropy=150` SHOULD still pass because the
  BC↔controller handshake (Session 04) anneals the
  effective target over `bc_target_entropy_warmup_eps`
  (default 5) episodes, so the 3-ep smoke sees the
  anneal's early phase where low entropy is acceptable.
- **If smoke FAILS on the tracking-error assertion**:
  capture the failure modal and log into
  `lessons_learnt.md` — likely the handshake anneal is
  mistimed. Don't proceed to full run until understood.
- **If smoke FAILS on the reward invariant**: STOP. This
  is a §24 correctness-gate failure. Roll back, fix, and
  redo Session 02/03 before relaunching.

### 3. Full 33-agent / 4-gen run

Watch the learning-curves panel for the criteria from
`purpose.md §What success looks like`:

1. **≥ 80 % of agents remain active through ep15** (bets
   > 0). Previous runs hit ≈ 50 %.
2. **`arbs_closed / arbs_naked` ratio > 15 % on ≥ 3
   agents by ep15**.
3. **`policy_loss` stays O(1)+ through ep15 on ≥ 50 %
   of agents**.
4. **≥ 3 agents reach `total_reward > 0` by end of
   gen 3**.
5. **Invariant `raw + shaped ≈ total` holds every
   episode** (spot-check 10 random rows).

Check also:

- `bc_final_signal_loss` / `bc_final_arb_spread_loss` on
  ep 1 rows — should be O(0.1) or smaller, indicating BC
  converged. If O(1) or higher, BC didn't get enough
  steps (rerun with higher `bc_pretrain_steps`).
- `matured_arb_bonus_active` / `naked_loss_scale_active`
  — confirm the expected values are in effect (varied
  per agent by gene; scale annealing toward 1.0 by
  gen 2).
- `curriculum_day_order` — confirms `"density_desc"`
  on every row.
- `arb_events` activity log lines — look for actual
  locked pairs rather than naked dominance.

### 4. Capture findings

Write a Validation entry in
`plans/arb-curriculum/progress.md` mirroring the format
of `plans/reward-densification/progress.md`'s Validation
entries. Include:

- Run duration, generations completed.
- Per-agent summary: architecture, reward trajectory
  shape (healthy / bleeder / passive / hyperactive),
  key gene values (BC steps, MTM weight, matured-arb
  bonus weight, naked_loss_scale).
- Population-level criteria scoring (1–5 from above):
  pass / partial / fail per criterion.
- Representative trajectories: at least one from each
  observed behavioural class (at least one from each
  architecture).
- Headline number: how many agents cleared "total_reward
  > 0". Any agent with a POSITIVE reward trajectory is a
  first — document per-ep details.
- Invariant spot-check: 10 random rows, `raw + shaped -
  total` absolute differences.
- Gotchas: anything surprising the operator observed
  (BC-related instabilities, entropy drift, close_signal
  atrophy despite BC, etc.).

### 5. Decide next plan

Based on validation outcome, pick from
`master_todo.md §Queued follow-ons`:

- **If ≥ 4 criteria pass**: open
  `arb-curriculum-scale` plan for 16-agent × 10-gen
  × full-date-window run.
- **If criteria 5 fails**: rollback. Do NOT ship this.
  Fix the invariant-breaking session (02 or 03) and
  redo Session 06.
- **If 1–4 all fail**: open `observation-space-audit` —
  the policy's obs may lack features needed to recognise
  good arbs in real time.
- **If partial success (some agents thrive)**: open
  `arb-curriculum-tune` — tighten gene ranges around the
  winning combinations; re-run with selection pressure
  focused on what worked.

## Cross-session rules

- Do NOT commit code or tests in this session.
- Do NOT bundle the launch into Session 06's commit (§38).
- If something in the validation surfaces a bug in
  Sessions 01–05, roll back and fix the offending
  session rather than patching in a follow-on.
- If the run completes without any criterion passing but
  also doesn't crash, it's a "planned experiment with
  null result" — write the Validation entry honestly
  (many RL plans fail; that's data).

## After Session 07

This plan is complete. Next steps depend entirely on the
Validation outcome — see
`master_todo.md §After Session 07` and
`§Queued follow-ons`.
