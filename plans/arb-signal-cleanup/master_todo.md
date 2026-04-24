# Master TODO — Arb Signal Cleanup

Three sessions, one commit per session, constrained by
`hard_constraints.md`. Sessions 01–02 are automatable;
Session 03 is operator-gated.

---

## Session 01 — Force-close at T−N + entropy velocity gene + transformer ctx widening

**Status:** pending.

**Deliverables:**

- `env/betfair_env.py`:
  - Read `constraints.force_close_before_off_seconds`
    (default 0 = disabled).
  - In the step loop, BEFORE the action-handling loop,
    iterate open pairs with unfilled second legs and call
    the existing `_attempt_close` path when
    `time_to_off ≤ threshold`. Mark each placed close leg
    `close_leg = True, force_close = True`.
  - In `_settle_current_race`, classify force-closed
    pairs into a new `scalping_arbs_force_closed` count
    distinct from `scalping_arbs_closed`.
  - Accumulate `scalping_force_closed_pnl` separately.
  - Update `race_pnl` to sum
    `locked + closed + force_closed + scaled_naked`.
  - Matured-arb bonus: `n_matured = completed + closed`
    (force-closes NOT included per §7).
  - `close_signal` success bonus: agent-initiated closes
    only (force-closes NOT included per §14).
  - Record `arbs_force_closed`,
    `scalping_force_closed_pnl`,
    `force_close_before_off_seconds` in `info` and JSONL
    row.
- `env/bet_manager.py`:
  - New `Bet.force_close: bool = False` attribute.
  - `scalping_arbs_force_closed` counter on BetManager if
    count aggregation lives there.
- `agents/ppo_trainer.py`:
  - Accept `alpha_lr` on init; default `1e-2` unchanged
    if not passed.
  - Construct `self._alpha_optimizer = torch.optim.SGD(
    [self._log_alpha], lr=alpha_lr, momentum=0)`.
  - Whitelist `alpha_lr` in the trainer-side gene
    override path (add `_TRAINER_GENE_MAP` if it doesn't
    exist; plumb through `PPOConfig` or equivalent).
- `agents/policy_network.py::PPOTransformerPolicy`:
  - Widen the documented + validated range for
    `transformer_ctx_ticks` from `{32, 64, 128}` to
    `{32, 64, 128, 256}`. Strictly additive — no
    existing value removed.
  - Update the class docstring at line ~1235 to reflect
    the new choice set.
  - If `agents/architecture_registry.py` or a sibling
    file enumerates the allowed values, update that
    list too (Session 01's first step is to locate any
    such enumeration — the range may live in more than
    one place).
  - No architectural changes — `position_embedding`
    and `causal_mask` already size off `ctx_ticks`.
- `EpisodeStats` + `_log_episode`: gain
  `arbs_force_closed`, `scalping_force_closed_pnl`,
  `force_close_before_off_seconds`, `alpha_lr_active`.
- `config.yaml`:
  - `constraints.force_close_before_off_seconds: 0`
    (default).
  - `agents.ppo.alpha_lr: 0.01` (default for runs without
    a gene override).
- Tests per `hard_constraints.md` §31 (10 tests across
  force-close + entropy-velocity + transformer ctx=256
  build-and-forward).
- CLAUDE.md: three new dated paragraphs —
  - "Force-close at T−N (2026-04-21)" under "Bet
    accounting" section.
  - "Entropy velocity as gene (2026-04-21)" under
    "Entropy control" section.
  - "Transformer context window — 256 available
    (2026-04-21)" under whatever section documents
    architecture choices (or a new subsection if
    none exists).

**Exit criteria:**

- `pytest tests/arb_signal_cleanup/ -x` green.
- Full `pytest tests/ -q` green (run only if no training
  is active).
- Invariant test parametrised over
  `force_close_before_off_seconds ∈ {0, 30}` and
  `alpha_lr ∈ {1e-2, 5e-2}` stays green.
- `PPOTransformerPolicy` instantiates and forwards at
  `transformer_ctx_ticks=256` without OOM or shape
  errors.
- Scripted integration: 1 race with 5 open pairs at
  T−29s, `force_close_before_off_seconds=30` →
  `arbs_force_closed=5` (assuming all are matchable),
  `scalping_force_closed_pnl` non-zero, `arbs_naked=0`.

**Acceptance:** with both defaults (threshold=0, alpha_lr
unchanged), rollouts byte-identical to pre-change. With
threshold=30 on a race that previously settled naked, the
pair is force-closed and its cost lands in `race_pnl`
instead of the naked term.

**Commit:**
`feat(env+arch): force-close before off + per-agent alpha_lr gene + transformer ctx 256 option`.

**Session prompt:**
[`session_prompts/01_force_close_and_entropy_velocity.md`](session_prompts/01_force_close_and_entropy_velocity.md).

---

## Session 02 — Shaped-penalty warmup

**Status:** pending (requires Session 01 invariant
parametrisation for stacking).

**Deliverables:**

- `env/betfair_env.py` or `agents/ppo_trainer.py`
  (whichever owns the per-episode shaped accumulation —
  Session 01's prompt locates this precisely):
  - Read `training.shaped_penalty_warmup_eps` from
    config; default 0.
  - Track per-agent episode count since PPO training
    started. BC pretrain episodes do NOT count toward
    the warmup index.
  - Compute
    `scale = min(1.0, episode_idx / max(1, warmup_eps))`.
  - Apply `scale` multiplicatively to `efficiency_cost`
    and `precision_reward` ONLY. Other shaping terms
    unchanged.
  - Record `shaped_penalty_warmup_scale` in `info` and
    JSONL row.
- `config.yaml`:
  - `training.shaped_penalty_warmup_eps: 0` (default).
- Tests per §32 (6 tests).
- CLAUDE.md: new dated paragraph "Shaped-penalty warmup
  (2026-04-21)" under "Reward function: raw vs shaped".

**Exit criteria:**

- `pytest tests/arb_signal_cleanup/ -x` green.
- Full `pytest tests/ -q` green.
- Invariant parametrised over
  `shaped_penalty_warmup_eps ∈ {0, 5}` at episode_idx in
  `{0, 2, 4, 5, 10}` stays green.

**Acceptance:** with `warmup_eps=0`, rollouts byte-
identical. With `warmup_eps=10`, a scripted 20-episode
run shows `efficiency_cost` contribution scaling
0.0 → 1.0 linearly across ep1–ep10, then constant 1.0.
No discontinuity at the transition.

**Commit:**
`feat(env): shaped-penalty warmup across first N episodes (default disabled)`.

**Session prompt:**
[`session_prompts/02_shaped_penalty_warmup.md`](session_prompts/02_shaped_penalty_warmup.md).

---

## Session 03 — Plan draft + validator + launch (operator-gated)

**Status:** pending.

**Deliverables:**

- Verify the `TrainingPlan` data model supports cohort-
  level overrides (per-sub-population gene ranges). If
  NOT:
  - Fall back to three plan files
    `arb-signal-cleanup-probe-A.json`,
    `-B.json`, `-C.json` run serially. Document the
    chosen path in `progress.md`.
- Write the plan file(s) per §24–§26 of
  `hard_constraints.md`:
  - Population 48 total (3 architectures × 16 each), or
    16 per cohort × 3 plans.
  - 4 generations, `auto_continue: true`,
    `generations_per_session: 1`.
  - Gene schema includes `alpha_lr` (new), plus the
    existing genes from `arb-curriculum-probe`.
  - **Pin `transformer_ctx_ticks = 128`** (not drawn
    from the existing `{32, 64, 128}` range). See
    `hard_constraints.md` §24 and the 2026-04-21
    transformer-audit entry in `lessons_learnt.md`.
  - `naked_loss_anneal: {0, 2}`.
  - `training.curriculum_day_order: "density_desc"`.
  - `training.shaped_penalty_warmup_eps: 10` (A and C
    cohorts; `0` on B).
  - `constraints.force_close_before_off_seconds: 30`
    (A and C; `0` on B).
  - `transformer_ctx_ticks` pinned to 256 (all
    cohorts — this is an architectural improvement,
    not a mechanism under test).
  - `seed: 8101`.
  - `status: "draft"`.
- Pre-launch checklist script
  `scripts/check_arb_signal_cleanup_prereqs.py`, mirrors
  `check_arb_curriculum_prereqs.py` shape:
  1. `config.yaml
     constraints.force_close_before_off_seconds == 0` at
     the config level (cohort override is where it flips
     on — config is the floor).
  2. `config.yaml training.shaped_penalty_warmup_eps ==
     0` at config level (same reasoning).
  3. Plan file(s) exist with status=draft, validate
     population + gens + new knobs per §24.
  4. `data/oracle_cache/` has ≥ 1 date (BC prerequisite).
  5. `registry/training_plans/277bbf49-…json` has
     `status == failed` (prior probe blocks nothing;
     this is a sanity check, not a gate).
- Post-run validator
  `scripts/validate_arb_signal_cleanup.py`:
  - Same 5 criteria as `validate_arb_curriculum.py`.
  - Additional output: per-cohort pass/fail matrix.
  - Additional output: naked vs force-closed rate per
    agent at ep15 (diagnostic for the force-close
    mechanism working as designed).
- `plans/INDEX.md`: new row for `arb-signal-cleanup`.
- `progress.md` in this plan: Session 03 entry with
  launch sequence (prereqs → launch via admin UI →
  validator → fill Validation template).

**Exit criteria:**

- `scripts/check_arb_signal_cleanup_prereqs.py` exits 0.
- Plan file(s) validate via
  `PlanRegistry('registry/training_plans').list()`.
- Validator exits 2 (no data) on a fresh run (sanity
  check before any episodes are logged).
- Sanity check: validator run against
  `arb-curriculum-probe` logs produces the SAME 3/5
  result the `arb-curriculum/progress.md` entry recorded.

**Acceptance:** operator can tick "Smoke test first",
select the plan(s), and launch.

**Commit:**
`chore(registry): arb-signal-cleanup-probe plan + prereq/validator scripts`.

**Session prompt:**
[`session_prompts/03_plan_draft_validator_launch.md`](session_prompts/03_plan_draft_validator_launch.md).

---

## Validation (operator-gated, NOT a session)

Operator launches the probe(s) via admin UI with smoke
test first. Full run completes. Validation entry written
to `progress.md` mirroring the format of
`plans/arb-curriculum/progress.md` Validation entry:

- 5 criteria pass/fail with detail columns.
- Per-cohort pass/fail matrix (A / B / C).
- BC diagnostics table.
- Naked vs force-closed rate per agent at ep15.
- Representative agents per cohort × architecture
  quadrant.

**Exit criteria:**

- All 5 criteria results recorded.
- Cohort attribution recorded (which mechanism is
  load-bearing).
- Invariant spot-check passes (10 random JSONL rows).

**Commit:** none (validation writes back into
`progress.md` only, per §39 of `hard_constraints.md`).

---

## After Validation: follow-ons

**If criteria 1–5 all hold:**
- Move to a 16-agent × 10-gen × full-date-window scale
  run with whichever subset of mechanisms the ablation
  showed is load-bearing.
- Seed the scale run from the top cash-positive agents of
  the probe.
- Open the next plan (scale).

**If C1 passes but C4 fails:** the entropy fix works,
reward shape is deeper than this plan reaches. Next plan
is `observation-space-audit`.

**If C4 passes but C1 fails:** controller structure
problem, not velocity problem. Next plan is a controller
architecture change (PI/Adam).

**If 1–4 all fail:** diagnosis was wrong. Stop shaping,
open `observation-space-audit`.

## Queued follow-ons (for context; not in this plan)

- **`missed-opportunity-shaping`** — reward/penalty per
  tick based on fill_prob × arb_feasibility. Queued from
  `plans/arb-curriculum/master_todo.md`. Not needed if
  this plan clears the first-10-ep signal problem; if
  it doesn't, still relevant.
- **`arb-curriculum-scale`** — superseded; if this plan
  passes, the scale run is scoped fresh here rather than
  inheriting the arb-curriculum scaffolding.
- **`observation-space-audit`** — if this plan fails on
  C4 despite the combined fixes, the policy may not have
  features to distinguish good arbs from bad.
