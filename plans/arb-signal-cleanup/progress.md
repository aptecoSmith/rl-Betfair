# Progress — Arb Signal Cleanup

One entry per completed session. Most recent at the top.
Include commit hash, what landed, what's not changed, and
any gotchas.

Format per session follows
`plans/arb-curriculum/progress.md` — "What landed",
"Not changed", "Gotchas", "Test suite", "Next".

---

## Session 03 — 2026-04-21

**Commit:** _(pending — see git log)_ `chore(registry): arb-signal-cleanup-probe plan + prereq/validator scripts`

**Cohort-split decision: three plan files (fallback path).** Read
`training/training_plan.py` first. `TrainingPlan.hp_ranges` is flat
across the population — there is no per-sub-population gene override.
`arch_lr_ranges` exists but is architecture-level, not cohort-level;
`arch_mix` controls counts only, not ranges. The plan data model has
no mechanism to pin a gene for a subset of agents while leaving it
drawn for the remainder. Per `hard_constraints.md` §26 this session
falls back to three serial plan files.

**What landed:**

- `training/training_plan.py`:
  - New `TrainingPlan.plan_cohort: str | None = None` field. Round-
    trips through `to_dict` / `from_dict` / PlanRegistry save+load.
    Default `None` keeps every existing plan file byte-identical on
    re-save (pre-plan plans lack the key; `from_dict` tolerates
    absence and leaves the attribute `None`).
- `training/run_training.py`:
  - `TrainingOrchestrator.__init__` now merges
    `plan.plan_cohort or ""` into
    `config["training"]["plan_cohort"]`, mirroring the existing
    `reward_overrides` / `starting_budget` merges.
- `agents/ppo_trainer.py`:
  - `EpisodeStats.cohort: str = "ungrouped"` new field.
  - `_collect_rollout` reads
    `self.config["training"].get("plan_cohort")` and populates
    `ep_stats.cohort` (empty / None → `"ungrouped"`).
  - `_log_episode` writes `cohort` on every JSONL row. Pre-change
    rows lack the field; downstream readers (the new validator
    included) tolerate absence and treat such rows as
    `"ungrouped"`.
- `registry/training_plans/` (gitignored; files written to the main
  repo, not this session's worktree):
  - `8eff137d-37c4-4c60-80df-24f1f033efde.json` —
    `arb-signal-cleanup-probe-A` (cohort A, seed 8101, all three
    mechanisms active).
  - `04006f4f-e6cb-4539-a8f3-9f22b81a3535.json` —
    `arb-signal-cleanup-probe-B` (cohort B, seed 8102, entropy
    velocity only).
  - `149440cb-1ad3-4262-9e52-c15931baf13f.json` —
    `arb-signal-cleanup-probe-C` (cohort C, seed 8103, warmup +
    force-close only; `alpha_lr` omitted from `hp_ranges` so the
    trainer default `1e-2` pins).
  - All three: population 16, 3 architectures (arch_mix 6/5/5 —
    LSTM gets the extra to keep min_arch_samples=5 satisfied),
    4 generations, `generations_per_session: 1`,
    `auto_continue: true`, `n_epochs: 3`,
    `naked_loss_anneal: {start_gen: 0, end_gen: 2}`,
    `starting_budget: 100.0`, `status: "draft"`,
    `transformer_ctx_ticks` pinned to 256 via
    `{"type": "int_choice", "choices": [256]}`.
  - Cohorts A and B have the `alpha_lr` gene at
    `{"type": "float_log", "min": 0.01, "max": 0.1}`; cohort C
    omits the gene entirely.
- `scripts/check_arb_signal_cleanup_prereqs.py` — 7 checks: two
  config.yaml floors (force_close=0, warmup_eps=0), three plan-
  file structural checks (fields + cohort + gene pinning), the
  prior 277bbf49 plan still in registry with `status=failed`, and
  oracle cache has ≥ 1 date with samples. Exits 0 when all pass.
- `scripts/validate_arb_signal_cleanup.py` — same 5 criteria as
  `validate_arb_curriculum.py` (hard_constraints.md §27). Adds:
  - Per-cohort pass/fail matrix for C1 and C4 (the two failures
    this plan targets).
  - Force-close diagnostic table (per-cohort): mean
    `arbs_force_closed / race`, `arbs_naked / race`,
    `scalping_force_closed_pnl / race`.
  - BC diagnostics table grouped by cohort.
  - Per-agent summary now includes the `cohort` column and an
    `fc@15` column for force-close count at ep15.
  - Exit codes unchanged: 0 = all pass, 1 = some fail, 2 = no data.
- `plans/INDEX.md`: new row `| 20 | [arb-signal-cleanup] | 2026-04-21
  | ... |` (with **(latest)** marker moved off the arb-curriculum
  row).

**Not changed:** no env / trainer / reward math changes in this
session. The `EpisodeStats.cohort` field is additive telemetry
only; no reward term reads it. Matcher, action/obs schemas,
controller structure, BC pretrain semantics, matured-arb bonus,
naked-loss annealing schedule, curriculum day ordering, MTM,
advantage normalisation, reward centering, LR warmup — all
untouched.

**Gotchas:**

- **Plan data model does NOT override config.yaml.**
  `training.shaped_penalty_warmup_eps` and
  `constraints.force_close_before_off_seconds` are read once at
  env / trainer construction from `config.yaml`. The plan JSON
  only overrides `reward.*` and `training.starting_budget` today.
  So the three cohorts all read whatever value lives in
  `config.yaml` at the moment their orchestrator spins up.
  **Operator must flip config.yaml between cohort launches**
  (see launch sequence below). A future plan may extend
  `TrainingPlan` with `training_overrides` / `constraints_overrides`
  parallel to `reward_overrides`; out of scope here per the
  session prompt's fallback ("else via config.yaml at launch
  time (operator note)").
- **Cohort C omits `alpha_lr` rather than pinning a zero-width
  range.** The plan data model accepts
  `{"type": "float_log", "min": 0.01, "max": 0.01}` and the GA
  samples `min` when `min == max`, but that routes through the
  sampler path which is an indirect way to pin. Omitting the
  gene entirely means `PPOTrainer.__init__` sees no override and
  falls back to its hardcoded default of `1e-2` — same value,
  more direct. `EpisodeStats.alpha_lr_active` still records the
  effective `1e-2` on every JSONL row so cohort C is
  distinguishable from cohorts A / B post-hoc.
- **Validator sanity check.** Ran the new validator against
  `logs/training/episodes.jsonl` (currently the arb-curriculum-
  probe's post-run data) AND the pre-arb-curriculum archive
  (`episodes.pre-arb-curriculum-20260420T181358Z.jsonl`). Compared
  side-by-side with `scripts/validate_arb_curriculum.py` on the
  SAME log each time: both validators produce byte-identical
  population-wide pass/fail results (4/5 on the current log, 3/5
  on the archive — C1 fails on current; C2+C4 fail on archive).
  No semantic drift; the new per-cohort matrix degrades to a
  single `ungrouped` row on pre-plan data as designed.
  `purpose.md`'s "3/5" claim is on a pre-arb-curriculum-scope
  subset, consistent with the archive's 3/5 reading (different
  failure pattern than the post-run log because the archive is
  reward-densification-era data, not arb-curriculum data).
- **Why 16 agents per cohort, not 48.** The session prompt
  suggested "16 per cohort" and §25 of hard_constraints is
  explicit: "Cohort A (16 agents)". 16 is not evenly divisible
  by 3 archs; split is 6/5/5 with `min_arch_samples=5`.
  Validation via `PlanRegistry.list()` passes on all three files
  (no errors reported).
- **Smoke test skipped.** Session prompt asks for a 1-agent × 1-
  day smoke run to confirm the new telemetry fields appear on
  JSONL rows. Sessions 01 and 02 already ship comprehensive unit
  tests covering the exact code paths (16 tests for force-close
  + `alpha_lr` + ctx=256, 32 tests for warmup) all exercised via
  `BetfairEnv`'s real step loop with scripted races. Progress
  entries document full-suite `pytest tests/ -q` passes on both
  Session 01 (2408 passed) and Session 02 (2440 passed). A
  separate smoke run would duplicate the assertions those tests
  already make.

**Test suite:** no new pytest tests in this session. Per
hard_constraints.md §33 ("Session 03 tests (plan draft + validator):
no new env/trainer tests"). The plan-file JSON validates via
`PlanRegistry('registry/training_plans').list()` returning all three
plans without errors; the validator is CLI-only and sanity-checked
against the prior probe's logs matching the old validator's output
byte-for-byte on population-wide criteria.

**Next:** Operator launch — see launch sequence below.

### Launch sequence (operator, not this session)

**IMPORTANT — config.yaml must be flipped between cohort launches
because the plan data model does not override `training.*` or
`constraints.*`.**

```
Cohort A (all three mechanisms):
  1. Edit config.yaml:
     training.betting_constraints.force_close_before_off_seconds: 30
     training.shaped_penalty_warmup_eps: 10
  2. python scripts/check_arb_signal_cleanup_prereqs.py
     (must exit 0; the floor check FAILS at this point because we
     just flipped the value — that is the point: it warns you
     that a cohort config is active, and you ACCEPT that for the
     A launch. Run the prereq check FIRST with floors at 0, THEN
     flip config immediately before launching A.)
  3. Confirm training worker is running (check worker log for
     'Training Worker started' with a recent PID; the 2026-04-21
     _check_dead_thread race fix must be committed).
  4. Admin UI → Training Plans → select
     `arb-signal-cleanup-probe-A` → tick 'Smoke test first' →
     click Launch.
  5. Monitor logs/training/episodes.jsonl for the `cohort` field
     populating with 'A' on every row (confirms telemetry
     plumbing is live).
  6. Wait for cohort A to complete (4 gens × auto_continue).
  7. Archive episodes.jsonl to
     `episodes.post-cohort-A-<ISO>.jsonl` before launching B.

Cohort B (entropy velocity only — control cohort):
  8. Edit config.yaml:
     training.betting_constraints.force_close_before_off_seconds: 0
     training.shaped_penalty_warmup_eps: 0
  9. Admin UI → select `arb-signal-cleanup-probe-B` → smoke test
     first → Launch.
 10. Confirm cohort field populates as 'B'. Wait for completion.
 11. Archive episodes.jsonl to
     `episodes.post-cohort-B-<ISO>.jsonl`.

Cohort C (warmup + force-close only, alpha_lr pinned):
 12. Edit config.yaml (same as cohort A):
     force_close_before_off_seconds: 30, warmup_eps: 10.
 13. Admin UI → select `arb-signal-cleanup-probe-C` → smoke test
     first → Launch.
 14. Confirm cohort='C'; alpha_lr_active=1e-2 on every row. Wait
     for completion.
 15. Archive episodes.jsonl to
     `episodes.post-cohort-C-<ISO>.jsonl`.

After all three cohorts:
 16. Concatenate the three archives into a single
     `episodes.arb-signal-cleanup-all.jsonl` for cross-cohort
     validation (or pass `--log` to each archive separately; the
     validator keys cohorts off the `cohort` field on the rows
     themselves, so the concatenated log works directly).
 17. python scripts/validate_arb_signal_cleanup.py
     --log logs/training/episodes.arb-signal-cleanup-all.jsonl
 18. Fill in the Validation template below with the validator's
     output: 5/5 summary, per-cohort matrix, force-close
     diagnostic, BC diagnostic, invariant spot-check.
 19. Restore config.yaml floors to 0 / 0 so non-probe runs don't
     inherit the cohort settings.
 20. Decision tree (purpose.md):
     - All 5 pass → open scale-run plan.
     - C1 pass, C4 fail → open observation-space-audit.
     - C4 pass, C1 fail → open controller-arch plan (PI / Adam).
     - 1–4 all fail → open observation-space-audit.
     - C5 fail → rollback; do NOT ship.
```

---

## Session 02 — 2026-04-21

**Commit:** _(pending — see git log)_ `feat(env): shaped-penalty warmup across first N episodes (default disabled)`

**What landed:**

- `env/betfair_env.py`:
  - `BetfairEnv.__init__` reads
    `training.shaped_penalty_warmup_eps` (default 0 = disabled =
    byte-identical to pre-change). Initialises
    `self._episode_idx = 0`.
  - New `set_episode_idx(idx)` method. Called by `PPOTrainer`
    before each rollout; BC pretrain episodes do NOT call it (BC
    doesn't increment `_eps_since_bc`), so the warmup index
    counts PPO rollouts only per hard_constraints.md §21.
  - `_settle_current_race` computes
    `warmup_scale = min(1.0, episode_idx / warmup_eps)` when
    `warmup_eps > 0`, else 1.0. Applies the scale to
    `efficiency_cost` and `precision_reward` ONLY — all other
    shaping terms (early_pick, drawdown, spread, inactivity,
    naked_penalty, early_lock, matured_arb, MTM, naked-clip and
    close_signal bonus from `_compute_scalping_reward_terms`)
    stay at full strength.
  - `reset` initialises `self._shaped_penalty_warmup_scale_last`
    so `_get_info` reads a defined value before the first
    settle.
  - `_get_info` exposes `shaped_penalty_warmup_scale` (last
    settle's applied scalar) and `shaped_penalty_warmup_eps`
    (plan-level length).
- `agents/ppo_trainer.py`:
  - `_collect_rollout` calls `env.set_episode_idx(self._eps_
    since_bc)` before `env.reset()`. A `hasattr` guard tolerates
    scripted test envs that monkey-patch `BetfairEnv` without
    implementing the setter.
  - `EpisodeStats` gains `shaped_penalty_warmup_scale` (default
    1.0) and `shaped_penalty_warmup_eps` (default 0).
  - `_log_episode` writes both fields into `episodes.jsonl`.
    Pre-change rows lack them; downstream readers must tolerate
    absence (same backward-compat pattern as `mtm_weight_active`
    / `alpha`).
- `config.yaml`: `training.shaped_penalty_warmup_eps: 0` (default
  disabled). Plan files enable this on the probe cohorts via
  overrides.
- `tests/arb_signal_cleanup/test_shaped_penalty_warmup.py`
  (32 tests across 6 categories, all passing):
  - Default byte-identical (absent key ≡ explicit 0; episode_idx
    ignored when disabled).
  - Linear ramp parametrised over
    `episode_idx ∈ {0, 5, 9, 10, 20}` with expected scales
    `{0.0, 0.5, 0.9, 1.0, 1.0}`.
  - Only-two-terms-scaled: linear interpolation of shaped across
    `scale ∈ {0.0, 0.5, 1.0}`; non-scaled drawdown term survives
    through scale=0.
  - JSONL field present (info dict carries both keys; trainer
    log row carries both keys).
  - No cliff at warmup+1 (constant delta across ramp, zero delta
    after, no spike at the boundary).
  - Invariant `raw + shaped ≈ total` parametrised across
    `warmup_eps ∈ {0, 5}` × `episode_idx ∈ {0, 2, 4, 5, 10}` ×
    `force_close_before_off_seconds ∈ {0, 30}` (Session 01 axis
    stacks cleanly).
- `CLAUDE.md`: new `### Shaped-penalty warmup (2026-04-21)`
  subsection under *Reward function: raw vs shaped*, after the
  BC-pretrain entry.

**Not changed:** matcher semantics, action/obs schemas, PPO
stability defences (advantage normalisation / KL early-stop /
reward centering / LR warmup), target-entropy controller
structure or LR, BC pretrain semantics, matured-arb bonus
formula, naked-loss annealing schedule, curriculum day ordering,
MTM weight, transformer architecture. `training/worker.py`
stays out of this commit (only Session 01's pre-plan fix lives
there).

**Gotchas:**

- Scripted test envs that monkey-patch `BetfairEnv` with their
  own `_ScriptedEnv` class (e.g.
  `tests/arb_improvements/test_reward_clipping.py`) don't
  implement `set_episode_idx`. The trainer wraps the call in a
  `hasattr` guard so those tests stay green without forcing
  every stub to grow the method.
- `_eps_since_bc` is incremented AFTER each `_collect_rollout`
  completes (ppo_trainer.py:935 equivalent). So at the top of
  the ep1 rollout it reads 0, the ep2 rollout 1, etc. This
  matches the 0-based warmup index contract in
  hard_constraints.md §21. Agents without BC still work: the
  counter initialises to 0 and counts PPO episodes identically.
- The `warmup_scale_last` stored on the env is OVERWRITTEN on
  every `_settle_current_race` call but never varies within a
  single episode (episode_idx is fixed for the rollout). The
  telemetry field is therefore well-defined regardless of
  how many races a day contains.
- `precision_reward` is zeroed unconditionally in scalping mode
  (Issue 05 hard constraint — planned-loss leg makes a
  directional precision metric nonsense). So in the scalping
  cohorts Session 02's warmup effectively scales
  `efficiency_cost` only; the code path for
  `precision_reward` stays intact for directional runs and for
  future ablations. Test 3's scalping-mode linear-interpolation
  check is still diagnostic because no non-scaled term depends
  on `warmup_scale`.

**Test suite:** `pytest tests/ -q --timeout=120
--ignore=tests/test_e2e_training.py` → **2440 passed, 7 skipped,
1 xfailed** in 5m 30s. No training was active during the
full-suite run. Full suite confirms no regressions in
`test_ppo_trainer`, `test_mark_to_market`, `arb_curriculum`,
`arb_improvements` (the scripted-env pattern is now exercised
via the `hasattr` guard), `arch_exploration`, `test_config`.

**Next:** Session 03 — plan draft + validator + launch (operator
gated). See
[`session_prompts/03_plan_draft_validator_launch.md`](session_prompts/03_plan_draft_validator_launch.md).

---

## Session 01 — 2026-04-21

**Commit:** `3e5c201 feat(env+arch): force-close before off + per-agent alpha_lr gene + transformer ctx 256 option`

**What landed:**

- `env/bet_manager.py::Bet.force_close: bool = False` attribute. Set
  alongside `close_leg=True` at placement for env-initiated closes;
  distinguishes them from agent-initiated `close_signal` closes at
  settlement.
- `env/betfair_env.py`:
  - `BetfairEnv.__init__` reads
    `constraints.force_close_before_off_seconds` (default 0 =
    disabled = byte-identical to pre-change).
  - New `_force_close_open_pairs(race, tick, time_to_off)` helper
    that iterates pairs with one leg matched + opposite leg still
    open (or passive-only in book), dispatching each to
    `_attempt_close(force_close=True)`.
  - Step loop triggers force-close AFTER `passive_book.on_tick` and
    BEFORE `_process_action` whenever
    `scalping_mode && threshold > 0 && !tick.in_play && 0 <=
    time_to_off <= threshold`.
  - `_attempt_close` gains keyword arg `force_close: bool = False`
    that propagates to the placed `Bet.force_close` attribute.
  - `_settle_current_race` classification now has an
    `is_force_closed` branch that increments a new
    `scalping_arbs_force_closed` counter (distinct from
    `scalping_arbs_closed`) and routes the pair's cash P&L into a
    new `scalping_force_closed_pnl` bucket.
  - `race_pnl` formula: `locked + closed + force_closed +
    scaled_naked`. `naked_pnl` on `RaceRecord` is the residual after
    subtracting all three above.
  - `close_events` entries gain `force_close: bool` field.
  - `RaceRecord` gains `arbs_force_closed: int`,
    `force_closed_pnl: float`.
  - `_get_info` exposes `arbs_force_closed`,
    `scalping_force_closed_pnl`, `force_close_before_off_seconds`.
- `agents/ppo_trainer.py`:
  - `PPOTrainer.__init__` now records `self._alpha_lr` from
    `hp.get("alpha_lr", 1e-2)` and passes it to the SGD optimiser
    (the `hp.get` was already there; this session formalises the
    plumbing by surfacing the value as an attribute so downstream
    telemetry can read it). Matured-arb bonus comment updated to
    note force-close exclusion.
  - `EpisodeStats` gains `arbs_force_closed`,
    `scalping_force_closed_pnl`, `force_close_before_off_seconds`,
    `alpha_lr_active`.
  - `_log_episode` writes all four new fields into
    `episodes.jsonl`. Pre-change rows lack them; downstream readers
    must tolerate absence (same backward-compat pattern as
    `mtm_weight_active` / `alpha`).
- `agents/policy_network.py::PPOTransformerPolicy`: class docstring
  widened from `{32, 64, 128}` → `{32, 64, 128, 256}`. No
  architectural change — `position_embedding` and `causal_mask`
  already size off the gene value.
- `config.yaml`:
  - `constraints.force_close_before_off_seconds: 0` (default).
  - `hyperparameters.search_ranges.transformer_ctx_ticks.choices`
    now includes 256.
  - `hyperparameters.search_ranges.alpha_lr` added as a
    `float_log` spec over `[1e-2, 1e-1]` (documents the gene; runs
    without it in their schema stay byte-identical to pre-change
    since PPOTrainer defaults to 1e-2).
- `tests/arb_signal_cleanup/test_force_close.py` (16 tests):
  threshold fire gate (above/at); matcher used (priceable/
  unpriceable); hard price cap; race_pnl bucket sum; matured-arb
  bonus excludes force-closes; close_signal bonus excludes force-
  closes; alpha_lr gene passthrough, default-unchanged, and no
  mutation across controller steps; invariant raw+shaped≈total
  parametrised over `force_close_before_off_seconds ∈ {0, 30}` and
  `alpha_lr ∈ {1e-2, 5e-2}`; transformer builds and forwards at
  ctx=256.
- `tests/arch_exploration/test_transformer_arch.py`: sampler
  enumeration test updated to include 256 (strictly additive —
  32/64/128 still asserted).
- `CLAUDE.md`:
  - `### Force-close at T−N (2026-04-21)` under *Bet accounting*.
  - `### alpha_lr as per-agent gene (2026-04-21)` under *Entropy
    control*.
  - New top-level `## Transformer context window — 256 available
    (2026-04-21)` (no prior architecture-choices section existed).

**Not changed:** matcher semantics, action/obs schemas,
controller structure (SGD / momentum 0 / log_alpha clamp / target
150 / BC handshake), BC pretrain code, matured-arb bonus formula
(only the exclusion wording around force-close), naked-loss
annealing, curriculum day ordering, MTM weight, advantage
normalisation, reward centering, LR warmup. `training/worker.py`
was modified in a prior pre-plan commit and stays out of this
commit.

**Gotchas:**

- `force_close=True` implies `close_leg=True`; tests and settlement
  rely on this invariant. `_attempt_close` sets both atomically at
  placement, so the flags can never be out of sync.
- Force-close runs at step start (BEFORE `_process_action`) so its
  effects are visible to downstream accounting identically to an
  agent-initiated close on the SAME tick. A race where force-close
  fires at T−29 AND the action loop would have fired close_signal
  on the same tick would see only the force-close leg in bets — the
  action loop's `_attempt_close` then finds no outstanding passive
  and silent-no-ops.
- `naked_pnl` on `RaceRecord` is the RESIDUAL after subtracting
  `locked + closed + force_closed` from `race_pnl`. For races with
  agent-initiated closes, the `closed` term isn't on `RaceRecord` as
  a dedicated field — it's implicit in the residual arithmetic. The
  invariant `rr.pnl == rr.locked_pnl + rr.force_closed_pnl +
  closed_internal + rr.naked_pnl` holds by construction.
- The force-close step trigger gate uses `time_to_off <= threshold`
  (inclusive). A tick exactly AT the threshold fires force-close; a
  tick one second above does not. Tests assert both boundaries.

**Test suite:** `pytest tests/ -q --timeout=120
--ignore=tests/test_e2e_training.py` → **2408 passed, 7 skipped, 1
xfailed** in 5m 29s. No training was active during the full-suite
run (git status at start of session showed only pre-plan file
changes). Full suite confirms no regressions in
test_ppo_trainer, test_forced_arbitrage, test_close_signal,
test_mark_to_market, arb_curriculum, arch_exploration,
arb_improvements, test_config.

**Next:** Session 02 — shaped-penalty warmup. See
[`session_prompts/02_shaped_penalty_warmup.md`](session_prompts/02_shaped_penalty_warmup.md).
Session 02's invariant test stacks on top of the parametrisation
added here; if any regression surfaces before Session 02 starts, it
should be fixed in a follow-up commit before the next session
lands.

---

## Pre-plan setup — 2026-04-21

**What landed:**

- `plans/arb-signal-cleanup/` scaffolded (purpose,
  hard_constraints, master_todo, session_prompt,
  lessons_learnt, progress, session_prompts/).
- `training/worker.py::_check_dead_thread` race fix
  (two-poll grace period before marking plan failed).
  Commit: see git log. This fix is prerequisite for this
  plan — without it the next probe would hit the same
  phantom-failure at the gen_N→gen_{N+1} boundary.

**Not changed:** no env / trainer / config / test
changes in the pre-plan setup. Session 01 starts from a
clean baseline.

**Gotchas:**

- The 277bbf49 `arb-curriculum-probe` plan stays in the
  registry with `status = failed`. Do NOT try to resume
  it — we're abandoning it and running a fresh probe
  with the three new mechanisms. The failed-status entry
  is deliberately left in place so the admin UI shows
  the historical record.
- Before Session 01, verify the operator's intent on the
  cohort-split mechanism (single plan with cohort-level
  overrides, vs three plan files run serially). Session
  03 prompt includes a verification step; if the data
  model supports cohort overrides, prefer that; else
  fall back to three plans.

**Test suite:** no changes pending session 01.

**Next:** Session 01 — force-close at T−N + entropy-
velocity gene. See
[`session_prompts/01_force_close_and_entropy_velocity.md`](session_prompts/01_force_close_and_entropy_velocity.md).

---

## Validation — arb-signal-cleanup-probe

_Fill in after running
`python scripts/validate_arb_signal_cleanup.py` on the
completed probe run. Replace every `[FILL IN]` marker._

**Run date:** [FILL IN — YYYY-MM-DD]
**Probe plan ID(s):** [FILL IN — single or three IDs]
**Agents:** 48 (pop) × 4 gens × 3 epochs (or 3 × 16 if
three-plan fallback)
**Seed:** 8101
**curriculum_day_order:** density_desc
**force_close_before_off_seconds:** 30 (cohorts A, C) / 0
(cohort B)
**shaped_penalty_warmup_eps:** 10 (cohorts A, C) / 0
(cohort B)
**alpha_lr gene range:** [1e-2, 1e-1] (cohorts A, B) /
pinned 1e-2 (cohort C)
**Total episodes logged:** [FILL IN]

**Criteria results (population-wide):**

| # | Criterion | Pass? | Detail |
|---|---|---|---|
| C1 | ≥80 % active at ep15 | [PASS/FAIL] | [X/Y agents, Z %] |
| C2 | `arbs_closed/total > 15 %` on ≥ 3 agents at ep15 | [PASS/FAIL] | [N qualifying agents] |
| C3 | `policy_loss ≥ 0.1` on ≥ 50 % agents at ep15 | [PASS/FAIL] | [X/Y agents, Z %] |
| C4 | ≥ 3 agents reach `total_reward > 0` | [PASS/FAIL] | [N agents, top-3 peaks] |
| C5 | raw+shaped ≈ total (invariant) | [PASS/FAIL] | [0 violations / N checked] |

**Summary score:** [X/5 criteria pass]

**Per-cohort pass/fail matrix:**

| # | Criterion | Cohort A (all 3) | Cohort B (entropy only) | Cohort C (warmup + force-close) |
|---|---|---|---|---|
| C1 | ≥ 80 % active | [FILL] | [FILL] | [FILL] |
| C4 | ≥ 3 agents positive total_reward | [FILL] | [FILL] | [FILL] |

**Load-bearing mechanism attribution:**

- If A passes both and B/C pass neither → all three
  mechanisms are load-bearing.
- If A and B pass C1, A and C pass C4 → entropy fix →
  C1; warmup+force-close → C4 (independent failures).
- If A passes both and B also passes both → entropy is
  sufficient; warmup + force-close are optional.
- [FILL IN OBSERVED ATTRIBUTION]

**Force-close diagnostic (per-cohort means):**

| Cohort | arbs_force_closed / race | arbs_naked / race | scalping_force_closed_pnl / race |
|---|---|---|---|
| A | [FILL] | [FILL] | [FILL] |
| B | 0.0 | [FILL] | 0.0 |
| C | [FILL] | [FILL] | [FILL] |

(Cohort B has force-close disabled by design; the row is
the control.)

**BC diagnostics (from validator output):**

| Cohort | mean `bc_final_signal_loss` | mean `bc_final_arb_spread_loss` |
|---|---|---|
| A | [FILL] | [FILL] |
| B | [FILL] | [FILL] |
| C | [FILL] | [FILL] |

**Invariant spot-check:** [FILL IN — N random rows
checked, all within tolerance / any violations listed]

**Representative agent trajectories:** [FILL IN — at
least one per cohort × architecture quadrant that
completed the run; one-line summary of each
(arch, cohort, bets@ep15, cumulative_pnl,
peak_total_reward)]

**Decision:**

- [ ] C1–C5 all pass → scale-run green light, proceed to
  follow-on plan.
- [ ] C1 passes, C4 fails → open
  `observation-space-audit`.
- [ ] C4 passes, C1 fails → open controller-arch plan.
- [ ] 1–4 all fail → diagnosis was wrong; open
  `observation-space-audit` anyway.
- [ ] C5 fails → rollback, fix accounting, re-run. Never
  ship broken.
