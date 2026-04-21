# Progress — Arb Signal Cleanup

One entry per completed session. Most recent at the top.
Include commit hash, what landed, what's not changed, and
any gotchas.

Format per session follows
`plans/arb-curriculum/progress.md` — "What landed",
"Not changed", "Gotchas", "Test suite", "Next".

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
