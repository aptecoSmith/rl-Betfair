# Master TODO — Arb Curriculum

Seven sessions, one commit per session, constrained by
`hard_constraints.md`. Sessions 01–05 are automatable;
Sessions 06–07 are operator-gated.

---

## Session 01 — Offline arb oracle scan + per-day density metric

**Status:** pending (blocks Session 04 + Session 05).

**Deliverables:**

- New module `training/arb_oracle.py`:
  - `OracleSample` dataclass (tick_index, runner_idx, obs,
    arb_spread_ticks, expected_locked_pnl).
  - `scan_day(date, data_dir, config) -> list[OracleSample]`.
  - `load_samples(date, data_dir) -> list[OracleSample]`.
  - CLI: `python -m training.arb_oracle scan --date D
    [--dates D1,D2,...]`.
- Export the matcher's filter predicates as pure functions
  from `env/exchange_matcher.py` if not already exposed.
  Oracle consumes them directly — no re-implementation.
- Cache layout:
  `data/oracle_cache/{date}/oracle_samples.npz` with
  header fields `obs_schema_version`,
  `action_schema_version`, `scalping_mode`, `created_at`.
  Gitignored (add to `.gitignore` if not already covered).
- Per-day density in the CLI output:
  `samples=X ticks=Y density=X/Y`.
- Tests: 8 per §27 of `hard_constraints.md`.

**Exit criteria:**

- `pytest tests/arb_curriculum/test_arb_oracle.py -x` green.
- Existing suite still green.
- CLI runs end-to-end on at least 3 of the 2026-04-19
  gene-sweep's training days; per-day densities logged in
  `progress.md`.

**Acceptance:** oracle cache exists on disk for the current
training-date window, loadable via `load_samples`, with
non-zero sample counts on high-activity days.

**Commit:** `feat(training): offline arb oracle scan with per-day density cache`.

**Session prompt:**
[`session_prompts/01_oracle_scan.md`](session_prompts/01_oracle_scan.md).

---

## Session 02 — Matured-arb bonus (knob at 0 default)

**Status:** pending.

**Deliverables:**

- `env/betfair_env.py`:
  - Read `matured_arb_bonus_weight` from reward-config;
    default 0.0. Read `matured_arb_bonus_cap` (config-
    only, not a gene; default 10.0).
  - In `_settle_current_race`: after pair enumeration,
    count `n_matured_pairs = arbs_completed + arbs_closed`
    (pairs where second leg filled). Apply zero-mean
    correction per §10: shaped contribution is
    `matured_arb_bonus_weight * (n_matured_pairs -
    expected_random_pairs)`, capped at `matured_arb_bonus_cap`.
  - `expected_random_pairs` starts as config constant
    `2.0` (documented in config.yaml) — a first-cut
    estimate; can be an EMA in a follow-on.
  - Record `matured_arb_bonus_active` in info dict and
    JSONL row.
- Whitelist `matured_arb_bonus_weight` in
  `_REWARD_OVERRIDE_KEYS` and add it to the trainer's
  `_REWARD_GENE_MAP`.
- New reward-config key
  `config.reward.matured_arb_bonus_weight: 0.0` +
  `config.reward.matured_arb_bonus_cap: 10.0`.
- Tests per §28.
- CLAUDE.md: new paragraph under "Reward function: raw vs
  shaped" → "Matured-arb bonus (2026-04-19)".

**Exit criteria:**

- `pytest tests/ -q` green; +~5 net tests.
- Invariant test parametrised over
  `matured_arb_bonus_weight ∈ {0.0, 1.0}` stays green.

**Acceptance:** with `weight=0`, rollouts byte-identical to
pre-change. With `weight=1.0`, episode reward gains
non-trivial shaped contribution on arb-completing
agents, bounded by the cap.

**Commit:** `feat(env): per-pair matured-arb shaped bonus (weight=0 default)`.

**Session prompt:**
[`session_prompts/02_matured_arb_bonus.md`](session_prompts/02_matured_arb_bonus.md).

---

## Session 03 — Naked-loss annealing knob

**Status:** pending.

**Deliverables:**

- `env/betfair_env.py`:
  - Read `naked_loss_scale` from reward-config; default
    1.0.
  - In `_settle_current_race`, wrap the per-pair naked
    loss sum with the scale (per §13).
  - Record `naked_loss_scale_active` in info + JSONL.
- Whitelist in `_REWARD_OVERRIDE_KEYS` and add to
  `_REWARD_GENE_MAP`.
- `training/plan_manager.py` (or the equivalent): support
  new plan-JSON field
  `naked_loss_anneal: {start_gen: int, end_gen: int}`.
  Expose the computed per-generation interpolation factor
  to the agent-spawning path so each agent's effective
  `naked_loss_scale` is `gene_value + (1.0 - gene_value)
  * anneal_progress` where anneal_progress ∈ [0, 1].
- Add `naked_loss_scale` gene schema entry.
- Tests per §29.
- CLAUDE.md: new paragraph → "Naked-loss annealing
  (2026-04-19)".

**Exit criteria:**

- `pytest tests/ -q` green.
- Invariant test parametrised over `naked_loss_scale ∈
  {0.5, 1.0}` stays green.
- A scripted 3-gen annealing test confirms the gene's
  effective value traces the configured curve.

**Acceptance:** with default (no annealing), runs
byte-identical. With `anneal: {0, 3}` and gene 0.1,
gen-0 effective scale is 0.1, gen-3 is 1.0, gen-1/2 are
linearly interpolated.

**Commit:** `feat(env): per-pair naked-loss scale gene + generation-level annealing`.

**Session prompt:**
[`session_prompts/03_naked_loss_annealing.md`](session_prompts/03_naked_loss_annealing.md).

---

## Session 04 — BC pretrainer + trainer integration

**Status:** pending (requires Session 01).

**Deliverables:**

- New module `agents/bc_pretrainer.py`:
  - `BCPretrainer` class with `pretrain(policy,
    oracle_samples, n_steps, lr) -> LossHistory`.
  - Cross-entropy on `signal` head (target runner's BACK
    slot), MSE on `arb_spread` head; other heads frozen
    (parameters untouched per §17).
  - Separate optimiser; PPO's Adam state untouched.
- `training/worker.py`:
  - When `scalping_mode` on AND `bc_pretrain_steps > 0`:
    - Load concatenated oracle samples for the agent's
      training dates.
    - If empty union, skip with warning (per §20).
    - Otherwise run BC, emit a `phase: "bc_warmup"`
      progress event with per-step loss curve.
    - Record `bc_final_signal_loss`,
      `bc_final_arb_spread_loss` on the first
      post-BC episode's JSONL row.
- New genes in the schema:
  - `bc_pretrain_steps: int`, range `[0, 2000]`, default 0.
  - `bc_learning_rate: float`, range `[1e-5, 1e-3]`,
    default 3e-4.
  - `bc_target_entropy_warmup_eps: int`, range `[0, 20]`,
    default 5.
- `agents/ppo_trainer.py`:
  - Accept `bc_target_entropy_warmup_eps` on init.
  - Anneal `self._target_entropy` from the first-post-BC
    measured entropy up to the config target (150) over
    the warmup window (per §18).
  - Integration test spying on `_update_reward_baseline`
    post-BC (per §30 / 2026-04-18 lesson).
- Tests per §30.
- CLAUDE.md: new "BC pretrain and target-entropy warmup"
  paragraph.

**Exit criteria:**

- `pytest tests/ -q` green; ≥ 8 new tests.
- Short integration smoke (tiny synthetic day, 3 agents,
  1 episode each) with BC on: BC phase fires, BC loss
  curve emitted, PPO runs normally afterward,
  per-step-units assertion green on post-BC update.

**Acceptance:** an agent with `bc_pretrain_steps=500` on
the real oracle cache has post-BC policy that prefers
BACK on oracle samples' target runner vs control tick
samples.

**Commit:** `feat(training): per-agent BC pretrain on arb oracle + target-entropy warmup handshake`.

**Session prompt:**
[`session_prompts/04_bc_pretrainer.md`](session_prompts/04_bc_pretrainer.md).

---

## Session 05 — Curriculum day ordering

**Status:** pending (requires Session 01).

**Deliverables:**

- `training/worker.py` (or the per-agent day-iteration
  path): consult `training.curriculum_day_order` config.
- Support values `"random"` (default),
  `"density_desc"`, `"density_asc"` per §21.
- Ordering driven by per-date density reading from
  `data/oracle_cache/{date}/oracle_samples.npz` header.
  Missing cache → density 0 (slot at end, log warning).
- Preserves "every day seen exactly once per epoch" (§22).
- Record `curriculum_day_order` in each JSONL row.
- Tests per §31.

**Exit criteria:**

- `pytest tests/ -q` green.
- A 3-day scripted test confirms each ordering mode
  produces the expected day sequence given synthetic
  density inputs.

**Acceptance:** switching
`curriculum_day_order: density_desc` produces arb-rich
days first in the episode sequence; the
`curriculum_day_order` field appears on every new JSONL
row.

**Commit:** `feat(training): opt-in curriculum day ordering driven by oracle density`.

**Session prompt:**
[`session_prompts/05_curriculum_day_ordering.md`](session_prompts/05_curriculum_day_ordering.md).

---

## Session 06 — Registry reset + training plan redraft (operator-gated)

**Status:** pending.

**Deliverables:**

- Archive the current registry into
  `registry/archive_<ISO>/`:
  - `models.db`
  - `weights/`
  - `training_plans/` (snapshot; live plans stay
    in-place)
- Move `logs/training/episodes.jsonl` to
  `logs/training/episodes.pre-arb-curriculum-<ISO>.jsonl`.
- Fresh registry:
  - New `models.db` via `ModelStore()`.
  - `weights/` recreated empty.
  - `episodes.jsonl` truncated.
- New training plan
  `registry/training_plans/arb-curriculum-probe.json`:
  - Population 33 (11 per arch).
  - 4 generations, `auto_continue: true`.
  - `reward_overrides`: leave new reward-config defaults
    apply (`matured_arb_bonus_weight` and
    `naked_loss_scale` from config.yaml as tuned in
    Sessions 02/03 plan-level updates, if any).
  - `hp_ranges` — INCLUDE:
    - `mark_to_market_weight` — carry over from
      `reward-densification-gene-sweep` range [0.0, 0.5].
    - `matured_arb_bonus_weight` — new gene; range
      [0.0, 2.0].
    - `naked_loss_scale` — new gene; range [0.05, 1.0].
    - `bc_pretrain_steps` — new gene; range [0, 1500].
    - `bc_learning_rate` — new gene; range
      [1e-5, 1e-3].
    - `bc_target_entropy_warmup_eps` — new gene; range
      [2, 15].
    - All other genes carry over from the gene-sweep
      plan (entropy_coefficient, entropy_floor,
      inactivity_penalty, naked_penalty_weight, etc.).
  - `naked_loss_anneal: {start_gen: 0, end_gen: 2}` —
    naked-loss anneals from gene value to 1.0 across the
    first two generations, full strength from gen 2.
  - `training.curriculum_day_order: "density_desc"` —
    arb-rich days first for the curriculum.
  - `seed` different from gene-sweep (pick 7919).
  - `status: "draft"`; all runtime fields null.
- `plans/INDEX.md`: new row for `arb-curriculum`.

**Exit criteria:**

- New `models.db` has 0 models.
- `episodes.jsonl` is 0 bytes.
- New plan JSON validates via
  `PlanRegistry('registry/training_plans').list()`.
- Previous gene-sweep state captured in archive folder.

**Acceptance:** operator can tick "Smoke test first,"
select `arb-curriculum-probe`, and launch.

**Commit:** `chore(registry): archive pre-arb-curriculum registry + redraft probe plan`.

**Session prompt:**
[`session_prompts/06_registry_reset_and_plan_redraft.md`](session_prompts/06_registry_reset_and_plan_redraft.md).

---

## Session 07 — Validation launch (operator-gated)

**Status:** pending.

This is a manual operator step (hard_constraints §35).
The plan folder's Session 07 prompt is instructional;
execution is operator-gated.

**Deliverables:**

- Operator launches `arb-curriculum-probe` with smoke test
  first.
- Smoke test passes (tracking-error gate at
  `target_entropy=150` — should be unaffected by the new
  work; BC warmup handles post-BC entropy).
- Full 33-agent / 4-gen run completes.
- Validation entry written to
  `plans/arb-curriculum/progress.md` mirroring the format
  of `reward-densification/progress.md`'s validation
  entries.

**Exit criteria:**

- Validation entry covers the 5 success criteria from
  `purpose.md`.
- Representative per-agent trajectories captured (for at
  least one from each architecture × each failure /
  success mode observed).
- Invariant `raw + shaped ≈ total` spot-checked on 10
  randomly-sampled episodes.jsonl rows — must hold within
  float tolerance.

**Commit:** none (validation writes back into
`progress.md` only, per §38).

**Session prompt:**
[`session_prompts/07_validation_launch.md`](session_prompts/07_validation_launch.md).

---

## After Session 07: follow-ons

If criteria 1–5 all hold: move to a 16-agent multi-day
scale run to confirm stability. Revisit the 2026-04-19
`reward-densification-gene-sweep` findings to see whether
MTM shaping now compounds with a warm-started policy.

If the population succeeds in aggregate, a follow-on
`arb-curriculum-scale` plan lifts the probe to a larger
training-date window.

If criterion 5 (invariant) fails: rollback, fix
accounting, re-test. Do not ship broken.

If 1–4 fail: next plan is `observation-space-audit` — the
policy's obs may not encode what it needs to recognise
good arbs in real time.

## Queued follow-ons (for context; not in this plan)

- **`force-close-curriculum`** — force-close all nakeds at
  T-30s before off as a curriculum parameter. Operator
  suggested this 2026-04-19 but sequenced AFTER the
  BC/curriculum bootstrap lands. Attacks the same local
  minimum from a different angle (removes the "hoped-for
  win" exit on nakeds).
- **`missed-opportunity-shaping`** — reward/penalty per
  tick based on fill_prob × arb_feasibility. Operator
  suggested 2026-04-19; zero-mean variant to avoid
  active-bleeding regression.
- **`arb-curriculum-scale`** — if the probe succeeds, lift
  to 16-agent × 10-gen × full-date-window.
- **`observation-space-audit`** — if the probe fails on
  the same passive/bleeding bifurcation despite BC,
  diagnose whether obs encodes the features the policy
  needs.
