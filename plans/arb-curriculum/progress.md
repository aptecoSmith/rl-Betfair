# Progress — Arb Curriculum

One entry per completed session. Most recent at the top.
Include commit hash, what landed, what's not changed, and
any gotchas.

Format per session follows
`plans/reward-densification/progress.md` — "What landed",
"Not changed", "Gotchas", "Test suite", "Next".

---

## Session 06 — 2026-04-20

**Commit:** (see git log)

**What landed (tracked files — registry is gitignored):**

- `plans/arb-curriculum/progress.md` — this entry.

**What landed on disk (gitignored, not in diff):**

- `registry/training_plans/277bbf49-8a2b-4d84-b8a3-3b9286e115eb.json` —
  `arb-curriculum-probe` plan (status=draft, seed=7919, pop=33, gens=4,
  naked_loss_anneal={start_gen:0, end_gen:2}).
  Written to `registry/training_plans/` in the MAIN REPO (the worktree
  does not share gitignored files with master).

**What operator must do BEFORE Session 07 (archive + reset):**

1. Confirm no active training: `tasklist | findstr python.exe` (Windows)
2. Run `pytest tests/ -q` on the latest commit — must be green.
3. Archive the current registry:
   ```powershell
   $ISO = (Get-Date -Format "yyyyMMddTHHmmssZ")
   New-Item -ItemType Directory -Path "registry/archive_$ISO"
   Move-Item registry/models.db "registry/archive_$ISO/"
   Move-Item registry/weights "registry/archive_$ISO/"
   Copy-Item -Recurse registry/training_plans "registry/archive_$ISO/"
   Move-Item logs/training/episodes.jsonl "logs/training/episodes.pre-arb-curriculum-$ISO.jsonl"
   ```
4. Initialize fresh registry:
   ```python
   from registry.model_store import ModelStore
   ModelStore()   # creates a fresh models.db with 0 models
   ```
   ```powershell
   New-Item -ItemType Directory -Force -Path registry/weights
   New-Item -ItemType File -Force -Path logs/training/episodes.jsonl
   ```
5. Set `config.yaml training.curriculum_day_order: density_desc` before
   launching (the probe plan JSON stores `training_overrides.curriculum_day_order`
   as documentation metadata, but the current code reads it from config.yaml —
   see Gotchas below).
6. Run oracle scan on the current training-date window if not already done:
   `python -m training.arb_oracle scan --dates <dates>` — BC pretrainer and
   curriculum ordering depend on `data/oracle_cache/` being populated.

**Probe plan details:**

- Plan ID: `277bbf49-8a2b-4d84-b8a3-3b9286e115eb`
- Name: `arb-curriculum-probe`
- Population: 33 (11 per arch × 3 archs)
- Generations: 4, epochs: 3, auto_continue: true
- Seed: 7919 (differs from gene-sweep seed 1337)
- `naked_loss_anneal`: `{start_gen: 0, end_gen: 2}` — naked-loss-scale
  anneals from each agent's gene value to 1.0 across gen 0-1; full scale
  from gen 2 onward.
- hp_ranges: gene-sweep ranges + 5 new arb-curriculum genes:
  - `matured_arb_bonus_weight` [0.0, 2.0]
  - `naked_loss_scale` [0.05, 1.0]
  - `bc_pretrain_steps` [0, 1500]
  - `bc_learning_rate` [1e-5, 1e-3]
  - `bc_target_entropy_warmup_eps` [2, 15]

**Validation (ran from worktree):**

```
probe validates OK: 277bbf49-8a2b-4d84-b8a3-3b9286e115eb
hp_ranges keys: ['arb_spread_scale', 'architecture_name', 'bc_learning_rate',
  'bc_pretrain_steps', 'bc_target_entropy_warmup_eps', 'early_lock_bonus_weight',
  'entropy_coefficient', 'entropy_floor', 'fill_prob_loss_weight',
  'inactivity_penalty', 'mark_to_market_weight', 'market_type_filter',
  'matured_arb_bonus_weight', 'naked_loss_scale', 'naked_penalty_weight',
  'reward_clip', 'reward_spread_cost_weight', 'risk_loss_weight']
```

**Not changed:** no code or test changes in this session (hard_constraints §35).

**Gotchas:**
- `training_overrides.curriculum_day_order: density_desc` is stored in the
  plan JSON as documentation metadata. `TrainingPlan.from_dict` ignores unknown
  keys, so this field survives the round-trip as JSON but is NOT applied by
  the current code. Operator must set `config.yaml training.curriculum_day_order:
  density_desc` before launching. A follow-on session can wire up
  `training_overrides` to `TrainingOrchestrator.__init__` alongside the
  existing `reward_overrides` pattern.
- The probe plan JSON lives in the MAIN REPO's `registry/training_plans/`
  (created before the archive step). Since the archive step uses `cp -r`
  (not `mv`) for training_plans, the plan survives the archive intact.
- After the registry reset, `registry/models.db` will have 0 models and
  the UI will show an empty population. This is expected — the probe launches
  a fresh Gen 0.
- Worktree commits (Sessions 01–05 of this plan) must be merged to master
  BEFORE the operator runs the archive+reset. The worktree branch is
  `claude/romantic-hodgkin-b2897d`.

**Test suite:** no new tests in this session. Full suite still green on
  Session 05 commit.

**Next:** Operator: (1) merge branch to master, (2) run archive+reset steps
  above, (3) set config.yaml curriculum_day_order: density_desc, (4) run
  oracle scan on training dates, (5) launch arb-curriculum-probe with
  smoke test first. Session 07 writes the validation entry.

---

## Session 05 — 2026-04-20

**Commit:** (see git log)

**What landed:**

- `training/arb_oracle.py` — 2 new public functions:
  - `density_for_date(date, data_dir)`: reads only `data_dir/{date}/header.json`
    (no .npz load); returns `float(data["density"])` or `0.0` on any error or
    missing cache. `data_dir` is the oracle_cache dir directly (e.g.
    `Path("data/oracle_cache")`), NOT the processed-data dir.
  - `order_days_by_density(dates, mode, data_dir, rng)`: reorders dates per
    mode (`random` | `density_desc` | `density_asc`); invalid mode logs
    `logger.error` and falls back to random. Membership always preserved.
    `random` mode uses `rng.sample(dates, len(dates))`.
  - Added `import random` at module level.
- `agents/ppo_trainer.py` — 3 changes:
  - `EpisodeStats`: added `curriculum_day_order: str = "random"`.
  - `_collect_rollout`: populates `curriculum_day_order` from
    `self.config.get("training", {}).get("curriculum_day_order", "random")`.
  - `_log_episode`: always emits `"curriculum_day_order": ep.curriculum_day_order`
    to JSONL row.
- `training/run_training.py` — 3 changes:
  - Added `import random` to module imports.
  - Updated `from training.arb_oracle import load_samples` →
    `from training.arb_oracle import load_samples, order_days_by_density`.
  - Per-agent loop: before `trainer.train(...)`, reorders `train_days` via
    `order_days_by_density` with a `random.Random(hash(agent.model_id) & 0xFFFFFFFF)`
    seed; passes `_ordered_train_days` to trainer instead of raw `train_days`.
    Logs the first 5 dates of the final order.
- `config.yaml`: added `curriculum_day_order: random` under `training:` with
  doc comment explaining the three modes and missing-cache behaviour.
- `CLAUDE.md`: new "Curriculum day ordering (2026-04-19)" subsection after the
  BC-pretrain warmup handshake section.
- `tests/arb_curriculum/test_curriculum_ordering.py`: 9 tests across 7 categories
  (parametrize expands membership test to 3). All 9 pass.

**Not changed:** oracle scan semantics, BC pretrainer, reward path, matcher,
  controller, action/obs schemas.

**Gotchas:**
- `density_for_date(date, data_dir)` takes the oracle_cache dir directly
  (e.g. `Path("data/oracle_cache")`), whereas `load_samples(date, data_dir)`
  takes the processed-data dir and derives `data_dir.parent / "oracle_cache" / date`.
  These two conventions differ — don't confuse them.
- `order_days_by_density` with `density_asc` places missing-cache dates
  (density=0) at the START (sparsest first); with `density_desc` they land at
  the END. Both emit a warning via `logger.warning`.
- Agent seed for per-agent rng: `hash(agent.model_id) & 0xFFFFFFFF`. UUID
  model_ids hash deterministically within a Python session but not across
  restarts — this is acceptable because curriculum order is a training
  heuristic, not a reproducibility requirement.

**Test suite:** `pytest tests/arb_curriculum/ -v` → 90 passed, 1 skipped.

**Next:** Session 06 — Registry reset + plan redraft.

---

## Session 04 — 2026-04-20

**Commit:** (see git log)

**What landed:**

- `agents/bc_pretrainer.py` — new module:
  - `BCLossHistory` dataclass (signal_losses, arb_spread_losses, total_losses,
    final_signal_loss, final_arb_spread_loss).
  - `BCPretrainer.pretrain(policy, samples, n_steps)`: freezes all non-actor_head
    params, trains signal + arb_spread dims via separate Adam, restores
    requires_grad=True. Returns BCLossHistory.
  - `measure_entropy(policy, samples)`: forward pass on up to 256 samples,
    returns mean Normal entropy; used post-BC to seed controller warmup.
  - `_is_bc_target_head(name)`: True iff `"actor_head" in name`.
  - `_sample_batch(samples, batch_size)`: sampling with replacement.
- `agents/ppo_trainer.py` — 5 changes:
  - `EpisodeStats`: added `bc_pretrain_steps: int = 0`,
    `bc_final_signal_loss: float = 0.0`, `bc_final_arb_spread_loss: float = 0.0`.
  - `__init__`: added `_bc_target_entropy_warmup_eps`, `_post_bc_entropy`,
    `_eps_since_bc`, `_bc_loss_history`, `_bc_pretrain_steps_done` fields.
  - `_effective_target_entropy()`: new method; anneals effective target from
    post-BC measured entropy up to `_target_entropy` over warmup episodes.
  - `_update_entropy_coefficient`: routes to `_effective_target_entropy()` instead
    of `self._target_entropy` so the controller respects the BC warmup.
  - `train()`: populates BC EpisodeStats fields on first post-BC episode;
    increments `_eps_since_bc` after each episode.
  - `_log_episode`: writes `bc_pretrain_steps`, `bc_final_signal_loss`,
    `bc_final_arb_spread_loss` to JSONL on first post-BC episode; logs effective
    target (not raw `_target_entropy`) in `target_entropy` field.
- `training/run_training.py` — 2 changes:
  - Imports: `BCPretrainer`, `measure_entropy` from `agents.bc_pretrainer`;
    `load_samples` from `training.arb_oracle`.
  - BC block after `PPOTrainer(...)` and before `trainer.train()`: unions oracle
    samples across all training dates, runs BC, sets trainer BC state fields.
    Skips gracefully on missing cache (FileNotFoundError) or schema mismatch
    (ValueError) with logger.warning. No-op when `bc_pretrain_steps=0` or
    `scalping_mode=False`.
- `config.yaml`: added 3 genes under `hyperparameters.search_ranges`:
  `bc_pretrain_steps` [0, 2000], `bc_learning_rate` [float_log, 1e-5,1e-3],
  `bc_target_entropy_warmup_eps` [0, 20].
- `CLAUDE.md`: new "BC pretrain (2026-04-19)" subsection under
  "Symmetry around random betting"; new "BC-pretrain warmup handshake
  (2026-04-19)" subsection under "Entropy control".
- `tests/arb_curriculum/test_bc_pretrainer.py`: 22 tests across 9 categories.
  All 22 pass.

**Not changed:** matcher, obs/action schemas, raw P&L accounting, oracle scan,
  matured-arb bonus, naked-loss annealing (all orthogonal).

**Gotchas:**
- `_is_bc_target_head` checks for `"actor_head"` (not `"signal_head"` or
  `"arb_spread_head"` — those don't exist as separate modules). All three
  policy architectures share a single `actor_head` MLP. BC targets dims
  0 (signal) and 4 (arb_spread) within that shared MLP's output; the other
  5 dims (stake, agg, cancel, requote, close) receive zero BC gradient.
- `load_samples(date, data_dir, strict=True)` looks in
  `data_dir.parent/oracle_cache/{date}/`. Pass `Path("data/processed")` to
  find cache at `data/oracle_cache/{date}/`. Tests: use `tmp_path/processed`
  so cache lands in `tmp_path/oracle_cache/`.
- `measure_entropy` uses `torch.no_grad()` — no grad pollution to the policy
  between BC and the first PPO update.
- `_effective_target_entropy` is called inside `_log_episode` to log the
  EFFECTIVE target (not raw), so operators see the warmup trajectory.

**Test suite:** `pytest tests/arb_curriculum/ -v` → 81 passed, 1 skipped.

**Next:** Session 05 — Curriculum day ordering.

---

## Session 03 — 2026-04-20

**Commit:** (see git log)

**What landed:**

- `env/betfair_env.py` — 4 changes:
  - `_REWARD_OVERRIDE_KEYS`: added `"naked_loss_scale"`.
  - `__init__`: reads `naked_loss_scale` (default 1.0) from reward config;
    clamps to [0, 1] with a logger warning on bad gene values.
  - `_compute_scalping_reward_terms`: new `naked_loss_scale` parameter (default 1.0);
    applies `race_pnl_adjusted = race_pnl - (1-scale) * loss_sum` to scale the loss
    side of naked cash flows. Winners untouched. At scale=1.0 is byte-identical.
  - `_get_info()`: emits `"naked_loss_scale_active": self._naked_loss_scale`.
- `agents/ppo_trainer.py` — 3 changes:
  - `_REWARD_GENE_MAP`: added `"naked_loss_scale": ("naked_loss_scale",)`.
  - `EpisodeStats`: added `naked_loss_scale_active: float = 1.0`.
  - `_build_episode_stats` / `_log_episode`: reads and writes the field.
- `training/arb_annealing.py` — new module: pure functions `anneal_factor` and
  `effective_naked_loss_scale` for generation-level linear interpolation of
  `naked_loss_scale` toward 1.0 over a `{start_gen, end_gen}` window.
- `training/training_plan.py`: `TrainingPlan` gains `naked_loss_anneal: dict | None = None`
  field; serialise/deserialise methods updated to round-trip it.
- `training/run_training.py`: per-agent HP dict is shallow-copied before trainer
  creation; if `training_plan.naked_loss_anneal` is set and the gene is present, the
  effective scale is computed and written into the copy before `PPOTrainer` sees it.
- `config.yaml`: added `naked_loss_scale: 1.0` under `reward:`.
- `CLAUDE.md`: new "Naked-loss annealing (2026-04-19)" subsection.
- `tests/arb_curriculum/test_naked_loss_annealing.py`: 25 tests across 8 categories.
  All 25 pass.

**Not changed:** matcher, obs/action schemas, raw P&L accounting for scale=1.0,
  oracle scan, BC pretrain plumbing (Sessions 04-05 scope), other reward knobs.

**Gotchas:**
- The annealing operates on a SHALLOW COPY of `agent.hyperparameters` so the
  original gene survives for breeding — the orchestrator never mutates the live agent.
- `_compute_scalping_reward_terms` receives pre-computed `race_pnl` which already
  includes full unscaled naked P&L. The adjustment formula
  `race_pnl - (1-scale)*loss_sum` correctly backs out the excess loss signal;
  it does NOT re-sum the naked array (which would double-count locked and closed P&L).

**Test suite:** `pytest tests/arb_curriculum/ -v` → 59 passed, 1 skipped.

**Scoreboard comparability:** Runs with `naked_loss_scale < 1.0` are NOT comparable
  to scale=1.0 runs on `raw_pnl_reward`. Document this in any plan that activates
  the gene.

**Next:** Session 04 — BC pretrainer (the big one).

---

## Session 02 — 2026-04-20

**Commit:** (see git log)

**What landed:**

- `env/betfair_env.py` — 4 changes:
  - `_REWARD_OVERRIDE_KEYS`: added `"matured_arb_bonus_weight"`.
  - `__init__`: reads `matured_arb_bonus_weight` (0.0), `matured_arb_bonus_cap` (10.0),
    `matured_arb_expected_random` (2.0) from reward config.
  - `_settle_current_race`: computes `matured_arb_term` using
    `clip(weight * (n_matured - expected_random), -cap, +cap)` where
    `n_matured = scalping_arbs_completed + scalping_arbs_closed`. Added to
    `shaped` sum. Term is 0.0 when `weight == 0.0` (no-op by construction).
  - `_get_info()`: emits `"matured_arb_bonus_active": self._matured_arb_bonus_weight`.
- `agents/ppo_trainer.py` — 3 changes:
  - `_REWARD_GENE_MAP`: added `"matured_arb_bonus_weight": ("matured_arb_bonus_weight",)`.
  - `EpisodeStats`: added `matured_arb_bonus_active: float = 0.0`.
  - `_build_episode_stats`: reads `info.get("matured_arb_bonus_active", 0.0)`.
  - `_log_episode`: writes `"matured_arb_bonus_active"` to JSONL row.
- `config.yaml`: added `matured_arb_bonus_weight: 0.0`, `matured_arb_bonus_cap: 10.0`,
  `matured_arb_expected_random: 2.0` under `reward:`.
- `CLAUDE.md`: new "Matured-arb bonus (2026-04-19)" subsection under
  "Symmetry around random betting".
- `tests/arb_curriculum/test_matured_arb_bonus.py`: 15 tests across 8 categories.
  All 15 pass.

**Not changed:** matcher behaviour, env schemas, raw P&L accounting, oracle scan,
  PPO stability defences, other reward knobs, action/obs schemas.

**Gotchas:**
- The env's `action_space` is a flat Box of shape `(max_runners × actions_per_runner,)`;
  tests that step the env must pass a float32 array, not scalar 0. Use
  `np.zeros(env.action_space.shape, dtype=np.float32)` as the noop action.
- `BetfairEnv.__init__` hard-requires `early_pick_bonus_min`, `early_pick_bonus_max`,
  `early_pick_min_seconds`, and `efficiency_penalty` in the reward config dict.
  Minimal test configs must include these four keys.

**Test suite:** `pytest tests/arb_curriculum/ -v` → 34 passed, 1 skipped.

**Next:** Session 03 — Naked-loss annealing.

---

## Session 01 — 2026-04-20

**Commit:** (see git log)

**What landed:**

- `env/exchange_matcher.py` — exported two pure functions:
  `passes_junk_filter(price, reference_price, max_dev_pct) -> bool` and
  `passes_price_cap(price, max_price) -> bool`. Used internally by the
  oracle. Existing class behaviour and all 33 matcher tests unchanged.
- `training/arb_oracle.py` — new module:
  - `OracleSample` dataclass (tick_index, runner_idx, obs, arb_spread_ticks,
    expected_locked_pnl).
  - `scan_day(date, data_dir, config) -> list[OracleSample]`: scans every
    pre-race tick via back-first arb check (junk filter → price cap →
    min_arb_ticks_for_profit → passive lay junk filter → lay price cap →
    freed-budget reservation → locked_pnl > 0). Uses a BetfairEnv for
    static obs (scalping obs v6). Samples sorted by (tick_index, runner_idx)
    for determinism.
  - `save_samples(...)`: writes `data/oracle_cache/{date}/oracle_samples.npz`
    + `header.json`. Atomic write (temp rename). Also writes
    `unique_arb_ticks_density` in the header.
  - `load_samples(date, data_dir, strict=True)`: loads cache; hard error
    on obs/action schema version mismatch (§9).
  - CLI: `python -m training.arb_oracle scan --date D [--dates D1,D2,...]`.
    Prints `samples=X ticks=Y density=X/Y unique_arb_ticks=A unique_arb_density=B`.
- `data/oracle_cache/` — added to `.gitignore`.
- `tests/arb_curriculum/test_arb_oracle.py` — 19 tests across the 8
  required categories (§27 of hard_constraints). 19 pass, 1 skipped
  (real-data obs-dim test skips cleanly when no processed data present).

**Not changed:** matcher behaviour, env schemas, reward path, PPO,
  controller, BetfairEnv observation_space, any training loop code.

**Gotchas:**
- `np.savez(path, ...)` appends `.npz` automatically. The temp file must
  be named with a `_tmp` stem, not a `.tmp` suffix — `with_suffix(".npz")`
  replaces the suffix, so `oracle_samples.tmp` → `.with_suffix(".npz")`
  → `oracle_samples.npz`, not `oracle_samples.tmp.npz`. Fixed by naming
  the stem `oracle_samples_tmp` and the temp file `oracle_samples_tmp.npz`.
- At price ~5.0 with 5% commission, `min_arb_ticks_for_profit` returns 9
  ticks (need to get from 5.0 to ~4.1). Tests at typical horse prices
  naturally produce samples without needing a crossed book.
- Pre-existing test failure: `test_session_4_9.py::TestStartEndpoint::
  test_start_returns_run_config` — confirmed pre-existing, unrelated.

**Test suite:** `pytest tests/arb_curriculum/ -v` → 19 passed, 1 skipped.
  Full suite → 1875 passed, 67 skipped, 1 pre-existing fail (session 4.9).

**Per-day densities (CLI output pending actual data run):**
  Oracle CLI not yet run against the training-date window because the
  training run may be active. Operator to run:
  `python -m training.arb_oracle scan --dates <dates>` after confirming
  no active training. Append results here. Flag any day with density < 0.001.

**Next:** Session 03 — Naked-loss annealing.

---

_Plan folder created 2026-04-19. See `purpose.md` for the
structural diagnosis flowing from the
`reward-densification-probe` 2026-04-19 failure and the
gene-sweep currently running: the policy finds "arb less"
before "arb better" because random arbing is expected-
negative. This plan attacks the local minimum via four
coordinated interventions: offline oracle scan, matured-arb
bonus, naked-loss annealing, BC pretrain + curriculum day
ordering._

_Operator observation 2026-04-19: one training day had
only 3 possible arbs across the whole day. That's bad
curriculum material at agent init — nothing for the policy
to imitate or exploit, but still ambient cost on any
random arbing. Curriculum day ordering (Session 05) uses
Session 01's oracle density to front-load arb-rich days._
