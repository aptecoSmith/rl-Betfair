# Master TODO — Arb Improvements

Work is organised into sessions. Each session has its own
`session_N_name.md` prompt file. Sessions are ordered so **each one
delivers a usable intermediate result** — stabilisation before
features, features before behaviour cloning, BC before aux head,
verification last.

Tick items off as they land. When a session completes, update
`progress.md` with what shipped and record anything surprising in
`lessons_learnt.md`. UI work discovered during a session goes into
`ui_additions.md`, which Session 8 consolidates.

---

## Phase 1 — Stop the collapse

Goal: by end of Phase 1, the `90fcb25f` failure mode no longer
reproduces. No epoch-1 loss > 10⁷, no policy collapse into
"don't bet" after the first outlier race.

- [x] **Session 1 — Reward & advantage clipping**
  (`session_1_reward_clipping.md`)
  - Add `reward.reward_clip` (training-signal only, unclipped raw
    reward still logged), `training.advantage_clip`, and
    `training.value_loss_clip`. All three default to 0 (off).
  - `raw + shaped ≈ total_reward` invariant still holds.
  - CPU-only tests: clipping caps training signal, leaves telemetry
    untouched, all defaults off = byte-identical rollout.

- [ ] **Session 2 — Entropy floor & per-head entropy logging**
  (`session_2_entropy_floor.md`)
  - Add `training.entropy_floor` (default 0 = off). When rolling
    mean policy entropy drops below the floor, scale
    `entropy_coefficient` up until recovered.
  - Per-head entropy (signal, stake, aggression, cancel, arb_spread)
    logged to the training monitor progress events.
  - CPU-only tests: adaptive scaling triggers and releases correctly,
    no effect when off.

- [ ] **Session 3 — Signal-bias warmup & bet-rate diagnostics**
  (`session_3_signal_bias_warmup.md`)
  - Add `training.signal_bias_warmup` (epochs) and
    `training.signal_bias_magnitude`. During warmup, add a linearly
    decaying positive constant to the `signal` action mean.
  - Training monitor emits `action_stats`: `bet_rate`, `arb_rate`,
    per-head `mean_entropy`, `bias_active`.
  - UI work added to `ui_additions.md`: bet-rate sparkline in the
    training monitor.
  - CPU-only tests: bias decays to zero, no effect after warmup.

## Phase 2 — Make arbs perceivable

Goal: the policy has explicit features that say "there is an arb
here right now, worth X ticks / Y% after commission". These are
useful to directional models too, so they ship unconditionally.

- [ ] **Session 4 — Pure arb feature functions**
  (`session_4_arb_feature_functions.md`)
  - Add to `env/features.py` (stdlib only, duck-typed):
    - `compute_arb_lock_profit_pct(back_levels, lay_levels, ltp,
      commission_rate) -> float`
    - `compute_arb_spread_ticks(back_levels, lay_levels, ltp,
      max_ticks) -> float`
    - `compute_arb_fill_time_norm(passive_size, traded_delta,
      max_norm) -> float`
    - `compute_arb_opportunity_density(history, window_s, now_ts)
      -> float`
  - Commission constant imported from the same place
    `BetManager.get_paired_positions` uses.
  - CPU-only tests: uncrossed book → 0, crossed book → hand-computed
    value, unpriceable runner → 0 not NaN, ladder-band transitions
    for spread-ticks.

- [ ] **Session 5 — Wire features into env + schema bump**
  (`session_5_arb_features_wiring.md`)
  - Extend `RUNNER_KEYS` with 3 new keys; extend `MARKET_KEYS` with
    `arb_opportunity_density_60s`.
  - Bump `OBS_SCHEMA_VERSION`.
  - Mirror the new keys in `data/feature_engineer.py` — maintain
    the "keys produced match env exactly" invariant
    (`env/betfair_env.py:97`).
  - CPU-only tests: obs vector shape matches expected dim, schema
    bump refuses pre-bump checkpoints, feature values equal the
    pure-function output for a known synthetic tick.

## Phase 3 — Oracle scan + behaviour-cloning warm start

Goal: every new agent has already seen "if this feature pattern
fires, place an arb" before PPO begins, using the real training-day
data as ground truth.

- [ ] **Session 6 — Arb oracle scan**
  (`session_6_oracle_scan.md`)
  - New module `training/arb_oracle.py` with `scan_day(date,
    data_dir)` and a CLI entrypoint.
  - Output: one deterministic `.npz` per date under
    `data/oracle_cache/{date}/oracle_samples.npz`.
  - Filters match `ExchangeMatcher` — an oracle sample the env
    would reject is not emitted.
  - CPU-only tests: synthetic day with one known arb moment
    produces exactly one sample; env-rejected moments excluded;
    empty day → empty dataset, no crash.

- [ ] **Session 7 — BC pretrainer + trainer integration**
  (`session_7_bc_pretrainer.md`)
  - New module `agents/bc_pretrainer.py`: cross-entropy on `signal`,
    MSE on `arb_spread`; other heads untouched.
  - `training.bc_pretrain_steps` (int, default 0 = off) gene added.
  - In `training/worker.py`: when scalping_mode and
    `bc_pretrain_steps > 0`, each agent pretrains on its own oracle
    samples before PPO.
  - Per-agent pretrain — diversity preserved. Empty oracle →
    skip BC with warning.
  - CPU-only tests: pretrain reduces loss on synthetic oracle,
    policy prefers oracle action after training, empty dataset
    skipped cleanly.

- [ ] **Session 8 — Wizard UI, evaluator, and UI consolidation**
  (`session_8_bc_ui_evaluator.md`)
  - Wizard: "BC pretrain steps" input on the scalping step. Help
    text explaining recommended range (500–2000).
  - Evaluator: record `bc_pretrain_steps` per model; warn if oracle
    cache is stale relative to the training-day episode file.
  - Consolidate every UI task accumulated in `ui_additions.md`
    across sessions 1–7 into the training monitor frontend.
  - Frontend `ng build` clean; CPU-only tests on backend glue;
    manual verification checklist for the wizard.

## Phase 4 — Optional auxiliary head

Goal: if BC pretraining alone doesn't stabilise the representation
through PPO updates, add a supervised head that keeps the trunk
arb-aware.

- [ ] **Session 9 — Auxiliary arb-availability head (gated)**
  (`session_9_aux_head.md`)
  - Add `training.aux_arb_head` (bool, default False).
  - When on, shared-trunk MLP head predicts "arb lockable at
    runner R within next K ticks" per-runner binary. Target
    computed from the oracle scan.
  - Loss added to PPO total with configurable weight.
  - CPU-only tests: forward pass shape, gradient flow to trunk,
    default off → no change to forward pass / memory.

## Phase 5 — Head-to-head verification

Goal: prove the stack works on the same config that originally
failed.

- [ ] **Session 10 — Full verification run vs 90fcb25f baseline**
  (`session_10_verification.md`)
  - Re-run the `90fcb25f` gene / arch / date config with all
    Phase 1–3 knobs on (aux head only if it shipped).
  - Compare against the baseline table in `progress.md`:
    arb_rate per episode, mean reward progression, mean
    locked_pnl.
  - Write findings to `progress.md`. Any surprise → entry in
    `lessons_learnt.md`.
  - This is the only session where a full GPU training run is
    expected. All prior sessions are CPU-only (see `testing.md`).
