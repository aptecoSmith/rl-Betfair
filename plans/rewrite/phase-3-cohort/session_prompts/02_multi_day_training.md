# Session prompt — Phase 3, Session 02: multi-day training

Use this prompt to open a new session in a fresh context. The prompt
is self-contained — it briefs you on the task, the design decisions
already locked, and the constraints. Do not require any context from
the session that scaffolded this prompt.

---

## The task

Extend Session 01's GPU-capable single-day train CLI to loop over N
days. Each day is one full rollout-and-update episode. Day boundaries
are episode boundaries. **No cross-day GAE bootstrapping; no shared
hidden state across days.**

End-of-session bar:

1. `python -m training_v2.discrete_ppo.train --days 7 --device cuda` runs end-to-end and writes one JSONL row per (day × episode) pair.
2. Value loss descends across the 7-day run (Phase 2 Bar 2 AMBER → GREEN).
3. Day shuffling is deterministic given a seed.

## What you need to read first

1. `plans/rewrite/phase-3-cohort/purpose.md` — multi-day contract,
   success bar 3.
2. `plans/rewrite/phase-2-trainer/findings.md` §"Bar 2 detail" —
   why one-day Bar 2 was AMBER; the multi-day argument that
   resolves it.
3. `plans/rewrite/phase-3-cohort/session_prompts/01_gpu_saturation.md`
   — Session 01's CLI shape; this session extends it.
4. `data/episode_builder.py::load_day` — single-day loader. The
   multi-day loop calls this N times.
5. `training_v2/discrete_ppo/train.py` (post-Session-01) — the
   file you're modifying.
6. `agents/ppo_trainer.py::_train_loop` — v1's multi-day reference
   for the loop shape. **Read, don't import.**

## What to do

### 1. CLI: `--days` and `--day-shuffle-seed` flags (~15 min)

Replace `--day` (single date) with **either**:

- `--day 2026-04-23` (single day, backwards-compatible with
  Phase 2 / Session 01); or
- `--days 7` (use 7 days from the data dir, deterministic order).

Mutually exclusive. If `--days` is supplied:

- Enumerate parquet files in `--data-dir` matching
  `YYYY-MM-DD.parquet`.
- Sort lexicographically. Take the last N (most recent).
- Hold out the LAST one as the eval day (Phase 3 cohort needs a
  held-out day; Session 02 just generates one trajectory, but
  the eval-day split lands here so Session 03 inherits the
  contract).
- Shuffle the remaining N-1 with `random.Random(day_shuffle_seed).shuffle(...)`.

`--day-shuffle-seed` defaults to the same value as `--seed`. Phase
3 explicitly does NOT add a curriculum — it's a follow-on plan if
ordering shows order-dependence problems.

### 2. Multi-day loop in `main()` (~30 min)

Phase 2's structure was:

```
day = load_day(day_str)
env = BetfairEnv(day, cfg)
shim = DiscreteActionShim(env, ...)
trainer = DiscretePPOTrainer(policy, shim, ...)
for ep_idx in range(n_episodes):
    stats = trainer.train_episode()
```

Multi-day extends this to:

```
days_to_train = _select_days(...)
for day_idx, day_str in enumerate(days_to_train):
    day = load_day(day_str, data_dir=data_dir)
    env = BetfairEnv(day, cfg)
    shim = DiscreteActionShim(env, scorer_dir=scorer_dir)
    # Re-bind the trainer's shim. Same policy, same optimiser,
    # same alpha state — only the env changes.
    trainer.shim = shim
    trainer._collector = RolloutCollector(
        shim=shim, policy=trainer.policy, device=device,
    )
    for ep_idx in range(epochs_per_day):
        stats = trainer.train_episode()
        _emit_jsonl_row(stats, day_idx=day_idx, day_str=day_str, ep_idx=ep_idx)
```

`epochs_per_day` is a new flag, default 1. (Phase 2 ran 5 epochs on
ONE day; Phase 3 trains 1 epoch on EACH of N days. The total
update count is comparable; the data diversity is much higher.)

Multi-day does NOT carry hidden state across days. Each
`trainer.train_episode()` calls `RolloutCollector.collect_episode`
which calls `policy.init_hidden(batch=1)` fresh. This is correct:
each day IS a fresh episode.

### 3. JSONL row schema additions (~10 min)

Per-row fields added on top of Session 01's schema:

- `day_str` — ISO date this row's training corresponds to.
- `day_idx` — 0-based index of the day in the run order.
- `epoch_idx` — 0-based index of the epoch within the day (default
  0 if `epochs_per_day=1`).
- `cumulative_episode_idx` — 0-based across all (day, epoch)
  pairs so far. Lets the findings.md table sort the trajectory
  without reconstructing it.

### 4. Episode-level diagnostics aggregation (~20 min)

End-of-run summary in `main()`:

- Per-day mean of `total_reward`, `day_pnl`, `value_loss_mean`,
  `policy_loss_mean`, `approx_kl_mean`.
- Across-day trend of `value_loss_mean` (the bar 2 verdict).

Print a table at end of run:

```
Day idx | Date       | total_reward | day_pnl | value_loss | approx_kl
   0    | 2026-04-15 |    -1455.6   |  -578.1 |    2.541   |   0.036
   ...
```

### 5. Multi-day run + findings (~60 min)

Run on the 7 most recent training days (excluding 2026-04-25, the
held-out eval day):

```
python -m training_v2.discrete_ppo.train \
    --days 7 \
    --device cuda \
    --seed 42 \
    --out logs/discrete_ppo_v2/multi_day_run.jsonl
```

Capture:

1. Per-day value loss trajectory. Does it descend monotone (with
   day-level noise) across the 7 days?
2. Per-day day_pnl. Does it trend up?
3. Across-day approx_kl. Does it stay < 0.5 for every day?
4. Wall time. 7 days × ~50 s/episode = ~6 min total expected on a
   real GPU.

Append to `plans/rewrite/phase-3-cohort/findings.md` a Session-02
section. The bar 3 verdict (multi-day Bar 2 GREEN) is this
session's deliverable.

If value loss is STILL non-monotone across 7 days: that's a real
finding, not noise. Stop and write up.

### 6. Test (~20 min)

`tests/test_v2_multi_day_train.py`:

```python
def test_multi_day_loop_uses_each_day_once(tmp_path, monkeypatch):
    """Validate the day-selection + iteration logic without running PPO."""
    # Pseudo-implementation: monkey-patch load_day to return tiny
    # synthetic days and assert the train loop calls it N times in
    # the shuffled-but-deterministic order, with the eval day held
    # out of training.
    ...

def test_episode_boundary_is_day_boundary(tmp_path):
    """Hidden state is reset between days."""
    # Run two synthetic days; assert the second day's first
    # transition has hidden_state_in[0] == zeros (init_hidden).
```

The tests cover the loop logic, not the gradient pathway —
Session 01's parity test plus Phase 2's existing trainer tests
already cover that.

## Stop conditions

- Bar 3 fails (value loss flat or oscillating across 7 days) →
  **stop**. Likely cause: the day shuffle is biasing the
  trajectory (e.g., all hard days first), OR the per-day re-bind
  of `shim` and `_collector` is leaking state. Investigate before
  Session 03.
- Wall time > 30 min for 7 days on GPU → **document, proceed**.
  Throughput optimisation is a follow-on plan; the bar is "works",
  not "optimal."
- approx_kl spikes above 0.5 on any day → **stop**. The KL
  pathway was rock-solid in Phase 2 + Session 01; a multi-day
  spike means the per-day env/shim re-bind disrupts the policy's
  hidden-state contract.

## Hard constraints

- **No env edits.** Same as all phases.
- **No cross-day hidden state.** Each day is a fresh episode.
  Phase 1's `init_hidden(batch=1)` is called at the start of every
  day's rollout.
- **No curriculum.** Day order is deterministic shuffle. If
  curriculum looks like it'd help, that's a follow-on plan.
- **No reward changes.** Same scalping config Phase 2 used.

## Out of scope

- GA cohort scaffolding (Session 03).
- Frontend events (Session 04).
- Curriculum day ordering (follow-on if needed).
- Multi-epoch-per-day beyond `epochs_per_day=1` for Phase 3 (the
  flag exists; Phase 3 cohort uses 1).
- Per-day eval (Phase 3 evaluates on the held-out day at end of
  cohort run, not after every training day).

## Useful pointers

- v1 multi-day loop: `agents/ppo_trainer.py::_train_loop` (read,
  don't import).
- Phase 2's CLI shape: `training_v2/discrete_ppo/train.py`.
- Phase 2's per-day baseline: 11872 steps, 113 s/episode (CPU). On
  GPU expect ~10–20 s/episode for 7 days = ~2 min total.
- Day enumeration pattern: data files are
  `data/processed/YYYY-MM-DD.parquet` + `_runners.parquet`. The
  loader takes the date string and the data-dir; existence check
  is implicit.

## Estimate

2.5 hours.

- 15 min: CLI flags.
- 30 min: multi-day loop.
- 10 min: JSONL schema.
- 20 min: end-of-run aggregation.
- 60 min: real run + findings.
- 20 min: tests.

If past 4 hours, stop and check scope. The most likely overrun is
diagnosing why value loss isn't descending if Bar 3 fails. Don't
chase fixes — write the finding and step back.
