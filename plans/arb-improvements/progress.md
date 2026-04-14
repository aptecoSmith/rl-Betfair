# Progress — Arb Improvements

*Filled in as each session completes. Each entry names the session,
the date, the commit hash, and the files changed. Anything
surprising goes into `lessons_learnt.md` — this file is a log.*

---

## Reference baseline — run 90fcb25f (2026-04-14)

The failure this plan exists to fix. All Phase-5 verification
compares against this table.

| Agent | Arch | Mean reward | Mean P&L | Eps with arb activity |
|-------|------|-------------|----------|------------------------|
| 575183fe-edf | ppo_time_lstm_v1 | -20.207 | +8.54 | 3 / 18 |
| 7a709d01-ead | ppo_lstm_v1 | -151.238 | +24.10 | 18 / 18 |
| c1d695f4-b94 | ppo_transformer_v1 | -21.127 | -15.79 | 5 / 18 |
| afa3881f-e2d | ppo_transformer_v1 | -29.654 | +2.00 | 5 / 18 |
| f63de5ac-cfe | ppo_transformer_v1 | -37.745 | +1.49 | 3 / 18 |
| f9b1289a-f61 | ppo_transformer_v1 | -136.248 | +109.59 | 18 / 18 |
| 2954352a-2ee | ppo_transformer_v1 | (partial — run cut short) | — | — |

Characteristic pattern: episode 1 loss in the `10⁹–10¹²` range
(e.g. `291555954607` for 575183fe, `1632888614073` for afa3881f),
then either total collapse (most agents) or pathological per-day
memorisation (7a709d01 and f9b1289a repeat the exact same per-day
reward from ep 4 onwards).

Success criteria post-Phase-3:

- No episode-1 loss above `10⁷` after clipping lands.
- `bet_rate > 0` across all 18 episodes for at least 5 / 6 agents.
- Mean locked_pnl across the population strictly improves vs
  baseline.

---

## Phase 1 — Stop the collapse

### Session 1 — Reward & advantage clipping

**Shipped 2026-04-14.** Added three independent training-signal clip
knobs, all defaulting to `0.0` (off) so existing runs are byte-identical:

- `reward.reward_clip` — per-step reward clipped to `[-c, +c]` *only*
  on the path into the GAE advantage/return computation. Raw reward
  still flows into `EpisodeStats.total_reward`, `info["day_pnl"]`,
  the `episodes.jsonl` log line, and the monitor progress event.
- `training.advantage_clip` — per-transition advantage magnitude
  clamped before the PPO surrogate ratio multiplies it.
- `training.value_loss_clip` — per-sample squared residual capped at
  `value_loss_clip ** 2` before the batch mean.

`max_grad_norm` is unchanged — the new clips sit *in front of* it,
not in place of it.

Files changed:

- `agents/ppo_trainer.py` — new hp fields, `Transition.training_reward`,
  reward clip in rollout, advantage / value-loss clamps in PPO update,
  `EpisodeStats.clipped_reward_total`, telemetry plumbed to
  `episodes.jsonl` and the progress event. `_REWARD_GENE_MAP` gains a
  `reward_clip` passthrough so the gene rides the same per-agent route.
- `env/betfair_env.py` — `reward_clip` added to
  `_REWARD_OVERRIDE_KEYS` (env itself ignores it; the trainer reads it
  off `self.reward_overrides`).
- `tests/arb_improvements/test_reward_clipping.py` — 8 CPU-only tests
  covering training-signal isolation, default-off byte-equivalence,
  the advantage / value-loss clamps, env override whitelist, the
  `raw + shaped ≈ total_reward` invariant, and progress-event
  telemetry.
- `plans/arb-improvements/ui_additions.md` — Session 1 wizard +
  monitor checkboxes already present from planning.

Test results: 8 / 8 new tests pass; full non-gpu/non-slow suite green
(1858 pass, 7 skip, 1 xfail) at parent commit `332ed86`.

UI work deferred to Session 8 (consolidation pass).

### Session 2 — Entropy floor & per-head logging

**Shipped 2026-04-14.** Adaptive entropy-coefficient controller + per-head
entropy diagnostics. All four new hyperparameters default to values that
leave training byte-identical (`entropy_floor=0` = off).

New hyperparameters on `PPOTrainer`:

- `entropy_floor` (float, default `0.0` = off). When the rolling mean
  entropy drops below this value, `entropy_coefficient` is scaled to
  `floor / rolling_mean × base`, capped at `entropy_boost_max`. When the
  rolling mean recovers, the coefficient snaps back to the baseline.
- `entropy_floor_window` (int, default `10` batches).
- `entropy_boost_max` (float, default `10.0`) — caps the multiplier.
- `entropy_collapse_patience` (int, default `5`) — consecutive batches a
  single head must sit below the floor before the `entropy_collapse`
  warning flag fires.

The controller scales the *coefficient* only; the policy's action
distribution is never touched directly (hard_constraints.md §Stabilisation).

Progress events now carry an `action_stats` dict at the top level:

```
action_stats = {
    "mean_entropy_signal":     <rolling mean>,
    "mean_entropy_stake":      <rolling mean>,
    "mean_entropy_aggression": <rolling mean>,
    "mean_entropy_cancel":     <rolling mean>,
    "mean_entropy_arb_spread": <rolling mean>,
    "entropy_collapse":        <bool>,
    "entropy_coeff_active":    <float>,
}
```

Per-head entropy is sliced out of `dist.entropy()` using the policy's
`max_runners` and `_per_runner_action_dim` — no policy-network change
needed. Heads not present in the policy (e.g. `arb_spread` on a
directional run) are reported as `0.0` for a stable schema but never
trip the collapse detector.

Files changed:

- `agents/ppo_trainer.py` — `_HEAD_NAMES` constant; entropy-floor hp
  plumbing; rolling window / per-head collapse streak; new
  `_update_entropy_controller` + `_compute_per_head_entropy` methods;
  per-head entropy accumulated through the mini-batch loop and flushed
  at the end of each `_ppo_update`; `action_stats` routed through
  `loss_info` into `_publish_progress`.
- `tests/arb_improvements/test_entropy_floor.py` — 7 CPU-only tests:
  floor triggers scaling, recovery restores baseline, floor off = no
  coefficient change, per-head entropy in progress event, collapse flag
  sets / clears, boost_max caps multiplier, raw+shaped invariant holds
  with floor armed.

Test results: 7 / 7 new tests pass; `tests/arb_improvements/`,
`tests/test_ppo_trainer.py`, `tests/test_forced_arbitrage.py` = 90
pass, no regressions. The 17 pre-existing real-data integration
failures (empty parquet, missing each-way divisors) are unchanged by
this session.

UI work deferred to Session 8 (consolidation pass).

### Session 3 — Signal-bias warmup & bet-rate diagnostics

Not yet started.

## Phase 2 — Make arbs perceivable

### Session 4 — Pure arb feature functions

Not yet started.

### Session 5 — Wire features into env + schema bump

Not yet started.

## Phase 3 — Oracle scan + BC warm start

### Session 6 — Arb oracle scan

Not yet started.

### Session 7 — BC pretrainer + trainer integration

Not yet started.

### Session 8 — Wizard UI, evaluator, UI consolidation

Not yet started.

## Phase 4 — Optional auxiliary head

### Session 9 — Aux arb-availability head

Not yet started.

## Phase 5 — Verification

### Session 10 — Full verification run

Not yet started.
