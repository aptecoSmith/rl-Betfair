# Session prompt — Phase 2, Session 03: first real-day training run

Use this prompt to open a new session in a fresh context. The prompt
is self-contained — it briefs you on the task, the design decisions
already locked, and the constraints. Do not require any context from
the session that scaffolded this prompt.

---

## The task

Run the first real PPO training pass on a real day. **One agent.
One day. One full multi-episode training run.** Wire the train CLI,
log loss curves + KL trajectory + per-runner advantage stats, write
the phase-level findings.

End-of-session bar: `python -m training_v2.discrete_ppo.train --day
2026-04-23` completes cleanly, all 4 algorithmic success bars
(value loss descends, KL stays < 0.5 median, per-runner advantage
shape correct, no env changes) PASS, findings.md written.

## What you need to read first

1. `plans/rewrite/phase-2-trainer/purpose.md` — success bar table
   (1–5), hyperparameter table, hard constraints.
2. `plans/rewrite/phase-2-trainer/session_prompts/01_rollout_collector_and_gae.md`
   + `02_ppo_update_and_trainer.md` and the resulting code under
   `training_v2/discrete_ppo/` — import these verbatim.
3. `plans/rewrite/phase-1-policy-and-env-wiring/findings.md` — the
   smoke-run reference numbers (random-init policy on 2026-04-23:
   `total_reward = -55.115`, `day_pnl = -£41.37`, 11.15 ms / step
   on CPU). Those are the "before training" baseline; Session 03's
   job is to confirm training does *something* — direction is more
   important than magnitude.

## What to do

### 1. `training_v2/discrete_ppo/train.py` (~45 min)

CLI entry point. Mirror Phase 1's `agents_v2/smoke_test.py` shape:

```
python -m training_v2.discrete_ppo.train \
    --day 2026-04-23 \
    --n-episodes 5 \
    --seed 42 \
    --out logs/discrete_ppo_v2/run.jsonl
```

Loads the day, builds env + shim + policy + trainer, runs `n_episodes`
of `trainer.train_episode()`, logs an `EpisodeStats` row to JSONL per
episode plus a console summary. Default `n_episodes = 5` so the run
fits in the 10-minute budget on CPU; the operator can bump it.

### 2. The actual run + observations (~60 min)

Run the CLI:

```
python -m training_v2.discrete_ppo.train --day 2026-04-23 --n-episodes 5
```

Capture:

1. **Loss curves.** `policy_loss_mean`, `value_loss_mean`,
   `entropy_mean` per episode.
2. **KL trajectory.** `approx_kl_mean`, `approx_kl_max`,
   `n_updates_run` per episode. Watch for the KL early-stop
   tripping (n_updates < full budget) — that's the early signal
   that something's drifting between rollout and update.
3. **Per-runner advantage stats.** Mean / std / max-abs of
   `advantages` per episode. A near-zero std would mean the value
   head is already perfect (suspicious on episode 1) or the GAE
   collapsed (bug).
4. **Action histogram.** Across each episode — does the policy
   start to favour OPEN_BACK over OPEN_LAY (or vice versa) by
   episode 5? Random policy was 320 / 375 (Phase 1 finding); a
   trained policy should drift away from 50/50.
5. **Reward.** `total_reward` per episode. Does it improve
   monotonically? (Probably not — 5 episodes is too few. But the
   *direction* over the 5 episodes is the signal.)

### 3. Findings writeup (~60 min)

`plans/rewrite/phase-2-trainer/findings.md`:

- Success bar table (1–5 from purpose.md PASS/FAIL).
- Loss curves table (5 rows × 3 columns).
- KL trajectory table (5 rows × 3 columns) with explicit early-stop
  count.
- Per-runner advantage stats.
- Action histogram per episode.
- Phase 3 implications: anything in the trainer you expect to need
  to change once GA + multi-day + frontend land. Specifically:
  - Does training on one day overfit aggressively? (One-day
    training is a Phase 2 simplification; Phase 3 trains on N days.)
  - Are the locked hyperparameters defensible, or did you have to
    change one to make the bar pass? (If you changed any, document
    which and why; that's information Phase 3 needs.)
  - Does the per-runner advantage actually have meaningful
    variation across runners, or is it dominated by 1–2 runners?
    (A dominated advantage signal would mean per-runner GAE isn't
    paying for itself; flag for Phase 3 ablation.)

## Stop conditions

- All 5 success bar conditions PASS → write findings.md GREEN,
  message operator "Phase 2 GREEN, ready for Phase 3", **stop**.
- Bar 1 fails (training crashes) → triage which session needs
  revisiting (Session 01 if rollout shape is wrong; Session 02 if
  the update path is wrong), **stop**.
- Bar 2 fails (value loss flat or exploding) → **stop**. Likely
  cause: per-runner reward attribution wrong (Session 01 bug) or
  GAE bootstrap wrong (Session 01 / 02 boundary bug). Don't reach
  for advantage normalisation — that's the v1 mistake the rewrite
  is undoing.
- Bar 3 fails (median KL > 0.5) → **stop**. Likely cause: hidden-
  state mismatch between rollout and update. v1's `ppo-kl-fix`
  plan is the canonical reference for this failure mode.
- Bar 4 fails (per-runner advantage shape wrong) → **stop**. This
  is a Session 01 bug; the in-session test should have caught it.
- Bar 5 fails (env changes) → revert the env diff immediately,
  file as a Phase −1 follow-on, **stop**.

## Hard constraints

- **No GA, no cohort, no multi-day.** One agent, one day. (Phase
  3.)
- **No frontend wiring.** Logs go to plain text / JSONL. (Phase 3.)
- **No new shaped rewards.** (Rewrite hard constraint §5.)
- **No entropy controller, no advantage normalisation, no LR
  warmup, no reward centering.** (Rewrite hard constraint §6.) If
  any of these would "fix" a bar failure, that's a finding to
  write up, not a fix to apply.
- **No env edits.** (Rewrite hard constraint §1.)
- **No hyperparameter search.** The locked values in purpose.md
  are the run. If you must change one, document which and why in
  findings.md.

## Out of scope

- GA breeding / mutation (Phase 3).
- Multi-day training (Phase 3).
- Cohort comparison vs v1 (Phase 3).
- Frontend / websocket events (Phase 3).
- Removing v1 code (after Phase 3 success per rewrite README).
- Performance optimisation (Phase 3 cares about wall time when
  it's running 64 agents × 30 days; Phase 2 cares about
  correctness).

## Useful pointers

- `training_v2/discrete_ppo/trainer.py::DiscretePPOTrainer` —
  Session 02.
- `agents_v2/smoke_test.py` — Phase 1 reference for the CLI shape
  (argparse, JSONL output, console summary).
- `data/episode_builder.py::load_day` — day loader.
- `models/scorer_v1/` — Phase 0 artefacts; the shim loads them.
- `plans/per-runner-credit/findings.md` — v1 investigation that
  motivated the per-runner value head + per-runner GAE in the
  rewrite. Useful context for "why do we expect per-runner
  advantages to actually vary".

## Estimate

2.5–3.5 hours.

- 45 min: `train.py` CLI.
- 60 min: real-day run + iteration on observations.
- 60 min: findings.md.
- 15 min: session-end summary message.

If past 4 hours, stop and check scope. The most likely overrun is
the findings.md — there's a temptation to write a long analysis. The
phase verdict is a one-page table + 5 short observations; ship that
and let Phase 3 do the deep analysis.
