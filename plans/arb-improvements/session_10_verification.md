# Session 10 — Head-to-head verification run

## Before you start — read these

- `plans/arb-improvements/purpose.md`
- `plans/arb-improvements/master_todo.md` — Phase 5, Session 10.
- `plans/arb-improvements/testing.md` — **this is the one session
  where GPU is allowed**. Full training runs are the point.
- `plans/arb-improvements/progress.md` — the `90fcb25f` baseline
  table is the comparison target.
- `plans/arb-improvements/lessons_learnt.md` — read all prior
  entries; this session is where we validate or refute everything
  upstream.

## Goal

Prove the stack works on the exact configuration that originally
failed. Re-run the `90fcb25f` gene / architecture / date spec with
all Phase 1–3 knobs enabled (aux head if Session 9 shipped).
Compare against the baseline table; record results; decide whether
the plan is done or whether a follow-up plan is needed.

## Scope

**In scope:**

- Replay the `90fcb25f` run configuration:
  - 15 agents, 1 elite, 50% survival, 30% mutation.
  - 3 generations, 6 epochs per day.
  - Architectures: `ppo_lstm_v1`, `ppo_time_lstm_v1`,
    `ppo_transformer_v1` (same mix).
  - Budget £20, max 20 bets/race, max_back=50.
  - Training dates: 2026-04-06, 07, 08.
  - Same seed if recoverable; if not, run 3 seeds and report each.
- Phase 1–3 knobs enabled:
  - `reward.reward_clip = 5.0`
  - `training.advantage_clip = 2.0`
  - `training.value_loss_clip = 3.0`
  - `training.entropy_floor = 0.5`
  - `training.signal_bias_warmup = 3`
  - `training.signal_bias_magnitude = 0.3`
  - `training.bc_pretrain_steps = 1000`
  - `training.aux_arb_head = <True if Session 9 shipped else False>`
- Full run on GPU with the same hardware target as the baseline
  (RTX 3090 via the existing infra).

**Out of scope:**

- Hyperparameter sweeps over the new knobs. This is verification
  of the settings recommended in each session's session_prompt,
  not a search.
- Production shipping decisions. If verification fails, the
  output is a new plan, not a hotfix here.

## Measurements & comparison

Record per-agent across the run:

- Episode-1 loss (compare: baseline was `10⁹–10¹²`, target `< 10⁷`).
- Bet-rate across episodes 1–18 (compare: baseline was ~1/6 agents
  had activity past ep 3; target is ≥ 5/6 agents with positive
  bet-rate through ep 18).
- `arbs_completed` per episode (target: > 0 for most episodes on
  most agents).
- Mean `locked_pnl` per agent across the run.
- Mean `total_reward` progression (target: monotonic improvement
  across generations for at least half the population).

Also record, for operator-facing sanity:

- `entropy_coeff_active` history — did the floor trigger?
- `clipped_reward_total` magnitude — how often did clipping bite?
- BC loss curves — did pretraining converge on every agent that
  had oracle samples?

Write findings into `progress.md` Session 10 entry as a table
structurally identical to the baseline table, plus a short prose
summary.

## Tests to add

No new unit tests in this session. Integration / verification
only. However:

- Confirm that the short integration smoke tests from Sessions 1,
  3, 7 still pass on the current main.
- Confirm `pytest tests/ -m "not gpu and not slow"` is green
  before kicking off the GPU run.

## Session exit criteria

- Full GPU run completes. Results table in `progress.md` with
  per-agent metrics matching the baseline table format.
- Verdict in `progress.md`: one of
  - **Green**: baseline failure mode is fixed. No follow-up
    needed. Close the plan.
  - **Yellow**: collapse is fixed but arb-rate / locked_pnl isn't
    materially better. Open a new plan or re-scope this one with
    a specific follow-up session.
  - **Red**: collapse still occurs. Root-cause analysis goes into
    `lessons_learnt.md`; new plan required.
- `lessons_learnt.md` has a "Verification findings" section with
  anything surprising.
- No commits unless operational fixes were needed during the run;
  in that case commit each independently with clear messages.
- `git push all`.

## Do not

- Do not tune knobs during the run. If a knob needs changing, that's
  a new plan / new session, not a mid-run adjustment.
- Do not declare success on a single seed if reproducibility is in
  question. If the seed from `90fcb25f` can't be recovered, run 3
  seeds and report all of them.
- Do not skip reporting failures. Yellow/Red verdicts are data.
- Do not leave `progress.md` without a verdict line. The whole
  plan exists to fix one specific failure; this session's output
  is a clear answer about whether it did.
