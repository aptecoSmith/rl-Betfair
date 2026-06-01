# Session prompt — execute `training-speedup-v2` autonomously

You are picking up a training-speedup refactor for the rl-betfair project.
Work autonomously through the plan, but honour the stop-points below. This
plan exists *because* a documented incompatibility (`--batched` silently
drops BC) was sailed into twice — so the entire ethos here is **verify, gate,
and never trust "it should be equivalent."**

---

## 0. Read these before touching anything (non-negotiable)

1. `plans/training-speedup-v2/purpose.md` — the why, the target, the spine.
2. `plans/training-speedup-v2/hard_constraints.md` — **inviolable.** Re-read
   #1 (bit-identical gate), #2 (no silent feature drops), #7 (current env is
   golden), #8 (no dynamics change as a "speedup").
3. `plans/training-speedup-v2/master_todo.md` — your checklist + per-step
   gates. Update its `[ ]/[~]/[x]` status as you go.
4. `plans/cohort_training_speedup/deferred.md` and `phase_3.md` — the PRIOR
   speedup work. **The landmines are here.** Specifically: `--batched`
   silently ignores `bc_pretrain_steps` and `per_transition_credit`
   (runner.py:1176-1198). Profiling showed PPO update is only 7.6% of wall;
   rollout is 92%. They correctly *refused* bet-aggregate caching and
   extract-vectorisation for being correctness-risky at small gain — those
   are now allowed ONLY behind the bit-identical gate.
5. `CLAUDE.md` (repo root + `rl-betfair/CLAUDE.md`) and your memory files.

## 1. The discipline (these were violated; that's why you're here)

- **VERIFY, DON'T GUESS.** Before any run that costs compute, grep/read the
  actual code, a doc, or a measurement — never act on "this flag should do
  X." If you're tempted to assume a load-bearing fact, STOP and check it.
- **The bit-identical gate is law.** Nothing is trusted until it reproduces
  the current env's golden trajectories (exact on discrete quantities,
  declared per-quantity float tolerance on continuous — GPU reductions
  reorder sums, so set tolerances that pass legit reordering and still catch
  logic drift).
- **Current env is golden.** When the fast path disagrees, the current env
  is right until proven otherwise. Never tune golden to make the fast path
  pass.
- **No silent feature drops.** If you touch the batched/rollout path, add
  the HC#2 regression test that asserts each flag's effect is present (or
  its drop is logged).

## 2. Work order and autonomy boundaries

Do these autonomously, in order, each gated:

- **Step 0 — profile.** Read-only; fully autonomous. Multiprocess-aware
  (cProfile misses worker processes — use per-phase wall timers in the
  worker, or attach to it). Reconcile to 867s/agent-day ±15%. **→ STOP and
  report the breakdown** so the operator can steer Step 2 vs Step 3 focus.
- **Step 1 — the harness.** THE keystone deliverable; invest in it. Capture
  golden from the current env across all 8 case classes in master_todo.
  Prove it catches a deliberately-injected 1-tick perturbation. **→ STOP and
  report** that it's built + discriminating before relying on it.
- **Step 2 — hot-path vectorisation.** Autonomous; each change gated by
  Step 1 golden. Start with the deferred `extract_array`.
- **Step 3A — true batch=N policy forward** in
  `training_v2/discrete_ppo/batched_rollout.py`. Autonomous; gated. Add the
  HC#2 no-silent-drop test here.
- **Step 3B — vectorise the obs/market path.** Autonomous; gated.

**HARD STOP before Step 3C (env-core rewrite).** Do the feasibility spike
(prototype matching+settlement as batched tensor ops on a slice; measure the
realistic multiplier + the un-vectorisable branching fraction), then **STOP
and report the spike numbers for a human go/no-go.** This is the high-risk,
high-cost rewrite — it does not start without an operator decision.

**Step 4 (BC ablation)** needs training runs; do it after the speedups land
and report. Preliminary record says BC may NOT be load-bearing (supervised
AUC success ≠ PPO-warm-start P&L win; c1 succeeded with BC off).

## 3. How to validate every step (the gate)

Run the candidate (optimised) env/path on identical inputs (same day, same
seed, same policy weights) and diff against the Step-1 golden:
- discrete quantities (actions, bet counts, pair_ids, match/naked/force-close
  classifications, settle outcomes) — **exact**;
- continuous (reward, value, price, P&L) — within the declared per-quantity
  tolerance, justified as float-reordering only.
A failing diff means the optimisation is wrong, not the golden. Fix or revert.

## 4. Stop and report to the operator when

- Step 0 profile done · Step 1 harness built+proven · before Step 3C full
  build (post-spike) · Step 4 ablation result.
- The gate fails in a way you can't cleanly attribute to legitimate float
  reordering.
- You would otherwise have to **guess** a load-bearing fact.
- Any hard constraint would have to bend.

## 5. Record as you go

- Update `master_todo.md` status live.
- Start `plans/training-speedup-v2/lessons_learnt.md`; log anything that
  bites (especially gate failures and their root cause).
- Append the Step-0 profile + each stage's measured speedup to
  `plans/EXPERIMENTS.md`.
- Correct the record: the c1/c2 EXPERIMENTS entries say "BC 500 steps" — it
  was a no-op under `--batched`. Annotate both.
- Match writeup size to result (confirmations short; analyses fuller).

## 6. Environment specifics (verify, don't assume)

- GPU: default `--device cuda` for any training/profile run that uses it.
- Tests: run the suites named in `deferred.md`'s pre-launch checklist plus
  your new `tests/test_env_golden_parity.py`; the golden-parity test is the
  new load-bearing guard.
- Long runs: drive them with a durable bash polling chain (the repo's proven
  mechanism — smart daemons fail here); announce up-front what you're
  waiting on; don't go silent.
- Held-out days `2026-05-20,21,22,25,27,28,29` stay sealed — never
  train/select/tune on them (Step 4 ablation evals on them read-only).

## 7. Done means

Every step's master_todo GATE passes: Step 0 profile reconciles; Step 1
harness discriminates; Steps 2/3A/3B are bit-identical to golden with logged
speedups; Step 3C is a recorded go/no-go decision; Step 4 is a recorded BC
decision with no silent state. The headline number — agent-train-day wall
vs the 867s baseline — is logged in EXPERIMENTS.
