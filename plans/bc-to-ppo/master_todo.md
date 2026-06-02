# BC → PPO — master TODO (the steps + status)

**START HERE (fresh session):** read `purpose.md` + `hard_constraints.md`.
The BC warm-start is `plans/bc-getting-it-right/_scripts/stepB_alltrain_
wd3e-3.pt` (holdout maturation AUC 0.745). The force-close safety barrier
is landed (config `force_close_max_deviation_pct: 0.50`) and MUST stay on.
GPU: `--device cuda`. Caches in `data/oracle_cache_v2/` +
`plans/bc-getting-it-right/_cache/`.

Status legend: [ ] todo · [~] in progress · [x] done.

---

## Step 0 — Confirm the barrier-honest rollout baseline. [~]

- Re-run the greedy-by-mature_prob holdout rollout WITH the barrier
  (`stepE_rollout.py --fc-max-dev config`, T=0.20 + 0.30). Confirm
  mat%/locked ≈ the pre-barrier numbers (expected: barrier touches ~1/341
  fills). This is the trustworthy deployment-economics baseline.
- Record in `plans/bc-getting-it-right/findings.md` (with/without barrier).

## Step 1 — Map mat% vs realistic budget (pick the operating point). [ ]

- Greedy rollout at T=0.20, barrier on, budgets {£100, £200, £300} on the 7
  holdout days. Output: mat% / locked / day_pnl / fc% / peak_open_liability
  per budget. (`stepE_rollout.py --starting-budget`, instrumented for spend.)
- **Decide the PPO training budget** from the knee — the smallest budget
  where passives mostly post (low `budget_lay` rejects) without absurd
  capital-at-risk. Likely £100–300/race. Record the choice + rationale.

## Step 2 — Build the BC→PPO warm-start (actor opens + mature head selects). [ ]

- The greedy rollout bypassed the actor. PPO needs a single policy whose
  **actor actually opens**, gated by `mature_prob`. Extend
  `bc-getting-it-right/_scripts/mature_head_bc.py` (or a new combined-BC
  script): actor CE on opportunity opens + the existing mature_prob BCE,
  input-norm ON. Save the combined policy.
- Sanity: greedy AND actor-driven rollouts both open selectively at the
  chosen threshold (mat% ≫ baseline, locked positive) before PPO.

## Step 3 — BC→PPO canary (the profit step). [ ]  ← operator-authorised

- Drive `DiscretePPOTrainer` standalone (like the `_step1` canaries),
  warm-started from Step 2's policy.
- Reward: `maturation_reward_mode` ON (pays only matured / profit-closed
  pairs; force/stop/naked → 0) + `open_cost` toll + MTM densification.
- Pins (§4): budget from Step 1, fc=120, close_walk=10, input_norm, barrier
  0.50. Select on LOCKED.
- Single config. Eval the 7 holdout days ONCE, fully hedged. Report BOTH
  optimistic (overdraft) and conservative numbers (§7).
- **GATE (success bar):** holdout LOCKED cleanly positive with mat% above
  the BC's 14–15% via LOWER fc% (more selective than the oracle). If yes →
  Step 4 (gated). If no → write the finding (oracle edge too thin at
  deployment fill rates → richer data); do NOT keep knob-tuning.

## Step 4 — Scale + validate. [ ]  ← OPERATOR-GATED (come back first)

- GA cohort: wire `input_norm` through `cohort/worker.py` (GREP every
  callsite — §8), evolve the genes (`arb_spread_target_lock_pct`,
  `open_cost`, `mature_prob_open_threshold`, `mature_prob_loss_weight`, …),
  select on `locked_per_std`.
- One fully-hedged holdout eval. Then dry-run via `ai-betfair` (no money).
- Do NOT start without operator sign-off (§9).

---

## Decisions already made (carry forward)
- Force-close deviation barrier = **0.50** (config default); no-LTP closes
  **refused** (operator delegated the call 2026-05-31).
- Select on **LOCKED**, train at the **deployment budget**, barrier **ON**
  in training. Step 4 is **operator-gated**.

## Verification (every step)
- [ ] Reported on held-out LOCKED P&L (§1), holdout split (§2), barrier ON
  (§3), warm-started (§5).
- [ ] Both optimistic + conservative numbers (§7).
- [ ] No Step 4 without operator sign-off (§9).
