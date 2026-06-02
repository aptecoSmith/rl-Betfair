# GA recipe search — hard constraints

Non-negotiables, each tied to a lesson that already cost us time.

## §P — PINS (do NOT evolve these)

1. **`force_close_before_off_seconds = 120` PINNED.** If FC is a gene
   the GA WILL rediscover fc=0 naked-luck winners (+£287 in-sample →
   -£175 held-out; HV cells). The safety rail is not negotiable.
2. **`close_walk_ticks` PINNED** (10 for deploy-honesty, or 0 — pick one
   and hold it cohort-wide). Round W showed close-walk is a deploy
   safety rail, not a profit lever; it strips directional variance at a
   small spread cost. Evolving it just adds a confound. Recommend pin
   = 10 so held-out day_pnl is the honest fully-hedged number.
3. **BC stays PER-AGENT, never shared.** Sharing BC-pretrained weights
   across the population collapses GA diversity irreparably (inherited
   lesson). Mixing BC/no-BC across agents is fine; the code already
   freezes/restores per-agent.

## §D — DATA SPLIT (corrected 2026-05-30 — the campaign trained on 6% of the data)

We have **49 processed days** (2026-04-06 → 05-29). Recipes trained on 3.
Fixed split, deployment-realistic (train before holdout):

1. **Train = 42 days, Apr 6 → May 19** (every processed day in that range;
   ~3,360 races). `--training-days-explicit` with the explicit list.
2. **Holdout = latest 7: May 20, 21, 22, 25, 27, 28, 29** (verified
   75-91 markets each, ~570 races). `--cohort-eval-days` with the
   explicit list. NEVER train on these.
3. **Explicit day lists ONLY — never `select_days(n)`** (the n_days≥14
   Apr-30 leak foot-gun in memory). Re-verify train ∩ holdout = ∅ at
   launch.
4. Old "held-out" numbers on May 07-19 become TRAINING data now → no
   longer a comparison surface. The May 20-29 holdout (almost entirely
   unused before) is the new one.

## §O — FULL OBS (drop the lean-obs compression)

1. **Drop `--predictor-lean-obs`** → 143-feature-per-runner full obs
   (was 23). KEEP the predictor bundle + direction head (free features).
2. Normalization is handled — `feature_engineer` log-norms
   volumes/sizes/depths + scales time/ranks; policy applies per-runner
   LayerNorm at input. Verified low-risk, BUT run the §V value-domain
   audit before the canary (a stale/leaky/outlier dim lean-obs never
   exposed is the residual risk).
3. Full obs = new input dim → arch-hash break → fresh init, no cross-
   load from lean-obs weights. Expected.
4. **Oracle + feature caches must be rebuilt at full-obs dim** across
   all 42 training days before any run (the preflight check enforces
   it; pre-build to avoid a 30s-in crash). See master_todo step 1.

## §S — REWARD SPARSITY AT SCALE (the "500k" concern; densification is mandatory)

A rollout = ONE day (~5-13k transitions, verified), so 42 days = more
*episodes*, not a bigger buffer — per-rollout credit assignment is
unchanged. The real issue is sparsity (reward-densification plan): the
reward arrives at settle, ~0 gradient on 99% of steps; full obs (143-d)
sharpens it (more params, same sparse signal). Densification is NOT
optional here:

1. **BC pretrain on the oracle ON + substantial** — the bridge from
   "oracle knows scalps" to "policy initialised toward them." The single
   most important mitigation; the head learns the oracle mapping from
   full obs.
2. **Per-tick credit delivery** — maturation reward + `open_cost` land
   at the OPEN tick / RESOLUTION tick, not smeared across 13k steps by
   GAE. Confirm the §R maturation-reward mode is per-tick, not settle-only.
3. **MTM shaping ON** (`mark_to_market_weight`) — per-tick position
   valuation for density.
4. **Collapse-vs-learn watch:** if even with all three the policy can't
   move held-out maturation/locked, that's the strong negative (signal
   not learnable end-to-end from this data) — stop, don't scale to GA.

## §H — HELD-OUT PROTOCOL (the E7 trap, amplified ×gens)

N gens of selecting on the eval set = N× the overfit pressure that
turned E7's -£34 into -£227 on unseen days.

1. **GA fitness = held-out (May 20-29) LOCKED P&L via `locked_per_std`**,
   NOT day_pnl (naked is zero-EV variance) and NOT in-sample reward.
   mat% is a SECONDARY diagnostic, tracked per gen.
2. **Monitor tripwire ON:** `monitor_eval_top_k` re-evals the top-K each
   gen on the holdout; if a separate iteration metric rises while the
   holdout flattens/regresses → overfit fingerprint; surface and stop.
3. **Final number on the holdout, reported once** with the honest
   fully-hedged (close_walk ON) number — no cherry-picking the best of N.

## §R — REWARD (operator's maturation-only shape; see purpose.md)

1. Positive channel = matured locked P&L + `matured_arb_bonus_weight ×
   (n_matured − expected_random)`. Negative channel = `− open_cost ×
   n_opens`. Naked windfalls 95%-clipped (already default).
2. **`matured_arb_expected_random` MUST be set to the real base-rate
   matured count** (NOT 0 — current base flags zeroed it, which
   re-enables spam-farming of the count term). Calibrate from a base
   recipe's observed mat% × opens.
3. **Naked + force-close CASH must be excluded from the positive
   channel** so the agent isn't paid for directional luck. If existing
   overrides can't isolate "matured-only" raw P&L, add a minimal
   `maturation_reward_mode` env flag (default OFF = byte-identical) —
   scope this in master_todo before launch. The `raw + shaped ≈ total`
   invariant must still hold.
4. **Collapse guard:** under the open_cost toll, if maturation isn't
   predictable the optimal policy is ~zero bets. Track `pairs_opened`
   per gen; a population trending to bets≈0 is the "maturation not
   selectable" signal (consistent with a flat Round T) — stop and
   report, don't burn the full budget.

## §G — GENES (evolve only maturation-relevant knobs)

Evolve: `open_cost`, `matured_arb_bonus_weight`,
`arb_spread_target_lock_pct`, `mature_prob_loss_weight` (+
`mature_prob_open_threshold` if gating), `predictor_p_win_back_threshold`
/ `_max_threshold` (band), `bc_pretrain_steps` (incl. 0), `bc_learning_rate`,
plus core PPO genes (lr, entropy, clip, gae_lambda) already in schema.

Do NOT add `force_close_*` or `close_walk_ticks` to the evolved set (§P).
`expected_random` is a calibrated constant, not a gene (§R.2).

## §C — RESUME / CHECKPOINT (hard prerequisite — currently MISSING)

Verified 2026-05-30: `runner.py` has NO resume support — it builds gen 0
and loops `for generation in range(n_generations)` in-process; a crash
restarts from gen 0, losing the whole multi-day run. The "Saved ~N gens"
logs are early-STOP messages, not checkpoints.

Build before launch (~half-day):
1. At each generation boundary, persist the surviving population —
   per-agent genes + weights_path + fitness — to
   `<output_dir>/gen_<N>_population.json` (weights already saved to the
   registry by `train_one_agent`).
2. Add `--resume-from <output_dir>`: if a `gen_<N>_population.json`
   exists, load the latest, set the generation counter to N+1, and
   continue the loop (skip already-completed gens). Idempotent.
3. Regression test: run 2 gens, kill after gen 1, `--resume-from` →
   completes gen 2 with the gen-1 elites as parents (not fresh-init).

## §B — COMPUTE BUDGET

Naive 100 agents × 10 gens × (3 train + 7 eval days) ≈ ~160 GPU-hours ≈
a week on one GPU. To hit "a few days":
- 64 agents (not 100); eval on a 3-day rotating SAMPLE per gen for
  fitness, full 7-day held-out only for the top-K each gen; use the
  `--batched` worker path. Target ≈ 2-3 days.
- Always `--device cuda` (memory: never inherit CPU on cohort runs).
- Resumable (§C) so a reboot/crash doesn't restart from zero.

## §L — LAUNCH GATE (revised 2026-05-30)

Round T already reported (RED on lean-obs/3-day: mature-gate didn't lift
mat%). This plan removes those two handicaps, so the gate is no longer
Round T — it is the **single-config CANARY** (master_todo step 3): one
promising config at full obs + 42-day train + BC, run to real length on
the holdout BEFORE the multi-agent GA.

- Canary shows held-out learning (locked/mat% moves off the ~5% floor,
  no scale/overfit blowup) → launch the GA horde.
- Canary flat/blowup → do NOT scale to the GA; that's the strong
  negative (full data + full obs + oracle BC still can't reach it) and
  we rethink (features / deep book) with real evidence.

Rationale for the canary gate (operator, 2026-05-30): surface compute,
scale, reward-attribution, and overfit issues on ONE config before
multiplying them across 64 agents × N gens.
