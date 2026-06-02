# Imitation-first — master TODO (executable, 3 steps)

## STATUS (2026-05-30) — diagnostic gates DONE, all PASS; Step 2 is the remaining work

See `findings.md` for full numbers. Summary:
- **Step 0 PASS→0.5:** hindsight oracle bleeds −£474/day holdout, 88.7% fc
  (§8b confirmed — labels "spread placeable" not "will mature").
- **Step 0.5 PASS:** maturation-conditioned oracle (NEW: `scan_day(
  maturation_conditioned=...)` + env-faithful passive-fill forward-walk,
  `training_v2/arb_oracle.py`; +7 tests). Holdout transformed to
  ~breakeven (+£559 locked, fc 89%→31%).
- **Step 1b:** BLOCKER — full obs unnormalized (max 190k) + policy has NO
  input-norm on the actor path. A BC-only probe would be confounded.
- **Step 1 PASS (via LightGBM proxy, confound-free):** maturation is
  predictable from features — **holdout AUC 0.76, top-decile lift 2.45x**.
- **Opportunity exists AND is learnable AND lift is economically
  meaningful → proceed to Step 2.**

**Remaining = Step 2** (reward-aware selective policy). First sub-task is
the policy input-norm UNBLOCKER (architecture change — needs sign-off).
See `findings.md` "Path forward" for the validated recipe. The 42-day
maturation-conditioned full-obs+predictor caches are scanned and ready.

Artifacts: `_step0/` (oracle-eval harness + results), `_step1/`
(obs_audit, predictability probe + results), `findings.md`.

---

**START HERE (fresh session):** read `purpose.md` + `hard_constraints.md`
first. Then do **Step 1a (oracle-cache rebuild)** and **Step 0** — they're
the cheapest, most decisive gates and need no training. Do NOT jump to a
GA. Each step gates the next.

GPU: default `--device cuda` (memory: never inherit CPU on cohort runs).

---

## Step 0 — Oracle's own held-out P&L (the ceiling / opportunity check)

**Question:** does the hindsight-optimal oracle make money on the 7 UNSEEN
holdout days, after commission, with the real matcher + fc=120 +
close_walk=10? It's the upper bound on anything that imitates it.

**Why first:** cheapest possible (no training), highest information. If
even the oracle isn't profitable out-of-sample, nothing downstream can be,
and we rethink before spending GPU.

**Actions:**
- [ ] Scan the oracle for the **7 holdout days** at full-obs dim
  (`scan_day` — deterministic, NOT a retrain; §8). Eval-only, no leak.
  Inject predictors so obs match (§8). See how `bc_pretrain` /
  `load_oracle_samples_for_dates` resolve cache paths.
- [ ] Build a small **oracle-as-policy eval harness**: for each holdout
  day, take the oracle's labeled opens (`OracleSample`: tick, runner,
  side, `arb_spread_ticks`), execute them through `BetfairEnv`
  (scalping_mode, fc=120, close_walk=10, commission 0.05), let them
  mature/force-close/settle, and sum **locked / day_pnl / mat%**. Reuse
  the env's settle accounting — do NOT recompute P&L by hand (commission
  + matcher must be honest). Check for an existing oracle-eval harness
  before building (grep `arb_oracle` callers; the BC path loads samples
  but may not *execute* them).
- [ ] Report per-day + aggregate locked/day_pnl/mat% on the holdout.

Also report the **force-close / naked rate** of the oracle's labels — per
§8b the current oracle labels "spread placeable," NOT "will mature," so a
high fc-rate here is the smoking gun that the oracle (not the policy) is
the problem.

**GATE:**
- Oracle profitable on holdout (LOCKED > 0, **low fc-rate**, sane mat%) →
  opportunity confirmed AND the oracle's labels mature → **Step 1.**
- Oracle bleeds with a **high force-close rate** → the oracle is labelling
  non-maturing opens (§8b) → **do Step 0.5 (redesign), then re-run Step 0.**
- Oracle bleeds even with low fc-rate → matcher/commission eats the edge,
  or the opportunity isn't in the unseen window → STOP, re-examine. (Major,
  cheap finding either way.)

---

## Step 0.5 — Maturation-conditioned oracle redesign (likely prerequisite)

**Trigger:** Step 0 shows the current oracle's labels force-close heavily
(expected per §8b).

**Change:** in `scan_day`, after a candidate (tick, runner) passes the
placeable-spread checks (Steps 1-8), **forward-walk the actual future
ticks and emit the OPEN label ONLY if the passive lay would fill before
fc=120** (the close/hold augmentation already forward-walks — reuse that
machinery). Optionally store the realised matured locked P&L on the
sample. Result: a true "will-mature" oracle whose OPEN labels are the
scalps that actually complete, not every placeable spread.

**Validate:** re-run Step 0 on the redesigned oracle — fc-rate should
collapse, LOCKED should be cleanly positive. THEN Step 1 BC has a target
worth imitating. Keep the change behind a flag so the old
"spread-placeable" oracle stays reproducible for comparison.

---

## Step 1 — BC to convergence (the learnability test, sparsity-free)

**Question:** is the oracle's behavior LEARNABLE from observable features?
Train the policy purely by behavioural cloning on the full oracle dataset,
full obs, 42 days, to convergence — eval directly on holdout, NO PPO.
Dense supervised gradient on every labeled decision → the sparse-reward
problem disappears entirely.

**Actions:**
- [ ] **Step 1a — scan oracle + build feature caches at full-obs across
  all 42 train days** (the long pole; do this first, it also serves Step
  2). Deterministic scan, NOT a retrain (§8); inject predictors so obs
  match. Verify obs_dim is the full 143-d/runner, not 23-d lean.
- [ ] **Step 1b — value-domain audit of full obs** (§2 / memory
  `feedback_feature_engineering_diagnostics`): build one env at full obs,
  `head(30)` a named runner's per-runner block + z-score min/max across
  all 143 dims. Catch any unnormalized/leaky/degenerate dim before training.
- [ ] **Step 1c — BC-to-convergence runner.** Extend
  `training_v2/discrete_ppo/bc_pretrain.py` (currently a ~500-step
  warm-start) to: train to convergence on the 42-day oracle set, with a
  BC-validation split (hold ~2 train days out of BC for early-stop), full
  obs, LSTM h256. Save weights. Standalone — does NOT hand off to PPO.
- [ ] **Step 1d — eval the BC'd policy on the 7 holdout days** via env
  rollout (the cohort eval path / `RolloutCollector`), fc=120,
  close_walk=10. Report held-out locked / day_pnl / mat% / fc%.
  Compare to Step 0's ceiling.

**GATE:**
- BC'd policy moves held-out locked/mat% materially off the ~5% / −£78
  floor (ideally approaching Step 0's ceiling) → behavior IS learnable
  from obs → **proceed to Step 2** (close the distribution-shift gap).
- Flat (≈ campaign baseline) → the oracle's decisions are NOT predictable
  from current features (consistent with direction AUC 0.70 + Round T).
  Strong, cheap negative → the real unlock is richer data (deeper book,
  the StreamRecorder 10-level work, `findings.md` Book-depth investigation).
  Do NOT proceed to a full GA.

**Caveats (hard_constraints §7):** hindsight gap (BC ≤ what's predictable
from obs); distribution shift (BC imitates oracle-visited states, eval
visits the policy's own) — that gap is exactly Step 2's job.

---

## Step 2 — Reward-aware polish (ONLY if Step 1 promises)

**Question:** Step 1's BC policy works on oracle-visited states but
compounds errors on its own. Add reward-awareness to fix the gap. PPO's
role here is **polish, not primary discovery** — the sparsity it suffers
is now bounded because BC already put the policy in the right region.

**Prerequisites (build before any run):**
- [ ] **`maturation_reward_mode` env flag + settle-wiring** — gather
  per-pair outcomes in `_settle_current_race`, classify
  matured/agent_closed/force/stop/naked, and when on set `race_reward_pnl
  = maturation_only_reward(pair_outcomes)` (pure helper already built+tested
  in `env/betfair_env.py`, `tests/test_maturation_reward.py`). Default OFF
  = byte-identical. Don't disturb existing accumulators / the raw+shaped
  invariant. Add to `_REWARD_OVERRIDE_KEYS`.
- [ ] **Wire the maturation reward through `per_pair_reward_at_resolution`**
  (already exists) so the payoff lands at the maturation tick, not settle.
- [ ] Integration test: maturation_reward_mode ON → raw == matured-locked
  + agent-close-profit; force/naked → 0 raw.

**Routes (pick per Step 1's result; cheapest first):**
- [ ] **2A — BC→PPO fine-tune (DAgger-flavoured).** Warm-start from Step 1
  BC weights, then short PPO with the maturation reward (per-pair-at-
  resolution + open_cost + MTM), full obs, 42/7 split, fc=120 +
  close_walk=10 pinned, `--composite-score-mode locked_per_std`. This is
  the **`plans/ga-recipe-search/` vehicle** — its hard_constraints (data,
  obs, reward, selection, resume) apply; use the auto-resume wrapper.
  Start single-config (the canary in ga-recipe-search master_todo Step 3)
  before any multi-agent GA horde.
- [ ] **2B — offline RL** (IQL / AWAC / CQL) or **Decision Transformer**
  (transformer + offline return-conditioning — fuses the architecture and
  no-online-exploration ideas). More build; only if 2A's online fine-tune
  still fights sparsity. Not scaffolded — design when reached.

**GATE:** held-out locked positive, fully-hedged, on the reserved 7 days,
reported once → deployable candidate. Else iterate route/features.

---

## Build state (carry-over, this session 2026-05-30)

**Done + tested:** close-walk (`close_walk_ticks`), GA resume
(`--resume-from`), `maturation_only_reward` pure helper, `mature_prob`
open-gate. (Round T showed the gate flat on lean-obs/3-day — this plan
removes both handicaps.)

**TODO:** Step 1a cache rebuild · Step 0 oracle-eval harness · Step 1c BC-
to-convergence runner + 1d holdout eval · Step 2 maturation_reward_mode
settle-wiring + per_pair_reward_at_resolution wiring + 2A/2B.

## Verification (every run)
- [ ] obs_dim == full 143-d/runner (NOT 23-d lean); train ∩ holdout = ∅.
- [ ] Holdout reported once, fully-hedged (close_walk ON).
- [ ] For any PPO: select `locked_per_std`, not `total_reward`.
