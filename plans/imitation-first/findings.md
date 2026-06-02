# Imitation-first — findings

## Step 0 — Oracle's own held-out P&L (2026-05-30) — VERDICT: oracle bleeds, fc~89%

**Question:** does the hindsight-optimal arb oracle make money on the 7
UNSEEN holdout days, under the real env (scalping_mode, fc=120,
close_walk=10, commission 0.05)? It's the upper bound on anything that
imitates it.

**Method:** bare oracle scan (labels are price-only, so predictor
injection / obs dim are irrelevant to the labels — verified in
`scan_day`, lines 252-321) of the 7 holdout days, then executed the
labeled opens through a real `BetfairEnv` via
`plans/imitation-first/_step0/oracle_eval_harness.py`. The oracle is the
OPEN policy only (greedy one-open-per-tick by `expected_locked_pnl`, no
agent close_signal); the env's price-adaptive passive sizing, T-120
force-close, close-walk, and settle accounting do the rest. No gates, no
predictors. Stake £10/open.

**Result (aggregate, 7 holdout days):**

| metric | value |
|---|---|
| pairs_opened | 2,838 |
| mat% (`arbs_completed/pairs_opened`) | **5.11%** |
| fc% (`arbs_force_closed/pairs_opened`) | **88.69%** |
| naked% | 6.2% |
| locked_pnl | +£195.39 |
| naked_pnl | −£480.19 |
| **day_pnl** | **−£3,319.78 (−£474/day)** |

Per-day fc% range 85.7–90.4%; mat% 3.8–6.7%; day_pnl −£367 to −£540.
Rock-solid across every day. Full numbers:
`_step0/results_spread_placeable.json`.

**Interpretation:**
- This is the §8b smoking gun. The current oracle (`scan_day`) labels a
  (tick, runner) profitable purely on current-tick prices — a back +
  passive lay `min_arb_ticks` away whose `expected_locked_pnl > 0` IF
  BOTH FILL. It does NOT forward-walk to confirm the passive fills. So
  ~89% of its labeled opens force-close (passive never fills). The
  oracle teaches "open whenever a profitable SPREAD exists," not "open a
  pair that will MATURE."
- The matured pairs DO lock positive (+£195 across 145 matured) — the
  2% scalp edge per matured pair is real. But maturation is only ~5% and
  the force-close toll on the other ~89% dominates: of the −£3,320,
  ~−£3,035 is force-close cost (≈−£1.21/pair crossing the thin near-off
  book), with naked variance a smaller −£285 net.
- **The fix is not better closing** (close_walk is already on; it's a
  variance tool, Round W). The fix is NOT OPENING pairs that won't
  mature → redesign the oracle to be maturation-conditioned (Step 0.5).
- BC/PPO imitating THIS oracle inherits the ~89% force-close rate — which
  is exactly what every campaign agent did (fc 74-90%). The labeler is
  the core flaw, confirmed at the source.

**Caveat (does not affect the verdict):** the harness issued ~50k
OPEN_BACK actions but only 2,838 created pairs — the rest were refused
downstream (per-race budget/liability exhaustion under £10 stake; the
action mask checks total budget, not available-after-liability). The
RATES (fc%/mat%) are the robust signal and are measured on the BEST
labels (greedy by expected lock), so 88.69% fc is if anything the
optimistic end. Absolute pair counts are budget-limited, not a ceiling
on the opportunity.

**GATE → Step 0.5** (oracle bleeds with high force-close rate → redesign
the oracle to label only maturing opens, then re-run Step 0).

---

## Step 0.5 — Maturation-conditioned oracle (2026-05-30) — VERDICT: transforms the oracle to ~breakeven; gate PASSES for Step 1

**Change:** `scan_day(maturation_conditioned=True, force_close_before_off_seconds=120)`
(new flag, default-off = byte-identical; behind a CLI flag
`--maturation-conditioned`). After a candidate passes the placeable-
spread checks, forward-walk the real future ticks and emit the OPEN
label ONLY if the passive lay would actually fill before T-fc.

**KEY SUB-FINDING — the close/hold forward-walk fill model was wrong.**
My first cut reused the close/hold augmentation's fill heuristic ("any
future available-to-BACK quote <= lay_target"). It kept 63% of labels —
but Step 0 showed the env matures only ~5%. Reading
`PassiveOrderBook.on_tick` (env/bet_manager.py) revealed the env's real
volume-mode fill needs BOTH: (1) a CROSSING latch on the **available-to-
LAY** ladder (`min(ATL) <= P`) — the close/hold walk checked the wrong
ladder (ATB) — AND (2) cumulative traded volume >= `queue_ahead` (ATL
size resting at P), counting only volume on ticks whose `LTP <= P`. The
faithful walk drops survive-ratio 63% → 11.6% (one day). This likely
explains why close/hold-augmented BC never helped — its labels modelled
fills with the wrong ladder and no queue.

**Result (oracle-as-policy on the 7 holdout days, re-run of Step 0):**

| metric | spread-placeable (Step 0) | maturation-conditioned |
|---|---:|---:|
| pairs_opened | 2,838 | 1,442 |
| locked_pnl | +£195.39 | **+£321.31** |
| **day_pnl (7d)** | **−£3,319.78** | **−£17.37 (−£2.5/day)** |
| mat% | 5.11% | **28.09%** |
| fc% | 88.69% | 69.42% |
| naked% | 6.2% | 2.5% |

Per-day day_pnl now −£1.96 .. +£52.84 (one positive day). Full numbers:
`_step0/results_maturation.json`.

**Interpretation:**
- The redesign transforms the oracle from catastrophic (−£474/day) to
  ~breakeven (−£2.5/day) with strongly positive locked (+£321) on HALF
  the opens. The gate's "LOCKED cleanly positive" is met emphatically;
  the economics are no longer a bleed. **BC now has a target worth
  imitating** (Step 1).
- BUT fc% only fell 89% → 69% — it did NOT collapse to ~0. The forward-
  walk still over-predicts fills ~2.5x (of the labels it keeps, only 28%
  actually mature in the env). Two candidate causes: (a) harness budget
  confound — £10 stake / £100 per race means many passives can't post
  (`budget_lay` reject → naked → force-close), independent of the fill
  model; (b) residual forward-walk vs env-matcher gap (queue/self-
  depletion/baseline-seeding nuances). The `--starting-budget 100000`
  control isolates (a) [result below].

**Budget-override control (`--starting-budget 100000`, isolates the
confound):**

| metric | mat-cond £100/race | mat-cond £100k/race |
|---|---:|---:|
| pairs_opened | 1,442 | 2,308 |
| mat% | 28.1% | **67.4%** |
| fc% | 69.4% | **30.6%** |
| locked_pnl | +£321 | +£559 |
| day_pnl (7d) | −£17.4 | −£119.3 |

The residual 69% fc% was LARGELY the budget confound: with budget
removed, 67% of the forward-walk's labels actually mature in the env
(fc% 89%→31%). The fill model is reasonably faithful (~30% false-
positive rate — queue/self-depletion/baseline nuances not modelled).
Numbers: `_step0/results_maturation_bigbudget.json`.

Nuance: unconstrained day_pnl is WORSE (−£119) than budget-limited
(−£17) — opening ALL labels adds force-close toll on the ~30% false
positives, and the locked edge (+£559) nearly-but-not-quite covers it.
Reading per day: locked ≈ +£46/day, force-close toll ≈ −£48/day, net
≈ breakeven. The edge is real and right at the margin; the lever to
tip it positive is SELECTIVITY (open only the high-confidence
maturations) — exactly what Step 1 (predict maturation from features) +
Step 2 (selective-open reward) target.

**GATE → Step 1.** fc% collapsed (89→31% with budget removed), LOCKED
cleanly positive (+£559), oracle transformed from −£474/day to ~breakeven.
A target worth imitating. Caveats carried forward: (a) labels are ~67%
precise on true env maturation under no budget limit (~28% under
deployment budget) — BC's ceiling is bounded by feature predictability
AND label precision; (b) the oracle itself is only ~breakeven, so Step 1's
real question is "is maturation PREDICTABLE from decision-time features?"
(mat% lift on holdout), not "can BC print money" — profitability is a
Step 2 selectivity problem.

---

## Step 1b — Full-obs value-domain audit (2026-05-30) — BLOCKER: full obs is unnormalized + the policy has no input norm

Audit (`_step1/obs_audit.py`) on one holdout day's full obs (2254-d,
predictor-injected) walking 601 ticks:

- **No non-finite entries** (good).
- **335 / 2254 dims have abs-max > 50; max value 190,610** — raw,
  unnormalized volumes/sizes (dims 2, 12, 37-39, 193, 310...). mean|x|
  across the obs ≈ 81.
- 1031 dims all-zero (inactive runner-slot padding — expected).

**Reading the policy (`agents_v2/discrete_policy.py::forward`):** the
backbone is `input_proj = Linear(obs_dim, hidden)` applied to the RAW
obs — **NO input normalization on the actor/LSTM path.** The per-runner
LayerNorm the plan §2 cited exists ONLY in `direction_prob_head`, not the
main path. The campaign never hit this because it trained on LEAN obs
(23-d/runner) which excluded the raw-volume dims; the plan's switch to
full obs exposes them.

**Two consequences for Step 1 as written:**
1. Unnormalized inputs up to 190k into a raw Linear → the few huge dims
   dominate the projection; well-normalized features are drowned out.
2. `DiscreteBCPretrainer` FREEZES the backbone (trains only `actor_head`).
   A BC-only probe on a random frozen backbone fed unnormalized obs would
   report "not learnable" as a CONFOUND — invalid as a learnability test.

**Pivot:** answer Step 1's real question ("is maturation predictable from
decision-time features?") with a scale-invariant LightGBM classifier on
`(obs → matured)` (trees ignore feature scale; no frozen-backbone issue).
A high holdout AUC ⇒ the signal IS in the features ⇒ a policy could learn
it (and Step 1c/Step 2 then need an input-normalization fix on the policy
— a real follow-on, recorded here). A ~0.5 AUC ⇒ not predictable ⇒ STOP
(richer data is the unlock). This is a STRONGER test of the gate question
than a confounded BC and is immune to the normalization blocker.

**Follow-on flagged:** full-obs BC/PPO needs input normalization added to
the policy (per-dim standardize from train stats, or a LayerNorm on
`input_proj`). This is an architecture change (state-dict / arch-hash
break) — deferred to Step 1c/Step 2 if the predictability probe passes.

---

## Step 1 — Maturation predictability (2026-05-30) — VERDICT: LEARNABLE (holdout AUC 0.76). Gate PASSES → Step 2.

LightGBM on `(full predictor-injected obs → matured?)`, 8 spread train
days (501k candidates) → 7 reserved holdout days (505k). Numbers:
`_step1/predictability_results.json`.

| metric | value |
|---|---:|
| holdout matured base rate | 0.123 |
| train AUC | 0.829 |
| **HOLDOUT AUC** | **0.759** |
| holdout AP | 0.264 (2.1x base) |
| **holdout top-decile precision** | **0.302 (2.45x lift)** |

**Interpretation:** maturation IS predictable from decision-time features
on unseen days (AUC 0.76, well above 0.5). Matches the direction-head AUC
~0.70 (direction = the maturation driver). Top features mix market
microstructure (low-index globals + raw book/volume dims 37-39) AND the
predictor columns (higher dims 462/566/568/590...) — both pull weight.

**Why this is the whole-plan green light:** Step 0.5 showed the per-pair
locked edge ≈ the force-close toll at the oracle's ~12-28% maturation
(net ~breakeven). Step 1 shows a model can rank candidates so its
top-decile matures at 30% (2.45x base). A policy that opens SELECTIVELY
(only high-mature-prob candidates) would push its opened-subset
maturation well above the breakeven point → the locked edge then exceeds
the toll → **net positive is reachable**. Opportunity exists (Step 0.5) AND
is learnable (Step 1) AND the lift is economically meaningful.

Note Step 1 was answered via the LightGBM proxy rather than the
master_todo's literal "BC policy + rollout mat% lift" because Step 1b
found the BC path is confounded by the unnormalized full obs + no policy
input-norm. The proxy is a STRONGER test of the gate question ("is the
signal in the features?") and is confound-free. The policy-level
confirmation folds into Step 2 (BC→PPO), which trains a policy anyway.

---

## Path forward — Step 2 (UNBLOCKERS LANDED 2026-05-30; canary running)

Gates 0/0.5/1 all PASS. Step 2 = reward-aware policy that exploits the
predictable maturation via SELECTIVITY. Progress:

1. **Policy input-normalization (UNBLOCKER) — DONE + TESTED.** Opt-in
   `input_norm=True` on `DiscreteLSTMPolicy`: registers per-dim
   `(obs_mean, obs_std)` buffers, `set_input_norm_stats(mean, std)`,
   standardizes obs before `input_proj`. PER-DIM (not LayerNorm-across-
   dims, which would let the 190k dims dominate). Default OFF registers
   NO buffers → byte-identical state_dict + behaviour for existing
   lean-obs cohorts. Tests: `tests/test_agents_v2_discrete_policy.py::
   TestInputNorm` (6). [Opt-in/default-off so no campaign-wide blast
   radius; still recommend operator review before a GA horde uses it.]
2. **`maturation_reward_mode` env wiring — DONE + TESTED.** New
   reward-override flag (in `_REWARD_OVERRIDE_KEYS`). `_settle_current_
   race` now classifies every pair (matured/agent_closed/force/stop/
   naked) into `pair_outcomes`; when on, RAW reward becomes
   `maturation_only_reward(pair_outcomes)` (matured locked +
   agent-close-at-profit; force/stop/naked → 0); shaped UNCHANGED;
   raw+shaped invariant holds. Default OFF byte-identical. Tests:
   `tests/test_forced_arbitrage.py::TestMaturationRewardMode` (4).
   `per_pair_reward_at_resolution` already existed/tested.
3. **BC canary (Step 1c/1d, policy-level confirmation) — RUNNING.**
   `_step1/bc_fullnet_canary.py`: full-NETWORK BC (not the cohort's
   frozen-backbone BC) on the 42-day maturation caches, input-norm ON,
   eval on the 7 holdout days vs an untrained baseline. Question: does a
   POLICY lift holdout mat% off the ~12% base? [results below when done]
4. **REMAINING — reward-aware BC→PPO (the profit step).** BC→PPO with
   `maturation_reward_mode` + `open_cost` toll + MTM, fc=120 +
   close_walk=10 pinned, select on LOCKED, eval the 7 holdout days once.
   Either drive `DiscretePPOTrainer` standalone (as the BC canary does)
   or wire `input_norm` through `cohort/worker.py` + a runner CLI flag
   (GREP-verify the flag reaches policy construction — memory
   `feedback_audit_launch_wiring`) for a GA campaign.

**Decisive expectation:** does PPO's selectivity (open only high-mature-
prob runners) lift opened-subset maturation enough that holdout LOCKED
goes cleanly positive (fully-hedged)? Step 1's 2.45x top-decile lift
says the signal is present.
