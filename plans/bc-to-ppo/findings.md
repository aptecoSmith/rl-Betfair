# BC → PPO — findings

Reported against held-out LOCKED P&L (§1), holdout split (§2), barrier ON
(§3). Warm-start = `plans/bc-getting-it-right/_scripts/stepB_alltrain_
wd3e-3.pt` (holdout maturation AUC 0.745). Canary:
`plans/bc-to-ppo/_scripts/bc_to_ppo_canary.py`.

---

## Steps 0–2 — done

- **Step 0 (barrier-honest baseline):** the greedy holdout rollout WITH the
  0.50 force-close barrier is byte-identical to without (T=0.30 exact; T=0.20
  day_pnl −£415.32 vs −£417.51, +£2.19 from refusing the one 2.08× outlier).
  Deployment economics confirmed, now barrier-honest.
- **Step 1 (budget):** chose **£100/race** for the canary (deployment-
  realistic; spend data: £100 binds with ~38 blocks/day, peak ~£450–507;
  £100k → £1740 peak). Selectivity, not capital, is the lever.
- **Step 2 (warm-start):** folded into the canary — `DiscreteBCPretrainer`
  trains `actor_head` (+ direction head) on oracle opportunities while the
  `mature_prob_head` stays frozen at AUC 0.745.

## Step 3 — BC→PPO canary: pipeline works; the per-pair economics are the wall

**Pipeline validated end-to-end** (warm-start → BC actor → PPO with
`maturation_reward_mode` + `open_cost` + barrier → holdout eval). Two
cold-start bugs found + fixed along the way:
1. **Device mismatch** — the trainer parks the policy on `rollout_device`
   (cpu) after updates; eval must read the policy's own device.
2. **Stake-head cold-start** — the env decodes `stake = stake_unit ×
   budget`, and the warm-start's UNTRAINED Beta stake head sits at
   `stake_unit ≈ 0.5` → **£50–100 stakes**, whose passive-lay liability is
   too big to post → everything goes naked → force-closes → mat% 0. Fixed
   with `--fixed-stake-unit 0.1` (pins + freezes the stake head to £10/open
   at £100 budget, isolating the OPEN-selectivity question).

**With the stake pinned, the warm-start reproduces the BC result** (canary
v2, ep0 = the warm-start policy's first rollout): **opened 278, mat% 13.3%,
locked +£16.4** at £10/open. So the BC warm-start opens selectively and
locks positive — as bc-getting-it-right found.

**But PPO with `open_cost` COLLAPSES the policy to ~0 opens** (canary v2,
`open_cost=0.1`): ep0 opened 278 → ep1 opened **1** → ep2 opened **0**. The
mechanism is the per-pair economics:
- The locked edge is **~£0.06 per open** (+£16.4 / 278 opens ≈ 0.6% of the
  £10 stake) — the scalp is a thin 2% on the ~13% that mature.
- `open_cost=0.1` charges the ~87% non-matured opens **−£24** vs the
  **+£16.4** locked → opening is net-NEGATIVE in the training reward → PPO
  correctly learns "stop opening."
- Break-even `open_cost` ≈ `locked_per_matured × mat_rate / (1−mat_rate)` ≈
  £0.44 × 0.133 / 0.867 ≈ **0.067**. Any `open_cost` above that collapses
  the policy; any below gives ~no selectivity pressure. A knife-edge.

**The deeper wall (the real finding):** `maturation_reward_mode` removes the
force-close toll from the *training* reward (so PPO *can* be taught to open),
but the **real day_pnl still pays the toll**. At the achievable per-pair
edge (~£0.06 locked/open) and ~13–15% maturation, the force-close toll on the
non-matured majority dominates: canary v2 ep0 locked **+£16.4** but day_pnl
**−£236**. Profitability (day_pnl > 0) needs maturation **≫ 50%** (where
locked clears the toll), but the mature head's best decile tops out at ~38%
(labelled) / ~30% (rollout). **LOCKED-positive is real and reachable;
day_pnl-positive is NOT reachable by selectivity alone** — the toll/edge
ratio is the binding constraint, not the maturation predictor.

**`open_cost=0` ceiling (canary v3, the non-collapsing config):** _[running
— rely on the gate + maturation reward for selectivity, no toll to cause
collapse; confirms whether PPO beats the warm-start and that day_pnl stays
negative]_

## pwin-as-direction probe (operator pivot 2026-05-31) — VERDICT: pwin mispricing does NOT predict maturation

Operator idea (NOT value betting / hold-to-settlement, which is directional
gambling — rejected): use the race-outcome predictor's mispricing
(`champion_p_win` vs market `implied_prob`) as a PRICE-DIRECTION signal — if
the market corrects toward the predictor, the price moves predictably, and a
predictable move is what matures a scalp. Stays pure scalping.

Probe (`pwin_direction_probe.py`, held-out 7 days, 505k candidates — reuses
the maturation dataset, no re-scan). Standalone AUC for predicting back-first
scalp maturation:

| feature | maturation AUC |
|---|---|
| **pwin − implied (mispricing)** | **0.502** (chance) |
| champion_p_win (raw) | 0.500 (chance) |
| dir_fire_shorten / dir_fire_drift | 0.500 / 0.503 (chance) |
| dir_q50_1m | 0.539 |
| **dir_q50_3m** | **0.599** (best single signal) |
| dir_q50_7m | 0.587 |
| (full-feature LightGBM, ref) | 0.745 |

Maturation by mispricing decile is **flat ~12–14%** (most under-priced decile
11.7%, most over-priced 13.1%, base 12.3%) — zero monotonic relationship.

**Reading:**
- **The specific idea is refuted.** The race-outcome mispricing has NO power
  to predict scalp maturation (AUC 0.50, flat). The market does not visibly
  correct toward the race-outcome predictor within our pre-off window (it may
  correct later / in-play — the [T−120, off] window we don't trade).
- **The general intuition is partly right but already captured.** A price-
  *direction* signal does help — the direction predictor's median-move
  quantile `dir_q50_3m` reaches 0.60 — but it's already in the obs and
  already used by the mature head (that's how it got to 0.745). The
  race-outcome pwin adds nothing on top.
- **Nothing reaches the toll-clearing bar.** Even the 0.745 full-feature
  ceiling realises only ~13–15% maturation in deployment; clearing the toll
  needs ~50–60%. The pwin pivot does not move the wall.

### REFRAME (operator reward push, 2026-05-31) — the "wall" verdict below was PREMATURE

The operator pushed back: PPO collapsing opens is a REWARD problem; look at
the reward fresh. That surfaced a lever I'd wrongly treated as fixed:

- **The current reward causes the collapse.** `maturation_reward_mode=full`
  pays ONLY for matured pairs and zeroes the force-close toll → the agent
  never sees the toll; `open_cost` is a flat per-open charge → opening is
  uniformly penalised → PPO rationally quits. The reward taught it to stop.
- **The −£1.25 toll is ~76% a DIRECTIONAL loss from RIDING losers to the
  T-120 force-close** (the actual close-cross is only ~£0.30). **We never cut
  losers.** A stop-loss that cuts a loser mid-race at ~−£0.30 (strict
  matcher, deeper book) drops break-even maturation **~59% → ~25%** — and we
  hit 22–38%. `stop_loss_pnl_threshold` is wired but defaults OFF / never
  used.
- **A clean reward = maximize REAL cash, densified by MTM** (no
  `maturation_reward_mode` distortion, no `open_cost` proxy). The agent then
  feels the loss building (MTM) and learns to cut losers — which the stop-
  loss mechanism enables.

**Stop-loss test (BC policy, 3 holdout days, T=0.30) — fires + helps
modestly, but not a clean flip.** (Units gotcha: `stop_loss_pnl_threshold`
is a FRACTION of stake, not £ — 2026-05-02 operator clarification. First
sweep at 0.5–2.0 = −£5 to −£20 never fired; corrected to 0.02–0.10 = −£0.20
to −£1.0 on a £10 stake.)

| stop (frac → £) | day_pnl | locked | mat% | fc% | sc% |
|---|---|---|---|---|---|
| none (baseline) | −41.85 | +8.75 | 13.2 | 83 | 0 |
| 0.02 (−£0.20) | **−26.81** | +1.01 | 2.9 | 23 | 74 |
| 0.05 (−£0.50) | −42.03 | +3.65 | 9.3 | 37 | — |
| 0.10 (−£1.0) | −35.50 | +8.06 | 12.0 | 52 | — |

Reading: the stop FIRES (sc% 74% at the tight end) and the best config cuts
the loss 36% (−41.85 → −26.81), one day flipped positive — but on the dumb
BC policy it does NOT flip aggregate-positive, and the tradeoff is sharp (the
tight stop caps losers but kills maturation 13%→3%). **Stop-loss alone isn't
the full unlock — but it caps the downside enough that opening is far less
negative-EV, so PPO won't collapse.** Next: the synthesis below.

## Clean-reward + stop-loss PPO canary (the synthesis)

Combines everything: **real-cash reward** (no `maturation_reward_mode`
distortion) + **MTM densification** (agent feels losses building) +
**stop-loss** (caps the downside, so opening isn't punished into collapse) +
**PPO** to open BETTER scalps (the operator's "open better scalps" goal).
Config: budget £100, fixed £10 stake, mature gate 0.30, open_cost 0,
mtm 0.05, stop-loss 0.05.

**RESULT: no early collapse (ep0–3 opened 123–189, ep1 pnl +£14, ep3 +£30 —
the open_cost-specific collapse mechanism is gone), BUT it collapses by ep5
(opened 0, stays 0 through ep17) and the holdout eval opens 0 → day_pnl 0.**
High KL (0.25–0.30) early → erratic, then settles on NOOP.

**Why it still collapses (the now-clear fundamental reason):** PPO correctly
finds opening is **negative-EV even with the clean reward + stop-loss**. The
BC stop-loss sweep already showed the best config is −£0.53/open (−£26.81 /
~50 opens) — still negative. NOOP = 0 beats trading at a per-open loss, so
PPO maximises reward by NOT trading. This is not a reward-shape artifact; it
is PPO finding the true optimum.

**The structural root — fill rate vs the spread (vs commission).** Scalping
earns the spread on passives that FILL and pays it on passives it must
CLOSE. Break-even needs **>50% fill**; we get ~13% (base) to ~38% (best
decile). The reason we can't get above ~38% is **Betfair's 5% commission**:
`min_arb_ticks_for_profit` forces the passive far enough to clear commission,
and at that distance the fill rate is ≤38%. So the passive fills too rarely
to beat the spread — a STRUCTURAL constraint, not a tuning/reward/PPO one.

### RETRACTED (operator correction, 2026-05-31): the "wall" framing was wrong

I reached "this can't work" three times; the operator is right that it's
wrong — people scalp this data profitably daily, so positive-EV trades
demonstrably EXIST. My "opening is −EV" was a sloppy average over the WRONG
trades the current weak policy opens. The job is to **open the RIGHT trades**
— a SELECTION + learning problem, exactly the operator's framing.

**The real blocker is a TRAINING-STABILITY failure, not an economic wall:**
PPO collapses to NOOP, which is an ABSORBING state (no opens → no reward
signal on opens → can't climb back out), so the run dies before it can learn
which trades are the right ones. The early (pre-collapse) episodes already
posted positive days — the signal was right there. Root causes:
`entropy_coeff=0.01` (too weak to hold the policy off deterministic NOOP) +
no reward-clip (the −£196 variance episodes cause violent lurches) + far too
few episodes (18). NONE of that is evidence about the strategy.

**THE CREDIT-ASSIGNMENT MISS (operator question, 2026-05-31) — likely the
real root cause.** Operator: "how does the model know which trades were good?
It can't learn to make better ones if we can't show it the mistakes." Dead
on. By default (and in every canary I ran) the env **lumps all cash P&L onto
the final settle tick** (`per_pair_reward_at_resolution` defaults OFF). So
PPO sees one number per race and CANNOT attribute a loss to the specific open
that caused it — the open is thousands of ticks earlier, and the credit has
to survive GAE across that whole gap (it doesn't). The model was being asked
to open better trades while blind to which of its 200 opens were the bad
ones.

`per_pair_reward_at_resolution=True` fixes this: it credits **each pair's
realised P&L — including force-close LOSSES (the mistakes), as a negative —
at the tick the pair RESOLVES**, per-pair (env line 5087→3516), not lumped at
settle. With the trainer's per-runner reward streams that attributes
"this open → this outcome" far more directly. This plausibly explains the
thrash/collapse as much as the variance does, and it's a `reward_override`
(so it carries into the cohort). Possible further enhancement: back-attribute
to the exact OPEN tick (cleanest signal) if resolution-time isn't enough.

**Operator reward ideas — both implemented + tested in single canaries
(2026-05-31). Correct, but downstream of the open-volume bottleneck.**
- **10× locked (`locked_pnl_reward_weight`, new env override).** Amplifies
  the POSITIVE locked (matured) reward in the shaped channel (losses stay 1×
  real cash). Verified firing (reward ≈ pnl + 9×locked). But the agent opens
  200–400 trades/race at £100 budget → ~95% can't post their passive → go
  naked → almost nothing matures → the locked bonus has nothing to amplify
  (locked stays ~£8) while nakeds drive −£200–400. Collapsed by ep8.
- Every single-agent config tried (entropy 0.01/0.05/0.15; per-pair credit;
  stop-loss; 10× locked) lands on **collapse or flood**. The binding
  constraint is **OPEN VOLUME / selectivity** — open fewer, better trades —
  which depends on balancing entropy × gate × open_cost × budget. That's a
  SEARCH problem (the GA's job), not a single-knob fix.

**→ Pivoted to the GA cohort (autonomous, 2026-05-31).** All our tools are
now genes/pins: `per_pair_reward_at_resolution` (credit fix),
`locked_pnl_reward_weight` (10× locked), `stop_loss_pnl_threshold`,
`reward_clip`, `entropy_coeff`, `mature_prob_open_threshold`, `open_cost`,
`arb_spread_target_lock_pct`. input_norm wired through `worker.py` (foot-gun
grep-verified, line 1265 + stats at 1370). The cohort evolves the
selectivity balance + selects survivors — robust to the single-agent
collapse/flood. See plans/EXPERIMENTS.md for the cohort runs.

**Corrected plan:** stabilise PPO so it keeps trading WHILE it learns —
crank entropy (keep it stochastic / exploring opens), tame reward variance,
run hundreds of episodes — then GA over the genes to SELECT the
right-trade-opening agents. The pwin/commission/fill-rate analysis above is
descriptive of the CURRENT weak policy, NOT a limit on what a well-trained
one can find. (`canary_stable.json`: first stabilised run, entropy 0.15.)

### (SUPERSEDED-PENDING-TEST) VERDICT — the per-pair economics wall is robust; the pwin pivot doesn't move it

The break-even arithmetic, from the deployment rollout (T=0.30): locked
+£0.88/matured vs −£1.25 toll/non-matured → **break-even needs ~59%
maturation**. Best achievable: ~30–38% (mature-head top decile),
22% (favourites, price 1–2 — the best price band), ~13–15% (realised
deployment). A 1.6–2.7× gap that NOTHING in the available signal set closes:

- PPO-selectivity collapses on the `open_cost` knife-edge; without `open_cost`
  it has no lever beyond the gate the BC already provides.
- The pwin race-outcome mispricing is **AUC 0.50** for maturation (refuted).
- The direction predictor's best feature (`dir_q50_3m`, 0.60) is already in
  the obs and already used (→ 0.745 ceiling).
- No price band clears the toll (best = favourites 22%).

**What IS validated (real, keep):** the maturation signal is learnable
(0.745 AUC) and the BC policy opens selectively + locks positive. The
force-close safety barrier is fixed. Scalping at this edge is **locked-
positive but cash-negative** (~−£10/day at £10 stake) — not deployable as-is.

**The wall is the toll-to-edge ratio.** Closing it needs either (a) a
**smaller toll** — deeper book data for better close execution (operator:
off the table) or a different close timing/mechanic (the one untested
lever), or (b) a **bigger per-pair edge** that doesn't take settlement risk
(none found). This is a strategic decision point for the operator — NOT a
tuning problem. Step 4 (GA cohort) should NOT run; it can't fix a signal the
data doesn't contain (memory `feedback_ga_selection_vs_reward_shaping`).
