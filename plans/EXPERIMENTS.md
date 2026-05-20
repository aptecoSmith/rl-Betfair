# Experiments

Chronological digest of launched cohort experiments and their verdicts.
Companion to `EXPLORATIONS.md` (which records analytical discussions
and strategic reasoning).

Per-entry structure:
- **Date(s)** as the H2 line
- **Plan name** as a slug — the plans/<dir>/ folder where the run lives
- **Intention** — what the experiment was meant to find out and why
- **Implementation brief** — the cohort config + new machinery that
  landed for it
- **Result** — held-out outcome, band verdict if any, and what the
  next plan inherited from it

Append-only. Don't rewrite historical entries; add new ones below.

---

## 2026-04-17 — scalping-close-signal

**Intention.** The pre-plan scalping env had no way for an agent to
exit an open pair before settle. Once both legs filled the trade
locked; once only one leg filled the pair rode to settle naked,
absorbing whatever directional move happened in the rest of the race.
The plan added a discrete `close_signal` head to the policy so the
agent could elect to close a pair at the current spread, converting
naked-leg variance into bounded spread cost.

**Implementation brief.** New action head emits a per-pair close
intent each tick. When set, the env places the opposite-side leg at
`equal_profit_lay_stake` / `equal_profit_back_stake` from
`env/scalping_math.py` — same equal-profit sizing used for the
auto-paired passive — and routes the pair through `_attempt_close`
with the strict matcher (LTP required, ±50 % junk filter). A
+£1 shaped bonus per successful close was added so PPO had a signal
distinct from the pair-level cash P&L. `Bet.force_close=False` for
agent-initiated closes; `scalping_closed_pnl` becomes a distinct
accumulator from the locked floor.

**Result.** Foundational change; not benchmarked as a standalone cohort
against a null. The infrastructure shipped clean and became the
substrate every subsequent scalping plan ran on. The 2026-04-21
force-close-at-T−N work was layered on top of this so the env could
also bail out pairs the agent left naked into the off; that lives in
`CLAUDE.md` "Force-close at T−N".

---

## 2026-04-25 — selective-open-shaping (cohort-O, cohort-O2)

**Intention.** Force-close costs were dominating per-race cash flow
(~£90/d per agent at deployment fc=120) yet the policy showed no
sensitivity to opening selectivity — agents opened ~620 pairs per
race and let ~77 % of them die at T−120. The hypothesis: charge a
small `open_cost` at the moment of decision and refund it iff the
pair resolves favourably, creating a per-tick gradient on "should
I open this pair?".

**Implementation brief.** New `open_cost` gene in `[0.0, 2.0]`. The
charge `-open_cost` lands on the OPEN tick; the refund `+open_cost`
lands on the resolution tick iff the pair matured or agent-closed.
Force-closed and naked outcomes leave the charge in place. Session 02
moved the credit assignment from settle-time to per-tick after the
cohort-O probe showed agents at gene values 0.06–0.83 had identical
76–77 % fc rates — GAE was smearing the per-race delta across 5,000
ticks. Eight regression tests in `tests/test_forced_arbitrage.py`
guard the zero-mean, charge/refund, and raw-untouched invariants.

**Result.** Cohort-O ρ(open_cost, fc_rate) = +0.055; cohort-O2 (with
matured_arb_bonus pinned to 0) = +0.314. The mechanism delivered its
gradient cleanly but the policy could not respond on the dimension we
wanted. Root cause traced in `plans/per-runner-credit/findings.md`:
the actor head consumed `(runner_emb_i, lstm_output)` only — no
per-runner forecast in the action's input. The policy had no
representational pathway to express "this runner's open will likely
fail to mature". Led directly to fill-prob-in-actor (2026-04-26) and
mature_prob-in-actor (2026-04-26).

---

## 2026-05-12 — scalping-pwin-gate

**Intention.** First plan to couple the trained price-direction
predictor into the scalping action mask. Hypothesis: most of the
naked-leg variance comes from races where the predictor has no
opinion (max p_win mid-band); gating per-runner on a confidence
threshold should remove the low-EV trades and leave the cleaner
ones.

**Implementation brief.** Hard action-mask gate on per-runner
`predictor_p_win`. Back side: only runners with `p_win ≥
predictor_p_win_back_threshold`. Lay side: only runners with
`p_win ≤ predictor_p_win_lay_threshold`. Cohort trained at
back_threshold=0.20, lay_threshold=0.40.

**Result.** 3/5 top-5 profitable on the 3-day held-out window;
mean −£13/day, median +£48/day. Locked floor +£75–£104/day held up
in-sample and out-of-sample but the naked channel still swung
±£200/day across the held-out 5. The result was "Modest improvement"
on locked but "No improvement" on day_pnl — the gate trimmed
race count without fixing the naked-side dispersion. Direct
predecessor to scalping-race-confidence-gate, which stacked a
per-race filter on top of the per-runner gate.

---

## 2026-05-13 — scalping-race-confidence-gate (MODEST)

**Intention.** pwin-gate filtered per-runner but admitted races
where every runner was mid-band — high uncertainty everywhere. Add
a per-race gate `max(p_win) ≥ race_confidence_threshold` so we only
trade races where the predictor sees at least one clear favourite.

**Implementation brief.** New env kwarg `race_confidence_threshold`,
initially locked at 0.30, raised to 0.50 mid-plan after the
2026-05-13 smoke FAIL. Stacked on top of pwin-gate's per-runner
thresholds (back=0.20, lay=0.40). Cohort tag
`_predictor_SCALPING_raceconf_1778661062`.

**Result.** Held-out 2026-04-28/29/30: 3/5 top-5 profitable, mean
+£39.40/day, median +£92.61/day. Modest success. Locked floor
+£87.91/day across the 5 (range +£77–£103). Naked channel still
dispersed −£197 to +£54 — the gate took out un-tradeable races
without strangling pair completion (maturation rate held at ~0.34)
but didn't fix naked variance. Findings noted explicitly that this
cohort trades the "lay-outsiders at price ~43" phenotype, while
the layq successor would trade "lay-favourites at price ~7" — the
two plans extract the predictor's edge from complementary parts of
the price ladder. See memory entry
`project_two_cohort_diversification.md`.

---

## 2026-05-13–14 — scalping-lay-quality-gate (STRONG)

**Intention.** The raceconf phenotype analysis surfaced a recurring
naked-loss pattern: agents lay-laying outsiders at very high prices
(>20) caught huge tail losses when the outsider went off favourite.
Hypothesis: tighten the lay-side pwin threshold and add a hard
lay-price cap to remove the structurally negative-EV lay region.

**Implementation brief.** Tightened `predictor_p_win_lay_threshold`
0.40 → 0.20. Added new env kwarg `lay_price_max = 20` (refused at
the action mask). Cohort tag `_predictor_SCALPING_layq_1778712871`.
Pre-launch held-out probe predicted +£0.098/£ EV; the cap also
dropped the >50 price bucket which had been fragile +EV in earlier
data.

**Result.** Strong success on the 3-day fc=0 held-out window: 5/5
top-5 profitable, mean **+£192.53/day** vs raceconf's +£39.40 — a 5×
improvement. Locked +£114, naked +£97. fc=120 deploy-realistic
numbers: 3/5 profitable, mean **+£26/day**, with ~£69/day eaten
by force-close costs because training ran at fc=0. This became the
explicit baseline for all subsequent plans — the "+£193 fc=0 / +£26
fc=120 layq null" referenced across `feedback_naked_variance_*`
and `project_force_close_train_vs_deploy.md`. The train-vs-deploy
fc asymmetry surfaced here is the central problem the
tight-naked-variance plan tried to attack next.

---

## 2026-05-14–16 — scalping-tight-naked-variance Phase 2A (tnv)

**Intention.** layq cleared Strong on fc=0 but only Modest on fc=120
because agents trained at fc=0 over-opened speculative late-pre-off
pairs they expected to ride to settle. Hypothesis: add a per-pair
naked-variance penalty on the shaped channel and a composite GA
selection score that values low naked variance, so the GA surfaces
agents that don't accumulate variance in the first place.

**Implementation brief.** Two new pieces of machinery (commits
`c1c5f19` + `5b7f3da`):
- Per-agent gene `naked_variance_penalty_beta ∈ [0, 0.05]` — L2
  symmetric per-pair penalty on the shaped channel.
- `composite_score_mode=tight_variance`:
  `locked − 0.5·σ_naked + 0.25·naked_mean` as the GA selector.

96 agents × 8 generations on the raceconf gate, 6 training days,
fc=0 in training (operator deferred fc=120 in training to a separate
ablation). Cohort tag `_predictor_SCALPING_tnv_raceconf_1778852093`.

**Result.** No band cleared. Best deployment candidate `32ed9e32`
(gen 0, β=0.00133) hit −£0.76/day at fc=120 new window with naked_std
£101 (Modest's ceiling) and 4/7 profitable days — break-even but
below Modest's ≥+£50/day bar. The fc=0 new-window cell beat the
layq null by +£30.77/day (−£9.73 vs −£40.50); fc=120 cells tied.
Variance-penalty mechanism CONFIRMED to produce selection pressure
in-sample — max naked span compressed 1216 → 658 across gens 0–3
(46 % reduction) and β_med rose from 0.016 to 0.030. But the
train-vs-deploy fc asymmetry was NOT resolved because fc=120 was
held out of training. Three follow-on recipes documented in
`findings.md`; the operator picked Option A (re-run with fc=120 in
training) which became tnv2.

---

## 2026-05-16–17 — tnv2 (REGRESSION on layq null)

**Intention.** Same machinery as tnv (variance penalty +
tight_variance composite) but train at fc=120 to close the
train-vs-deploy gap that limited tnv. Switched the selector to
`locked_per_std = locked_pnl / (1 + naked_std_daily)` so the
metric never reads naked-sign and rewards stable locked floor.

**Implementation brief.** Cohort tag
`_predictor_SCALPING_tnv2_raceconf_1778943297`. 12 agents × 8 gens
× raceconf gate, fc=120 in training. 10 in-sample-eval days (up
from 6 in tnv1, tighter σ estimate). Stopped early at 67/96 agents
when in-sample distribution made the verdict clear (only 1/67
agents in-sample positive PnL, and that one caught a naked
tailwind). Held-out reeval: 10 agents top-by-in-sample-day_pnl ×
2 windows × 2 fc settings = 4 JSONLs.

**Result.** **REGRESSION across all 4 cells** on the layq null:
- fc=0 newwindow (7d): mean −£50.53/d (vs null −£40.50, −£10/d worse), 3/10 prof
- fc=0 oldwindow (3d): mean −£201.73/d, 1/10 prof
- fc=120 newwindow (7d): **mean −£176.89/d** (vs null −£17, **−£160/d worse**), 0/10 prof
- fc=120 oldwindow (3d): mean −£210.02/d, 0/10 prof

The locked floor was actually HIGHER than the layq predecessor
(+£198/d fc=120 vs layq's +£122) but force-close cost climbed
3.6× (−£251/d vs −£69/d) because training at fc=120 with a
locked-rewarding selector produced volume-of-opens agents. Pair
counts climbed faster than maturation rate; the locked-share
gain was eaten by the fc-cost share growth. Single positive
signal `abd438ea` (+£116/d fc=0 newwindow, 5/7 prof) collapses
under fc=120 (−£131/d, 0/7 prof) — fc=0 positive is a leftover
naked-tail effect that deploy-time force-close caps. Root cause
diagnosed in EXPLORATIONS.md entry 3: `locked_per_std` is blind
to the fc cost it incentivises. Direct successor is tnv3 with
`day_pnl_per_std`.

## 2026-05-17 — tnv3 (stopped at gen 1 partial; mechanism REJECTED)

**Intention.** Postmortem of tnv2's selector (EXPLORATIONS.md entry
3): `locked_per_std` rewards opening volume and tight variance —
both achievable via the same phenotype that incurs heavy fc cost.
The metric is blind to the cost it incurs. Switch numerator to
day_pnl directly so the GA can't game the locked floor without
paying for fc.

**Implementation brief.** New `composite_score_mode=day_pnl_per_std`.
β range widened to [0, 0.10]. New `--early-stop-patience 3
--early-stop-min-gens 4` flag to halt cohorts that plateau. 10
in-sample-eval days (up from 6 in tnv2 — tightens σ estimate by
~1.8×). Cohort tag `_predictor_SCALPING_tnv3_raceconf_1779011408`.

**Result.** **Stopped at gen 1 partial (20/96 agents) on mechanism
analysis.** Gen 0 mean day_pnl −£46, gen 1 (n=8) drift settled at
−£31. The +£28/d gen-0→gen-1 lift at n=4 shrank to +£15/d at n=8
— curve was flattening fast. But the load-bearing observation was
that `mean_fc_pnl` was CLIMBING under selection (gen 0 −£86 → gen 1
−£91), not falling. day_pnl_per_std improved day_pnl through the
same levers tnv1 already used (tighter naked + higher locked) and
did NOT cut fc cost.

Diagnosed in-flight: **GA selection picks which agents reproduce;
it does not change what each agent learns during training.** PPO's
per-step gradient comes from the reward function, which was
unchanged between tnv2 and tnv3. Every gen-1 child inherits its
parent's policy weights and continues learning from the same
locked-rewarding gradient. The selection metric filters survival;
it doesn't tell PPO how to learn differently. We were asking GA
selection to fix a problem that lives in the reward function.

Second mechanism: fc cost is partially substitutable for naked
variance in the selector. Two agents with identical day_pnl can
score 3.8× differently if one converts variance into bounded fc
cost — the denominator's non-linear σ penalty rewards heavy fc use
when the numerator is held constant.

The "structural change → no improvement" result rejects
selection-side interventions for this problem class. The plan's
central thesis (variance-aware selection over existing populations
surfaces deployable agents) is rejected on **mechanism**, not just
on data. ~22h GPU saved by stopping early; rerouted to a reward-side
experiment (per-tick fc-cost shaped penalty + fc-prob feeding
actor_head are the candidate levers; see `findings_tnv3.md`).

---

## 2026-05-17 — fc-cost-probes (reward-side levers, 5-agent × 1-gen shape)

**Context.** tnv3 was rejected on mechanism (entry above). The
follow-up direction needed reward-side intervention rather than
another GA selector. To avoid burning ~28h on a full cohort that
might also fail to bite, the operator picked a **probe-shaped
budget**: spend 20h trying multiple small probes (5 agents × 1
gen × 3 train + 3 eval days, ~1h wall each) and only commit a
full cohort once a lever clearly bites.

Bite criteria are lever-specific; the common floor across all probes
is `locked ≥ +£70/d` and `naked_span ≤ baseline + 50` (variance
discipline must not collapse). The status panel (`tools/
show_probe_status.py`, new) renders per-agent chronological rows
+ cohort-mean deltas + bite verdict per metric.

**Baseline** (tnv3 gen 0, fc=120 in training, raceconf gate, 10
in-sample-eval days, n=12): pnl=−£46/d, fc_n=54, fc_£=−£86/d,
cl_n=9, span=£227, bets=178, locked=+£88/d, cl_£=−£13/d.

### Probe A — `close_signal_bonus` £1 → £10

**Intention.** Raise the per-close shaped bonus (from the
2026-04-18 `plans/naked-clip-and-stability` default of £1) to £10
to test whether the agent will substitute close_signal for
force-close once the close becomes meaningfully rewarded.

**Implementation.** New env-kwarg `close_signal_bonus` plumbed
through `_REWARD_OVERRIDE_KEYS` in `env/betfair_env.py`; default
1.0 stays byte-identical. Cohort tag
`_predictor_SCALPING_probe_a_close_bonus_1779036506`.

**Result.** **NO BITE.** 5/5 agents:
- Mean cl_n 9.7 (vs baseline 9 — **unchanged**)
- Mean fc_n 50.8 (vs 54 — barely changed)
- Mean fc_£ −£96.5 (vs −£86 — *worse* by £10/d)
- Mean pnl −£42.7 (vs −£46 — flat, within noise)
- Mean locked +£90.6 (stable)

The £10 bonus did not produce close_signal substitution in 3 train
days. Two candidate root causes carry through to subsequent
probes: (i) magnitude too small against ±£500/d naked variance
(probe O tests this), (ii) credit-assignment from "close at T−200
→ +£10 reward at race-end T−0" is the bottleneck — policy has no
representational pathway to anticipate force-close before it
happens (probe D addresses this).

### Probe B — `open_cost` pinned 0.5 cohort-wide

**Intention.** Test whether `selective_open_shaping`'s per-pair
charge mechanism (introduced 2026-04-25; gene-enabled in tnv2/tnv3
but mutated low) pulls down pair counts and fc cost when pinned
HIGH uniformly. The charge lands on the open tick and only refunds
on matured/agent-closed pairs — force-closes leave the charge in
place. Per-tick gradient on the open decision, theoretically the
correct credit assignment.

**Implementation.** `--reward-overrides open_cost=0.5` with the
gene NOT enabled (mutual-exclusion guard). Cohort tag
`_predictor_SCALPING_probe_b_open_cost_1779039318`.

**Result.** **NO BITE.** 5/5 agents:
- Mean fc_n 51.6 (vs baseline 54 — barely moved)
- Mean cl_n 9.0 (vs 9 — unchanged)
- Mean bets 174.2 (vs 178 — barely moved)
- Mean pnl −£17.2 (vs −£46 — +£29 improvement, but driven by
  agents 8c7bdabc/+£52 and 1c59ffd2/−£7 catching naked tailwinds,
  NOT the lever)
- Mean fc_£ −£96.3 (vs −£86 — actually *worse* by £10)
- Mean locked +£99.6 (slightly higher)

`open_cost=0.5` pinned cohort-wide did NOT pull down pair counts
or fc events in 3 train days. The +£29/d mean pnl improvement
looks like a hit but the per-agent decomposition shows it's
naked-tailwind luck on two of five agents, not a structural shift.
Same failure mode as probe A: the lever's gradient is delivered
but PPO doesn't respond in 3 train days at small sample size.
The pre-plan `selective-open-shaping` work had already shown that
opening volume + fc rate barely respond to `open_cost` in the
0.06-0.83 gene range; this probe extends that observation to
0.5 specifically.

### Probe C — combo (close_signal_bonus=10 + open_cost=0.5 + mature_prob_loss_weight=3.0)

**Intention.** Test whether A's substitution incentive + B's
per-tick gradient + a *trained* mature_prob_head bite together
when none bit alone. mature_prob's BCE label structurally puts
force-closed pairs in the negative class — training it hard
should give the actor a discriminative "this open is risky" feature
via the existing fill_prob/mature_prob actor-input wiring (CLAUDE.md
§"mature_prob_head feeds actor_head").

**Implementation.** Three `--reward-overrides` flags. Same cohort
shape. Cohort tag `_predictor_SCALPING_probe_c_combo_1779042022`.

**Result.** **NO BITE.** 5/5 agents:
- Mean fc_n 52.5 (vs baseline 54 — unchanged)
- Mean cl_n 8.9 (vs 9 — unchanged)
- Mean bets 174.2 (vs 178 — unchanged)
- Mean pnl −£7.5 (vs −£46 — +£38 improvement, but driven by agent
  ed51f840 alone (+£32 with naked +£45 — same tailwind story))
- Mean fc_£ −£96.7 (vs −£86 — *worse* by £11)
- Mean locked +£96.1 (stable)

The combo of close_signal_bonus + open_cost + mature_prob did NOT
multiplicatively unlock anything. Even with mature_prob_loss_weight
pinned high (3.0) to ensure the actor-input column carries
discriminative info, the policy doesn't fire close_signal more
(cl_n unchanged) or reduce opens (bets unchanged) in 3 train days.

**Consolidated read across A + B + C** (n=15 trained agents): the
**5-agent × 3-train-day shape is too short** for PPO to relearn
its open / close policy under any of these reward gradients. Mean
pnl improvements in the +£29-£44/d range across A/B/C are NOT
structural shifts — they reflect 1-2 of 5 agents catching naked
tailwinds inside the 3-day eval window. The lever-signal metrics
(fc_n, cl_n, bets) are flat against baseline ±2-3 events. This is
the same "gradient delivered but PPO unresponsive" failure mode
documented in `plans/selective-open-shaping/lessons_learnt.md`
(2026-04-25 cohort-O / O2 sessions).

Implications for probe budget:
1. Magnitude follow-up (O, £50) tests whether *strength* of close
   bonus matters at all.
2. Timing follow-up (H, T-180) tests an env-only knob that doesn't
   require PPO learning.
3. The architectural intervention (D, fc_prob_head) is the
   structural lever — but it also needs sufficient training time
   for the aux head to discriminate. Probably needs a longer
   training window (5+ days minimum) rather than 3.

### Probe O — `close_signal_bonus` £1 → £50 (magnitude follow-up to A)

**Intention.** If A at £10 didn't move cl_n, does £50? Tests
whether magnitude alone was the bottleneck. If O also flat, mechanism
is the issue, not size.

**Implementation.** `--reward-overrides close_signal_bonus=50.0`.
Cohort tag `_predictor_SCALPING_probe_o_close_50_1779045080`.

**Result.** **NO BITE — magnitude conclusively ruled out.** 5/5 agents:
- Mean cl_n **8.4** (vs baseline 9 — *LOWER* with 5× the bonus)
- Mean fc_n 51.8 (vs 54 — unchanged)
- Mean bets 171.8 (vs 178 — unchanged)
- Mean pnl −£3.9 (+£42 vs baseline — driven by agents 22518d17/+£69
  and 554ac85d/+£18 catching naked tailwinds; same pattern as A/B/C)
- Mean fc_£ −£93.8 (vs −£86 — worse)

The £50 bonus produced FEWER closes per day (8.4) than the default
£1 (baseline 9). This is the cleanest possible refutation of the
"magnitude" hypothesis. PPO simply does not learn to use
close_signal more in 3 train days — the credit assignment from
"close at T-200 → +£50 reward at race-end T-0" is the wall.

### Probe A2 — close_signal_bonus=10, 7 train + 3 eval days (meta-test)

**Intention.** After A/B/C all failed to bite on the per-agent
lever signals (fc_n/cl_n/bets unchanged across 15 agents), the
consolidated read pointed to **sample size, not mechanism**, as
the likely bottleneck: 3 train days × ~250 episodes/day might be
too few for PPO to relearn its open / close policy against the
new shaping gradient.

A2 keeps probe A's lever identical (`close_signal_bonus=10`,
otherwise default) but extends training to 7 days (`--days 10
--n-eval-days 3`). Cheapest possible test of the meta-hypothesis.
If A2 bites where A didn't → bump all subsequent probes to 7-day
training. If A2 also flat → the bottleneck is structural and the
architectural probe (D) becomes the only worth-trying lever.

**Implementation.** Launcher `C:\tmp\probe_a2_close_bonus_7day.ps1`.
Same gates + overrides as A; only `--days 10 --n-eval-days 3`
changes (was `--days 6 --n-eval-days 3`). Wall ~2.1h (vs A's ~40
min). Cohort tag
`_predictor_SCALPING_probe_a2_7day_1779051255`.

**Result.** **NO BITE — sample-size hypothesis REFUTED.** 5/5 agents:
- Mean cl_n **7.7** (vs baseline 9 — *LOWER* even with 2.3× training)
- Mean fc_n 51.2 (vs 54 — unchanged)
- Mean fc_£ −£98.3 (vs −£86 — worse by £12)
- Mean bets 171.2 (vs 178 — unchanged)
- Mean pnl −£15.3 (vs −£46 — +£31 driven by agent c321fb90's
  naked +£43 luck again, same pattern as A/B/C/O)
- Mean locked +£100.3 (slightly higher)

The meta-test conclusively rules out training length as the
bottleneck. Doubling training time (3 → 7 days) at the same
close_signal_bonus=10 produced **fewer** closes per day, not more.
PPO does not learn to substitute close_signal for force-close
under this gradient regardless of magnitude (£10, £50) OR
training length (3, 7 days).

**Cross-probe meta-finding** (now 30 trained agents across A/B/C/
O/H/A2): the close_signal / open_cost / mature_prob /
force_close_timing levers cannot move the per-agent lever signals
(cl_n, fc_n, bets) at this cohort scale. The "+£30-40/d pnl
mean" lifts across multiple probes are consistently traceable to
1-2 agents per probe catching positive naked variance days,
**not** to the lever working. This is hard evidence for a
**representational bottleneck**: the policy has no per-runner
"this open will force-close" feature to condition on, so any
shaped gradient against fc cost has to propagate through
hundreds of ticks against ±£500/day naked variance — which
swamps the signal at PPO's value-function level.

Probe D is the targeted intervention.

### Probe H — `force_close_before_off_seconds` 120 → 180 (timing test)

**Intention.** Pure env-knob change, no policy retraining of new
shaping. Tests whether the *timing* of force-close matters: earlier
flatten might reduce per-pair spread cost (close at a less-thin
book) at the cost of more events.

**Implementation.** `--reward-overrides
force_close_before_off_seconds=180`. Cohort tag
`_predictor_SCALPING_probe_h_fc180_1779048195`.

**Result.** **NO BITE.** 5/5 agents:
- Mean pnl −£38.7 (vs baseline −£46 — +£7 within noise)
- Mean fc_n 52.1 (vs 54 — unchanged)
- Mean fc_£ −£96.4 (vs −£86 — *worse* by £10)
- Mean cl_n 8.7 (vs 9 — unchanged)
- Mean bets 173.4 (vs 178 — barely moved)
- Mean locked +£96.2 (slightly higher)

Moving force-close from T-120 to T-180 traded slightly fewer fc
events (51-53 vs baseline 54) for slightly more expensive ones
(per-event cost up). Net wash. The timing of fc bail-out is not
the binding constraint — book liquidity at T-180 isn't materially
thicker than at T-120, and the policy's behaviour pre-T-180 is
unchanged from pre-T-120.

### Probe D — `fc_prob_head` (new aux head, deferred)

**Intention.** Add a third per-runner aux head with strict label
`1.0 if pair has count >= 2 AND any leg has force_close=True else 0.0`.
Feed sigmoid into `actor_head` as a 5th column (joining fill_prob,
mature_prob, direction_back/lay_prob). Gives the policy a dedicated
representational pathway to anticipate fc per-runner. mature_prob
is close but conflates naked + force-closed both as negative class;
fc_prob isolates the fc class specifically.

**Implementation.** Architectural change (~2-3h scaffolding):
new head in `agents_v2/discrete_policy.py`, new label path in
`training_v2/discrete_ppo/aux_labels.py`, new BCE loss path in
`training_v2/discrete_ppo/trainer.py`, new gene
`fc_prob_loss_weight` in `training_v2/cohort/genes.py`. Behaviour-
flag default `False` preserves current architecture; probe D
launcher sets it to `True` and pins `fc_prob_loss_weight=3.0`.
Cannot land mid-flight (breaks A/B/C architecture comparability) —
scaffolded in parallel with O/H runs and launched after H.

**Training window amendment (2026-05-17 19:11).** Original spec
was 3 train + 3 eval days mirroring A/B/C. Post-A/B/C meta-finding
(15 agents, no lever-signal movement) means D must run with at
least 7 train days — the aux head needs training time to
discriminate, on top of PPO needing time to learn to use the
discriminative output. Probe D launcher will use `--days 10
--n-eval-days 3` (matching A2's shape). Wall ~2.5h.

**Implementation landed (2026-05-17 22:18).** Files touched:
- `agents_v2/discrete_policy.py`: new `fc_prob_head: nn.Linear(hidden, max_runners)`
  gated behind `enable_fc_prob_head=False` default. Sigmoid feeds
  actor_input as a 5th column (joining fill_prob, mature_prob,
  direction_back, direction_lay) WITHOUT detach — surrogate-loss
  gradient flows back through the head. Default off keeps the
  arch byte-identical to A/B/C/O/H/A2.
- `training_v2/discrete_ppo/aux_labels.py`: new
  `assign_per_transition_fc_labels` — strict label
  `1.0 iff pair force-closed; else 0.0`.
- `training_v2/discrete_ppo/trainer.py`: reads
  `fc_prob_loss_weight` from `hp`. New
  `_compute_per_transition_fc_loss` mirrors the mature path.
- `env/betfair_env.py`: `_REWARD_OVERRIDE_KEYS` adds
  `fc_prob_loss_weight` + `enable_fc_prob_head` (passthrough).
- `training_v2/cohort/worker.py`: reads `enable_fc_prob_head`
  from `per_agent_reward_overrides` and passes to policy ctor.
  `fc_prob_loss_weight` added to `_PHASE7_TRAINER_HP_KEYS`.

Regression: 53 existing v2 policy + aux-head + direction-prob
tests pass with the default `enable_fc_prob_head=False` path.

Cohort tag `_predictor_SCALPING_probe_d_fcprob_1779056297`.
Launched 22:18. Overrides confirmed:
`{force_close_before_off_seconds=120, enable_fc_prob_head=True,
fc_prob_loss_weight=3.0}`.

**Result.** **NO BITE — representational-pathway hypothesis
refuted at this cohort scale.** 5/5 agents:
- Mean fc_n 51.5 (vs baseline 54 — unchanged)
- Mean cl_n 7.7 (vs 9 — *LOWER*, same as A2)
- Mean bets 164.8 (vs 178 — modest −13 drop; the only metric
  showing any movement, and it's locked floor dragging down with
  it from +£88 to +£86)
- Mean pnl −£33.3 (vs −£46 — +£12 lift, no naked tailwind
  outliers this round so the lift is the smallest of any probe)
- Mean fc_£ −£102 (vs −£86 — worse by £16)

The architectural intervention (new fc_prob_head feeding actor_input
+ trained at BCE weight 3.0) DID nudge the policy toward fewer
opens (bets 165 vs 178) — the new column is providing SOME signal —
but the change is too small at 5-agent × 7-day scale to clear the
naked-variance noise, and the locked floor dropped marginally too,
suggesting the new column is adding mild actor noise without the
selective-open behaviour we wanted.

**See `plans/scalping-tight-naked-variance/findings_probes.md`**
for the consolidated 7-probe meta-finding + recommended next
steps (preferred: 12-agent × 8-generation full cohort with
fc_prob_head enabled + day_pnl_per_std selector + tighter
lay_price_max).

### Probes queued (medium effort, not yet scaffolded)

- **E** — per-runner `open_cost × (1 − mature_prob[runner])`. Makes
  the open charge fc-risk-aware. Composes with B; depends on
  mature_prob_head being meaningfully trained (probe C feedback).
- **F** — time-windowed close_signal_bonus. Only pay the bonus on
  closes in `[T−180, T−60]`. Surgically targets the decision window
  rather than rewarding any close. Composes with A/O.
- **I** — cap force-close `pnl` magnitude in the matcher (band-aid,
  not learning fix). Tests whether the cost MAGNITUDE matters or
  just the existence of fc events.
- **N** — small "no-op" shaped bonus (e.g. +£0.01/tick on
  zero-open ticks). Counterbalance to PPO's open-more gradient.
  Risk: bets=0 collapse if too strong.

### Probes flagged as speculative (not on critical path)

- **M** — direction_predictor as action-mask gate (use existing
  trained head as a per-tick "should I close NOW" trigger).
- **K** — RNN over recent fc events (architectural).
- **P** — multi-tick "commit" on opens (architectural).

---

## 2026-05-18 — E1: per-tick close credit [QUEUED]

**Intention.** The 7-probe fc-cost-probes finding (entry above)
ruled out close-bonus *magnitude* and training *length* as bottlenecks.
The one thing every probe shared — and never tested — was the
credit-assignment *timing*. `close_signal_bonus` lands inside
`race_shaping` at race-settle (`env/betfair_env.py:280-325`), so a
close at T−200 sees its bonus arrive ~150+ ticks later. With γ=0.99,
λ=0.95, the gradient reaching the close decision is
`0.95^150 ≈ 4.6e-4` of the bonus magnitude. A £50 bonus arrives at
the action as ~£0.023 of gradient — invisible against ±£500/d naked
variance.

This is the **same trap** `open_cost` hit on 2026-04-25 and was fixed
in Session 02 of `selective-open-shaping` by moving to per-tick
credit. The lessons file is explicit: "GAE smeared the per-race
delta back across 5,000 ticks, drowning the per-tick gradient at the
open decision in value-function noise. Per-tick delivery puts the
gradient at the right place." That fix bit. The same fix has never
been applied to `close_signal_bonus`.

Hypothesis: deliver `close_signal_bonus` at the close-tick (mirror
`_charge_open_cost` / `_resolve_open_cost_pairs` pattern). PPO will
finally see the close-decision gradient and `cl_n` will move.

**Implementation brief.** Landed in commit `4d4a5b6`. New
`_step_close_bonus_pnl` accumulator reset each step;
`_attempt_close` increments it by `close_signal_bonus` on
agent-initiated successes (`force_close=False AND stop_close=False`)
when `per_tick_close_bonus=True`; end-of-`step()` adds it to
`reward` and `_cum_shaped_reward`. The matching contribution is
suppressed from `_compute_scalping_reward_terms`'s `race_shaping`
(passed `close_signal_bonus=0.0`) so totals don't double-count.
5 regression tests guard the path (TestPerTickCloseBonus).
Cohort tag `_predictor_SCALPING_probe_e1_per_tick_close_1779134372`,
overrides `per_tick_close_bonus=true close_signal_bonus=10.0`.

**Result.** **NO BITE.** 5/5 agents finished 2026-05-18 21:05.

| Metric | Baseline (tnv3 gen 0) | E1 mean (5 agents) | Δ |
|---|---:|---:|---:|
| pnl | −£46/d | −£47/d | −£1 (flat) |
| **cl_n** | **9** | **9.2** | **+0.2 (within noise)** |
| fc_n | 54 | 49 | −5 |
| fc_£ | −£86/d | −£97/d | −£11 (worse per event) |
| locked | +£88/d | +£80/d | −£8 |
| bets | 178 | 158 | −20 (mild drop in opens) |
| naked_span | £227 | £67 | — (smaller eval window dominates) |

Per-agent cl_n: 10, 11, 11, 9, 6 — three above baseline, two
at-or-below. Mean is statistically indistinguishable from baseline
on n=5. The hypothesis that "moving bonus to per-tick will move
cl_n" is **refuted at this cohort scale**, same conclusion the
7-probe meta-finding reached for magnitude-and-timing variations.

The mechanism IS working — the test suite proves the credit lands
on the close tick and the settle path doesn't double-count. PPO
just doesn't respond to a £10 per-close gradient in 7 days, regardless
of when it's delivered. Consistent with `findings_probes.md` final
hypothesis: "cohort-scale signal-to-noise is the binding constraint."

Per the user gate, NOT escalating to full cohort. Moving to E2.

---

## 2026-05-18 — E2: asymmetric MTM [QUEUED]

**Intention.** The current per-tick mark-to-market shaping (CLAUDE.md
"Per-step mark-to-market shaping") is symmetric — gains and losses
both telescope to zero across a race. That gives the policy a
per-tick redistribution of the realised P&L signal but no asymmetric
pressure to close drawdowns vs run winners.

Asymmetric weighting (`mtm_weight_loss > mtm_weight_gain`) makes
holding an underwater pair painful at every tick the drawdown
exists. The cumulative shaped contribution per race is no longer
zero-mean — losing races pay more shaped reward than winning ones
gain — which intentionally violates the existing "raw + shaped ≈
total" invariant. This is the same trade-off `naked_loss_anneal`
made (CLAUDE.md "Naked-loss annealing"). The user's intuition
("agents should run scared from naked bets") maps directly to this
asymmetry.

**Implementation brief.** Landed in commit `4d4a5b6`. Two new
reward-override keys: `mark_to_market_weight_loss` and
`mark_to_market_weight_gain`. When either is set, the per-tick
MTM shaped contribution becomes `weight_loss * min(0, delta) +
weight_gain * max(0, delta)`. Defaults fall back to
`mark_to_market_weight` so symmetric weights are byte-identical to
pre-E2. Telemetry on info dict: `mtm_asymmetry_active`,
`mtm_weight_loss`, `mtm_weight_gain`. 6 regression tests guard the
path. Cohort tag
`_predictor_SCALPING_probe_e2_asym_mtm_1779138407`, overrides
`mark_to_market_weight_loss=0.15 mark_to_market_weight_gain=0.05`
(3× asymmetry, drawdowns hurt more than wins).

**Result.** **NO BITE.** 5/5 agents finished 2026-05-18 22:12.

| Metric | Baseline | E2 mean | Δ |
|---|---:|---:|---:|
| pnl | −£46/d | −£38/d | +£8 (naked-tailwind noise) |
| **cl_n** | **9** | **8.0** | **−1 (LOWER, opposite of intended)** |
| fc_n | 54 | 51 | −3 |
| **fc_£** | **−£86/d** | **−£102/d** | **−£16 (WORSE per event)** |
| locked | +£88/d | +£84/d | −£4 |
| bets | 178 | 162 | −16 |

Per-agent cl_n: 8, 9, 11, 4, 8 — one agent (dee54213) dropped to
4 closes, opposite of what the per-tick drawdown pressure was
meant to produce. The +£8/d pnl lift is fully traceable to two
agents (ee88c6d5 +£3, dee54213 −£7) catching modest naked
tailwinds — same pattern as A/A2/O.

Mechanism diagnosis: the asymmetric MTM gradient IS landing
(verified by tests). PPO sees continuous loss-side pressure. But
the policy's response is to slightly reduce opens (bets 178→162)
rather than to substitute close_signal. The shaped channel pressure
didn't unlock the close action. Worse, fc cost per event climbed
£16 — likely because reduced opens left the remaining pairs in
worse spots when force-close triggered.

The hypothesis "punish drawdowns continuously → policy learns to
close" is **refuted at 5×7d scale**. Same conclusion as E1.

---

## 2026-05-18 — E3: open-with-close-feasibility gate [QUEUED]

**Intention.** The user's principle — *"If they can't close it,
they should never have opened it"* — applied at the env level
without any learning. At open time, peek at the opposite-side ladder
and compute the hypothetical cost-to-close at the current spread.
If the cost exceeds a configurable fraction of stake, refuse the
open at the matcher level.

This is a structural prior the policy doesn't have to learn. It
kills the 0.3% outsider-residue trades that EXPLORATIONS.md entry 2
identified (back-first opens at price 15-30 with 0% positive naked
outcomes), and more importantly it kills the pre-off thin-liquidity
opens that drive the bulk of fc cost.

**Implementation brief.** Landed in commit `f4e4a06`. New
reward-override `close_feasibility_max_spread_pct` (default None
= disabled, byte-identical). At each open candidate in
`_process_action`, peek the opposite-side top price (junk-filtered
via the matcher's `pick_top_price`) and the projected aggressive
match price. Refuse the open if
`|agg_price - close_price| / close_price > threshold` OR the close
side is unpriceable. Telemetry: `opens_refused_close_feasibility`
counter on info dict. 5 regression tests guard the gate. Cohort
tag `_predictor_SCALPING_probe_e3_close_feas_1779142421`,
override `close_feasibility_max_spread_pct=0.05` (refuse opens at
>5% spread).

**Result.** **BITES — first probe to clearly move the metrics
across all 8 probes attempted (A/B/C/O/H/A2/D + E1/E2/E3).**
5/5 agents finished 2026-05-18 23:18.

| Metric | Baseline | E3 mean (5 agents) | Δ |
|---|---:|---:|---:|
| **pnl** | **−£46/d** | **+£59.4/d** | **+£105.4 ⭐** |
| **fc_n** | **54** | **34.8** | **−19.2 (−36 %)** |
| **fc_£** | **−£86** | **−£56** | **+£30 (35 % improvement)** |
| locked | +£88 | +£107 | +£19 |
| **maturation rate** | **~0.34** | **0.50** | **+0.16** |
| bets | 178 | 147 | −31 |
| cl_n | 9 | 6.1 | −2.9 |

Per-agent pnl: +£72, +£86, −£29, +£136, +£32 — **4/5
profitable**. The mechanism is exactly what the user's principle
predicted: refusing opens whose close path would be too expensive
prevents the worst pre-off thin-liquidity opens. Fewer bad opens
→ remaining ones mature more reliably (mr 0.34→0.50), fc cost
drops 35%, locked floor climbs.

Naked variance still contributes ±£40+ swings per agent
(c997b601 +£32, 8c8ea6c6 +£43, 1ea86cfe −£45, 3b73dfd7 +£73,
2b462882 −£28) — the agent who got the worst naked draw
(1ea86cfe) is the only loss. So the floor is real but the naked
ceiling/cellar still in play. A full cohort at 12-agent × 8-gen
should average this out.

**Verdict: ESCALATE to full cohort.** Per the user's gate this
goes into the queue for after all other probes finish (E4, E5,
E6 still to run). Recipe: 12-agent × 8-gen × 13/10 day split,
override `close_feasibility_max_spread_pct=0.05`, raceconf gate,
fc=120 in training. Wall ~28h.

### E3 full cohort — STRONG band confirmed (2026-05-19, stopped at gen 3)

Cohort tag `_predictor_SCALPING_e3_full_cohort_1779172530`.
Launched 2026-05-19 ~17:35, stopped 2026-05-19 ~20:20 after gen
3 partial (44/96 agents trained). Stopped on mechanism analysis —
the trajectory peaked at gen 2 (+£28/d mean) and dipped at gen 3
partial (+£16/d), classic inverted-U; remaining 14h GPU rerouted
to the robust-phenotype plan (R1+R3+R4).

**In-sample trajectory (10-day eval):**

| Gen | n | mean pnl | best | profitable | mean locked | mean fc£ |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 12 | +£9 | +£41 | 9/12 | +£110 | −£62 |
| 1 | 12 | +£24 | +£65 | 9/12 | +£116 | −£63 |
| 2 | 12 | +£28 | +£65 | **11/12** | +£116 | −£62 |
| 3 (8/12) | 8 | +£16 | +£63 | 7/8 | +£111 | −£64 |

**Held-out reeval (7-day forward window 2026-05-07..05-13)** —
top-5 by deployable composite (positive in-sample pnl, penalised
by worst-day < −£30 AND naked_worst < −£40):

| Agent | Gen | In-sample | **fc=120** | **fc=0** | locked (fc=120) | naked (fc=120) |
|---|---:|---:|---:|---:|---:|---:|
| cea2ee94 | 1 | +£65 | **+£72** | +£215 | +£114 | +£9 |
| f89b9b94 | 3 | +£63 | **+£63** | +£25 | +£101 | +£11 |
| 11099f65 | 2 | +£32 | **+£49** | +£117 | +£101 | +£5 |
| 57a42db5 | 2 | +£48 | **+£48** | +£114 | +£101 | −£6 |
| 850522b9 | 2 | +£65 | +£45 | +£12 | +£109 | −£10 |
| **MEAN** | — | — | **+£55.4** | **+£96.7** | **+£105** | **+£2** |

**5/5 agents profitable on BOTH fc=0 AND fc=120.** Compare:
- tnv2 fc=120 newwindow: 0/10 profitable, mean −£177/d (regression baseline)
- layq fc=120 newwindow: mean +£26/d (prior deployment best)
- **E3 cohort fc=120 newwindow: +£55/d (+£29 vs layq, +£232 vs tnv2)**

**Band: STRONG.** Clears `mean ≥ +£50/d AND ≥4/5 prof AND fc=120
deeply positive`. First strong-band cohort in the project's
history.

**Phenotype reads.** Three distinct deployment shapes surfaced:

1. **11099f65 + 57a42db5 (twins, gen 2):** ROBUST shape. Locked
   ~£100/d, naked modest-positive, fc cost ~£40, span tight.
   Held up cleanly on BOTH fc settings (+£48-49 fc=120, +£114-117
   fc=0). The recommended deployment candidates — consistent
   £45-50/d after force-close cost, with modest +£20-30 naked
   tailwind available in non-fc days.
2. **cea2ee94 (gen 1):** HIGH-UPSIDE shape. Locked +£114, naked
   +£115 on fc=0 (some tailwind contribution), +£9 on fc=120.
   Most profitable in absolute terms but partly tailwind-dependent;
   smaller naked-positive expected on a different forward window.
3. **850522b9 (gen 2, in-sample gold standard) + f89b9b94 (gen 3):**
   OVERFIT shapes. 850522b9 in-sample +£65 → held-out +£45 (still
   profitable but dropped £20); f89b9b94's naked turned negative
   on fc=0 (locked floor still kept it +£25). Real but degraded
   shapes.

**Decision: deploy candidates are 11099f65 and 57a42db5.** Cea2ee94
is a candidate too if next-window reeval confirms the naked-tail
holds. The cohort verdict supports E3 as the production lever
going forward.

Next: E3+E4 combo probe (auto-firing now); R1+R3+R4 next big
bet (plans/robust-phenotype/).

---

## 2026-05-18 — E4: inverted keep_open action + MTM stop-loss [QUEUED]

**Intention.** Reframe the close action. Env auto-closes any pair
whose MTM drops below `−£mtm_stop_loss` (a fixed gene), and the
agent's action becomes a `keep_open` override (default off) instead
of `close_signal` (default off). Inverts the gradient direction —
the agent must actively *defend* losing positions instead of actively
*closing* them.

Defensive-action framing is empirically easier for RL on sparse
events: zero exploration cost (action defaults to a safe behaviour)
rather than positive exploration cost (default behaviour is the bad
one). The expected base rate of "agent decided to keep underwater
position" is much lower than "agent decided to close any pair", so
the gradient SNR per `keep_open` invocation is much higher.

**Implementation brief.** Landed in commit `69d2263`. Less
invasive than first sketched: the existing `close_signal` action
column is REINTERPRETED at action-processing time when
`keep_open_inversion=True`. Raised → mark runner's pairs as
keep-open this tick (suppress stop-loss auto-close). Lowered →
env's stop-loss runs as normal. Agent loses ability to initiate
close — by design. Stop-loss path already existed (uses
`stop_loss_pnl_threshold`); E4 just adds the override layer.
Action-space unchanged, no architecture-hash break.
Cohort tag `_predictor_SCALPING_probe_e4_keep_open_1779150400`,
overrides `keep_open_inversion=true stop_loss_pnl_threshold=0.10`.

**Result.** **BITES — second probe to clearly move metrics.**
5/5 agents finished 2026-05-19 00:32.

| Metric | Baseline | E4 mean | Δ |
|---|---:|---:|---:|
| **pnl** | **−£46/d** | **−£2.4/d** | **+£43.6 (vs E3's +£105)** |
| cl_n | 9 | **0** | by design (close suppressed) |
| fc_n | 54 | 45.3 | −8.7 |
| fc_£ | −£86 | −£80 | +£5.8 |
| **locked** | **+£88** | **+£100** | **+£12** |
| naked_span | £227 | £128 | −£99 (tighter dispersion) |
| bets | 178 | 164 | −14 |
| cl_£ | −£13 | £0 | by design (no close-leg losses) |

Per-agent pnl: +£9, +£36, −£24, −£5, −£28. 2/5 profitable.
Locked floor solid across all 5 (+£82 to +£118 range).

**CAVEAT — bite is confounded.** E4 stacks two interventions
together: keep_open inversion AND stop_loss=0.10. The bite could
be entirely from the env's stop-loss mechanism (which existed
pre-E4 but was never tested with this gate combination). To
attribute properly: **queue an E4b ablation with
`stop_loss_pnl_threshold=0.10` only (no inversion, agent still
controls close_signal)**. If E4b matches E4 → stop-loss is doing
the work, inversion is dead weight. If E4b ≪ E4 → the action
reframe matters too.

**Verdict: ESCALATE.** Same as E3, queued for full-cohort run
after probes finish. The mean lift is half of E3's, so E3 is the
higher-priority full cohort. But E4 is a complementary lever
(close-side via env automation) and the COMBINED cohort
(E3 + E4) is the obvious next-tier candidate.

---

## 2026-05-18 — E5: per-pair reward at resolution tick [QUEUED]

**Intention.** The structural fix. Currently 600 opens/race + their
600 outcomes share one summed reward signal at race-settle. PPO sees
all per-decision gradients smeared by GAE through trajectories of
length thousands. SNR per decision is awful.

Replace with: each pair's realised P&L (locked / closed /
force-closed / naked) lands as a per-step reward AT THAT PAIR'S
RESOLUTION TICK. Race total = sum of per-pair P&Ls (invariant
preserved). PPO's existing per-runner value head can predict
per-pair returns much more cleanly than per-race noise. E1 is a
special case of this for close events; E5 covers the natural-mature,
force-close, and naked-at-settle cases too.

**Implementation brief.** Landed in commit `6596b17`. Minimal
shaped-channel telescope version (not the full raw-channel
rewrite). New reward-override
`per_pair_reward_at_resolution` (default False, byte-identical).
When True: `_emit_per_pair_resolution_pnl` walks bm.bets each step,
finds newly-resolved pairs (both legs matched, not in
`_paid_pair_ids`), computes locked P&L = `min(win_pnl, lose_pnl)`
using the un-floored BetManager formula, pushes into a per-step
accumulator. End-of-step folds into reward + cum_shaped (mirror of
E1 close-bonus path). Settle subtracts the cumulative emitted
amount from `shaped` — per-race telescope so raw+shaped sum is
preserved; only delivery TIMING changes. 5 regression tests.
Cohort tag `_predictor_SCALPING_probe_e5_per_pair_1779154427`.

**Result.** **MARGINAL BITE.** 5/5 agents finished 2026-05-19
01:39.

| Metric | Baseline | E5 mean | Δ |
|---|---:|---:|---:|
| pnl | −£46/d | −£16.3/d | +£29.7 |
| **locked** | **+£88** | **+£106.7** | **+£18.7 (HIGHEST of any probe)** |
| fc_n | 54 | 52.8 | −1.2 |
| fc_£ | −£86 | −£98 | −£12 (worse per event) |
| cl_n | 9 | 7.3 | −1.7 |
| bets | 178 | 176 | flat |

Per-agent pnl: −£11, +£13, −£45, +£4, −£43 — 2/5 profitable but
mean dragged by two −£40+ losses (c4e74275 and 7abeffa0 both
caught significant naked losses).

The mechanism worked — credit lands at the resolution tick (verified
in tests), and the policy responds by locking more reliably
(+£18.7 locked floor — the highest of any probe across all 12
probes tried, including E3/E4). But unlike E3, the open-rate is
unchanged (bets ~176), so the pair count keeps growing, fc cost
grows in proportion (−£98 vs −£86), and the net pnl improvement
is moderate.

**Verdict: BITE on locked-floor mechanism, weak on cohort pnl.**
Probably benefits more from full-cohort GA selection (where the
"open less, lock cleaner" subset gets selected for) than from a
small probe. Worth escalating but at LOWER priority than E3
and E4.

---

## 2026-05-18 — E6: hard pair-count budget [QUEUED]

**Intention.** Cap `max_open_pairs_per_race = 30` (currently 600+).
Forces selectivity at the open by making opens scarce. Reduces
naked-side variance by construction — fewer pairs = less stuff to
manage. Tests whether the over-opening phenotype is the *binding
constraint* or merely *symptomatic* of bad credit assignment.

Expected outcome: locked floor drops materially (fewer matured
pairs). If overall day_pnl improves anyway, that's evidence the
phenotype is binding and the cleaner training-side fixes (E1, E2,
E5) might be able to reach the same place by other means. If
day_pnl gets worse, locked-floor was load-bearing and the next bet
is on E1/E2/E5.

**Implementation brief.** Landed in commit `f4e4a06`. New
reward-override `max_open_pairs_per_race` (default 0 = no cap =
byte-identical). When `_pairs_opened_this_race >=
max_open_pairs_per_race`, refuse the open at the earliest exit
(before generating a pair_id). Telemetry:
`opens_refused_pair_budget` counter on info dict. 5 regression
tests guard the gate. Cohort tag
`_predictor_SCALPING_probe_e6_pair_budget_1779146398`, override
`max_open_pairs_per_race=30`.

**Result.** **NO BITE — but CAVEAT: cap never bound.** 5/5
finished 2026-05-18 23:25.

| Metric | Baseline | E6 mean | Δ |
|---|---:|---:|---:|
| pnl | −£46/d | −£38.5/d | +£7.5 (within noise) |
| **bets** | **178** | **171** | **−7 (NOT 30/race × races/day = much lower)** |
| fc_n | 54 | 52.8 | −1.2 |
| locked | +£88 | +£95 | +£7 |
| cl_n | 9 | 7.1 | −1.9 |

The cap of 30 pairs/race was set too high. Looking at the
baseline bets=178/day across ~3 races/day ≈ 60 bets/race ÷ 2 legs
= 30 pairs/race — at the CAP, not above it. So the gate fired
rarely if at all. The earlier "600 pairs/race" figure from
`findings_probes.md` was from a different gate config (no
raceconf, no lay_quality cap). Under raceconf the natural
opens/race is much lower.

Conclusion: this probe tells us nothing about whether forcing
selectivity via a scarcity gate works. To get a real test would
need `max_open_pairs_per_race=5` or similar (a binding cap).
**Queue an E6b re-run with cap=5** if the open-phase intervention
direction proves out further (E3 already strongly supports it).

---

## 2026-05-18 — E7: curriculum — train close in isolation first [DEFERRED]

**Intention.** Last-resort structural fix. Two-phase training.
Phase A: agent's open decision is heuristically forced (e.g.
predictor-driven, fixed stake and arb_spread); only `close_signal`
is policy-trainable. Phase B: unfreeze open, fine-tune both.

By isolating close from the noise of varying open decisions in
Phase A, PPO sees clean gradient on the close action. If even a
tight Phase A doesn't move `cl_n`, the bottleneck is so structural
that no amount of credit-fixing can save it — and the conclusion is
that the close action just isn't usefully learnable on this reward
shape. Worth running only if E1+E2+E5 don't bite.

**Implementation brief.** _Deferred to a follow-on session._
The full implementation requires either (a) trainer-loop changes
to switch reward shape between phases, (b) env-level heuristic
open injection with simultaneous masking of agent open actions,
or (c) action-space partial freezing with explicit parameter
groups. All three approaches are ~1-2 days of careful work and
benefit from the data E1-E6 will produce — if any of those probes
bite, the curriculum need is reduced. Skipping for now per
"start with the cheap probes" gate.

**Result.** _Pending implementation._

---

## 2026-05-19 — R1: Sortino-style composite selector [QUEUED]

**Intention.** E3 full cohort (in-flight as of writing) surfaced
agents with similar mean pnl but wildly different per-day spans —
top-pnl 571f6eda had worst day −£105 / best +£313 (span £418)
while top-pnl 850522b9 had worst −£20 / best +£160 (span £180).
The `day_pnl_per_std` selector treats positive and negative
variance the same in the denominator, so it can't distinguish
850522b9's left-tail-truncated shape from 571f6eda's symmetric
high-variance shape.

R1 replaces the denominator with downside-deviation
(`sqrt(mean(min(0, day_pnl)²))`) — penalises ONLY sub-zero days.
Re-ranks the population to surface the bounded-worst-day
phenotype the user wants for deployment.

**Implementation brief.** _Pending._ New
`composite_score_mode=sortino` branch in
`training_v2/cohort/runner.py::_score_one_agent`. Additive form
(`mean - λ × downside_dev`) for numerical stability per
`plans/robust-phenotype/hard_constraints.md §4`. Pure selection
change; doesn't touch reward or env.

**Result.** _Pending._

---

## 2026-05-19 — R2: worst-day floor selector [QUEUED]

**Intention.** Hard-quadratic alternative to R1's Sortino — a
selector that's flat above the worst-day threshold and
quadratically punishing below it. Cleaner ranking signal than
Sortino under small eval-window noise (Sortino's denominator can
swing on 1-2 bad days when n_eval_days is small).

`composite = mean(pnl) − λ × max(0, −X − worst_day)²` where X
defaults to £30. Above the floor, score = mean(pnl); below, the
penalty grows quadratically with the breach.

**Implementation brief.** _Pending._ New
`composite_score_mode=worst_day_floor` branch. Two CLI knobs:
`--worst-day-floor` (default −30.0) and
`--worst-day-floor-lambda` (default 1.0).

**Result.** _Pending._

---

## 2026-05-19 — R3: quadratic per-pair naked-loss penalty [QUEUED]

**Intention.** Reward-side intervention that makes the agent's
PPO gradient feel concentrated naked losses much more painfully
than dispersed ones. A −£100 single-pair naked costs β×10,000;
ten −£10 nakeds cost β×1,000 — 10× difference even though
aggregate £-loss is the same. Trains the policy to avoid the
trades that produce the deep worst-days in the E3 cohort.

Replaces the existing symmetric `naked_variance_penalty_beta`
with a loss-only quadratic form. Naked WINNERS untouched (the
existing `naked_winner_clip` already neutralises 95% of them per
`plans/naked-clip-and-stability`).

**Implementation brief.** _Pending._ New reward-override
`naked_loss_quadratic_beta` (default 0.0 = byte-identical). Lands
in shaped channel per `plans/robust-phenotype/hard_constraints.md
§2`. Formula:
`shaped -= β × sum(min(0, p)² for p in naked_per_pair)`.

**Result.** _Pending._

---

## 2026-05-19 — R4: liquidity-floor open gate [QUEUED]

**Intention.** Extends E3 (close-feasibility spread gate) with a
ladder-depth check. Most of the −£80 to −£125 worst days in the
E3 cohort came from thin pre-off books where the projected close
was technically priceable (passed E3's 5% spread gate) but only
against a £5 sitting opposite — once gone, the next level was
junk-far. R4 adds a depth check that complements the spread
check.

Refuse opens when post-junk-filter opposite-side ladder depth at
top level < £X. New refusal counter
`opens_refused_liquidity_floor`.

**Implementation brief.** _Pending._ New env kwarg
`opposite_side_depth_floor` (default None = disabled). Read
post-junk-filter via the existing matcher accessors per
`hard_constraints.md §5`. Refusal in same code path as E3 (after
joint-budget, before placement). Adds the depth check AFTER E3's
spread check so they compose cleanly when both engaged.

**Result.** _Pending._

---

## 2026-05-19 — R5: velocity-aware open mask [QUEUED]

**Intention.** When recent LTP velocity is high on a runner, the
market is moving — scalp opens are likely to drift adversely
before maturation. Reading the existing `ltp_velocity_30` feature
(already in the obs slice; no new feature engineering), mask
OPEN_BACK / OPEN_LAY actions for that runner when velocity
exceeds threshold Y.

Queued as follow-on if R1+R3+R4 leaves residual fragility on the
worst-day metric.

**Implementation brief.** _Pending._ New env kwarg
`open_mask_max_ltp_velocity` (default None). In `_process_action`
aggressive path, check the current tick's
`runner.ltp_velocity_30` (computed by feature_engineer.py); if
above threshold, refuse the open. Counter
`opens_refused_velocity_mask`.

**Result.** _Pending._

---

## 2026-05-19 — R1+R3+R4 combined probe [QUEUED]

**Intention.** Test all three angles simultaneously at probe
scale (5×7d, ~2h GPU). The three mechanisms are clearly
non-overlapping (selection × reward × env-side), so attribution
is clean: a bite tells us the combination works without needing
to disentangle.

Composes with E3's `close_feasibility_max_spread_pct=0.05` —
this run stacks four mechanisms total (E3 + R1 + R3 + R4) and is
the prime candidate for full-cohort escalation if it bites.

**Implementation brief.** Landed commit `0a176b7`. Probe
launcher `C:\tmp\probe_r1r3r4_robust_phenotype.ps1`. Combines:
- `--composite-score-mode sortino` (R1)
- `--reward-overrides naked_loss_quadratic_beta=0.001
  opposite_side_depth_floor=10.0
  close_feasibility_max_spread_pct=0.05
  force_close_before_off_seconds=120`

Cohort tag prefix `_predictor_SCALPING_probe_r1r3r4_*`. 14
regression tests pass (TestSortinoComposite,
TestQuadraticNakedLossPenalty, TestLiquidityFloorGate).

**Result.** **NET NEGATIVE vs E3 alone** (probe scale). 5/5
finished 2026-05-19 23:30.

| Metric | E3 alone (probe) | R1+R3+R4 combined | Δ |
|---|---:|---:|---:|
| pnl mean | +£59 | +£28.9 | **−£30** |
| profitable | 4/5 | 4/5 | tie |
| pnl peak | +£136 | +£64 | −£72 |
| pnl trough | −£29 | −£38 | −£9 |
| locked | +£107 | +£111 | +£4 |
| fc_n | 35 | 36 | flat |
| cl_n | 6.1 | 6.3 | **preserved** (unlike E3+E4 combo) |
| bets | 147 | 150 | +3 |

Per-agent pnl: +£64, +£59, **−£38**, +£31, +£28. The
distribution is TIGHTER than E3 alone (range £102 vs E3's
£165) but the mean is lower because the upside is capped much
more than the downside. One agent still caught a deep naked
(a0011a52, naked −£56) — R3 (β=0.001) and R4 (depth floor
£10) didn't prevent it.

**R1 is inactive at probe scale.** ``composite_score_mode=
sortino`` with ``--generations 1`` means no breeding round, so
the selector never feeds back into reproduction. The probe
effectively tested **R3+R4** stacked on E3. R1 needs the full
cohort test (queued at
`C:\tmp\cohort_e3_sortino.ps1`) for its real signal.

**Mechanism diagnosis: R3+R4 push in the right direction at
TOO-WEAK strength.** Three candidate explanations:

1. **R3 β=0.001 too weak.** A −£100 naked costs only β×10,000 =
   £10 in shaped, vs the £100 in raw. The shaped term is
   one-tenth of the natural raw signal — barely registers in
   PPO's gradient. Probably needs β ≥ 0.01 to actually shift
   behaviour.
2. **R4 depth floor £10 too lenient.** Most opposite-side books
   already > £10 at the top level. Gate rarely fires. Needs a
   higher floor (£30-50) to actually catch thin books.
3. **The mechanisms compete with each other** — both refuse
   "borderline" opens but in different dimensions, so their
   aggregate effect is a smaller set of refusals than each
   would produce alone.

Critically, cl_n is **preserved** at 6.3 (vs 6.1 baseline,
unlike E3+E4 which forced cl_n=0). The agent still actively
manages closes; the R3+R4 stack just doesn't add enough gradient
or constraint pressure to move the worst-case agent's behaviour.

**Verdict: tentatively net negative; ablations + retuning
needed.** Next steps queued:
1. R3 alone — does R3's quadratic loss penalty help by itself?
2. R4 alone — does R4's depth floor help by itself?
3. E4b — attributes the E4 combo's subtraction.
4. E3+Sortino full cohort — R1's REAL test at multi-gen scale.

If R3 alone or R4 alone bites, that mechanism gets retuned
(higher β / higher floor) and stacked. If neither bites alone,
the direction is rejected at probe scale and we focus on R1's
cohort + the layq-style baseline.

---

## 2026-05-19 — R3 alone (ablation) — IDENTICAL to combined

**Intention.** Ablate R3 (quadratic naked-loss penalty β=0.001)
on top of E3 (close_feasibility=0.05) with R4 removed. Attributes
R3's contribution vs the R1+R3+R4 combined probe.

**Result.** **IDENTICAL to R1+R3+R4 combined**, agent-by-agent
to the penny:

| Metric | R1+R3+R4 combined | R3 alone |
|---|---:|---:|
| pnl mean | +£28.9 | +£28.9 |
| per-agent pnl | +£64, +£59, −£38, +£31, +£28 | **+£64, +£59, −£38, +£31, +£28** |
| locked | +£111 | +£111 |
| fc_n | 36 | 36 |
| cl_n | 6.3 | 6.3 |

**Key attribution:** R4 (depth_floor=£10) is **inert** at this
strength — never refused enough opens to change behaviour.
Typical opposite-side top-level size on the cohort's gate config
exceeds £10 nearly always. **R3 alone accounts for the entire
£30/d subtraction vs E3.**

β=0.001 is enough to perturb policy behaviour but in the wrong
net direction: the gradient pressure away from naked losses
doesn't translate to fewer naked losses (a0011a52 still caught
−£56 naked despite the penalty being active). PPO either:
- Can't distinguish "naked likely" from "naked unlikely" at the
  open decision (no representational pathway for it)
- OR overcorrects on safe opens to avoid hypothetical naked
  risk, suppressing profitable trades

Cohort tag `_predictor_SCALPING_probe_r3_alone_1779233480`.

**Verdict: R3 net negative at β=0.001.** Retune candidates:
- β=0.01 (10× stronger gradient pressure) — would a −£100 naked
  costing β×10,000=£100 of shaped (matching the raw cost)
  shift the policy meaningfully? Worth a probe.
- β=0.0001 (10× weaker) — at the limit of what PPO can
  distinguish from noise; probably indistinguishable from E3
  alone.

R4 retune: depth floor would need to be ~£30-50 to actually
fire on the cohort's gate. Queued for follow-on if needed.

---

## 2026-05-19/20 — R4 alone (ablation) — INERT at £10

**Intention.** Confirm R4's inertness by running it alone (no R3).
Predicted to match E3 alone exactly if R4 at depth_floor=£10
never fires.

**Result.** **IDENTICAL to E3 alone, agent-by-agent:**

| Agent | E3 alone pnl | R4 alone pnl |
|---|---:|---:|
| 1 | +£72 | **+£72** |
| 2 | +£86 | **+£86** |
| 3 | −£29 | **−£29** |
| 4 | +£136 | **+£136** |
| 5 | +£32 | **+£32** |
| **mean** | **+£59.4** | **+£59.4** |

Locked, fc_n, fc_£, cl_n, bets — every metric matches. R4 at
depth_floor=£10 is a complete no-op on the cohort's gate config
(raceconf + pwin gates + lay_price_max). Cohort tag
`_predictor_SCALPING_probe_r4_alone_1779237489`.

**Verdict: R4 needs retuning to bind.** Floor at £10 too lenient;
opposite-side top-level size routinely exceeds it on the
admitted races. Strong-variant probe at floor=£30 queued
(`C:\tmp\probe_r4_strong.ps1`) — if it bites the depth-floor
mechanism is real but underweight here; if it ALSO matches E3
alone, the mechanism is structurally inactive on this gate
config and we'd need much higher floors (£100+) to fire at all.

---

---

## 2026-05-19 — E3+E4 combo probe — NEGATIVE add-on

**Intention.** Stack E3 (close-feasibility refusal) + E4
(keep_open inversion + stop_loss=0.10) at probe scale to test
whether the two biters compound. E3 attacks the open phase
(refuse high-spread opens), E4 attacks the close phase
(automate via stop-loss, agent loses active close). The
mechanisms operate on different bet-lifecycle points so might
compose.

**Implementation brief.** Probe launcher
`C:\tmp\probe_e34_combo.ps1`. Cohort tag
`_predictor_SCALPING_probe_e34_combo_1779225377`. Same 5×7d
shape as E1-E6.

**Result.** **NEGATIVE ADD-ON.** E4 mildly subtracts from E3.
5/5 finished 2026-05-19 22:23.

| Metric | Baseline | E3 alone (probe) | E3+E4 combo | Δ vs E3 alone |
|---|---:|---:|---:|---:|
| pnl mean | −£46 | +£59 | **+£34** | **−£25** |
| profitable | — | 4/5 | 4/5 | tie |
| pnl peak | — | +£136 | +£95 | −£41 |
| pnl trough | — | −£29 | −£26 | +£3 |
| locked mean | +£88 | +£107 | +£118 | +£11 |
| fc_£ | −£86 | −£56 | −£48 | +£8 |
| cl_n | 9 | 6.1 | **0** | by-design (suppressed) |

Per-agent pnl: −£26, +£17, +£22, +£61, +£95 — tighter
distribution than E3 alone (range £121 vs E3's £165). The combo
caps both upside AND downside. Locked floor up £11, fc cost
down £8 — small wins. But the upside loss (peak −£41) outweighs
those gains in the cohort mean.

**Mechanism diagnosis.** Inversion forces `cl_n=0` (agent loses
active close). The env stop-loss at 10% MTM drop handles
obvious drawdowns but doesn't trigger on naked legs that aren't
pair-MTM-losing at the close tick. Result: trades that the
agent's `close_signal` would have actively closed for £-£5 of
spread cost now ride to settle naked and pay full naked
variance instead. The combo trades the £-bounded close-leg
losses (−£8 fc improvement) for unbounded naked variance.
Net negative.

**Verdict: drop E4 from forward stacks.** The R1+R3+R4 launcher
already excludes E4 (correct by accident — the plan
prioritises R-series mechanisms). Queued the E4b ablation
(`stop_loss=0.10` alone, no inversion) to attribute whether
the inversion or the stop_loss was the load-bearing subtractor;
useful intelligence but lower priority than the R1+R3+R4
probe.

### E4b ablation (stop_loss alone, no inversion) — confirms stop_loss is the subtractor

5/5 finished 2026-05-20 01:50.

| Stack | pnl mean | Δ vs E3 alone |
|---|---:|---:|
| E3 alone | +£59 | baseline |
| E3 + E4 (inversion + stop_loss) | +£34 | −£25 |
| **E3 + stop_loss alone (E4b)** | **+£36** | **−£23** |

E4b matches E4 combo to within £2. **The stop_loss IS the load-
bearing subtractor; inversion alone was roughly neutral or
marginally helpful.** Cohort tag
`_predictor_SCALPING_probe_e4b_stoploss_only_1779241480`.

**Mechanism diagnosis:** `cl_n` dropped from baseline 9 to 4.7
(vs E3's 6.1). The stop_loss closes ~4 pairs/race that the
agent's `close_signal` would have managed itself, AT WORSE
PRICES — stop_loss fires once MTM has drifted 10% against the
pair, by which point the cost-to-close is higher than the
agent's optimal earlier exit. Earlier-but-bigger losses instead
of later-but-smaller losses.

Also: stop_loss can auto-close pairs that would naturally have
matured profitably (the 10% MTM drift was a temporary mid-race
swing; the price recovers and the pair completes positively).
Stop_loss kills those by force-closing prematurely.

**Combined attribution: ALL three add-on stacks tested net-
subtract from E3 alone:**

| Add-on | Mechanism class | Δ vs E3 |
|---|---|---:|
| R3 β=0.001 | reward (quadratic naked-loss) | −£30 |
| R4 floor=£10 | env (depth gate) | ±£0 (inert) |
| E4 inv + stop_loss | env+action (close-side) | −£25 |
| stop_loss alone | env (close-side) | −£23 |

E3 alone is a strong local optimum; every close-side or
naked-side mechanism tested adds net negative at probe scale.
The remaining hypothesis to test is that the magnitudes were
all *too weak* (R3, R4) or the mechanisms were *wrong-
direction* (stop_loss, inversion). R3-strong and R4-strong
probes queued.

### R3-strong (β=0.01) — partial recovery, still −£17 vs E3 alone

5/5 finished 2026-05-20 02:57.

| Stack | pnl mean | locked | per-agent range |
|---|---:|---:|---:|
| E3 alone | +£59 | +£107 | −£29 to +£136 |
| R3 weak (β=0.001) | +£29 | +£111 | −£38 to +£64 |
| **R3 strong (β=0.01)** | **+£42** | **+£113** | **−£25 to +£76** |

**+£13 lift from weak → strong; gradient magnitude DID matter.**
Still −£17 vs E3 alone. But the per-agent shifts vindicate the
mechanism's design intent:

- **Best agent: +£64 → +£76** (responders amplify with β↑)
- **Worst agent: −£38 → −£25** (left-tail truncation IS working)
- **Locked floor: +£111 → +£113** (modest lift)

The cohort mean is dragged by middle-bucket high-variance
agents (+£31/+£28 in R3-weak became +£16/+£69 in R3-strong).
Per-agent response variance grew with β. This is the typical
high-variance-response signature that **benefits from GA
selection across generations** — breeding amplifies the strong
responders. Probe scale (n=5, no breeding) just averages a
high-variance population.

Cohort tag `_predictor_SCALPING_probe_r3_strong_1779245500`.

**Verdict: R3 at β=0.01 is the most promising add-on we've
seen.** Worst-day truncation works; locked floor lifts;
upside-responders amplify. The probe-scale mean isn't biting
but the per-agent signal is. A full cohort with R3 β=0.01 +
Sortino selection should compound this — Sortino specifically
selects for the bounded-worst-day phenotype that R3 is producing.

**Queued: R3 even-stronger (β=0.05) probe** to test whether
going further continues the pattern or breaks it (a β too
strong would over-clip the safe trades and collapse locked
floor).

### R4-strong (floor=£30) — over-aggressive

5/5 finished 2026-05-20 04:04.

| Stack | pnl mean | locked | bets | fc_n | fc_£ |
|---|---:|---:|---:|---:|---:|
| E3 alone | +£59 | +£107 | 147 | 35 | −£56 |
| R4 weak (£10) | +£59 | +£107 | 147 | 35 | −£56 (inert) |
| **R4 strong (£30)** | **+£36** | **+£71** | **101** | **27.5** | **−£43** |

Per-agent: −£26, **+£137**, −£7, +£3, +£72. The +£137 agent
caught a +£105 naked tailwind. Strip that outlier and the
remaining 4 average +£10.5 — clearly worse than E3.

Cohort tag `_predictor_SCALPING_probe_r4_strong_1779249521`.

**Mechanism diagnosis:** £30 floor refuses ~30 % of opens (bets
147 → 101). Catches some bad opens (fc cost down £13/d) but
also a lot of profitable opens (locked floor crashed £36/d).
The cohort's raceconf + pwin 0.20/0.40 gate already restricts to
relatively-liquid races; a £30 floor is over-aggressive — many
profitable scalps happen against £15-25 books.

Sweet-spot floor for R4 (if any) is between £10 (inert) and
£30 (over-aggressive). Probably £15-20. **Not worth another
probe** — R4's mechanism is real but the bandwidth where it
helps without over-clipping is narrow, and R3's mechanism (at
β=0.01) showed more promising probe-scale per-agent dynamics.

---


