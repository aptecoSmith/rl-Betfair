---
status: review-document
opened: 2026-04-26
author: H2 diagnostic session
type: first-principles architecture review
---

# Greenfield review — what would we build from scratch?

## Plain-English summary

**The bet you're trying to make.** For each horse on each tick, the
agent should decide "do I open a back/lay pair on this horse, or
not?" That's basically a yes/no question, one horse at a time.

**What we built instead.** Every tick, the agent outputs 70 numbers
all at once — five sliders for each of 14 horses (one for "do I
bet?", one for "how much?", one for "how aggressively?", etc.). The
env interprets those sliders into actual bets.

That's like asking a chess player to make a move by adjusting 64
sliders (one per square) instead of just saying "rook to e4". You
can do it, but you're fighting the natural shape of the decision.

**Why it matters.** Every problem we keep running into is a knock-on
effect of that choice:

- **The H2 finding** (per-tick credit measurement): when the agent
  makes a bad bet on horse #5, the "that was bad" feedback gets
  spread thinly across all 14 horses, not focused on horse #5. We
  measured the agent only receives about a quarter of the feedback
  signal it should.
- **The H1 finding** (label conflation): we had to bolt extra
  "discrimination heads" onto the agent so it could tell horses
  apart per-runner. One of those heads got mis-labelled and made
  things worse.
- **The reward shaping** (8+ different bonuses and penalties): we
  keep adding new ones because the agent struggles to learn from
  the natural P&L signal alone. Each one is a patch for the same
  root issue.
- **The entropy controller, KL early-stop, advantage normalisation,
  LR warmup**: all defending against the agent's own instability,
  which mostly comes from having 70 outputs to balance.

**What greenfield-me would build instead.** Two simpler pieces:

1. A **standalone "is this a good bet?" model** trained on historical
   data with a clear yes/no label. Train it once, freeze it.
2. A **thin RL agent on top** that just picks which of the highest-
   scored opportunities to actually take given the budget. One bet at
   a time, not 70 sliders.

Plus realised P&L as the only reward — no shaping. The architecture
does the work the shaping is currently trying to do.

**Have we taken a wrong turn?** Yes — but it's an old one (action-
space design), not a recent one. Every individual decision since
then was sensible given what came before. The cumulative system is
solving the wrong shape of the problem.

**Should we throw it all out?** No. The fix can be staged:

1. **Cheap fix first** (per-runner value head) — gets the feedback
   to focus on the right horse without changing the action space.
   Try this if cohort-M fails.
2. **Medium fix next** (standalone scoring model fed to the agent as
   features) — tests whether the agent can learn selectivity if it's
   HANDED a calibrated "is this a good bet?" score.
3. **Expensive rewrite last** (change the action to "pick one
   horse"). Only if both above fail.

The discrete-action rewrite stays as the last resort because it
breaks every saved weight and you can't tell what fixed things if it
works (could be the action change, could be the credit assignment,
could be the exploration shape — all confounded).

The technical detail behind every claim above is in Parts 1–6 below.

---

## What this document is

This document does an exercise: forget the current system, design the
RL pipeline you would actually build for this problem from first
principles, then compare to what exists. The goal is to surface
candidate "wrong turns" — places where the current architecture
diverges from the principled design in ways that are causing the
problems we keep running into (force-close rate stuck at ~75 %,
selectivity gap across cohorts O / O2 / F / M, partial H2 attenuation
of per-tick credit).

This is a thinking document, not a recommendation to scrap and rebuild.
Some divergences will look like real wrong turns; others will look
reasonable in hindsight.

---

## Part 1 — What is the problem, really?

Strip away "RL" and "PPO" for a moment. What is the user actually
trying to accomplish?

> Given live Betfair price/depth/LTP feeds for ~200–400 GB horse
> markets per day, place pairs of orders (back at price A, lay at
> price B, A > B) such that **both fill before the off**, locking in
> the spread A − B as profit. Avoid pairs that fill on only one leg
> ("naked"), because then you carry race-outcome risk for thin
> compensation.

That problem decomposes cleanly into two sub-problems:

**Sub-problem A — the per-opportunity decision.**
For a (runner, tick, side) tuple, the question is binary: "if I open
a paired bet here, will both legs fill before the off?" If yes,
expected value = (spread − commission) × stake. If no, expected
value = E[naked loss | this opportunity] × stake (negative).

This is a **supervised classification problem**, not an RL problem.
The label is observable from historical data (did the second leg
fill before the off, and if so, did it fill before the agent had to
force-close?). Features are the price book state, runner attributes,
time-to-off, traded volume velocity, etc. A standard binary classifier
trained on a few weeks of historical data should produce a calibrated
P(mature | features) in the 0.0–1.0 range.

**Sub-problem B — the portfolio decision.**
Given a budget that's small relative to the universe of opportunities,
which subset of high-EV opportunities do you take? When do you cancel
an open passive that hasn't filled? When do you cross out (close) a
matured pair that's drifted against you? Are there cross-runner
correlations that mean "open on runner A" implies "don't open on
runner B in the same race"?

This is closer to an RL problem — sequential, budget-constrained,
path-dependent. But it's a SMALL RL problem — the policy class is
essentially "open the top-K opportunities that fit budget", with K
depending on portfolio state.

---

## Part 2 — Greenfield design

If I were starting from scratch tomorrow, this is what I would build.

### Layer 1 — Supervised opportunity scorer

A standalone supervised model that consumes (runner_features,
market_features, time_to_off) and outputs three calibrated heads:

1. **P(mature | open here, side, …)** — binary, BCE loss against the
   historical label "did this open mature within the close window
   without needing a force-close".
2. **E[realised P&L | open here, mature]** — regression, MSE against
   realised locked spread on matured pairs.
3. **E[loss | open here, naked]** — regression, MSE against realised
   naked-leg P&L on the negative class.

Training data: a few weeks of historical races, one row per
(runner, tick, side) where the agent COULD have opened. The labels
are derivable post-hoc from the historical price book by simulating
"if I opened at this tick what would have happened" — exactly what
the env's matcher already does.

This model is trained ONCE, frozen, and serves as a feature provider
for the policy.

**Why supervised first, not RL?** Because the per-opportunity question
has a clear ground truth and doesn't suffer from credit assignment
problems. RL is the right tool when the policy class is unknown; for
"score this opportunity" the policy class is literally "is the score
above a threshold?".

### Layer 2 — Greedy or thin-RL allocator

Given the scorer's output for every (runner, tick, side) opportunity
in the current race, the policy decides which to act on:

**Option A — Pure greedy with one tunable threshold.**
At each tick: for each runner, compute
`EV = P(mature) × E[realised | mature] − (1 − P(mature)) × E[loss | naked]`
. Open on every (runner, side) with `EV > threshold` AND `EV / stake_required > yield_threshold`, subject to remaining budget. Tune the
two thresholds against historical P&L on a held-out window.

**Option B — Thin RL on portfolio allocation only.**
The action space is one Categorical per tick: `{do_nothing, open(runner_i, side)} for i in active_runners`. The reward is realised P&L per pair at settlement. The state includes the scorer's outputs as features. The RL agent learns when to defer vs act, and learns from the residual signal not captured by the supervised scorer.

Both options have the **same architectural property**: per-opportunity
credit assignment is structural, not statistical. In option A, no RL.
In option B, the discrete action concentrates the policy gradient on
the chosen runner.

### Layer 3 — Portfolio risk envelope

Independently of layers 1 and 2, a hard rules-based layer:

- Max open pairs per race
- Max budget per race (and per day)
- Force-close window (the existing T−N seconds rule is fine)
- Max simultaneous opens on correlated runners (top-2 favourites
  often have inverse-correlated price moves)

This layer has zero learned parameters. It just refuses risky
combinations.

### Layer 4 — Online evaluation

Every day, the supervised scorer's calibration is re-checked on the
day's data. If P(mature) starts diverging from realised mature rate,
the scorer is re-trained. The allocator's threshold is also tracked
against per-day yield.

### Reward, action, and critic shape

If we use Option B, the design choices are:

- **Action**: Categorical over `{noop} ∪ {(runner_i, side) for i in 0..N-1}`. Mutually exclusive — one action per tick. Plus one continuous head for stake (could be discrete bins too).
- **Critic**: per-runner, NOT scalar. Each runner gets its own value estimate `V(s, runner_i) → R`. GAE bootstraps per-runner. The value head is just a small MLP over (runner_features, portfolio_state).
- **Reward**: realised P&L at settlement, NO shaping terms. The supervised scorer already provides per-opportunity guidance via its features; the RL layer doesn't need shaped pressure.
- **Entropy / exploration**: the discrete action space makes entropy a clean scalar (categorical entropy). No need for the per-head entropy floor controller.

### Hyperparameter strategy

For ~25 hyperparameters, **Bayesian optimisation or grid search**, not GA. The GA's value is when you have hundreds of agents running for many generations and can afford the throughput. For the actual scale here (12 agents × 18 episodes × ~70 min) BayesOpt over 30 trials would converge faster and give better single-run results.

### Data scale

Currently: 12 days of training data. For supervised learning the rule
of thumb is ~10× more data than parameters. A small classifier (a few
hundred K params) needs millions of examples. ~12 days × ~300 markets
× ~14 runners × ~1000 ticks × 2 sides ≈ 100M examples. Plenty.

For RL training: 6 days × 18 episodes is tight but workable for the
thin allocator (option B).

---

## Part 3 — What we actually built

The current system has:

| Layer | Greenfield | Actual |
|---|---|---|
| **Supervised scorer** | Standalone, trained once, frozen | None as a separate stage. `fill_prob_head` and `mature_prob_head` are auxiliary heads INSIDE the policy, trained jointly with the actor on the same rollout data. BCE labels are computed at episode-end backfill from `env.all_settled_bets`. |
| **Action space** | Categorical over runners + small continuous heads | 70-dim continuous: 5 heads × 14 runners (`signal`, `stake`, `aggression`, `cancel`, `arb_spread`), each independent Gaussian. The agent acts on every runner every tick. |
| **Critic** | Per-runner value head, GAE per-runner | Single scalar value head. GAE on a single trajectory. This is the H2 finding: ~75 % of per-runner credit is smeared. |
| **Reward** | Realised P&L only | Realised P&L + 8+ shaped terms competing: `early_pick_bonus`, `precision_reward`, `efficiency_penalty`, `mark_to_market`, `naked_loss_anneal`, `matured_arb_bonus`, `open_cost`, `+£1 per close_signal`, BC pretrain offset, … |
| **Entropy / exploration** | Categorical entropy, no controller needed | Per-head entropy floor controller + target-entropy SAC-style alpha controller (`entropy-control-v2`) with hand-tuned target = 150 for 70 Gaussian dims. |
| **Hyperparameter strategy** | BayesOpt or grid over ~25 knobs | Genetic algorithm over 25 genes, 12 agents per cohort, 18 episodes per agent, multi-cohort probes. |
| **Risk envelope** | Hard rules-based layer above the policy | Force-close at T−N is rules-based ✅. Other risk constraints are mostly emergent from shaped reward terms (`naked_loss_scale`, `naked_penalty_weight`, `efficiency_penalty`). |
| **Online evaluation** | Per-day calibration check on the supervised scorer | Per-cohort scoreboard against held-out test days; no per-day calibration of the auxiliary heads. |

---

## Part 4 — Where the current design differs (diff-by-diff)

Each row below is a divergence between greenfield and actual. The
"verdict" column is my honest call on whether it looks like a wrong
turn, a reasonable choice, or a "jury's out".

### D1. Action space: continuous-multi-head vs discrete-over-runners

**Greenfield**: Categorical over runners ensures structural per-runner
credit assignment.

**Actual**: 70-dim continuous Gaussians. Every runner gets a policy-
gradient update on every tick, regardless of which runner the agent
acted on.

**Why it matters**: This is the structural cause of the H2 partial
attenuation we just measured. The per-tick credit signal at the open
tick is ~25 % of theoretical maximum because the gradient is
distributed across ALL 14 runners (and across 5 heads each), not
focused on the runner the agent actually opened.

**Verdict: Likely wrong turn.** The continuous-multi-head design was
inherited from generic continuous-control PPO setups (think MuJoCo,
robot locomotion) where every actuator does need a per-step continuous
update. This problem is fundamentally discrete: "which runner do I
open on this tick, if any". The continuous formulation is fighting
the natural shape of the decision.

**Confidence: HIGH** that the action space is misaligned with the
decision problem. **MEDIUM** on whether changing it would unlock
selectivity — the discrete reformulation also constrains the policy
class (one open per tick vs many), which has its own costs.

### D2. Critic: scalar vs per-runner

**Greenfield**: Per-runner value head; GAE per-runner.

**Actual**: Scalar value head; GAE on aggregate trajectory.

**Why it matters**: The H2 diagnostic showed ~75 % of per-tick credit
is consumed by the value head's bootstrap producing similar predictions
for similar states regardless of which runner was opened. A per-runner
value head closes this directly.

**Verdict: Likely wrong turn**, but a smaller one than D1. This is
fixable as a surgical change without touching the action space.

**Confidence: HIGH** that this is a real attenuation source.
**MEDIUM-HIGH** that fixing it alone would close the selectivity gap
(the diagnostic showed the residual signal IS direction-correct, just
quiet — amplifying it via per-runner critic is the proportionate fix).

### D3. Supervised heads inside the policy vs separate scorer

**Greenfield**: Supervised scorer trained standalone, frozen, used as
features.

**Actual**: `fill_prob_head` / `mature_prob_head` / `risk_head` train
jointly with the actor on the rollout data the policy itself produces.

**Why it matters**: Several second-order issues stem from this:

1. **The labels reflect what the CURRENT policy did, not the universe
   of possible decisions.** If the policy never opens on a runner, the
   head never gets training data on that runner. The scorer's
   coverage is policy-dependent.
2. **Joint training tangles the loss landscape.** The actor and the
   head are coupled: a head update changes the actor's input, which
   changes the rollout, which changes the head's labels next epoch.
3. **The labels are computed post-hoc from `env.all_settled_bets`,
   which is what found the H1 bug** (label conflated force-closes with
   matures). A standalone scorer trained on historical data with
   carefully-defined labels would have surfaced this immediately.

**Verdict: Probable wrong turn**, but the fix is partly underway. The
mature_prob_head plan landed two days ago and it's the right
direction. The cleaner version — train it standalone first, then
freeze and feed to actor — would be a moderate refactor.

**Confidence: HIGH** on the limitations of joint training. **MEDIUM**
that the fix requires going fully greenfield here; the existing aux
heads with the strict label could be sufficient if D1 / D2 are fixed.

### D4. Reward shape: 8+ competing terms vs P&L-only

**Greenfield**: Realised P&L at settlement. Done.

**Actual**: 8+ shaped terms with hand-tuned magnitudes:

- `early_pick_bonus`, `precision_reward`, `efficiency_penalty` — original directional-betting terms (now nearly inactive in scalping mode).
- `mark_to_market` — per-step shaping on unrealised P&L delta. Telescopes to zero at settle.
- `naked_loss_scale`, `naked_penalty_weight` — penalise unhedged exposure.
- `matured_arb_bonus`, `+£1 per close_signal` — reward closing pair lifecycles.
- `open_cost` — selective-open shaping; charge at open, refund at favourable resolve.
- BC pretrain offset on entropy controller.
- `shaped_penalty_warmup_eps` — schedule scale on penalty terms in the first N episodes.

**Why it matters**: Each shaping term was added in response to a
specific observed failure of the previous setup. The result is a
balance of forces tuned by hand, with multiple terms pushing the same
direction (e.g. `naked_penalty_weight` AND `naked_loss_scale` AND the
asymmetric raw-naked-loss accounting). The selectivity gap could
plausibly be "the shaped terms are louder than the per-runner credit
signal H2 leaves intact, and the policy optimises the louder thing".

**Verdict: Reasonable in evolution, problematic in aggregate.** Each
addition was justified at the time. The CUMULATIVE complexity is the
problem — it's hard to know which term is doing the work and which is
fighting the others. A clean redesign would be P&L only with the
architecture (D1, D2, D3) doing the credit-assignment work that the
shaping terms are currently trying to do.

**Confidence: HIGH** that the reward stack is over-engineered.
**LOW-MEDIUM** that simplifying it WITHOUT fixing D1/D2/D3 would help
— the shaped terms exist BECAUSE the architecture can't do credit
assignment unaided. They are symptoms of the structural problem, not
the cause.

### D5. Entropy / exploration: SAC-style controller vs categorical

**Greenfield**: Categorical entropy is a clean scalar; basic
entropy bonus is enough.

**Actual**: Two-stage controller (per-head floor + target-entropy SAC
alpha) with target = 150 hand-tuned for 70 Gaussian dims, plus BC
warmup handshake.

**Why it matters**: This complexity exists BECAUSE the action space
is high-dim continuous (D1). Fix D1 and most of this evaporates.

**Verdict: Symptom, not cause.** The entropy controller is doing
useful work given the action space; it's not itself a wrong turn.
But it would be wholesale unnecessary in the greenfield design.

**Confidence: HIGH** that this is downstream of D1.

### D6. Hyperparameter search: GA vs BayesOpt/grid

**Greenfield**: BayesOpt over 25 knobs, 30–50 trials.

**Actual**: GA with 12 agents × 1+ generations, breeding pool, mutation rate, adaptive mutation cap, etc.

**Why it matters**: GA is well-suited when you can afford 1000s of evaluations and want diversity preservation. At 12 agents × 70 min per cohort, you're paying GA's diversity overhead without getting its scale benefits. BayesOpt on the same compute budget would converge faster on the optimum (single-best rather than diverse-population).

**Verdict: Reasonable choice given the framing**, but possibly a wrong
choice given the actual scale. The GA was set up expecting much larger
populations and many generations; in practice each cohort is one
generation of 12.

**Confidence: MEDIUM-HIGH** that BayesOpt would beat GA at this scale
on single-best-agent metrics. **LOW** on whether it matters for the
core problem (the architectural issues are upstream of HP search).

### D7. Single trajectory vs per-runner trajectory in GAE

**Greenfield**: GAE runs once per (runner, tick) pair, on per-runner
rewards.

**Actual**: GAE runs once on the aggregate trajectory; rewards are
race-aggregate.

**Why it matters**: This is just a restatement of D2 from a different
angle. The fix is the same: per-runner critic + per-runner advantage
computation.

**Verdict: Same as D2.**

### D8. No supervised pre-training of the scorer

**Greenfield**: Train P(mature) classifier on historical data
standalone first.

**Actual**: BC pretrain on arb-oracle samples is the closest analog —
behavioural cloning of a hand-coded oracle policy. But this is BC of
ACTIONS (which arb to take), not supervised forecasting of OUTCOMES
(will this arb mature).

**Why it matters**: BC teaches the policy to MIMIC the oracle. It
doesn't teach the policy to FORECAST opportunity quality. A trained
P(mature) head would let the actor reason about opportunities the
oracle didn't take.

**Verdict: Different tools for different jobs.** BC + supervised
scorer would both be useful. The current setup has BC; a standalone
supervised scorer is the missing piece.

**Confidence: MEDIUM** that adding a supervised scorer would help.
**HIGH** that the absence of one is a real gap.

---

## Part 5 — Diagnosis: have we taken a wrong turn?

My honest read is: **YES, but the wrong turn is older and bigger than
H1 or H2.**

The wrong turn was choosing a 70-dim continuous-multi-head action
space (D1) for a problem that's fundamentally about discrete per-
opportunity decisions. Once that choice was made:

- A scalar value head was unavoidable (D2) — per-runner critic on a
  continuous-multi-head action space is awkward.
- Auxiliary heads inside the policy became necessary (D3) — without
  per-runner discrimination in the action structure, you have to bolt
  it on.
- Heavy reward shaping became necessary (D4) — to compensate for the
  credit-assignment loss that the architecture can't fix.
- A complex entropy controller became necessary (D5) — to manage
  exploration in 70 continuous dims.

Each individual addition was the right local move given the existing
constraints. The cumulative system is solving the wrong shape of
problem.

The "selectivity gap" (force-close rate stuck at ~75 % across cohorts
O / O2 / F) and the "H2 partial attenuation" finding from this
session are both symptoms of D1. The H1 finding was a symptom of D3
(joint training of aux heads on policy-generated labels). All three
findings would be much smaller problems in the greenfield design.

**That said:** going back to D1 is expensive. It's the action-space
change discussed in the prior turn. It breaks every saved weight,
every test pattern, every diagnostic intuition. The smaller fixes
(D2 = per-runner value head, D3 = standalone supervised scorer) are
real improvements that don't require a wholesale rewrite.

### Concrete recommendations from this exercise

1. **Keep cohort-M running.** The H1 fix may be sufficient on its own
   given the existing architecture's limitations. Selectivity might
   recover even with the residual H2 attenuation.

2. **If cohort-M fails (ρ ≈ 0):** open `per-runner-value-head` (D2).
   This is the surgical-minimum H2 fix and matches the diagnostic's
   exact failure mode. Expected to close 60–80 % of the credit gap.

3. **If per-runner-value-head also fails:** the next move is NOT
   discrete action over runners (D1). It's a **standalone supervised
   scorer** (D3) trained on historical data with carefully-defined
   labels for `P(mature without force-close)`. Use it as a frozen
   feature into the existing actor. This is moderate-surgery and
   tests "is the per-runner discrimination signal itself the
   bottleneck, independent of the credit-assignment problem".

4. **Discrete-action reformulation (D1) is the last resort, not the
   second.** It's the biggest change with the least clear attribution
   path (a win could be due to action-space change OR per-runner
   credit OR exploration shape change — all confounded). Reserve for
   "we've tried everything else and the policy still won't be
   selective".

5. **Reward simplification (D4) is the long-term direction.** Each
   architectural fix (D2, D3) lets you remove one or two shaping
   terms. Track the reward stack's complexity as a debt metric and
   peel it back as the architecture absorbs more of the work.

6. **HP search (D6) is downstream of everything else.** Fix the
   architecture first; whether you use GA or BayesOpt on the resulting
   smaller hyperparameter surface is a second-order concern.

---

## Part 6 — What this exercise does NOT say

Just to be explicit about the bounds:

- **It does not say "the work to date was wrong."** Every change was
  defensible given what was known. The H1 finding two days ago was
  load-bearing diagnostic work that any future architecture would
  benefit from. The KL-fix work was real instability that needed
  fixing regardless of the architectural direction.
- **It does not say "rewrite everything."** D1 is expensive. D2 is
  cheap. D3 is moderate. The recommendation is to take the cheap fix
  first, the moderate fix second, the expensive fix only if necessary.
- **It does not say "RL was the wrong choice."** Layer 2 in the
  greenfield design IS still RL — it's just thin RL on top of a
  supervised feature stack. The choice to use RL for portfolio
  allocation is reasonable. The choice to use RL for per-opportunity
  scoring (which is what the current architecture effectively does)
  is the questionable one.
- **It does not predict that fixing D1/D2/D3 will produce a profitable
  agent.** It predicts that fixing them will remove ARCHITECTURAL
  obstacles. The data scale, the market dynamics, the commission
  structure — all those still need to admit a profitable strategy.
  An architectural fix that surfaces "actually this is a hard market
  to scalp profitably" would itself be valuable information.

---

## Stop

This is a review document, not a plan. The next concrete decision
point is cohort-M's outcome. From there:

- ρ ≤ −0.5: H1 fix sufficient. Defer this whole document.
- ρ ≈ 0: open `per-runner-value-head` plan. Re-read this document to
  decide whether to bundle the standalone supervised scorer at the
  same time.
- ρ ≥ +0.2: something is structurally wrong; come back to this
  document in detail and consider whether the action-space rewrite
  (D1) needs to happen sooner rather than later.
