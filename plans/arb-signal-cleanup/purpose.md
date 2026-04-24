# Purpose — Arb Signal Cleanup

## Why this work exists

The `arb-curriculum-probe` (plan 277bbf49, 2026-04-20/21, 66
agents × 2 gens completed) validated the structural premise
of `plans/arb-curriculum/` but flagged two concrete failures
that block a scale run. Full validator result: 3/5 (C2, C3,
C5 pass; C1 and C4 fail). See
`plans/arb-curriculum/progress.md` Validation entry.

The two failures trace to a single root cause: **the first
~10 episodes after BC pretrain produce reward-signal noise
that swamps real learning.** Three mechanisms contribute:

1. **Entropy drift outruns the controller.** 17/66 agents
   (26 %) stopped betting by ep15. Their entropy trajectory
   was uniformly 139 → 170–184 across ep1–ep10; the
   target-entropy controller's `alpha_lr = 1e-2` SGD step
   could not arrest drift once entropy passed ~157. Once the
   policy diffuses far enough above target, `close_signal`
   and `requote_signal` lose probability mass and the agent
   collapses into the zero-action attractor. C1 failure.
2. **Shaped penalties dominate early cash P&L.** Only 1/66
   agents reached positive `total_reward`, but 7/66 had
   positive cumulative *cash* P&L across the 18 episodes
   logged (58ee1816 +£4 611, 308236be +£4 301, 5b93d7b3
   +£2 544, …). Their reward went negative because
   `efficiency_penalty × bet_count` + centred
   `(precision − 0.5) × precision_bonus` overwhelmed the
   positive raw-pnl signal. The agents making money were
   getting punished for how they made it — the natural
   post-BC exploration shape (high count, mediocre
   precision) was exactly what the shaping terms penalise.
   C4 failure.
3. **Naked outcomes dominate reward variance.** Even the
   agents meeting C4 tended to do so via lucky naked
   windfalls, not controlled arbing. 2ac21f95 (the one C4
   winner) had 0 close_signal uses and 95 naked outcomes;
   its +31.51 peak was a directional accident. Meanwhile
   naked losses at `naked_loss_scale = 1.0` moved agent
   rewards by ±£100s per race, dwarfing both the matured-
   arb bonus (capped at ±£10) and the MTM shaping (~±£1 per
   race). The training signal tracks luck, not skill.

These three mechanisms all bite hardest in the first ~10
episodes — exactly the window where BC's warm-start is
meant to be compounding. A fix to any one of them in
isolation is dominated by the other two. We fix all three
in one probe.

## The plan in one sentence

**Cleanse the first-10-episode training signal at three
points simultaneously: (a) give the entropy controller real
authority so it can arrest drift; (b) warm up the shaped
penalties so the agent isn't punished for exploring; (c)
force-close open positions at T − 30s so naked variance
stops dominating the reward signal.**

## Design sketch

### (A) Entropy controller velocity

Widen the `alpha_lr` range via a new GA gene and let the
probe tell us *how much* velocity the controller needs —
not a blind retune. Current default is `alpha_lr = 1e-2`
(SGD, momentum 0) and it's hardcoded. The new gene range
lets the probe sample [1e-2, 1e-1] across the population;
successful cohorts tell us the right band. Log_alpha clamp
floor stays at `log(1e-5)` (no changes there — the problem
isn't clamp width, it's step magnitude).

Why a gene not a config retune: the controller's effective
authority depends on reward magnitude, entropy magnitude,
and episode cadence — all of which this plan *also*
changes. A flat retune would pick the wrong value. Let the
GA find it.

### (B) Shaped-penalty warmup

New plan-level knob `shaped_penalty_warmup_eps: int`. Over
the first N episodes of each agent's training, the
`efficiency_cost` and `precision_reward` terms are scaled
linearly from 0 → 1. Everything else — MTM, matured-arb
bonus, early_pick_bonus, drawdown, spread_cost, inactivity,
naked_penalty, early_lock — stays at full strength.

Why those two specifically: they're the shaping terms that
penalise the natural post-BC behaviour (exploration-level
bet counts, not-yet-calibrated precision). The other terms
either reward behaviour we want (MTM, matured-arb,
early_pick) or penalise behaviour we definitely don't want
at any episode (naked losses, drawdowns). Warming only the
penalties avoids rewarding "do nothing" — the agent still
gets positive gradient for good arbing from ep1.

Why plan-level not gene: this knob is a *calibration* of
the reward shape for the BC→PPO handshake, not a
behavioural axis to evolve. One value across the
population, like `bc_target_entropy_warmup_eps` already is.
Default `0` = byte-identical to pre-change.

### (C) Force-close at T−30s

New env config knob `force_close_before_off_seconds: int`
(default 0 = disabled, byte-identical). When > 0 and
`scalping_mode` is on, each env step first checks
`time_to_off` against the threshold. Below threshold, the
env iterates every open position with an unfilled second
leg and issues a best-available market-close through the
existing `ExchangeMatcher` (same junk filter, same LTP
guard). Closes the second leg at the best post-filter
opposite-side price; if the matcher can't find a priceable
counter-leg (unpriceable runner, runaway book), the
position stays naked and is handled by existing naked-term
accounting.

**What this replaces, not removes.** Force-close converts
"naked through the off (±£100s variance)" into "closed at
T−30s at best-available spread (±£0.50–£3.00 bounded
cost)." Nakedness still exists for the unpriceable-runner
edge case, but becomes the exception, not the rule. The
per-pair naked term keeps its current full-value accounting
for those residual cases.

**Credit assignment.** Force-close P&L lands at the
force-close step, not at settle. The agent who opened a
doomed position 200 ticks earlier will see the loss arrive
closer in time to the decision that caused it. MTM shaping
already smooths this within the race; force-close makes the
final unwind realised instead of a naked gamble.

**What this does to `close_signal`.** The existing agent-
initiated close action's job becomes *beat the force-close
spread by closing earlier*. That's the correct pressure —
the agent should learn spread-sensitivity as an active
skill, not a passive outcome. Matured-arb bonus keeps
crediting both natural matures and `close_signal` closes;
force-closes are excluded from the matured count (the agent
didn't choose to do them).

### Why these three together, not sequential

Each intervention addresses a distinct piece of the
first-10-episode signal problem, and each is dominated by
the others' noise if run alone:

- **(A) alone** — entropy controller has authority, but
  reward is still garbage (nakeds + penalised exploration),
  so stable exploration explores a noisy gradient.
- **(B) alone** — exploration is unpenalised, but entropy
  still drifts off target and nakeds still dominate
  variance.
- **(C) alone** — reward signal is bounded, but the agent
  still explodes entropy and still gets punished for
  post-BC exploration shape.

Sequential probes take ~18–24 wall-clock hours each (33
agents × 4 gens × 3 epochs ≈ the 277bbf49 timing). Running
them independently means three runs ≈ 72 hours with each
subsequent run's baseline contaminated by the prior fixes.
One combined 48-agent probe with a three-cohort ablation
(16 × all-three / 16 × entropy-only / 16 × no-entropy-but-
warmup-and-forceclose) tells us in one ~25-hour run which
fixes are load-bearing, or whether all three are needed.

## What success looks like

Same five criteria from `plans/arb-curriculum/purpose.md`,
re-evaluated on the new probe:

1. **≥ 80 % of agents remain active through ep15**
   (bets > 0). Tests (A) — if the entropy controller has
   real authority, the 17-agent zero-bet collapse should
   disappear.
2. **`arbs_closed / (arbs_closed + arbs_naked) > 15 %` on
   ≥ 3 agents at ep15**. Unchanged target. The prior probe
   already passed this at 9/66 agents; force-close changes
   the interpretation slightly (naked count now only
   includes unpriceable residuals) but the threshold is
   still calibrated to "agent is running controlled arbs,
   not gambling".
3. **`policy_loss ≥ 0.1` on ≥ 50 % of agents at ep15.**
   Unchanged target. Gradient survival.
4. **≥ 3 agents reach `total_reward > 0` by the end of
   gen 3.** Tests (B) and (C) jointly — if penalties are
   warmed up AND naked variance is bounded, the cash-
   positive agents from 277bbf49 should translate into
   reward-positive agents.
5. **Invariant `raw + shaped ≈ total` holds every episode.**
   Correctness gate; non-negotiable.

**Ablation signal** (secondary, not pass/fail):

- If the "all three" cohort passes C1+C4 but the other two
  don't, all three fixes are load-bearing.
- If "entropy-only" passes C1 alone and "no-entropy"
  passes C4 alone, the fixes attack independent failures
  (most likely outcome per the diagnosis).
- If "all three" passes both and "entropy-only" also
  passes C4, we over-scoped and (B)+(C) are optional —
  valuable to know.

## What this plan does NOT change

- **Matcher semantics.** `env/exchange_matcher.py` stays
  single-price, no-walking, LTP-filtered. Force-close calls
  the existing matcher; it does NOT introduce a "close at
  any price" mode.
- **Action / obs schemas.** No new dims. Force-close is
  env-initiated, not agent-initiated — the action space
  is unchanged.
- **BC pretrain.** The mechanism from
  `plans/arb-curriculum/` Session 04 stays wired. Post-BC
  policy still warms into training the same way; the BC↔
  target-entropy handshake is unchanged (the entropy
  controller's *authority* is what's new, not its target
  or warmup).
- **Matured-arb bonus.** Stays at its Session 02 formula
  and cap. Force-closes are excluded from the matured
  count so the agent isn't credited for env-initiated
  maturation it didn't earn.
- **Naked-loss annealing.** Stays as a gene. With
  force-close active, natural nakeds become rare, so the
  anneal window mostly controls the residual-nakeds cost.
  We leave the gene in play to test whether the combination
  interacts.
- **PPO stability defences.** All five stay: ratio clamp
  ±5, KL early-stop 0.03, advantage normalisation, reward
  centering per-step, LR warmup.
- **MTM shaping.** Stays at config default 0.05.
- **Curriculum day ordering.** Stays at `density_desc`.
- **`max_bets_per_race` and budget constraints.** Unchanged.

## Relationship to upstream plans

- Direct follow-on to `plans/arb-curriculum/`. That plan's
  four mechanisms (BC + matured-arb + naked-anneal +
  density ordering) all stay active in this probe. This
  plan adds three more mechanisms that attack the
  signal-noise problem those four didn't reach.
- Subsumes the `force-close-curriculum` follow-on queued
  in `plans/arb-curriculum/master_todo.md` "Queued
  follow-ons" (2026-04-19 operator suggestion, pulled
  forward).
- Orthogonal to `plans/entropy-control-v2/`. The
  controller's design is unchanged — we're widening the
  authority of an already-correct mechanism, not replacing
  it.
- Supersedes the "adaptive mutation" follow-on queued
  elsewhere; that was a diversification knob, not a
  signal-cleanup knob, and isn't relevant to this
  diagnosis.

## Failure modes (worth pre-articulating)

- **Force-close eats so much spread the agent just bleeds.**
  If most arb opportunities can't find a priceable close
  leg at T−30s, the agent loses the spread on every race
  and learns "don't open positions" — same zero-action
  attractor, different path. Detection: force-close cost
  per race > matured-arb bonus per race consistently;
  `bets` trending down rather than up across generations.
  Remedy: tighten the force-close threshold (e.g. T−60s
  instead of T−30s so closes happen when books are
  thicker), or add a force-close-aware early warning
  signal to obs (future plan, not this one).
- **Entropy velocity gene collapses to its floor.** GA
  discovers `alpha_lr = 1e-2` (the old default) is
  already enough, and all surviving agents cluster there.
  Interpretable outcome: entropy wasn't the load-bearing
  failure and the other two fixes carried the result.
  Remedy: none needed; this tells us the probe answered
  the question.
- **Shaped-penalty warmup creates an explore-cliff.** At
  ep = warmup_eps the shaping terms snap from 0 to full
  strength; the agent experiences a phase transition where
  its previously-rewarded exploration shape is suddenly
  penalised. Detection: `policy_loss` spike or bet-count
  collapse at exactly `warmup_eps + 1`. Remedy: smooth the
  ramp (sigmoid or cosine) rather than linear; caught in
  Session 02 testing via a scripted-episode invariant.
- **Invariant breaks.** Force-close touches the
  `raw_pnl_reward` path (the forced close P&L goes into
  `race_pnl` same as any other close) and the warmup
  touches `shaped_bonus`. If either path is mis-accounted,
  `raw + shaped ≠ total`. Must be caught in Session
  testing, NOT validation.
- **Ablation cohorts aren't independent.** The GA bred
  cross-cohort might blur the ablation. The plan-level
  cohort split (via separate hp-range blocks on plan-sub-
  populations) is something we'll need to verify the
  `TrainingPlan` data model supports, or implement as
  three separate plan files run serially with the same
  registry snapshot. Covered in Session 03.

## What happens next (if this works / doesn't)

**If criteria 1–5 all hold on validation:** move to a
16-agent multi-day scale run to confirm stability under
the combined mechanism set. Drop whichever mechanism the
ablation shows is redundant. Revisit the 277bbf49 cash-
positive agents' genes as a seed point for the scale run.

**If C1 passes but C4 still fails:** the entropy fix
works but the reward-shape problem is deeper than the
warmup + force-close combination handles. Next plan:
`observation-space-audit` — the policy may not have the
features it needs to distinguish good arbs from bad in
real time, and no amount of reward shaping around a
feature-poor obs will find the signal.

**If C4 passes but C1 still fails:** the reward-shape
fixes work but the entropy controller needs more than a
velocity bump. Next plan: controller architecture change
(Adam with per-parameter normalisation, or switch to PI
control with integral term).

**If criterion 5 (invariant) fails during test:** rollback,
fix the accounting, re-test. Never ships broken.

**If 1–4 all fail again:** the diagnosis was wrong. Stop
shaping, stop pretraining, and open
`observation-space-audit` as the next plan.

## Folder layout

```
plans/arb-signal-cleanup/
  purpose.md              <- this file
  hard_constraints.md
  master_todo.md
  progress.md
  lessons_learnt.md
  session_prompt.md
  session_prompts/
    01_force_close_and_entropy_velocity.md
    02_shaped_penalty_warmup.md
    03_plan_draft_validator_launch.md
```
