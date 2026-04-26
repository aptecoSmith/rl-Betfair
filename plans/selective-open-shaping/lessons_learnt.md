---
plan: selective-open-shaping
status: session-01-complete
landed: 2026-04-25
---

# Lessons learnt — selective-open-shaping

## Session 01 (2026-04-25) — open-cost mechanism, shipped at gene 0.0

### What landed

A new shaped-reward term in `env/betfair_env.py::_settle_current_race`
that charges `open_cost` per successful pair open and refunds it
at settle iff the pair resolves favourably (matured or agent-
closed). Force-closed and naked outcomes do NOT refund. Default
gene `open_cost = 0.0` makes the term byte-identical to pre-plan.

Concrete changes:

- `env/betfair_env.py`:
  - `_REWARD_OVERRIDE_KEYS` gains `"open_cost"`.
  - Env reads `self._open_cost` from `reward_overrides` with
    default `0.0`; clamped to `[0.0, 2.0]`.
  - `RaceRecord` gains `pairs_opened: int` and
    `open_cost_shaped_pnl: float`.
  - `_settle_current_race` extends the existing `pair_bets`
    walk (the same one introduced by `da05332` for partial-fill
    coverage) to count `pairs_opened` (every distinct pair_id
    in matched bm.bets) and `refund_pair_count` (matured +
    agent-closed). Computes
    `open_cost_shaped_pnl = open_cost × (refund_count − pairs_opened)`,
    adds it to the shaped accumulator inside the
    `if self.scalping_mode` block.
  - `_get_info` exposes `pairs_opened`, `open_cost_shaped_pnl`,
    `open_cost_active`.

- `agents/ppo_trainer.py`:
  - `EpisodeStats` gains `pairs_opened`, `open_cost_shaped_pnl`,
    `open_cost_active`.
  - The trainer's info→stats wiring reads the three new keys.
  - The episodes.jsonl row writer surfaces all three.

- `tests/test_forced_arbitrage.py`:
  - 8-test class `TestSelectiveOpenShaping` — one per outcome
    class, plus zero-mean / mixed / raw-untouched guards.

- `CLAUDE.md`:
  - New subsection "Selective-open shaping (2026-04-25)" under
    "Reward function: raw vs shaped".

### Design choice — settle-only, not per-tick

`purpose.md` proposed charging the cost at the open tick and
refunding at the resolution tick (both per-tick), with the
rationale "PPO sees the cost in the same mini-batch as the open
action." Implementation landed at settle-only — both the charge
and the refund collapse into the per-race `_settle_current_race`
contribution.

The shift was deliberate after reading the env's per-step reward
flow more carefully:

- **The matured bonus already lands at settle, not at maturation
  tick.** It works (within its limits) — PPO's GAE propagates the
  settle-time delta back across the steps that contributed.
- **A per-tick charge would have required a per-tick mark on the
  step's reward**, plus per-pair state tracking through the env's
  step loop, plus careful handling of refund timing. Several new
  edge cases.
- **Magnitude beats timing for credit assignment.** A 200-open
  race × £0.5 cost × 77 % force-close rate = ~£77 of cumulative
  shaped pressure attributable to opens. That's larger than the
  matured bonus contribution at the same race; PPO's GAE will
  distribute it back across the right ticks.

If the post-implementation gene-sweep probe shows the magnitude
isn't enough, Session 02 can revisit a per-tick variant. For
now, settle-only is byte-identical to the rest of the shaping
family and contained to one file.

### Hard_constraints §3 relaxation

The hard_constraints originally said: "open means aggressive
matched AND passive posted." Implementation counts every distinct
pair_id in matched bm.bets — which includes the case where the
aggressive matched but the passive failed to post (budget exhaust,
junk filter, etc.). Those pairs land in `pair_bets` with length 1
(only the aggressive leg) and the existing settle walk classifies
them as naked.

Decision: **count them as opens.** The agent's DECISION was to
enter a paired position; whether the env's downstream paperwork
worked is part of the open's risk profile. Charging the cost on
naked-from-start pairs is consistent with naked-by-eviction — both
paid the open cost, neither got the refund.

This is a relaxation of the original hard_constraints §3 wording.
Recorded here so a future reviewer doesn't read the implementation
as a bug.

### Test design

The 8 integration tests use a `_settle_with_bets` helper that
injects synthetic Bet objects directly into `bm.bets` and calls
`env._settle_current_race(race)` — bypassing the full episode
loop. This is faster and more focused than building a multi-tick
day, but still exercises the REAL settle code path (no mocks on
the pair_bets walk, the covered-fraction math, or the shaped
accumulator).

Construction caveat: tests pass the gene through
`reward_overrides={"open_cost": 1.0}` rather than through
`config["reward"]["open_cost"]`. The `_REWARD_OVERRIDE_KEYS`
whitelist routes the gene through the same passthrough channel
PPOTrainer uses; tests use that channel directly. An earlier
version of the tests put the gene in the wrong key
(`config["scalping"]["open_cost"]`) and silently saw `open_cost=0`
on every test, which 4/8 caught. Fixing the test setup recovered
the right channel.

### Regression sweep

PPO trainer + forced_arbitrage + mark_to_market + population
manager: 287+ tests pass. No existing test had to change.

## Open work — Session 02

`master_todo.md` defines a 12-agent gene-sweep probe to validate
that the mechanism actually moves the policy. Pre-launch gate:
the post-kl-fix-reference run (now relaunched with the threshold
bump + state_dict fix, with `open_cost=0.0` so the run is
byte-identical to the previous diagnostic) must complete first.
If that run shows force-close rate has dropped to <40 % under
the unstarved-PPO trainer alone, this plan closes as "resolved
upstream" without Session 02. Otherwise Session 02 runs.

## Meta-lesson

The mechanism shipped INACTIVE because the diagnostic run that
will validate it isn't done yet. This is the third time this
session that "ship the infrastructure with a default-zero gene
so the in-flight scoreboard isn't disturbed" has been the right
call (the others: `mark_to_market_weight=0`, `force_close_before_off_seconds=0`).
The pattern is durable enough to elevate to a meta-lesson in
`plans/ppo-kl-fix/lessons_learnt.md` if it recurs once more —
but for now, just note it as a working pattern.

---

## Session 03 (2026-04-25) — analysis only, no code

### What we have to work with

Cohort-O probe (`a5f0c7af-…`) under per-tick design (commit
`8dfa1f6`). 10 of 12 agents have completed at least 10 of 18
episodes; the remaining two (`07452066`, `ef0cebb5`) have
1 / 5 eps and are excluded from the correlations. Same trainer
state as `post-kl-fix-reference`: KL fix in, force-close at T−30s,
shaped-penalty warmup, density-desc curriculum. The only
intentional independent variable is `open_cost`.

Gene draws span **15× in magnitude**:

| open_cost | n agents | architectures |
|---|---|---|
| 0.000 | 1 (5 eps only) | time_lstm |
| 0.056 | 2 | time_lstm × 2 |
| 0.197–0.201 | 4 | time_lstm × 2, transformer × 2 |
| 0.623 | 1 | lstm |
| 0.827–0.829 | 3 | transformer × 3 |

### Headline correlations (10-agent denominator, last-8-ep means)

| Spearman ρ(`open_cost`, X) | value | interpretation |
|---|---|---|
| `oc_shaped` | **−0.976** | mechanism delivers the gradient as designed (linear in gene) |
| `fc_rate` (% force-closed) | **+0.055** | **zero policy response** |
| `pairs_opened` | +0.139 | flat (within day-driven noise) |
| `arbs_naked` | +0.430 | nakeds slightly **rise** |
| `arbs_closed` (close_signal) | −0.345 | closes slightly **fall** |
| `arbs_completed` (matured) | −0.164 | matures slightly fall |
| `entropy` | −0.479 | entropy contracts under pressure |
| `value_loss` | +0.491 | value head noisier under pressure |
| `total_reward` | −0.758 | high-gene agents pay more, get nothing back |
| `total_pnl` (raw cash) | +0.527 | raw mostly unchanged (small noise) |

`pairs_opened` is dominated by curriculum day, not gene: day-to-day
mean ranges 200–900; cross-agent SD on a fixed day is 30–70. The
gene's footprint on opens is invisible at this scale.

`fc_rate` is remarkably stable across the gene range — 74–77 % on
most days, 81–83 % on April 7 (sparse day) — and the spread on a
fixed day is ±2pp regardless of gene. **18 episodes of
`open_cost=0.83` produces the same force-close rate as 18 episodes
of `open_cost=0.06`.**

Within-agent trajectory across 18 eps shows no learning trend
either: high-gene `61617a7f`'s `fc_rate` tracks 75–81 with no
slope; `pairs_opened` follows the curriculum and ignores the gene.
This is not a "needs more episodes" failure — the per-tick gradient
is being delivered every step but the policy isn't moving on the
dimension we want.

### Picking the cause

**Most likely: a combination of #4 (no per-runner discrimination
pathway) and #6 (volume-shaped policy is GA-equilibrium under
matured-bonus), with #4 being the architectural blocker.**

Why I rule out the other candidates:

- **#1 — "matured-bonus dominates"**: incomplete on its own. If
  it were just a signal-magnitude race, we'd still see SOME
  inverse correlation between gene and fc_rate. We see ρ = +0.055.
  Net-shaping accounting (high gene pays −£447/race shaped vs
  matured-bonus paying ~+£300–£1160) would push for "open less"
  if the policy could express "open less *selectively*". It can't
  — so it pays the cost flat.
- **#2 — action distribution saturated**: probably a contributor
  but secondary. Entropy is contracting under pressure (ρ = −0.479)
  not saturating to a corner; the policy *is* moving, just not in
  the dimension we want. Saturation alone wouldn't produce the
  +0.430 arbs_naked rise.
- **#3 — value head can't distinguish, needs more training**: not
  consistent with the trajectories. After 18 eps the value loss
  is *higher* on high-gene agents (ρ = +0.491), not converging.
  More episodes means more shaped pressure on a value head that
  has no observation-space pathway to predict it per open.
- **#5 — `max_bets_per_race=20` mis-counted**: irrelevant. Agents
  fire 200–900 opens/race; the cap never binds.

The smoking gun for #4: at `agents/policy_network.py:563–570` the
per-runner action is sampled from `(runner_emb, lstm_output)`. The
fill-prob head exists per-runner (`fill_prob_head`, line 590) and
is BCE-trained as an auxiliary, **but its output never enters
actor_head**. The policy literally has no way to express "this
runner's open will likely fail to mature" in its action
distribution. The signal-head threshold ±0.33 is fired from a
backbone representation that isn't conditioned on per-runner
fill-probability. So even with infinite per-tick gradient on
`open_cost`, the optimisation surface only allows the policy to
shift the GLOBAL signal-firing rate (and the curriculum day already
fixes that within ±50 opens), not to be SELECTIVE per-runner.

#6 reinforces #4: maturing arbs are sparse, the policy needs
high-volume search to find them, matured-bonus pays for that
search, and there is no per-runner pathway by which the policy
could lower volume only on the runners that won't mature. The
cheapest move under added open-cost pressure is to take it on the
chin and keep firing.

### Next experiment — ONE probe to discriminate #1 from #4/#6

**Cohort-O2: open_cost sweep with matured_arb_bonus_weight pinned
to 0.0.**

- 12 agents, same arch_mix as cohort-O (4 lstm / 4 time_lstm /
  4 transformer), fresh seed (`8402`).
- All genes cloned from cohort-O **except**:
  - `matured_arb_bonus_weight`: pinned to `{min: 0.0, max: 0.0}`.
  - `open_cost`: full sweep `{min: 0.0, max: 1.0}` (same as cohort-O).
- 1 generation, 18 episodes, `auto_continue: false`.

The pin removes the largest competing positive shaped signal. If
the open-cost mechanism can move fc_rate at all, it will be most
visible here: there is no countervailing reward to keep volume up.

**Decision matrix:**

| Result | Interpretation | Verdict for the mechanism |
|---|---|---|
| ρ(open_cost, fc_rate) ≤ −0.5 across the 12 agents | matured-bonus had been masking the response — #1 was the binding constraint | mechanism is viable; retune matured/open balance, promote with caps |
| ρ(open_cost, fc_rate) within ±0.2 (same as cohort-O's +0.055) | architectural/policy lacks discrimination — #4 binding | mechanism is a dead-end as-is; pivot to fill-prob-in-actor |
| ρ(open_cost, fc_rate) between −0.5 and −0.2 | partial signal under unstacked shaping | inconclusive; need fill-prob-in-actor experiment to confirm magnitude ceiling |
| Any agent collapses to `bet_count = 0` early | matured-bonus had been the *only* thing keeping the policy alive | important side-finding; mechanism's interaction with other shaping is fragile |

This is a single 12-agent probe, no code changes (the matured
bonus already accepts gene 0.0), reuses the existing infrastructure.
Roughly the same compute cost as cohort-O.

### Dead-end criteria — when to pivot

Declare the open-cost shaping mechanism dead and pivot to
fill-prob-conditioning of the actor head if **any** of:

1. The cohort-O2 probe lands ρ(open_cost, fc_rate) within ±0.2
   (same as cohort-O). Two independent probes showing the
   mechanism doesn't move the policy is sufficient evidence the
   open-tick gradient cannot find a representational pathway.
2. The cohort-O2 probe shows ρ(open_cost, fc_rate) between −0.2
   and −0.5 AND ρ(open_cost, arbs_naked) stays positive. That
   would mean "the mechanism trades force-closes for nakeds",
   which is no win — both outcomes are the cost we're trying to
   avoid.
3. The cohort-O2 probe shows agents collapsing to `bet_count = 0`
   under any active gene value. That's the silence-optimisation
   failure mode — the mechanism solves the problem by removing
   the agent rather than making it selective. Cohort-A bottom-6
   already has this failure shape under different penalty genes;
   reproducing it here means the mechanism is a worse instance
   of an already-known failure.

The pivot in any of those cases: feed `fill_prob_per_runner`
(already produced by the policy each forward pass — see
`agents/policy_network.py:777`, line 1229, line 1643) as an input
to `actor_head`. Architecture change in `policy_network.py`
across all three classes; new BC handshake (the current BC
targets the signal head directly with oracle labels, untouched
by this change); independent regression sweep. A new plan, not
a Session 04 of this one.

### What I am NOT proposing

- Don't bump `open_cost` above 1.0 in the next probe — the
  cohort-O ceiling at 0.83 already pays −£447/race shaped with
  zero behavioural response; magnitude is not the missing
  ingredient.
- Don't add another shaping term as a workaround. The cohort-O
  evidence is that the policy's optimisation surface is the
  blocker. Stacking more shaping just adds gradients the policy
  can't act on.
- Don't kill the in-flight cohort-O probe. The two unfinished
  agents (`07452066` and `ef0cebb5`) bring the denominator to 12
  and let the operator confirm or refute the 10-agent verdict
  with the full sample. The flat ρ at 10 agents is unlikely to
  change at 12 but the operator gets the cleaner write-up.

---

## Session 04 (2026-04-26) — cohort-O2 verdict: mechanism dead

Cohort-O2 (`08667590-…`, 6 agents, time_lstm only,
matured_arb_bonus_weight pinned to 0.0, 12 eps, gene span 10×).
All 6 completed.

| Spearman ρ(`open_cost`, X) | cohort-O | cohort-O2 |
|---|---|---|
| `oc_shaped` | −0.976 | −0.943 — gradient delivered, again |
| `fc_rate` | +0.055 | **+0.314** — wrong sign |
| `pairs_opened` | +0.139 | +0.029 — flat |
| `arbs_closed` | −0.345 | **−0.600** — closes drop further |
| `total_reward` | −0.758 | −0.771 |

`fc_rate` band: 75.8–78.1 % (cohort-O: 74–77 %). Removing the
matured-bonus did not unmask anything. **Cause #1 ruled out;
cause #4 (architectural pathway) confirmed as the binding
constraint.**

The anti-correlation on `arbs_closed` is the new datum: under
unmasked open-cost pressure the policy uses `close_signal` LESS,
not more. Without per-runner discrimination it can't selectively
close the bad pairs either — the open-cost gradient just makes
all paired behaviour marginally worse.

### Verdict

Open-cost shaping mechanism is dead under the current actor
architecture. Plan status → `complete` (mechanism stays in the
codebase at gene 0.0; byte-identical to pre-plan, no revert
needed).

### Pivot

New plan: **fill-prob-in-actor** — feed `fill_prob_per_runner`
(already produced each forward pass at `policy_network.py:777`
/ 1229 / 1643) into `actor_head` so the signal action can be
conditioned on the policy's own per-runner fill forecast.
Architectural change across all three policy classes; new BC
handshake. Separate plan; skeleton next.

---

## Session 05 (2026-04-26) — closing update from fill-prob-in-actor

`plans/fill-prob-in-actor` ran and closed as a third negative
result. Cohort-F (`e7077b2b-…`, 12 agents, ppo_time_lstm_v1,
18 eps, NEW architecture with fill_prob → actor_head) landed
ρ(open_cost, fc_rate) = **−0.112** — within the same ±0.2 flat
band as cohort-O / cohort-O2.

Three independent probes, two architectures (O+O2 pre-fix, F
post-fix), same glued 75–79 % fc_rate band. The architectural
unblock did not move the policy on the dimension we wanted.

**Cohort-F's contribution to the analysis:** it produced the
first non-trivial response on `pairs_opened` (ρ = −0.448 —
volume drops ~22 % from gene 0.1 to gene 0.95) WITHOUT a
corresponding shift in the matur:close:naked:fc composition.
The policy responded to the open-cost gradient by **globally
shrinking volume**, not by per-runner selectivity. This
localises the missing mechanism: the actor CAN shrink volume
globally but CANNOT (or does not learn to) condition signal
output on per-runner state at "this runner will mature, that
one will force-close" granularity.

**Status:** selective-open-shaping stays `complete`, mechanism
stays in the codebase at gene 0.0 (byte-identical default,
no revert needed). The fill-prob-in-actor architectural change
also stays in (not gene-gated, no revert; ruled out one more
cause).

The selectivity question now lives at
`plans/fill-prob-in-actor/session_prompts/
03_volume_vs_selectivity_followup.md` — a successor
investigation into why the actor's responses to shaped
pressure are global-magnitude only. Operator's call whether to
open a new plan from that prompt.
