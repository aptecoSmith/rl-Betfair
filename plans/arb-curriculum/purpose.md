# Purpose — Arb Curriculum (bootstrap past the naked valley)

## Why this work exists

The 2026-04-19 `reward-densification-probe` and the 2026-04-19
`reward-densification-gene-sweep` converged on a structural
diagnosis that no reward-shaping knob alone can fix:

> **The policy finds "arb less" orders of magnitude faster than
> "arb better" because the aggregate gradient at init points
> *against* all arbing.** Random arbing loses money (naked
> losses land at full cash value; wins are commission-capped and
> 95%-clipped on nakeds). The stop-arbing gradient is large,
> obvious, per-step; the arb-selectively gradient is small,
> noisy, per-pair. PPO descends the hill it can see. "Do
> nothing" becomes a stable attractor with no gradient out
> (MTM = 0 when no positions).

Every run since 2026-04-14 bears this out:

- `activation-A-baseline` 2026-04-19: 16 agents, 15 eps each,
  all converged to either passive (bets=0) or active-bleeding
  (reward −1500 to −3000, `arbs_closed/arbs_naked` stuck at
  5–7%).
- `fill-prob-aux-probe` 2026-04-19: aux supervised signal at
  `weight=0.10` didn't shift the picture.
- `reward-densification-probe` 2026-04-19: per-step MTM
  shaping didn't either — agent 1 collapsed to `bets=0` by
  ep5, agent 2 lost `close_signal` by ep6.
- `reward-densification-gene-sweep` 2026-04-19: 33 agents × 4
  gens with every shaping knob widened. Running at time of
  writing; expectation from the structural diagnosis is that
  no gene combination breaks the bifurcation.

The operator's observation sharpens the diagnosis: one of the
training days had **only 3 arb opportunities across the whole
day**. Under the current curriculum (days presented in
calendar order), an agent faces a distribution where some
days structurally *can't* teach "arb well" — there's nothing
for the policy to imitate or exploit. Yet those days still
*punish* random arbing with naked losses. The signal-to-
noise ratio on arbs is brutal on those days, and a random
initialisation bleeds money before it can find the few real
moments.

## Diagnosis (precise form)

Three forces are trapping the policy:

1. **Sign inversion at init.** Expected reward from random
   arbing is negative because loss magnitudes dominate.
2. **Credit-assignment blur.** Per-race settlement can't
   distinguish "this arb was good" from "this arb was bad"
   when 20+ pairs settle together under GAE smearing.
3. **Gradient-free attractor.** Once the policy stops acting,
   MTM = 0, close_signal atrophies, and there's no signal
   pointing back out of the zero-action region.

Reward shaping addresses (2) partly (MTM per-step,
per-pair naked asymmetry, close_signal bonus). It does not
address (1) or (3). We need interventions that *change the
sign* of the early gradient or *warm-start the policy in a
region where the gradient already points the right way.*

## The plan in one sentence

Attack the local minimum at three points simultaneously:
**(a) warm-start the policy from a rule-based oracle that
only takes profitable arbs (BC pretrain); (b) anneal the
naked-loss magnitude early in training so the first real
mistakes cost less; (c) add a small per-pair lifecycle
bonus so "completing an arb" is a reward regardless of P&L;
(d) order the training curriculum by oracle density so
early episodes are on arb-rich days.**

### Design sketch

**Oracle scan** (Session 01). Offline, deterministic pass
over every training day. At each tick, for each runner,
check whether a profitable arb is lockable post-commission
AND the env's matcher would let it through (LTP-filter,
price cap, freed-budget reservation). Emit per-tick
samples to a gitignored `.npz` cache. Also emit a per-day
density metric: `arb_samples / total_ticks`.

**BC pretrainer** (Session 04). Per-agent, before the first
PPO rollout. Cross-entropy on the `signal` head (oracle
says "back"), MSE on the `arb_spread` head (oracle's ideal
tick distance), other heads untouched. BC uses a separate
optimiser so PPO's Adam state stays fresh at first update.

**Matured-arb bonus** (Session 02). A small fixed reward per
pair whose second leg fills — naturally OR via
`close_signal`, regardless of locked P&L sign. Shapes the
agent toward "complete arbs," not "make money on arbs."
Zero-mean corrected by subtracting its expected value on
the random-policy baseline so the invariant holds.

**Naked-loss annealing** (Session 03). A new
`naked_loss_scale` gene (float, 0.0..1.0) and env-side
scaler on the per-pair naked-loss term inside `race_pnl`.
Starts small early (bootstrap), grows toward 1.0 as
training matures. Annealing curve is an env config knob
(`generation`-dependent); default on = unchanged behaviour
at `scale=1.0`.

**Curriculum day ordering** (Session 05). Training worker
picks day order based on Session 01's density metric:
arb-rich days first (bootstrap), arb-sparse days later
(generalisation). Default off = random order as today.

**Registry reset + validation** (Sessions 06, 07 —
operator-gated). Archive the gene-sweep state, draft a new
training plan that turns on BC + matured-arb bonus +
naked-loss anneal + curriculum ordering, launch.

### Why these four together

Each intervention by itself is weak; together they attack
all three forces:

- **Force 1** (sign inversion): naked-loss annealing
  reduces early loss magnitudes; BC pretrain starts the
  policy already on the right side of the gradient;
  matured-arb bonus makes the win side less commission-
  capped.
- **Force 2** (credit-assignment blur): matured-arb bonus
  gives per-pair feedback that doesn't require end-of-race
  aggregation.
- **Force 3** (gradient-free attractor): BC pretrain means
  the policy starts acting. It may still retreat if
  training punishes it, but now it retreats *from a known
  better region* rather than from noise.

## What success looks like

Post-landing validation (33-agent / 4-gen run, same shape
as `reward-densification-gene-sweep`):

1. **≥ 80 % of agents remain active through ep15**
   (bets > 0). Passive collapse should be vanishingly rare
   with BC warm-start and annealed nakeds. Previous
   baseline ≈ 50 % active.
2. **`arbs_closed / arbs_naked` ratio > 15 % on ≥ 3 agents
   by ep15**. Current baseline 5–7 % population-wide; a
   successful run should at least triple this.
3. **`policy_loss` stays O(1)+ through ep15 on ≥ 50 % of
   agents.** Gradient survival.
4. **≥ 3 agents reach `total_reward > 0` by the end of
   gen 3.** No run has ever seen a positive-reward agent;
   getting one is the headline result.
5. **Invariant `raw + shaped ≈ total` holds every episode.**
   Correctness gate; non-negotiable.

Criteria 1–4 are goal-signs. Criterion 5 is the correctness
gate (same rule as every plan post-2026-04-18).

## What this plan does NOT change

- **Matcher.** `env/exchange_matcher.py` stays single-
  price, no-walking, LTP-filtered. The oracle reads the
  matcher's filter predicates but never modifies matching
  logic.
- **Action / obs schemas.** No new dims. BC trains a subset
  of existing dims; matured-arb bonus is a reward-path
  tweak.
- **Target-entropy controller.** Stays wired. Session 04
  (BC pretrainer) handles the BC↔controller interaction
  explicitly — after BC, policy entropy is low, controller
  will try to boost it; a small `bc_target_entropy_warmup`
  grace period prevents first-PPO-update from undoing BC.
- **PPO stability defences.** All five stay: ratio clamp
  ±5, KL early-stop 0.03, advantage normalisation, reward
  centering per-step, LR warmup.
- **Mark-to-market shaping.** Stays at config default 0.05;
  orthogonal to this plan.
- **Scoreboard comparability.** Matured-arb bonus and
  naked-loss annealing ARE reward-scale changes (`scale<1`
  reduces raw reward magnitude; matured-arb bonus is
  shaped, not raw). Pre-landing scoreboard rows are NOT
  comparable on `total_reward` but remain comparable on
  `raw_pnl_reward` when `naked_loss_scale=1.0`.

## Relationship to upstream plans

- Follow-on to `plans/reward-densification/`. That plan
  landed the MTM mechanism correctly; this plan tackles
  the local-minimum that reward shaping alone couldn't
  escape.
- Supersedes `plans/arb-improvements/` Phase 3 (Sessions
  6–10). That plan scoped BC + oracle + aux-head +
  verification in 2026-04-14 and was paused. The
  **design work from that folder is inherited here** —
  particularly the "per-agent BC, never shared"
  hard-constraint from its `lessons_learnt.md` (a prior
  footgun). This plan re-frames the work with the
  post-2026-04-14 deltas folded in and adds the new
  interventions (matured-arb bonus, naked-loss annealing,
  curriculum day ordering) that the 2026-04-19 failure
  evidence motivates.
- Orthogonal to `plans/entropy-control-v2/`. Controller
  stays as-is; BC handshake is scoped inside Session 04.

## Failure modes (worth pre-articulating)

- **BC overfits to oracle.** Agent learns the oracle's rule
  exactly and can't generalise. Detection: post-BC policy
  is deterministic; early PPO eps show entropy collapse.
  Remedy: cap BC training steps (e.g. 500), add small
  dropout, use a small oracle-sample subset per agent.
- **Matured-arb bonus dominates P&L signal.** Agent
  completes cheap unprofitable arbs to harvest the bonus.
  Detection: `bets` high, `arbs_completed` high, `locked_pnl`
  ≈ 0, `total_pnl` negative. Remedy: cap bonus total per
  episode; reduce bonus magnitude.
- **Naked-loss annealing leaves residual reward-shape
  dependency.** After anneal completes, agents bleed
  because they never learned to avoid nakeds. Detection:
  sudden reward drop at the anneal-complete generation.
  Remedy: extend anneal schedule; add stopping condition
  based on arbs_naked rate.
- **Curriculum day ordering leaks future info.** Agents
  effectively see the same arb-rich days repeated first,
  which isn't representative of deployment. Detection:
  reward curve diverges between curriculum and random
  orderings on held-out days. Remedy: shuffle-within-
  bucket rather than strict density sort.
- **Gene-sweep found a working combo.** If the 2026-04-19
  `reward-densification-gene-sweep` actually produces one
  or more agents meeting the success criteria, this plan's
  premise partially evaporates — reward shaping CAN escape
  the attractor at the right magnitude. Detection: that
  run's Validation entry. Remedy: re-scope this plan to
  add only the most cost-effective intervention (likely
  matured-arb bonus) and leave BC / annealing for later.
- **Invariant breaks.** Matured-arb bonus or naked-loss
  annealing mis-accounted → `raw + shaped ≠ total`.
  Detection: existing integration test
  `test_invariant_raw_plus_shaped_equals_total_reward`
  plus new variants added in Session 02 / 03. Must be
  caught before landing, not in validation.

## What happens next (if this works / doesn't)

**If criteria 1–5 all hold on validation:** move to a
multi-day scale run (16+ agents × 10+ gens) to confirm
stability; then revisit `reward-densification-gene-sweep`
findings to see if MTM shaping now compounds with a
warm-started policy.

**If criterion 5 (invariant) fails during test:** rollback,
fix the accounting, re-test. This should NEVER ship broken.

**If criteria 1–4 all fail:** the diagnosis shifts again.
Next suspects in priority order: (a) observation space
doesn't encode the features the policy needs to recognise
good arbs (open `observation-space-audit` plan); (b)
matcher semantics differ enough from live Betfair that
oracle-generated arbs aren't representative; (c) the
action space doesn't let the policy express the right
policy (e.g. finer stake control needed).

**If criteria 1–4 partially succeed:** the lever works but
the magnitude is wrong. Second-pass session to tune the
three knobs (BC steps, matured-arb bonus weight,
naked-loss anneal schedule) via GA.

## Folder layout

```
plans/arb-curriculum/
  purpose.md              <- this file
  hard_constraints.md
  master_todo.md
  progress.md
  lessons_learnt.md
  session_prompt.md
  session_prompts/
    01_oracle_scan.md
    02_matured_arb_bonus.md
    03_naked_loss_annealing.md
    04_bc_pretrainer.md
    05_curriculum_day_ordering.md
    06_registry_reset_and_plan_redraft.md
    07_validation_launch.md
```
