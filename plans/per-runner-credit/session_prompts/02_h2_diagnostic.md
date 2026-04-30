# Session prompt — H2 diagnostic: per-tick credit assignment via GAE

Use this prompt to open a new session in a fresh context. The
prompt is self-contained — it briefs you on the question, the
evidence, and the constraints. Do not require any context from
the session that scaffolded this prompt.

---

## The question

**At the "open tick" of a force-closed pair vs the "open tick" of
a matured pair, is the GAE-derived advantage actually different?**

If yes — the actor has a learnable per-open signal in the advantage
tensor; the failure is somewhere else (saturation, exploration,
optimisation surface).

If no — the scalar value head is smoothing the per-runner credit
out before the actor sees it. **H2 is binding** and a follow-on
plan needs to address it (per-runner value head, distributional
critic, or discrete-action reformulation over runners).

If inconclusive — write up what was tried, propose the next
experiment, stop.

## Why this question, why now

`plans/per-runner-credit/findings.md` confirmed H1 by code
inspection (the fill-prob label conflated force-closes with
maturations). The mature-prob-head plan was scaffolded and is
running as cohort-M (12 agents, 18 eps each, ~70 minutes total).

The probe answers a binary mechanism question: does the gene's
effect on `fc_rate` correlate? If cohort-M lands flat (ρ within
±0.2, same as cohort-O / O2 / F), the next investigation is
**why** the actor doesn't translate per-runner discriminative
input into per-runner action conditioning. H2 is the leading
hypothesis.

The H2 diagnostic is independent of cohort-M's verdict:

- If cohort-M is **positive** (ρ ≤ −0.5), the diagnostic still
  tells us how MUCH of the gradient is per-runner usable, which
  informs how aggressive future shaped pressure can get.
- If cohort-M is **negative** (ρ within ±0.2), the diagnostic
  identifies the binding architectural constraint and queues
  the follow-on plan.

## What you need to read first

1. `plans/per-runner-credit/findings.md` — full H1 evidence
   trail and the H2 hypothesis (referenced in §"What this means
   for H2 and H3").
2. `plans/fill-prob-in-actor/lessons_learnt.md` Sessions 01–02 —
   cohort-F's volume-vs-selectivity asymmetry finding.
3. `plans/selective-open-shaping/lessons_learnt.md` Sessions
   03–04 — the cohort-O / O2 reasoning that motivated cohort-F.
4. `CLAUDE.md` sections "Reward function: raw vs shaped",
   "Selective-open shaping (2026-04-25)", "fill_prob feeds
   actor_head (2026-04-26)", and "mature_prob_head feeds
   actor_head (2026-04-26)".
5. `agents/ppo_trainer.py::_compute_advantages` (line 1823) —
   the GAE implementation. This is what you will instrument.
6. `env/betfair_env.py::_settle_current_race` and the per-tick
   open-cost dispatch in `step` — how the open-cost charge /
   refund lands per tick.

## The hypothesis under test (H2 in detail)

The open-cost charge lands on the **OPEN tick** (the tick that
matched the aggressive leg) and the refund lands on the
**RESOLUTION tick** (matured-natural or agent-closed only;
force-closed and naked do not refund). Per-tick deliveries were
introduced 2026-04-25 by `plans/selective-open-shaping` Session 02
after cohort-O Session 01's settle-time delivery showed gradient
smearing across 5,000 ticks via GAE.

PPO's advantage assignment via GAE pushes the gradient back
across multiple ticks; the value head smooths the per-step delta
into a low-variance estimate of the future return.

**The hypothesised failure mode:**

- The shaped delta on the OPEN tick is `−open_cost` regardless
  of which runner the agent opened on. Same charge, every open.
- The signal that DIFFERENTIATES which open was bad arrives at
  a DIFFERENT tick (force-close at T-30s vs settle for matured
  pairs) and is attached to a different runner-index in the
  action vector.
- The value head outputs a SCALAR `V(s)`, not per-runner. So
  GAE bootstraps with a single value across ALL runners on each
  tick.
- If the value head can't predict per-open shaped pressure on a
  per-runner basis (it can't — the value head's input is the
  shared backbone, not per-runner state), the surrogate-loss
  gradient at the open tick sees the same advantage for all
  runners on that tick.
- The actor has a per-runner discrimination input
  (`fill_prob`, `mature_prob` from the auxiliary heads), but
  the ACTION GRADIENT pushing the actor to use that input is
  smeared across runners. The actor learns "open less globally"
  rather than "open less on the runners my heads say will fail".

This is testable: at the open tick of force-closed pairs vs
matured pairs, the per-tick advantage should be similar (close
to `−open_cost`) regardless of outcome, because the resolution-
tick refund is far in the future and gets discounted /
bootstrapped through the value head.

## What to do

### 1. Identify a cohort-M rollout to use

By the time this session runs, cohort-M will likely be complete
or far along. Pick **one completed agent's rollout** (one
agent × 18 episodes ≈ 100k transitions). Don't replay live —
use the saved checkpoint + replay the env from the same seed.

If cohort-M agents aren't yet in the registry, use a cohort-F
agent (`registry/archive_*` should have those). The diagnostic
isn't sensitive to which architecture variant — both have a
scalar value head.

### 2. Instrument `_compute_advantages` (feature-flagged)

In `agents/ppo_trainer.py::_compute_advantages` (line 1823 ish):

- Add an opt-in env flag (e.g. `H2_DIAGNOSTIC_DUMP_PATH`) that
  when set, logs per-transition diagnostics to a parquet file:
  `tick_idx`, `episode_idx`, `value_pred`, `td_residual`,
  `advantage`, `return`, the action vector (so you can identify
  open ticks downstream), and any `info["action_debug"]`
  metadata that was captured at rollout time.
- Default behaviour (env flag unset) MUST be byte-identical to
  pre-change. Any test in `tests/test_ppo_trainer.py` that
  passes pre-instrumentation must still pass.
- Do NOT change the gradient pathway. The diagnostic is read-
  only — same advantages flow into the policy update.

### 3. Identify open ticks and their pair outcomes

Walk `env.all_settled_bets` from the rollout to build a mapping
`(tick_idx, runner_idx) -> outcome` where outcome ∈
{`matured`, `agent_closed`, `force_closed`, `naked`}.
Same classification logic as the trainer's episode-end backfill
(`_collect_rollout`'s `pair_to_transition` walk).

Tag each open tick in the diagnostic dump with its outcome.

### 4. Compute the comparison

For each completed open whose outcome is one of the four
classes, extract `advantage[open_tick]` from the dump. Group:

```
adv_matured       = [advantage at open ticks of matured pairs]
adv_agent_closed  = [advantage at open ticks of agent-closed pairs]
adv_force_closed  = [advantage at open ticks of force-closed pairs]
adv_naked         = [advantage at open ticks of naked pairs]
```

Compute mean and stddev for each group. Plot the four
distributions on one histogram. Compute pairwise differences:

- `mean(adv_force_closed) − mean(adv_matured)` (the load-bearing
  number — should be **negative and large** if H2 is NOT binding,
  near-zero if H2 IS binding)
- `mean(adv_agent_closed) − mean(adv_matured)` (closes are
  voluntary — agent's choice; expected to be slightly negative
  vs matured but not as much as force-closes)
- `mean(adv_naked) − mean(adv_matured)` (naked = passive never
  filled; cost is the residual directional risk)

### 5. Statistical test

Welch's t-test between `adv_force_closed` and `adv_matured`. The
question is "are these two distributions distinguishable" — one-
sided test, `H1: mean(force_closed) < mean(matured)`. Report
p-value, effect size (Cohen's d), and the absolute mean
difference in reward units.

**Decision matrix:**

| `mean(adv_force_closed) − mean(adv_matured)` | Interpretation |
|---|---|
| ≤ −5 (large negative; significant p < 0.01) | H2 NOT binding — the actor has a clear per-tick signal saying "force-closed opens were bad". The selectivity gap is somewhere else (saturation? exploration? actor architecture?). |
| Within ±2 (effectively zero, p > 0.1) | **H2 binding** — the open-tick advantage is indistinguishable across outcomes. Per-runner credit assignment is the architectural blocker. |
| Between −5 and −2 (some signal, weak) | Inconclusive — the signal exists but is small. Probably means H2 is a partial constraint; other factors also at play. Worth a deeper look. |

### 6. Write up

A new file at `plans/per-runner-credit/h2_diagnostic.md` with:

- Method (which agent / rollout was used; instrumentation
  approach; classification rules).
- Per-class advantage distributions (histogram or kernel density;
  ASCII art if image rendering not available).
- The headline number: `mean(adv_force_closed) − mean(adv_matured)`
  with confidence interval.
- The verdict (H2 binding / not binding / inconclusive).
- If H2 binding: a SKETCH of the follow-on plan options ranked
  by surgery size. Do NOT implement.

| Option | Surgery size | Effect |
|---|---|---|
| Per-runner value head | Small — `value_head` parallel to `actor_head`, output `(batch, max_runners)` instead of `(batch, 1)`; GAE bootstraps per-runner | Direct fix; per-runner advantage flows by construction |
| Distributional critic | Larger — output a distribution over returns rather than a scalar; needs a quantile or categorical loss | Captures multimodal value, may help indirectly |
| Discrete action over runners | Largest — replaces the per-runner Gaussian with a Categorical over which runner to open on (mutually exclusive); action log-prob concentrates at the chosen runner | Strongest credit signal but biggest change to the action space |

## Stop conditions

- H2 confirmed binding → write report, draft follow-on plan
  skeleton, **stop**. Do not implement.
- H2 confirmed not binding → write report saying so, name the
  next thing to investigate, **stop**.
- Diagnostic inconclusive → write up what was tried, what was
  unclear, propose the next experiment, **stop**.

## Hard constraints

- The instrumentation in `_compute_advantages` MUST be feature-
  flagged. Default behaviour byte-identical. The full
  `tests/test_ppo_trainer.py` suite (66 tests, all 4
  `TestRecurrentStateThroughPpoUpdate`, the
  `test_real_ppo_update_feeds_per_step_mean_to_baseline`
  integration test) MUST still pass.
- The mature_prob_head architecture STAYS. Do not revert.
- The fill_prob → actor_head pathway STAYS. Do not revert.
- Don't re-run cohort-O / O2 / F / M. Use existing data.
- Don't touch env / reward / aux heads.

## Out of scope

- Implementing a per-runner value head.
- Implementing a distributional critic.
- Reformulating the action space as discrete over runners.
- Changing the env reward shape.
- Anything in the live-inference repo (`ai-betfair`).
- Re-running probes.

## Useful pointers

- `agents/ppo_trainer.py:1823` — `_compute_advantages`. This is
  where to add the diagnostic dump.
- `agents/ppo_trainer.py::_collect_rollout` — the
  `pair_to_transition` walk that classifies pair outcomes at
  episode end (line 1494 ish). Reuse this classification logic
  for tagging open ticks; don't reinvent it.
- `env/betfair_env.py::_settle_current_race` — how
  `open_cost_shaped_pnl` is accumulated. Per-tick dispatch is
  in the env's `step` function.
- `env/betfair_env.py::_attempt_close` — force-close placement
  (sets `Bet.force_close = True`).
- `registry/models.db` — agent metadata (model_id, gene values,
  weights_path).
- `registry/weights/<model_id>.pt` — saved policy weights for
  replay.
- `logs/training/episodes.jsonl` — per-episode summary rows
  (already has the per-class outcome counts; useful sanity-check
  alongside the per-tick advantage dump).

## Estimate

Single session, 1–2 hours of focused diagnostic work:

- 30 min: instrument `_compute_advantages` + per-tick dump.
- 30 min: rollout one agent, dump, build the per-tick →
  outcome mapping.
- 30 min: aggregation, statistical test, plot.
- 30 min: write up the report and the follow-on plan sketch.

If you find yourself doing more than that, pause and check
whether scope has crept beyond the diagnostic question. If yes,
write up what you have, flag the scope creep, and stop.
