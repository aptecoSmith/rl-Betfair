# Session prompt — successor investigation: volume-vs-selectivity asymmetry

Use this prompt to open a new session in a fresh context. The
prompt is self-contained — it briefs you on the question, the
evidence, and the constraints. The current session that
discovered this asymmetry has been closed.

---

## The question

**Why does the actor respond to shaped per-pair pressure by
globally shrinking volume, but never by being selective
per-runner?**

Three probes on the open-cost shaped-reward mechanism produced
the same flat 75–79 % force-close rate across a 10–15× gene
span:

| Probe | matured-bonus | architecture | ρ(open_cost, fc_rate) | ρ(open_cost, pairs_opened) |
|---|---|---|---|---|
| Cohort-O | active | pre-fix | +0.055 | +0.139 |
| Cohort-O2 | pinned to 0 | pre-fix | +0.314 | +0.029 |
| Cohort-F | active | **post-fix** (fill_prob → actor_head) | −0.112 | **−0.448** |

Cohort-F is the first probe that reduced volume (ρ = −0.448)
but the COMPOSITION of outcomes (matur:close:naked:fc ratio)
stayed identical. Adding a per-runner discrimination input to
actor_head was used by the policy as a global magnitude knob,
not a per-runner discriminator.

This isn't expected to be a quick fix. It's a representational
question about how the action distribution + advantage flow
combine to produce or fail to produce per-runner conditioning.

## What you need to read first

1. `plans/fill-prob-in-actor/lessons_learnt.md` — Sessions 01
   and 02 (the architectural change + cohort-F result).
   Section "What did move — the volume-vs-selectivity
   asymmetry" is the load-bearing finding.
2. `plans/selective-open-shaping/lessons_learnt.md` — Sessions
   03 and 04 (the analysis that motivated cohort-F). Section
   "Picking the cause" enumerates 6 candidate causes; cause #4
   is now empirically ruled out as the binding constraint.
3. `CLAUDE.md` sections "Reward function: raw vs shaped",
   "Selective-open shaping (2026-04-25)", and "fill_prob feeds
   actor_head (2026-04-26)" — the existing reward / policy
   architecture.

## Hypotheses to test (don't try to solve all of them)

The previous session sketched two candidate explanations. Both
are speculative; investigation should test rather than assume.

### H1 — fill_prob's BCE target ≠ what the operator wants

`fill_prob_head` is trained on oracle labels for "this PASSIVE
will fill before race-off." The operator's intent for
selectivity is "this aggressive open will MATURE" — i.e.
passive fills AND we don't have to force-close. Maturation ⊊
fill (you can fill and still need to force-close if the
passive's match comes too late).

`ρ(fill_prob_loss_weight, fc_rate) = +0.469` in cohort-F is
suggestive: well-trained fill_prob CORRELATES with HIGHER
force-close rate. The actor may be selecting runners that are
easy to passive-match but risky to maintain to settle.

**Test:** check the oracle scan code
(`plans/arb-curriculum/session_prompts/01_oracle_scan.md` and
the oracle label generation pipeline) — what does the BCE
label actually mean in the data? If it's "passive matched at
all" rather than "passive matched + race went to settle
without force-close intervention", that's the proximate cause.

A second oracle target ("this open will mature") would need
its own auxiliary head. That's a Session 02-shaped landing,
not a quick experiment.

### H2 — per-tick shaped reward → global update only

The open-cost charge lands on the OPEN tick (the tick that
matched the aggressive leg) and the refund lands on the
RESOLUTION tick. PPO's advantage assignment via GAE pushes the
gradient back across multiple ticks; the value head smooths
the per-step delta into a low-variance estimate of the future
return.

Hypothesis: the shaped delta on the OPEN tick is roughly the
same regardless of which runner the agent opened on (the
charge is `-open_cost` for any open). The signal that
DIFFERENTIATES which open was bad arrives at a DIFFERENT tick
(force-close at T-30s vs settle for matured pairs) and is
attached to a different runner-index in the action vector.

If the value head smooths these together (hard to predict per
runner because the value head outputs a SCALAR V(s), not
per-runner), the surrogate-loss gradient at the open tick
sees the same advantage for all runners on that tick. There's
no per-runner credit assignment in the gradient even though
there IS a per-runner discrimination dim in the input.

**Test:** instrument `_compute_advantages` to log the
advantage at the open tick of bad-outcome opens (force-close)
vs good-outcome opens (matured). If they're indistinguishable,
the credit-assignment pathway is the bottleneck regardless of
how rich the actor's input is.

This is a deeper architectural question and may motivate a
per-runner value head, distributional critic, or reformulating
the open as a discrete action over runners (so the
log-probability concentrates at the chosen runner).

### H3 — the actor's signal head IS conditioning per-runner but PPO smooths it back out

A weaker hypothesis: the actor IS producing per-runner
differentiated signal logits, but the global policy optimum
under per-tick shaped pressure happens to be "lower all
signal heads uniformly." This is testable by inspecting the
actor's logits across runners on a sample of bad-outcome ticks
(do they vary in the way fill_prob varies?). If yes, the
optimisation is finding the right local optimum and the
shaping is genuinely insufficient.

## What to deliver

This is a research question, not a build task. The expected
deliverable from a successor session is:

1. A diagnostic report (one or two markdown files in a new
   plan folder, e.g., `plans/per-runner-credit/`) that picks
   ONE of H1 / H2 / H3 to investigate first, runs a
   minimal experiment (no code changes if possible — log
   inspection / synthetic forward passes / a one-shot
   computation), and reports findings.
2. **Don't** attempt to fix the selectivity gap in this
   session unless the diagnostic clearly localises a
   one-edit cause. Most paths here lead to architectural
   changes (per-runner value head, discrete action over
   runners) that need their own plans.
3. **Don't** re-run cohort-O / cohort-O2 / cohort-F. Their
   results are settled.

## Hard constraints carried over

- The fill_prob → actor_head change in `agents/policy_network.py`
  STAYS. It's not gene-gated, not reverted. Three policy
  classes were updated; tests in
  `tests/test_policy_network.py::TestFillProbInActor` are the
  load-bearing regression guards.
- `open_cost` gene STAYS in env-init at default 0.0.
  Byte-identical to pre-plan when unset. Don't revert.
- Don't touch the BC pre-train target.
  `plans/arb-curriculum/hard_constraints.md` still applies.
- Don't touch the entropy controller. Its alpha-saturation
  observation in cohort-F is a separate concern, not
  load-bearing for this investigation.

## Stop conditions

- If a one-line investigation reveals H1 (oracle label is
  "fill" not "mature") to be the proximate cause — flag it,
  draft a follow-on plan to add a "will-mature" auxiliary
  head, and stop. Don't implement.
- If the diagnostic shows the value head's per-tick advantage
  is indistinguishable across runners (H2 binding) — that's
  a much bigger surgery than a single session. Flag and stop.
- If the diagnostic is inconclusive — write up what you tried,
  what was unclear, and propose the next experiment. Don't
  force a verdict.

## Out of scope

- Implementing a per-runner value head.
- Implementing a discrete-action reformulation.
- Changing the env's reward shape.
- Re-running the cohort-O/O2/F probes.
- Anything in the live-inference repo (`ai-betfair`).

## Useful pointers

- `agents/policy_network.py:759` (LSTM), `:1213` (TimeLSTM),
  `:1626` (Transformer) — the lines where fill_prob now feeds
  actor_input. Order is fill_prob → actor → critic.
- `agents/ppo_trainer.py::_compute_advantages` — the GAE
  implementation; this is where per-tick reward becomes
  per-tick advantage.
- `env/betfair_env.py::_settle_current_race` — open-cost
  charge / refund accounting; per-tick dispatch is in the
  step function.
- `registry/training_plans/e7077b2b-8e30-4fd8-b38a-b2148229892a.json`
  — cohort-F plan + result rollup.
- `logs/training/episodes.jsonl` — per-episode rows for
  cohort-F (model_ids start `f46961c9, 163067a1, 16da1f31,
  6706a462, 6121ef5f, 3e78228a, e2686156, 25251cab, 1f45310c,
  140c8ce6, ca0ae9e2, 512efcd5`).
