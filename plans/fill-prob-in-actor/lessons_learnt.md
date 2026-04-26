---
plan: fill-prob-in-actor
status: complete
landed: 2026-04-26
verdict: negative-result-mechanism-dead
---

# Lessons learnt — fill-prob-in-actor

## Session 01 (2026-04-26) — architectural change shipped

### What landed

`fill_prob_head` output is now concatenated into `actor_input`
across all three policy classes. Per-runner action parameters
are now sampled from
`(runner_emb_i, backbone, fill_prob_i)` instead of
`(runner_emb_i, backbone)`. `actor_head[0]`'s input dim is
exactly +1 wider — `runner_embed_dim + lstm_hidden + 1` for
LSTM/TimeLSTM, `runner_embed_dim + d_model + 1` for the
transformer.

Concrete changes:

- `agents/policy_network.py`:
  - `PPOLSTMPolicy.__init__`: `actor_input_dim` += 1; comments
    updated to name the new column.
  - `PPOLSTMPolicy.forward`: `fill_prob` is computed from
    `lstm_last` BEFORE the actor block; `actor_input` concat
    now includes `fill_prob.unsqueeze(-1)` as the third tensor.
  - `PPOTimeLSTMPolicy.__init__` / `forward`: same shape change,
    same forward reordering.
  - `PPOTransformerPolicy.__init__` / `forward`: same. The
    transformer's backbone is `out_last` (not `lstm_last`); the
    concat pattern is identical otherwise.
  - The risk_head computation order is unchanged — `risk_head`
    does NOT feed actor; only `fill_prob_head` does.
  - `migrate_*` helpers untouched (they migrate intra-
    architecture state-dicts; this is an inter-architecture
    change, so a hard refusal on cross-load is the
    correct-by-default behaviour).

- `tests/test_policy_network.py`:
  - New 12-test class `TestFillProbInActor` (4 tests × 3 classes):
    - input-dim guard, gradient-through (forward-side),
      gradient-through (backward-side), pre-plan weights fail
      to load.

- `CLAUDE.md`:
  - New top-level subsection "fill_prob feeds actor_head
    (2026-04-26)" after "Transformer context window — 256
    available". Names every load-bearing test; documents the
    architecture-hash break and the gradient-through choice.

### Architecture-hash break verified

PyTorch's `load_state_dict(strict=True)` already refuses to
load a state_dict whose `actor_head.0.weight` is one column
narrower than the new policy's expected shape. The variant
identity is therefore carried by the changed weight shape; no
new explicit version field needed (Hard_constraints §4).

The three `test_..._pre_plan_weights_fail_to_load` tests
construct a synthetic pre-plan state_dict by cloning every
parameter except `actor_head.0.weight`, which is replaced with
a tensor whose input width is `actor_head[0].weight.shape[1]
- 1`. `load_state_dict(sd, strict=True)` raises with a message
referencing the parameter — confirmed by an `excinfo.value`
substring assertion.

### Gradient-through, not detach

The forward-side gradient-through guard is the strictest of
the four checks per class. Naïvely, `fill_prob_head.weight.grad
is not None` would seem to be enough — but the BCE auxiliary
loss ALSO trains `fill_prob_head`, so a detach in the
actor-input concat could leave the head with a non-None
gradient that came entirely from the BCE side. The
forward-only check (perturb the head's weight; assert
`action_mean` changed for fixed obs / hidden) literally
verifies the actor's output depends on the head's weights.
Without it, an accidental
`fill_prob.unsqueeze(-1).detach()` refactor would silently
re-introduce the cohort-O / cohort-O2 dead-end.

The backward-side test
(`test_..._actor_loss_routes_grad_through_fill_prob_head`)
complements the forward-side test by checking the gradient
plumbing: `out.action_mean.sum().backward()` produces a
non-None gradient on the head's weight, with non-zero abs-max.
Both directions are covered.

### Pre-existing test interaction

77 of the 78-test `tests/test_policy_network.py` suite passed
unchanged after the architectural change. The pre-existing
`test_gradients_flow_through_actor` carved out
`fill_prob_head` from its "must have grad on actor-only loss"
loop — the carve-out's comment refers to a previous plan's §8.
With the new architecture the head DOES receive gradient on
actor-only loss, but the carve-out is a `continue` rather than
an inverse assertion, so the test still passes (it just
over-specifies what's not gradient-receiving — accurate before
this plan, conservative after). No edit needed.

### 237/237 protected guards pass

The Hard_constraints §11 protected set passed unchanged:

- `tests/test_ppo_trainer.py` — full suite (66 tests),
  including all 4 `TestRecurrentStateThroughPpoUpdate` and
  `test_real_ppo_update_feeds_per_step_mean_to_baseline`.
- `tests/test_forced_arbitrage.py` — full suite (108 tests),
  including all 8 `TestSelectiveOpenShaping`.
- `tests/test_mark_to_market.py` — full suite (11 tests),
  including `test_invariant_raw_plus_shaped_with_nonzero_weight`.
- `tests/test_population_manager.py` — full suite (52 tests).

### Two pre-existing seed-fragile guards now fail (NOT in protected set)

A wider sweep across `tests/` surfaced two failures that
weren't in the protected list:

- `tests/test_ppo_advantage_normalisation.py::TestRealTrainerUpdateBounded::test_real_update_policy_loss_bounded`
  — assert `abs(policy_loss) < 100`, got 112.94.
- `tests/test_ppo_stability.py::TestLargeRewardSmoke::test_large_reward_does_not_explode_policy_loss`
  — assert `abs(policy_loss) < 100`, got 157.84.

Both are fixed-seed (seed=0) heuristic guards on a synthesised
±£2000-advantage rollout. A seed sweep on master shows the
test is fragile pre-change too:

| seed | master | post-change |
|---|---|---|
| 0 | **49.7 ✓** | -112.9 ✗ |
| 1 | 49.7 ✓ | 49.8 ✓ |
| 2 | 189.8 ✗ | -94.3 ✓ |
| 3 | -94.8 ✓ | 188.9 ✗ |
| 4 | -722.7 ✗ | -723.4 ✗ |

The "100" threshold is not a robust bound for the synthesis —
it's the value seed=0 happens to produce on master. My change
reshuffles the orthogonal init RNG (because `actor_head[0]`'s
weight matrix went from `(32, 96)` to `(32, 97)`,
`torch.nn.init.orthogonal_` samples a slightly different
matrix), so seed=0 lands in a fail bucket and seeds=2,3 swap.

Decision: **leave the tests as-is.** They test a normalisation
mechanism, not the policy architecture. The defence is still
working — the magnitude swings are bounded `< 1000`, just not
`< 100` at this particular seed. Editing the threshold to
"make them pass" would mask future genuine regressions (e.g.,
a regression that produces 5,000+ would also pass at a relaxed
threshold). The right fix is a multi-seed median check, but
that's a separate refactor from this plan.

A flag-and-move-on rather than a paper-over: this lesson
exists so a future reader doesn't re-encounter the same
ambiguity. The test_training_worker.py websocket-handshake
timeout in the same sweep is unrelated (websocket flake on
heavy concurrent load during the sweep — passes in isolation).

### Smoke-checked all three classes

Out-of-pytest one-shot verification: each of the three policy
classes instantiated with default-shape hyperparams, fed a
random `(2, OBS_DIM)` obs, and confirmed `action_mean.shape ==
(2, ACTION_DIM)` and
`actor_head[0].weight.shape[1] == runner_embed_dim + backbone + 1`
in all three cases. No exception, no NaN.

## Session 02 (2026-04-26) — probe scaffolded, awaiting operator launch

The cohort-F probe JSON has been authored and lands in
`registry/training_plans/<uuid>.json` with `status='draft'`. It
is NOT running. Per the operator instruction at the start of
this session, "Do NOT launch the probe; that's an operator
decision after the JSON is on disk and the code is green."

### Probe shape (matches Hard_constraints §13)

- Population 12, single architecture (`ppo_time_lstm_v1`).
- 1 generation, n_epochs 3 (18 episodes per agent).
- `auto_continue: false`. Single-shot diagnostic.
- Cohort label `F` (Fill-prob).
- Seed 8403 (distinct from O=8401, O2=8402).
- `open_cost ∈ [0.0, 1.0]` (matches cohort-O / cohort-O2).
- `matured_arb_bonus_weight ∈ [5.0, 20.0]` (cohort-O setting,
  not cohort-O2's pin to 0; we want the realistic interaction).
- `fill_prob_loss_weight ∈ [0.0, 0.3]` (full range — separates
  "well-trained fill_prob" from "near-constant 0.5").

Other genes cloned from cohort-O2 verbatim so the only
intentional independent variables are `open_cost` and
`fill_prob_loss_weight`.

The 12/18 sizing was lifted from the initial cohort-O2-matched
6/12 design after the operator flagged 6–8 hours of compute
available (Session 02 conversation). N=12 / 18-eps matches
cohort-O exactly on both denominator and trajectory length, so
the primary metric `ρ(open_cost, fc_rate)` is directly
comparable to cohort-O's +0.055 baseline rather than
triangulated against a smaller cohort. The wider envelope also
collapses the decision-matrix's "between −0.5 and −0.2"
inconclusive band — the original "follow-on N=12 probe" branch
is now satisfied in this run.

Multi-architecture was rejected: cohort-O Session 03 found "the
signal was identically flat across all three architectures (no
arch interaction to resolve here)", so spreading 12 agents
across three archs would dilute the primary correlation's
statistical power on a known-uninformative axis.

### Decision criteria (when the operator runs the probe)

| ρ(open_cost, fc_rate) | Verdict |
|---|---|
| ≤ −0.5 | Mechanism works; promote architecture + promote `open_cost` to production gene set. |
| Between −0.5 and −0.2 | Partial. With N=12 already in hand, pivot to a gene-range refinement probe (narrower `open_cost` sweep around the most-responsive zone) rather than re-running the same 12 agents. |
| Within ±0.2 (same as cohort-O / cohort-O2) | Architectural change is also insufficient. Closes BOTH this plan AND selective-open-shaping. |

A secondary correlation worth measuring: `ρ(fill_prob_loss_weight,
fc_rate)` within the cohort. If positive, well-trained
fill_prob → better selectivity (confirmatory). If zero, the
actor is ignoring the new input dim.

### Result (2026-04-26) — mechanism dead, plan closes

Cohort-F (`e7077b2b-…`, 12 agents, ppo_time_lstm_v1, 18 eps)
ran in ~70 minutes (not the budgeted ~3 hours; the 12/18
estimate was conservative). All 12 agents completed.

**Headline correlations (last-8-eps means, N=12):**

| Spearman ρ(`open_cost`, X) | cohort-O | cohort-O2 | **cohort-F** |
|---|---|---|---|
| `oc_shaped` | −0.976 | −0.943 | **−0.951** — gradient delivered, third probe in a row |
| `fc_rate` | +0.055 | +0.314 | **−0.112** — same flat band, wrong sign band as cohort-O |
| `pairs_opened` | +0.139 | +0.029 | **−0.448** — new datum: high-gene agents DO open less |
| `arbs_closed_n` | −0.345 | −0.600 | **−0.511** — close_signal usage drops, same direction as O2 |
| `total_reward` | −0.758 | −0.771 | **−0.776** — high-gene agents pay, get nothing back |

**fc_rate band:** 75.5–79.5 % across all 12 agents (cohort-O
74–77 %, cohort-O2 75.8–78.1 %). Glued.

**Verdict per Hard_constraints §13 decision matrix:** within
±0.2 → architectural change is also insufficient. Closes BOTH
this plan AND selective-open-shaping with no further follow-on.

### What did move — the volume-vs-selectivity asymmetry

Cohort-F is the FIRST probe where `pairs_opened` correlates
meaningfully with `open_cost` (ρ = −0.448 vs +0.139 / +0.029
in O / O2). The agent table shows the response cleanly:

| open_cost | pairs_opened | fc_rate | matur_n | naked_n |
|---|---|---|---|---|
| 0.10 | 688 | 75.5 % | 63 | 85 |
| 0.19 | 597 | 77.4 % | 57 | 62 |
| 0.39 | 620 | 76.5 % | 61 | 66 |
| 0.53 | 679 | 76.3 % | 66 | 79 |
| 0.83 | 681 | 75.9 % | 65 | 78 |
| 0.95 | 542 | 76.4 % | 51 | 63 |

Volume drops ~22 % from gene 0.1 to gene 0.95 (688 → 542). But
the COMPOSITION (matur:close:naked:fc ratio) is unchanged. The
policy responded by **globally shrinking volume**, not by
discriminating per-runner.

This is exactly the failure mode cohort-O Session 03 predicted
("the optimisation surface only allows the policy to shift the
GLOBAL signal-firing rate, not to be SELECTIVE per-runner").
The new architectural pathway gave the actor a per-runner input
dimension; PPO used that capacity to reduce the global mean of
the signal head's output, not to vary signal output BETWEEN
runners within a tick.

**This is the load-bearing finding of the plan.** It localises
the missing mechanism: the actor CAN shrink volume globally; it
CANNOT (or does not learn to) condition signal magnitude on
per-runner state at the granularity of "this runner will
mature, that one will force-close." Three independent probes,
two architectures (O+O2 pre-fix, F post-fix), and the same flat
fc_rate band is the answer.

### Secondary: ρ(fill_prob_loss_weight, fc_rate) = +0.469

Wrong direction. Two reads (don't need to fully resolve here,
but this is the signal that points to the next investigation):

1. **The actor IS using the new input but conditioning the
   wrong way.** fill_prob's BCE target is "this PASSIVE will
   fill" — it forecasts whether a paired passive will match
   before race-off. The operator's intent for selectivity is
   "this aggressive open will MATURE (= passive fills AND we
   don't have to force-close)." Different events. Maturation
   ⊊ fill (you can fill and still not mature if force-close
   beats the passive's match). An actor conditioning signal on
   "fill_prob is high" might be choosing exactly the runners
   that are easiest to passive-match but still risky to
   maintain to settle.
2. **Or: 18 eps isn't long enough for selectivity to emerge.**
   Unlikely — cohort-O's 18-ep tail showed no within-agent
   trend, and within-cohort cross-section here is similarly
   flat. If selectivity were "just slow to emerge", we'd
   expect at least some heterogeneity in `fc_rate` ordering
   matching `open_cost` ordering. We don't.

(1) is more likely and motivates a successor investigation: is
the policy CAPABLE of per-runner discrimination at the scale
the data demands, regardless of which auxiliary head feeds
actor? See the successor session prompt under
`session_prompts/03_volume_vs_selectivity_followup.md`.

### Other observations

- ρ(open_cost, value_loss) = +0.608 — high-gene agents have
  noisier value head. Same direction as cohort-O (+0.491) but
  larger. Net pressure on the value head increases with the
  gene; consistent with the value head having no observation-
  space pathway to predict per-open shaped pressure.
- ρ(open_cost, entropy) = −0.413 — entropy contracts under
  pressure, same as cohort-O (−0.479). The controller (alpha
  saturated at upper clamp 0.1 on most agents) couldn't push
  entropy to its 150 target — pre-existing observation from
  the f46961c9 inspection earlier in the session.
- raw_pnl: ρ = −0.357 — high-gene agents lose more cash too,
  not just shaped reward. Driven by reduced volume × similar
  per-bet expectation, not by worse per-bet selection.

### What's broken in the JSONL output

`approx_kl`, `fill_prob_loss_weight_active`,
`matured_arb_bonus_weight_active` are not in
`logs/training/episodes.jsonl` rows for this run. The trainer
writes the gene VALUES into `models.hyperparameters` but not
the per-episode "active" snapshots. KL diagnostics aren't
logged at all — we couldn't sanity-check whether the
Session-02 KL fix held during this run. Worth a separate
ticket; doesn't change the verdict here (cohort-O / cohort-O2
results converge to the same place under presumably similar
KL behaviour).

### Plan status: `complete`

- The architectural change (fill_prob → actor_head) STAYS in
  the codebase. It is not gene-gated; there is no revert. The
  probe ruled out one more candidate cause of the selectivity
  gap; that's a useful negative result on its own.
- selective-open-shaping plan is updated with a closing note
  pointing here.
- Successor session prompt drafted at
  `session_prompts/03_volume_vs_selectivity_followup.md` to
  open the question of *why the actor can shrink volume
  globally but not discriminate per-runner*. Operator's call
  whether to start a new plan from it.
