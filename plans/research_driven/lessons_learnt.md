# Lessons Learnt — Research-Driven

Append-only. One entry per surprising finding. "Surprising" includes
*successful* non-obvious decisions, not only mistakes — the bias is
to overshare here, because the cost of writing an entry is one
paragraph and the cost of repeating an avoidable mistake in three
sessions' time is real.

Cross-references to `arch-exploration/lessons_learnt.md` and
`next_steps/lessons_learnt.md` are welcome — anything load-bearing
in those files still applies here, and entries that build on them
should cite them rather than repeat them.

Format per entry:

```
## YYYY-MM-DD — Session NN — Short title

**What happened:** factual.
**Why it was surprising:** what the prior expectation was.
**What changes because of it:** code, tests, docs, or just future
behaviour. If "nothing", say so explicitly.
```

---

## 2026-04-07 — Pre-session — Crossing the spread is the *only* mode

**What happened:** While reading `research/research.txt` on money
pressure / queue position / crossing the spread, the operator
noticed the model picks lay prices as bet prices for back bets, and
asked whether the matcher needed work to *enable* crossing the
spread. Auditing `env/exchange_matcher.py` and
`env/betfair_env.py::_process_action` showed the answer is the
opposite: the matcher *only* crosses the spread, and the agent has
no other action verb. The screenshots are correct, not buggy.

**Why it was surprising:** The framing of the research material
(join queue / cross spread / cancel as three distinct decisions)
implied the simulator was modelling at least two of the three. It
turns out it's modelling only one, and that one is forced.

**What changes because of it:** Two things. (1) The whole
research-driven planning folder was reframed around adding the
*passive* regime rather than enabling the aggressive one — see
`analysis.md` §1 and `proposals.md` P3. (2) `hard_constraints.md`
gained an explicit "no `modify` action; cancel + place is the
canonical move" rule, because the temptation to add `modify` only
appears once you're already thinking about passive orders, and we
want the rule recorded before that thinking starts.

---

## 2026-04-07 — Pre-session — Phantom fills exist in `ai-betfair`

**What happened:** The operator reported that `ai-betfair` has
declared "bets on today" for trades that demonstrably had no
liquidity to match against on the real exchange. This is the live
equivalent of the optimistic-fill assumption that the simulator
itself made before `exchange_matcher.py` was tightened.

**Why it was surprising:** We had assumed the live wrapper was
subscribing to the Betfair order stream and using it as the source
of truth. It is apparently not — it's treating the policy's action
emission as the fact of the fill. This means every research-driven
improvement we ship would be wasted while the bug is open: a policy
trained against ground truth in sim, deployed into a runtime that
fabricates state.

**What changes because of it:** Three things. (1) `bugs.md` got an
R-1 entry recording the bug. (2) `design_decisions.md` recorded the
prerequisite-not-co-task rule (the fix has to ship in `ai-betfair`
before any new policy from this folder lands in production).
(3) `master_todo.md` Phase 0 makes that prerequisite explicit at
the top of the execution queue.

---

## 2026-04-07 — Pre-session — Matcher does not deplete its own already-matched volume

**What happened:** The operator inspected the 14:00 race in
`ai-betfair` and found two bets on the winner, £12.10 followed by
£17 about 12 seconds later, against a visible book that had only
£21 of liquidity at that price. Audit traced the simulator side of
the same logic to `env/exchange_matcher.py::_match`, which is
stateless across ticks: it reads `top.size` from each tick's
ladder snapshot directly, with no deduction for fills the agent
already made at the same price level. Filed as `bugs.md` R-2.

**Why it was surprising:** The matcher is the file that already
got hardened against the phantom-profit incident — junk filtering,
single-price rule, hard cap after filter. It had been treated as
"the careful one" since that work landed. R-2 shows there's a
whole different category of phantom liquidity it doesn't address:
*self*-depletion. Realising the existing hardening only protects
against *third-party* junk in the book, not against the agent's
own stacked orders, was the surprise.

**What changes because of it:**
- `bugs.md` R-2 entry with sim-fix sketch (per-(market, selection,
  side, price) accumulator on `BetManager`).
- `downstream_knockon.md` §0a documents the live-side equivalent
  in `ai-betfair` and explains why §0 and §0a are distinct
  bugs that need parallel fixes.
- `master_todo.md` Phase 0 records R-2 as an in-repo task that
  should land before P3/P4 (passive orders make the bug worse by
  stretching the depletion window).
- `design_decisions.md` records R-2 as an independent sim bug,
  not a sub-item of any P1–P5 proposal.
- Operator follow-up flagged: pull the `ai-betfair` order-stream
  log for the actual 14:00 race to determine whether the live
  observation is R-1 (phantom fill, second match never happened),
  R-2 (over-placed against stale local view), or a snapshot
  artefact (real Betfair liquidity replenished between data
  ticks). All three are possible and the fix differs.

---

## 2026-04-07 — Pre-session — Phantom-fill prerequisite was over-scoped

**What happened:** The first draft of `master_todo.md` Phase 0
gated *all* research-driven work on the `ai-betfair` phantom-fill
fix. The operator pushed back: "isn't that the opposite of what we
are trying to do? Check if those constraints really make sense."
Re-reading the constraints showed an inconsistency:
`hard_constraints.md` #8 said the gate was on **deployment** of a
new policy, which was correct, but `master_todo.md` Phase 0 said
it was on **shipping any item**, which was overreach. The
inconsistency had been written into the same folder on the same
day.

**Why it was surprising:** The framing felt safe at draft time
("don't ship into a runtime that lies"), but it conflated two very
different gates. Training-side work in this repo can proceed
without ever touching `ai-betfair`; only the *deployment hand-over*
needs to wait for the live wrapper fix. The two streams are part
of the same family of "the simulator/wrapper has been too
optimistic about fills" and they should run *in parallel*, not
sequentially. Sequencing them masquerades as caution but is
actually a bottleneck.

**What changes because of it:**
- `master_todo.md` Phase 0 rewritten to be a deployment-gate, not
  a programme-wide block. Training-side sessions may merge to
  master without waiting for the cross-repo fix.
- `hard_constraints.md` #8 reworded to be explicit about the
  training-vs-deployment split, citing the new
  `design_decisions.md` entry.
- New `design_decisions.md` entry "Phantom-fill gate is on
  deployment, not on training-side work" captures the refinement
  and credits the operator's review for catching it.
- General lesson logged here: when a constraint feels safe but
  serialises work that could parallelise, it is probably
  over-scoped. The next time this folder writes a "prerequisite"
  rule, ask "what is the *narrowest* scope at which this
  prerequisite is actually load-bearing?" before committing to
  the broader version.

---

## 2026-04-08 — Session 22 — Single-seed PPO comparison is uninformative as a phase gate

**What happened:** The P1 vs baseline comparison ran two full PPO
training runs under identical hyperparameters. One policy collapsed
to 0 bets (entropy vanished, all actions mapped to "do nothing"); the
other continued betting normally. Which one collapsed differed across
runs — in the session's actual run the P1 policy collapsed and the
baseline did not, producing a spuriously large baseline lead. There
was no signal about whether the P1 features were useful.

**Why it was surprising:** The expectation was that two policies
trained from different initialisations on the same market data would
produce at minimum a directionally informative comparison — even if
noisy. In practice, PPO collapse is itself a high-variance event.
The comparison was completely dominated by "which run happened to
avoid the local entropy-minimum" rather than by obs quality. A
single seed cannot distinguish "P1 features hurt" from "this seed's
P1 run collapsed first".

**What changes because of it:**
- `master_todo.md` Phase 1 gate result recorded as INCONCLUSIVE,
  not as evidence against P1. Proceeding to P2 is still justified
  because the features are correctly wired (gradient flows at fresh
  init; confirmed in a separate targeted run).
- Future phase gates must use the evolutionary infrastructure
  (N≥10 agents per config, tournament selection) rather than
  single-seed PPO. The evolutionary framework was built precisely
  because single-seed runs are noisy; the decision-gate comparison
  should have used it from the start.
- `integration_testing.md` updated to call out that the comparison
  run result is not a strict pass/fail and that single-seed results
  should not be relied on.

---

## 2026-04-08 — Session 22 — Gradient norm is near-zero on collapsed policies

**What happened:** `check_p1_gradient_norm` reported a gradient
norm of ~1e-10 on the P1 policy after training — technically
non-zero (so the assertion `grad_norm != 0.0` passed) but
meaningless at 8 decimal places. The policy had already collapsed
to 0 bets before the check ran.

**Why it was surprising:** The check was designed to catch
mis-wiring (P1 columns silently ignored by the network). A
collapsed policy fails the check for a completely different reason:
when the actor-critic value head always returns a near-constant
(because the LSTM hidden state is constant — no bets → same state
every tick), the gradient of value w.r.t. the input is near-zero
regardless of wiring. The distinction matters: "gradient is zero
because features are ignored" and "gradient is zero because the
policy has collapsed" look identical to the assertion.

**What changes because of it:**
- `integration_testing.md` updated: the gradient-norm check must
  be run on a **fresh policy** (before or very early in training),
  not on one that has already been trained to completion. A
  collapsed policy will pass the non-zero assertion trivially while
  hiding the wiring question entirely.
- The check in `scripts/session_22_p1d_compare.py` is run before
  training as a dry-run step (`--dry-run` flag); this is the
  intended usage. Do not interpret the post-training gradient norm
  as a wiring check.

---

New entries get appended below this line in chronological order.
Don't reformat or rewrite earlier entries — if a finding turns out
to be wrong, write a new entry recording the reversal and cite the
earlier one.

---

## 2026-04-08 — Session 23 — Spread-cost term is intentionally non-zero-mean (asymmetric)

**What happened:** P2 added `spread_cost` to the shaped reward.  Unlike all previous
shaped terms (`early_pick_bonus`, `precision_reward`, `drawdown_shaping`), this term is
strictly non-positive — it is a *cost*, not a bonus that averages to zero.

**Why it was surprising (and why this entry is mandatory):** The standing rule in
`hard_constraints.md` #1 requires new shaped terms to be zero-mean for random policies,
specifically to prevent the "participation trophy" bug where the agent learns to bet
more because betting is free.  When writing the design pass for P2, it was not obvious
whether spread cost should be an exception to this rule or should be centered somehow.

The conclusion was that the zero-mean rule applies to *bonuses* (terms that can be
positive or negative depending on behaviour quality) but NOT to *costs* (terms that are
always non-positive).  For spread cost, the asymmetry IS the defence: a random policy
that bets indiscriminately will accumulate strictly negative expected spread cost, which
discourages random betting directly.  Centering the term (e.g. by subtracting the
expected random-policy cost) would nullify this signal and teach the agent that the
spread is irrelevant.

**What changes because of it:**

- This entry is the mandatory historical record required by `session_23_p2_spread_cost.md`
  and the design pass §6.  **Do NOT "fix" spread_cost to be zero-mean** — the asymmetry
  is not an implementation oversight but a deliberate exception to the zero-mean rule.
- `test_p2_spread_cost.py :: TestRandomPolicyAsymmetry :: test_aggressive_policy_expected_spread_cost_strictly_negative`
  pins the asymmetry with an assertion and an explicit comment so any future refactor
  that accidentally zeros it will fail loudly.
- The comment above the spread_cost computation in `_settle_current_race` cites this
  session by number and says "DO NOT add an offset to make it zero-mean".
- `hard_constraints.md` #1 already notes the exception: "pure-cost terms are the
  one exception to this rule".  No change to hard_constraints.md needed, but this
  lessons_learnt entry is the cross-reference cited there.

---

## 2026-04-10 — Session 28 — Hardcoded `* 2` action dims were everywhere

**What happened:** Adding the aggression flag changed the per-slot
action count from 2 to 3 (`ACTIONS_PER_RUNNER`). The action-space
definition in `betfair_env.py` was a one-line fix, but the knock-on
was enormous: `max_runners * 2` was hardcoded in 15+ test files,
2 scripts, the population manager, and all 3 policy architecture
classes. The policy architectures also hardcoded `output_dim=2` in
their actor heads and manually assembled the flat action vector as
`[signal, stake]` with explicit index slicing (`actor_out[:, :, 0]`,
`actor_out[:, :, 1]`). Every one of these was an `IndexError` or
shape-mismatch at runtime — no static analysis caught them.

**Why it was surprising:** The expectation was that adding a new
per-slot action dimension would be localised to `betfair_env.py` and
the test file. In reality the "action dim = max_runners * 2"
assumption was baked into ~30 locations across the codebase, and the
policy network's actor head hardcoded the number of output values
per runner. The most insidious failures were in integration tests
that build action arrays with `np.zeros(14 * 2, ...)` — these
produced correctly-shaped arrays that only failed when
`_process_action` tried to read index 28+ from a 28-element vector.

**What changes because of it:**
- Introduced `ACTIONS_PER_RUNNER` constant in `betfair_env.py` and
  replaced all `* 2` references with it. Future action-space
  extensions (session 29 cancel flag) should only need to bump this
  constant + update the dispatch and actor assembly.
- Policy architectures now compute `_per_runner_action_dim =
  action_dim // max_runners` and use it for the actor head output dim
  and for generic action assembly (`parts = [actor_out[:, :, i] for i
  in range(self._per_runner_action_dim)]`). No more hardcoded
  index slicing.
- Existing tests that were written for aggressive-only dispatch now
  use `actions.force_aggressive: true` in their config fixtures.
  This is the regression backstop from constraint 2 — it keeps
  pre-P3 tests byte-identical without rewriting every action array.
- `test_betfair_env.py` has 44 action-array constructions that were
  all `np.zeros(14 * 2, ...)`. Bulk-replaced to `14 * _APR`.
  If this pattern recurs, consider a test helper that builds
  correctly-shaped action arrays from a dict of slot intentions
  (like `_build_action` in the new P3a test file).

---

## 2026-04-10 — Session 28 — `force_aggressive` as a test migration strategy

**What happened:** Rather than rewriting every existing test to set
the new aggression slot to 1.0 (aggressive), the test config fixture
was updated to include `actions.force_aggressive: true`. This made
all pre-session-28 tests behave identically to before — the
aggression flag value is ignored and all bets go through the
aggressive path. Only the new P3a-specific tests exercise the
passive dispatch.

**Why it was surprising:** It wasn't surprising per se, but it was a
non-obvious application of the `force_aggressive` config. The plan
described it as a "regression backstop for operators who want to
reproduce the pre-P3 policy". It turned out to be equally valuable
as a test migration strategy — ~40 existing tests needed zero logic
changes, only action-array size updates.

**What changes because of it:** Future action-space extensions should
consider adding a similar config bypass from the start. When session
29 adds the cancel flag, it can follow the same pattern: existing
tests use `force_aggressive` and ignore the new cancel slot; only
the new P3b tests exercise cancel dispatch. This avoids an
O(n-tests) rewrite on every action-space bump.

---

### Session 30 — Larger action spaces collapse faster in single-seed PPO

**Date:** 2026-04-11

**What happened:** The P3+P4 policy (4 dims per slot: signal, stake,
aggression, cancel) collapsed to zero bets by epoch 3, while the
baseline (same architecture, same HP, but force_aggressive=true,
effectively 2 active dims) kept betting through all 20 episodes.
Diversity was healthy in the first 2 epochs (21.4% passive fraction,
82 passive placements, 82 cancels) but evaporated once the policy
entered the zero-bet basin.

**Why it was surprising:** The assumption was that doubling the action
dimensions from 2 to 4 was a small expansion since the new dimensions
(aggression, cancel) are binary-ish flags, not continuous controls.
In practice, the Gaussian policy's initial `action_log_std` covers a
56-dim space (14 runners × 4 dims) with a shared entropy coefficient
of 0.01. The extra dimensions spread entropy across more outputs
without increasing total exploration pressure — each dimension gets
less exploration budget.

A second surprise: **zero passive fills during the entire training
run.** The queue-ahead estimator never triggered a fill on this
fixture data, which means the passive path had no positive reward
signal. The policy could learn "aggressive bets produce P&L (positive
or negative)" but "passive placements produce nothing". With no
signal, the policy learned to avoid passive orders — which then left
the cancel action with nothing to cancel, making a third of the
action space inert.

**What changes because of it:**
1. When expanding the action space, consider scaling entropy
   coefficient proportionally (e.g. 0.01 × 4/2 = 0.02) or using
   per-dimension entropy tuning.
2. Zero passive fills on this fixture data is a training-data
   limitation, not a code bug. Future training should either use
   higher-liquidity data where traded volume deltas are large enough
   to clear queue-ahead, or seed the early training with synthetic
   fills to bootstrap the passive reward signal.
3. The lesson from session 22 — single-seed PPO comparison is
   uninformative as a phase gate — is **doubly true** when the
   action space is larger. Collapse probability increases with
   action dimensionality. Evolutionary infrastructure is not just
   desirable, it is mandatory for any gate involving action-space
   changes.
