# Session 12 — Hold-cost reward term (DESIGN PASS FIRST)

## Before you start — read these

- `../purpose.md`
- `../hard_constraints.md` — **constraint 1 (zero-mean shaping) is
  the single hardest constraint in this session.** Read it twice.
- `../master_todo.md` (Session 12)
- `../progress.md` — confirm Session 11 has landed and its findings
  are logged
- `../lessons_learnt.md`
- `../design_decisions.md`
- `../../arch-exploration/session_7_drawdown_shaping.md` — the
  design-pass template for this kind of work. Read the whole file.
  The Option D formulation is the template you should mentally
  pattern-match against.
- `../../arch-exploration/lessons_learnt.md` — Session 7 entries
  document the false starts and why they were wrong.
- `../ui_additions.md`
- `../initial_testing.md`
- repo-root `CLAUDE.md`

## Goal

Add a shaped reward term that penalises carrying open exposure for
long periods without closing it — encouraging the agent to manage
liabilities actively — **without** biasing random policies toward
under-betting.

This is the second "symmetric shaped term" session. Session 7 did
drawdown; this does hold-cost. They are different failure modes:
drawdown is about how deep the P&L goes; hold-cost is about how
long positions stay open.

## Why this session starts with a design pass

The obvious formulation is strictly non-positive:

```
hold_cost = − ε × Σ_open_bets (liability × ticks_open)
```

A random-betting agent opens positions and holds them for some
duration by construction, which means this term accumulates
negative reward for every agent regardless of skill. The policy
then learns "the best strategy is to not bet" — which is exactly
the asymmetric-shaping bug we burned three sessions fixing in the
previous phase.

We need a closed-form, zero-mean-in-expectation formulation. The
Option D template from Session 7 used "signed position inside the
running range" to achieve this for drawdown. For hold-cost, the
analogous idea might be "signed position relative to the running
mean hold duration" — penalise longer-than-own-average holds, and
reward shorter-than-own-average holds. But the design pass is your
job, not mine.

## Scope

### Phase A — Design pass (no code)

Write the design into this file below the `---` line, covering:

1. **Chosen formulation.** The closed-form expression for the
   shaped term, per race or per tick as appropriate.

2. **Where it lives.** Which method in `env/betfair_env.py`. Which
   accumulator it feeds (`_cum_shaped_reward`, not `_cum_raw_reward`
   — it is training signal, not cash).

3. **Zero-mean proof.** Either an algebraic proof (like the
   reflection-symmetry argument in Session 7) or a Monte Carlo
   justification (a large-N synthetic simulation with random
   policies showing mean shaped term is within 2σ of zero).
   Algebraic is preferred. If you can't prove it algebraically,
   document why and design the Monte Carlo check carefully.

4. **Worked examples.** At minimum:
   - A random policy — expected shaped term is zero.
   - A liability-churning policy (closes fast) — positive shaped
     term.
   - A liability-holding policy (holds to settlement) — negative
     shaped term.
   All three with numeric tables, not just prose.

5. **Gene, type, range.** Gene name matching the existing
   `reward_efficiency_penalty` / `reward_precision_bonus` / new
   `reward_drawdown_shaping` convention (i.e. `reward_hold_cost`).
   Env reward-config key (without the `reward_` prefix, matching
   Session 3 conventions — e.g. `hold_cost_weight`). Type, range,
   default.

6. **Interaction with existing shaping.** Specifically: does it
   overlap with `efficiency_penalty` (which already discourages
   churn) or with `reward_drawdown_shaping` (which penalises time
   near the running low)? If yes, document the overlap and why it
   is acceptable; if no, document why not.

**Commit the design pass before writing any implementation code.**
Commit message: `Session 12 design pass — hold-cost shaping`.

### Phase B — Implementation (after design pass is committed)

1. Plumb the new gene through `config.yaml` search ranges, the
   sampler, the `_REWARD_OVERRIDE_KEYS` list, the population
   manager's repair step (if the gene has a constraint), and
   `BetfairEnv.__init__`.
2. Add the term to the correct shaped accumulator in
   `_settle_current_race` or wherever the design-pass phase
   identified.
3. Wire to UI via `ui_additions.md`.

### Out of scope

- Any reward term that isn't hold-cost. Drawdown, precision, and
  efficiency stay as they are.
- A time-weighted reward formulation that would require the env
  to track per-tick hold duration if the matcher rules don't
  already expose that information. If the design pass reveals
  this dependency, **stop** and either simplify the design or
  scope a separate matcher-side session.

## Tests to add

Create `tests/next_steps/test_hold_cost_shaping.py`:

1. **Gene sampling.** Gene present, values in range.
2. **Env plumbing.** Extreme override flows through to the right
   env attribute. Analogous to Session 1 / 3 tests.
3. **Zero-mean for random policy.** N=1000 synthetic random-policy
   episodes; mean shaped contribution within 2 standard errors of
   zero. Budget ~5 seconds for this test. If it's slower, cut N
   and document the tradeoff.
4. **Liability-churning policy gets positive shaping.** Synthetic
   agent that closes positions fast; shaped term > 0.
5. **Liability-holding policy gets negative shaping.** Synthetic
   agent that holds to settlement; shaped term < 0. Magnitude
   should be comparable to test 4.
6. **Raw + shaped invariant.** Holds across all of the above.
7. **Bucketing.** The new term lands in `_cum_shaped_reward`, not
   `_cum_raw_reward`. Assert via `info["shaped_bonus"]`.
8. **Repair at two layers.** If the gene has a constraint with any
   other gene, test that both the population manager and the env
   repair it.

All CPU, all fast except the zero-mean test.

## Manual tests

- **M1 (UI smoke)** after the UI widget is added.

## Session exit criteria

- Design pass committed in a separate commit before implementation
  starts.
- All 8 tests pass.
- `raw + shaped ≈ total_reward` invariant still holds in the
  existing test suite.
- `progress.md` Session 12 entry with the chosen formulation
  briefly summarised.
- `lessons_learnt.md` updated — particularly with any false starts
  during the design pass. Those are the valuable lessons.
- `ui_additions.md` Session 12 items added.
- `master_todo.md` Session 12 ticked.
- Commit.

## Do not

- Do not skip the design pass. A session that commits implementation
  before design is a session that will be reverted.
- Do not use an asymmetric formulation "because it's simpler". The
  zero-mean test must pass by construction, not by luck.
- Do not route the new term through `_cum_raw_reward`. It is
  shaping.
- Do not introduce a per-tick reward callback into the env if the
  env currently only settles per-race. A structural change to the
  env's reward timing is its own session, not a hidden dependency
  of this one.

---

## DESIGN PASS (to be filled in before implementation)

*(leave blank until the session begins)*
