# Session 23 — P2: spread-cost shaped reward (DESIGN PASS FIRST)

## Before you start — read these

- `../purpose.md`
- `../hard_constraints.md` — **constraint 1 (zero-mean-ish
  shaping) and constraint 2 (raw vs shaped bucketing) are the
  two that can kill this session** if mis-handled. Read both.
  Note: for this session the shaping is NOT zero-mean — it's a
  pure cost. See the "intentional asymmetry" note below.
- `../analysis.md` §1 and §2
- `../proposals.md` P2
- `../master_todo.md` Phase 1
- `../progress.md` — confirm session 22's decision gate
  resolved "continue to P2".
- `../lessons_learnt.md`
- `../../arch-exploration/session_7_drawdown_shaping.md` — the
  design-pass template. This session uses the same pattern.
- `../initial_testing.md`
- `env/betfair_env.py::_settle_current_race`
- `env/exchange_matcher.py` — where fill price and filtered top
  are computed.

## Goal

Add a shaped reward term that charges the agent a fraction of
the spread it crossed on each matched bet, so the policy learns
that aggressive trading is not free even when there is plenty
of liquidity. After this session, a policy that bets the same
selections but uses tighter entry timing should score higher
than one that over-trades across wide spreads.

**Intentional asymmetry note.** Unlike `early_pick_bonus` and
`precision_reward`, this term is deliberately **not** zero-mean.
It is a *cost* — the spread is a real friction that always
removes value, so the shaped term is strictly non-positive. This
is the one exception to the zero-mean rule in
`hard_constraints.md` #1, and it is acceptable *because random
policies paying the spread should be discouraged from random
betting*. The rule the other terms obey — "random policies
should get zero expected shaped reward" — is a defence against
the specific bug where asymmetric shaping teaches the agent to
bet without caring. For a pure-cost term the asymmetry *is* the
defence.

This exception **must** be documented in the design pass and in
`lessons_learnt.md` so a future session doesn't "fix" it back to
zero-mean.

## Why this session starts with a design pass

Two design questions are worth working through before writing
code:

1. **What counts as "the spread you crossed"?** Options:
    - `fill_price − fair_mid` where `fair_mid` is the
      volume-weighted midpoint of the filtered top-of-book.
    - `fill_price − opposite_best` (i.e. the full spread,
      half-width).
    - `fill_price − own_side_best` (the price the agent *could*
      have joined the queue at, if passive orders existed).
   The last option is the most principled but only makes sense
   after P3/P4 land — you'd need "what the passive price would
   have been" to measure "what the agent gave up by crossing".
   For this session, design around option 1 or 2 and document
   the reasoning.

2. **How does this interact with `efficiency_penalty`?** The
   existing per-bet friction term already discourages churn.
   Spread cost would double-count it unless one of the two is
   reduced. The design pass must state which and by how much.

Commit the design pass before writing any implementation code.

## Inputs — constraints to obey

1. **New term lands in `_cum_shaped_reward`, not
   `_cum_raw_reward`.** It's training signal, not cash.
2. **`raw + shaped ≈ total_reward` invariant still holds.**
   Tested.
3. **Intentional asymmetry is documented in code and in
   `lessons_learnt.md`.** A comment above the new term in
   `_settle_current_race` cites this session by number and
   explains why the zero-mean rule is deliberately violated.
4. **The matcher already knows the fill price and the filtered
   top.** Do not recompute them. Either return them from the
   matcher or stash them on the resulting `Bet`.

## Steps

### Phase A — Design pass (no code)

Write the design into this file below the `---` line, covering:

1. **Chosen formulation.** The closed-form expression for
   `spread_cost` per matched bet, with the `fair_mid` or
   `opposite_best` definition you picked and why.
2. **Where it lives.** Which method, which accumulator.
3. **Interaction with `efficiency_penalty`.** Whether it is
   reduced, removed, or left alone, and why.
4. **Worked examples.** At minimum:
    - A tight-spread race (5 %) with the agent crossing → small
      positive cost, small penalty.
    - A wide-spread race (20 %) with the agent crossing → large
      positive cost, large penalty.
    - A no-bet policy → zero cost.
    Numeric tables, not prose.
5. **Gene, type, range.** `reward_spread_cost_weight`, float,
   default and range. Follow the `reward_*` convention from
   previous sessions.
6. **Asymmetry justification.** A two-paragraph note explaining
   why this term is allowed to be strictly non-positive, so a
   future reader does not "fix" it.

**Commit the design pass before implementation.** Commit
message: `Session 23 design pass — spread-cost shaping`.

### Phase B — Implementation

1. Return `fill_price` and `fair_mid` (or whatever the design
   chose) from the matcher, or stash on the `Bet` object at
   placement time.
2. In `_settle_current_race`, compute `spread_cost` per race as
   the sum over matched bets of
   `matched_stake × (fill_price − fair_mid) / fair_mid`
   (or your chosen formulation).
3. Add to `shaped_bonus`. Expose on `info["spread_cost"]`.
4. Plumb `reward_spread_cost_weight` through config.yaml,
   sampler, population manager repair, and
   `BetfairEnv.__init__`. Follow session 12's pattern.
5. Update `ui_additions.md` for the replay UI breakdown panel.

## Tests to add

Create `tests/research_driven/test_p2_spread_cost.py`:

1. **Pure computation.** Given a fabricated `Bet` with known
   fill price and fair mid, spread_cost equals the formula.
2. **No-bet policy.** Zero spread_cost, zero contribution to
   shaped bonus.
3. **All-aggressive on tight spread.** Strictly negative shaped
   contribution, small magnitude.
4. **All-aggressive on wide spread.** Strictly negative shaped
   contribution, larger magnitude than (3).
5. **Random policy over N races.** Expected spread_cost is
   strictly negative, not zero. This test exists to *pin* the
   intentional asymmetry so a future refactor can't silently
   zero-mean it.
6. **Invariant holds.** `raw + shaped ≈ total` across all of
   the above.
7. **Bucketing.** `spread_cost` lands in `shaped_bonus`, not in
   `raw_pnl_reward`.
8. **Gene sampling and plumbing.** Analogous to session 12's
   tests.

## Manual tests

- **Open a race in the replay UI** where the agent crossed at
  least one wide spread. Confirm the new `spread_cost` line is
  visible in the shaped-reward breakdown and is non-zero.
- **Confirm sign.** On a tight-spread race, the line is small;
  on a wide-spread race, the line is larger. If they look the
  same the computation is wrong.

## Session exit criteria

- Design pass committed in a separate commit before
  implementation.
- All 8 tests pass.
- Existing tests still pass.
- `raw + shaped ≈ total_reward` invariant still holds.
- `progress.md` Session 23 entry with the chosen formulation.
- `lessons_learnt.md` entry capturing the intentional-asymmetry
  decision. **This is mandatory**, not optional, because the
  deviation from the zero-mean rule needs a loud historical
  record.
- `ui_additions.md` entry for the spread-cost breakdown line.
- `master_todo.md` Phase 1 P2 box ticked.
- Commit.

## Do not

- Do not skip the design pass. A session that commits
  implementation before design is a session that will be
  reverted.
- Do not make the term zero-mean to match the other shaping
  terms. The asymmetry is deliberate and justified.
- Do not route it through `_cum_raw_reward`. It is shaping.
- Do not leave `efficiency_penalty` untouched without stating
  why in the design pass. Either the two overlap (and one
  shrinks) or they don't (and you have proven it).
- Do not bundle a UI change into this session beyond appending
  a row to `ui_additions.md`. The actual UI work is its own
  session if it isn't trivial.

---

## DESIGN PASS (to be filled in before implementation)

*(leave blank until the session begins)*
