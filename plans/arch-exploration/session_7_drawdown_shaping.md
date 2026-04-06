# Session 7 — Drawdown-aware shaping (DESIGN PASS FIRST)

## Before you start — read these

- `plans/arch-exploration/purpose.md`
- `plans/arch-exploration/master_todo.md` (Session 7)
- `plans/arch-exploration/testing.md`
- `plans/arch-exploration/progress.md` — confirm Sessions 1-6 done.
- `plans/arch-exploration/lessons_learnt.md`
- `plans/arch-exploration/ui_additions.md`
- Repo root `CLAUDE.md` — the zero-mean-for-random-policies rule is
  the single hardest constraint on this session.

## Goal

Add a shaped reward term that encourages the agent to limit
intra-day drawdown, without biasing random policies toward
conservatism.

## Why this session starts with design, not code

The obvious formulation is wrong:

```
drawdown_penalty = − ε × max(peak_day_pnl − current_day_pnl, 0) / starting_budget
```

This is strictly non-positive. A random-betting agent will
accumulate drawdowns and therefore accumulate negative shaped
reward. That teaches the agent "betting less is strictly better",
which is exactly the asymmetric-shaping bug we spent a whole session
fixing in the phantom-profit investigation.

We need a zero-mean formulation. Options:

### Option A — Centred around expected-random drawdown

Precompute (or estimate online) the expected drawdown of a random
betting policy given the day's variance. Subtract it from the
observed drawdown:

```
shaped = − ε × (observed_drawdown − expected_random_drawdown) / starting_budget
```

Agents doing worse than random drawdown lose reward; agents doing
better than random gain reward. Expected value for a random agent
is exactly zero.

Complication: "expected random drawdown" depends on the day's
realised variance, which you only know in hindsight. Could compute
it in post-hoc settlement using the actual bet magnitudes and a
permutation bootstrap. Ugly but tractable.

### Option B — Paired agent reference

Compare the agent's drawdown to the drawdown a fixed "null policy"
(e.g. "bet nothing" or "bet uniformly random of matching volume")
would have had on the same day. Use the difference, sign-preserved,
as the shaped term. Zero-mean by construction.

Complication: requires running a shadow policy per episode.
Non-trivial compute overhead.

### Option C — Drop the feature, use a different risk knob

Instead of drawdown, penalise high open liability (variance-of-P&L
proxy) using a formulation that's symmetric by construction. For
example, reward `-ε × (open_liability_zscore)²` centred on a moving
average of the agent's own liability — so "higher-than-usual-for-me"
is penalised and "lower-than-usual-for-me" is rewarded. Zero-mean
within the agent's own trajectory.

Complication: may just reproduce what `efficiency_penalty` is
already doing indirectly.

### Design-pass exercise

**Do not write any code until one of these (or a fourth option you
come up with) is chosen and written up in this file, below the line
of dashes, with:**

- A worked example showing the expected shaped reward for a random
  policy is exactly zero (or arbitrarily close).
- A worked example showing a drawdown-avoiding policy gets positive
  shaped reward.
- A worked example showing a drawdown-amplifying policy gets
  negative shaped reward of similar magnitude.
- A description of where in `_settle_current_race` the term is
  computed and which accumulator (`_cum_shaped_reward` — this is
  shaping, not real money) it lands in.
- A gene name, type, and sensible range.

Paste the design here, commit it as "design pass for Session 7", and
only then start implementation.

---

## After design approval — scope

**In scope once design is approved:**
- Implement the chosen formulation in `_settle_current_race`.
- Add the gene to `config.yaml` search_ranges.
- Plumb via the reward-overrides path from Session 1.
- Tests (see below).
- Append UI work to `ui_additions.md` Session 7 items.

**Out of scope:**
- The "hold cost per open-liability-tick" idea discussed in design
  review. That has the same asymmetric pitfalls and deserves its own
  design pass — not this session.

## Tests (once code exists)

Create `tests/arch_exploration/test_drawdown_shaping.py`:

1. **Gene sampling.** Gene present, in range.

2. **Zero-mean for random policy.** Build a synthetic day where the
   agent bets uniformly at random on every race. Run the full
   settlement. Average the shaped contribution over N=1000 seeds.
   Assert the mean is within 2 standard errors of zero. This is the
   critical invariant test.

3. **Drawdown-avoiding policy gets positive shaping.** Construct a
   synthetic agent that closes positions early (lower drawdown than
   random) and assert shaped contribution > 0.

4. **Drawdown-amplifying policy gets negative shaping.** Construct
   a synthetic agent that increases bet sizing after losses and
   assert shaped contribution < 0.

5. **Raw+shaped invariant.** `raw + shaped ≈ total_reward` on a
   synthetic day with the new term active.

6. **Bucketing.** The new term lands in `_cum_shaped_reward`, NOT
   `_cum_raw_reward`. Assert via `info["shaped_bonus"]`.

All CPU, all fast except the zero-mean test which may need N=1000
synthetic episodes — budget ~5 seconds, don't let it balloon.

## Session exit criteria

- Design pass approved and pasted into this file below the dashes.
- All tests pass.
- `progress.md` Session 7 entry describing the chosen formulation.
- `lessons_learnt.md` updated with any gotchas discovered during the
  design pass. **Especially** document the false starts — they are
  useful to future-you.
- `ui_additions.md` Session 7 items present.
- Commit.

## Do not

- Do not skip the design pass. The commit message "design pass for
  Session 7" should exist before any implementation code is
  committed.
- Do not use an asymmetric formulation, even if it looks harmless.
  The zero-mean test must pass by construction, not by luck.
- Do not route the new term through `_cum_raw_reward`. It is
  shaping; it must not pretend to be real money.

---

## DESIGN PASS (to be filled in before implementation)

*(leave blank until the session begins)*
