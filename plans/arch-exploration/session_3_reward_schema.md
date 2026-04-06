# Session 3 — Expand reward hyperparameter schema

## Before you start — read these

- `plans/arch-exploration/purpose.md`
- `plans/arch-exploration/master_todo.md` (you are on Session 3)
- `plans/arch-exploration/testing.md` — **CPU-only, fast feedback.**
- `plans/arch-exploration/progress.md` — confirm Sessions 1 & 2 are
  done. This session depends on Session 1's reward-overrides plumbing.
- `plans/arch-exploration/lessons_learnt.md`
- `plans/arch-exploration/ui_additions.md`
- Repo root `CLAUDE.md` — the sections on reward symmetry and raw vs
  shaped bucketing are critical. Every new gene in this session must
  keep the zero-mean property.

## Goal

Promote the following reward shaping parameters from hardcoded
`config.yaml` values to mutable genes:

- `early_pick_bonus_min`
- `early_pick_bonus_max`
- `early_pick_min_seconds`
- `terminal_bonus_weight` (currently locked at 1.0 via the line
  `terminal_bonus = day_pnl / self.starting_budget`; make it
  `terminal_bonus_weight * day_pnl / self.starting_budget`)

## Scope

**In scope:**
- Add the four genes to `config.yaml` search_ranges.
- Thread them into `BetfairEnv` via the `reward_overrides` kwarg added
  in Session 1. `PPOTrainer` should already be extracting
  `reward_*`-prefixed keys from `hyperparams` — extend that extractor
  if needed so the new keys flow through.
- Add server-side validation/repair: after sampling or mutation, if
  `early_pick_bonus_max < early_pick_bonus_min`, swap them (repair
  step) rather than rejecting the genome. Document the choice.
- The `terminal_bonus_weight` multiplier must be applied inside
  `_settle_current_race` exactly where the existing terminal bonus is
  computed, and the result continues to land in `_cum_raw_reward` (it
  is real money, not shaping).

**Out of scope:**
- No new shaping *formulas*. We are widening existing knobs, not
  inventing new reward terms. Drawdown / hold-cost / etc. land in
  Session 7 after a design pass.
- No architecture or PPO changes.

## Ranges to use

| Gene | Type | Range | Default | Where it's read |
|---|---|---|---|---|
| `early_pick_bonus_min` | float | [1.0, 1.3] | 1.2 | `_compute_early_pick_bonus` |
| `early_pick_bonus_max` | float | [1.1, 1.8] | 1.5 | `_compute_early_pick_bonus` |
| `early_pick_min_seconds` | int | [120, 900] | 300 | `_compute_early_pick_bonus` |
| `terminal_bonus_weight` | float | [0.5, 3.0] | 1.0 | `_settle_current_race` terminal block |

## Zero-mean / symmetry invariant

`early_pick_bonus` is already symmetric: the multiplier is applied to
`bet.pnl` (winners and losers alike), so raising the multiplier makes
good bets better AND bad bets worse. Promoting min/max/seconds to
genes does not break symmetry. **Verify this with a test** (see
below).

`terminal_bonus_weight` scales `day_pnl / starting_budget`. `day_pnl`
is real cash P&L and is raw (not shaped). Scaling a raw term up or
down does not break the zero-mean invariant because raw rewards are
not expected to be zero-mean — they are expected to reflect actual
money. The gene simply controls how much the agent cares about
end-of-day vs per-race settlement.

## Tests to add

Create `tests/arch_exploration/test_reward_schema.py`:

1. **All four genes sampled, in range.** Same shape as Session 2.

2. **Env picks up overrides.** Construct `BetfairEnv` with extreme
   overrides (e.g. `early_pick_bonus_min=1.0`, `max=1.8`,
   `early_pick_min_seconds=120`, `terminal_bonus_weight=3.0`) and
   assert the corresponding env attributes have changed. Extends the
   Session 1 plumbing test.

3. **Max < min is repaired, not crashed.** Deliberately construct
   overrides with `min=1.5, max=1.1`. Assert the env or upstream
   repair step corrects the ordering (swap or clamp — document
   which) and never crashes.

4. **Symmetry is preserved.** Compute `early_pick_bonus` for a
   synthetic winning bet and a synthetic losing bet with identical
   magnitude and identical placement time, with `min=1.0, max=1.8`.
   Assert `winner_bonus + loser_bonus == 0` within floating-point
   tolerance. This is the zero-mean check.

5. **Terminal bonus scaling is raw, not shaped.** Run a synthetic day
   with `terminal_bonus_weight=2.0`, settle it, and assert that
   the terminal contribution lands in `info["raw_pnl_reward"]`, not
   `info["shaped_bonus"]`. Also assert `raw + shaped ≈ total_reward`
   still holds.

6. **Mutation stays in range + repaired.** Seed RNG, sample, mutate
   200 times, assert ranges and `max >= min` invariant hold after
   each mutation.

## Session exit criteria

- All six tests pass.
- Full test suite (non-GPU, non-slow) still passes.
- `progress.md` Session 3 entry.
- `lessons_learnt.md` updated if needed.
- `ui_additions.md` Session 3 items already listed; verify the
  min-max validator requirement is noted there (it is).
- Commit.

## Do not

- Do not make the bonus asymmetric. Absolutely do not add "+ε per bet
  placed" or any variant. That is the exact bug `CLAUDE.md` warns
  about and it was the thing that made training converge on "bet
  more".
- Do not move `commission` into the genetic schema. Keep it
  config-wide.
