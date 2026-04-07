# Hard Constraints — Next Steps

Non-negotiables. If a session prompt appears to ask for something
that violates one of these, stop and push back before writing code.

Most of these are derived from repo-root `CLAUDE.md` and the lessons
of the phantom-profit incident that started the whole arch-exploration
phase. They are load-bearing: ignoring one is how a silent bug
teaches the agent to lose money while the scoreboard shows green.

## Reward correctness

1. **Zero-mean shaping for random policies.** Every shaped reward
   term must have zero expectation under a random betting policy.
   Option D (reflection-symmetric range position, Session 7 of
   arch-exploration) is the current template for constructing new
   closed-form shaped terms. Asymmetric "bonus per bet placed" and
   strictly-negative "penalty while holding" formulations are the
   exact bugs that killed the previous phase — do not reintroduce
   either.

2. **Raw vs shaped bucketing.** Every reward term is classified as
   either *raw* (reflects real money) or *shaped* (training signal
   only) and accumulated into the correct counter in
   `env/betfair_env.py::_settle_current_race`. The invariant
   `raw + shaped ≈ total_reward` is tested and must continue to
   hold to floating-point tolerance.

3. **`info["day_pnl"]` is the authoritative day P&L field.**
   `info["realised_pnl"]` is last-race-only and exists only for
   backward compatibility. No new code reads `realised_pnl` except
   where compat is explicitly required.

## Matching engine

4. **`ExchangeMatcher` single-price rule.** A bet matches at one
   price — the best level of the opposite-side book after
   filtering. No ladder walking. Stake exceeding that level's size
   is unmatched, not spilled into the next level.

5. **LTP-based junk filter is mandatory.** Levels more than
   `max_price_deviation_pct` from the runner's LTP are dropped
   before matching. Without LTP a runner is unpriceable and the
   bet is refused.

6. **Max-price cap runs after the junk filter.** Never gate on the
   unfiltered top-of-book. Gating on raw ladder levels is how the
   phantom-profit bug hid.

Any new feature (reward term, observation component, architecture
hook) that would require peeking at unfiltered ladder levels, or
that would need multi-level walking to be interesting, is rejected.

## Genes must be plumbed

7. **Sampled ≠ used is a bug.** Every gene in the search schema
   must have at least one test that asserts its sampled value
   actually reaches the object that consumes it, not just that it
   was sampled. A grep for a gene name must hit both a "read from
   hp" site and a "passed to downstream consumer" site.

8. **Repair genomes at two layers, not one.** If a constraint
   exists between genes (e.g. `early_pick_bonus_max >= min`), it
   is enforced both in `population_manager` *and* at the env
   construction boundary. Unit tests that bypass the population
   manager (e.g. ad-hoc `BetfairEnv(reward_overrides={...})`) must
   not produce invalid combinations.

## Testing discipline

9. **No GPU during development sessions.** Sessions are CPU-only
   for development and tests. The only exception is a dedicated
   integration-testing session that is explicitly labelled as such
   and uses `@pytest.mark.gpu` with skip-by-default. See
   `initial_testing.md` and `integration_testing.md`.

10. **Test after each feature, not at the end.** Batching tests
    until the end of a session is how the previous phase surfaced
    the "sampled ≠ used", "inverted range genome", and "shared
    config mutation" bugs — all of which would have been caught
    earlier with per-feature testing. Do not batch.

11. **No full training runs in unit tests.** Unit-test the pieces
    (`sample_hyperparams`, `BetfairEnv.__init__`, policy
    `forward()`, etc.) in isolation. If a test needs to exercise a
    real PPO loop it belongs in `integration_testing.md`, not the
    fast feedback set.

## Scope discipline

12. **No scope creep.** "While I'm here, let me also fix..." is
    rejected at code-review time. Drive-by fixes go into the next
    housekeeping sweep or get their own session. The one documented
    exception is fixing a regression *caused by the current session*
    — that stays in the session.

13. **No speculative work on parking-lot items.** Items marked
    parking-lot in `next_steps.md` are only promoted when a real
    run gives a concrete reason. Do not preemptively implement
    coverage-math upgrades, deeper LSTMs, or encoder changes.

## Documentation discipline

14. **Every session updates `progress.md`** with a factual entry:
    what shipped, what files changed, what tests were added, what
    did not ship and why.

15. **Every surprising thing goes in `lessons_learnt.md`** at the
    end of the session that surfaced it. "Surprising" includes
    successful non-obvious decisions, not only mistakes.

16. **Every new configurable value appended to `ui_additions.md`**
    in the same session that introduces it. A session is not
    complete until any new knob it adds has a UI task queued.
