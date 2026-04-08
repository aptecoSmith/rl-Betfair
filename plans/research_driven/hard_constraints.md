# Hard Constraints — Research-Driven

Non-negotiables for any session promoted from this folder. If a
session prompt appears to ask for something that violates one of
these, stop and push back before writing code.

These are derived from `CLAUDE.md`, `next_steps/hard_constraints.md`,
and the conclusions of `analysis.md` / `downstream_knockon.md` in
this folder. The first six are inherited from `next_steps/` because
they apply to any work that touches the matcher, reward, or obs
schema. The remainder are specific to research-driven work.

## Inherited from `next_steps/hard_constraints.md`

The following items continue to apply, full stop. Do not relitigate
any of them in a research-driven session — re-read the original file
if needed.

1. **Zero-mean shaping for random policies.** New shaped terms
   (notably the spread-cost term in P2) must integrate to ~zero
   under a random betting policy, or be paired with an offset that
   makes them so. The phantom-profit incident is the warning.
2. **Raw vs shaped bucketing.** Any new term lands in *exactly one*
   accumulator. The `raw + shaped ≈ total` invariant test stays
   green.
3. **`info["day_pnl"]` is authoritative.** New eval code reads
   `day_pnl`, never `realised_pnl`.
4. **Matcher single-price rule.** P3/P4 add a *new* code path for
   passive orders, but the existing aggressive path still matches
   at one price only. No ladder walking is added under any guise.
5. **LTP-based junk filter mandatory.** Even passive orders must
   not be allowed to rest at junk-filtered prices. A passive order
   placed outside the LTP tolerance is refused on placement, not
   silently rested.
6. **Max-price cap runs after junk filter.** Same rule as today,
   applies to both regimes.

## Research-driven additions

7. **Vendored matcher must stay simulation-only.** If P3 or P4 add
   queue-bookkeeping state to `exchange_matcher.py`, the vendored
   copy in `ai-betfair` is allowed to receive the same code, but
   the live inference path in `ai-betfair` must **not** depend on
   it. Live position-keeping comes from the Betfair order stream,
   period. The vendored matcher exists to support replay/backtest
   in `ai-betfair`, nothing more. See `downstream_knockon.md` §5.

8. **Phantom-fill bug gates *deployment*, not training-side work.**
   No research-driven session lands a new policy into **live**
   `ai-betfair` while the phantom-fill bug (`bugs.md` R-1,
   `downstream_knockon.md` §0) is open. "But the new policy works
   in sim" is not a counter-argument — the live wrapper would
   still fabricate state. **However**, training-side work in this
   repo (P1, P2, P5, etc.) may merge to master and run training
   while the cross-repo fix is in flight; the two streams run in
   parallel. The gate is on the *hand-over* of a policy for live
   trading, nothing earlier. See `design_decisions.md` 2026-04-07
   entry "Phantom-fill gate is on deployment, not on training-side
   work" for the reasoning.

9. **Cancel ships with passive orders or not at all.** P3 and P4
   are bundled. A session may not promote one without the other.
   The three-way decision (join / cross / cancel) only makes sense
   as a unit. `analysis.md` §2 and `proposals.md` P3 give the full
   reasoning.

10. **No `modify` action.** Cancel + new place is the canonical
    way to move a price. Adding a separate modify action is
    rejected as action-space bloat.

11. **Live data is the source of truth, not the simulator's
    estimator.** When the simulator's queue estimator and the live
    order stream disagree about a position, **live wins**. The
    estimator is a deliberate approximation built for training, not
    a model of physical reality. Code in `ai-betfair` that
    "corrects" live values to match the estimator is rejected.

12. **Hand-engineered features must be byte-identical sim ↔ live.**
    Any feature added in P1 must be computed by the same code path
    in `rl-betfair` and `ai-betfair`. If the cost of keeping that
    in sync is too high for a given feature, the feature is
    dropped, not approximated. See `downstream_knockon.md` §1.

13. **Schema bumps refuse old checkpoints.** P1 (obs) and P3
    (action) both bump the checkpoint schema. The loader must
    refuse the mismatched combination loudly, not silently zero-pad
    or zero-truncate. Precedent: the LSTM/transformer arch sessions
    in `arch-exploration/`.

## Scope discipline (inherited)

14. **No scope creep.** Same rule as `next_steps/`. "While I'm
    here, let me also..." → next session.

15. **No speculative work on parked items.** See `not_doing.md` —
    items there were parked for a reason. Promoting one requires a
    *new* reason, recorded as a `design_decisions.md` entry.

## Documentation discipline (inherited)

16. **Every session updates `progress.md`.**
17. **Every surprising thing goes in `lessons_learnt.md`.**
18. **Every new configurable value goes in `ui_additions.md`.**
19. **Every change to obs/action/matcher updates
    `downstream_knockon.md`** so the `ai-betfair` audit stays
    current. This is unique to research-driven work because the
    cross-repo impact is the whole reason this folder exists.
