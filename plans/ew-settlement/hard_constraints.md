# Hard Constraints — EW Settlement

Non-negotiables. Violation of any of these is rejected at review.

## Settlement correctness

1. **Win-market settlement must not regress.** The fix is additive:
   EW logic activates only when the race has `market_type ==
   "EACH_WAY"` AND valid `each_way_divisor`. All existing Win-market
   tests must pass unchanged.

2. **Place odds formula is `(W - 1) / D + 1`.** The fraction
   applies to the *profit* portion of the odds, not the full decimal
   odds. `W / D` is wrong. This is the standard UK each-way
   convention and matches Betfair's documented EW terms.

3. **Winner collects both legs.** A winning selection in an EW race
   pays out on the win leg at full win odds AND on the place leg at
   place-fraction odds. Code that only pays one leg for a winner is
   rejected.

4. **Placed-only runner collects place leg, loses win leg.** The
   net P&L for a placed-but-not-winning selection is:
   `-(S/2) + (S/2) x ((W-1)/D) x (1 - commission)`.

5. **Stake is halved per leg, not doubled overall.** The
   BetManager's `budget` deduction for an EW bet at requested
   stake S must be S (same as today). Internally, S is split into
   two S/2 legs. The agent is not surprised by a 2x budget hit.

## Reward integrity

6. **Raw + shaped ~ total invariant still holds.** EW settlement
   changes the *raw* component only. Shaped components (spread cost,
   early pick, etc.) are computed on the original stake, not the
   split legs. This keeps shaping zero-mean for random policies.

7. **`info["day_pnl"]` remains authoritative.** EW settlement
   updates day_pnl correctly. No separate EW P&L accumulator.

## Scope

8. **No new observation features in this plan.** The agent already
   has `market_type_each_way`, `each_way_divisor`,
   `place_odds_fraction`, and `number_of_each_way_places`. These
   are sufficient. Adding more EW features is a separate plan.

9. **No action-space changes.** The agent's action still emits one
   signal + stake per runner. The EW leg split is internal to
   BetManager, not visible to the policy.

10. **No scope creep.** This plan fixes settlement and reward only.
    Feature engineering, action-space redesign, or separate
    win/place market trading are future work.

## Documentation

11. **Every session updates `progress.md`.**
12. **Every surprising finding goes in `lessons_learnt.md`.**
13. **Incorrect comments are corrected, not left with "TODO" notes.**
