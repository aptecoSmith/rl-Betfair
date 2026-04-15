# Hard Constraints — Scalping Asymmetric Hedging

Non-negotiables. Violation of any of these is rejected at review.

## Reward integrity

1. **`raw + shaped ≈ total_reward` invariant still holds.** Every
   new reward term lives inside either the raw or the shaped
   accumulator in `_settle_current_race`. If the invariant test
   fails, fix the accounting — don't silence the test.

2. **Locked_pnl contributes only the guaranteed floor.** The new
   definition is `max(0, min(win_outcome_pnl, lose_outcome_pnl))`
   per pair. An equal-stake pair contributes £0. A pair where one
   outcome is a loss contributes £0. No locked credit for lucky
   outcomes — ever.

3. **Naked-loss asymmetry preserved.** The existing rule
   (`raw = locked + min(0, naked)`) stays. Naked windfalls remain
   excluded from raw reward. Only the *definition* of `locked`
   changes.

4. **Worst-case shaping is zero-mean-ish for random policies.**
   The term rewards `Δ worst_case` on closing legs. Closing
   randomly is as likely to narrow worst-case as widen it — the
   expected value should be near zero. If the scaled version
   produces a strong positive bias for random play, the
   coefficient must be reduced or the term reshaped until it
   doesn't. See CLAUDE.md "Symmetry around random betting".

## Scope discipline

5. **No changes to `ExchangeMatcher`.** No ladder walking, no
   dropping the LTP requirement, no gating price caps on the
   unfiltered top-of-book. These three independently caused the
   phantom-profit bug — any PR touching them is rejected. See
   CLAUDE.md "Order matching: single-price, no walking".

6. **No changes to `BetManager.settle_race`'s winner/placer
   distinction.** Scalping reward lives in the environment's
   post-settlement accumulator, not in the per-bet settlement
   path.

7. **Scalping logic activates only when paired bets exist on the
   same `(race_id, selection_id)`.** Unpaired matched orders fall
   through to the existing naked-loss / naked-windfall path with
   no change in behaviour.

8. **Opening-leg stake sizing unchanged.** The close-position
   action only applies to *closing* a pre-existing open position.
   Opening legs still use the existing discrete stake head.

## Pair definition

9. **A "pair" is a back + lay (or lay + back) on the same
   `(race_id, selection_id)` within the same race.** First-fill
   FIFO: the earliest unmatched opening leg is paired with the
   next closing leg of opposite side. Excess stake on either side
   becomes a new unpaired order (naked).

10. **Worst-case floor is computed on stake-weighted prices per
    pair, not globally netted across the race.** If an agent
    places back → lay → back → lay on the same runner, that's two
    pairs, each evaluated independently. Global netting would
    hide sizing mistakes inside a favourable overall P&L.

## UI honesty

11. **Bet Explorer classification is derived from pair worst-case
    floor only.** Realised P&L never feeds the badge. A lucky
    +£130 on an equal-stake pair displays as "directional", not
    "locked".

12. **Classification categories are exhaustive and disjoint:**
    locked, neutral (exactly £0 floor), directional (negative
    floor but paired), naked (unpaired). Every matched order
    belongs to exactly one.

## Action-space change (Session 04)

13. **Close-position action sizes the hedge via `back_stake ×
    back_price / lay_price`**, clamped to the opposite side's
    available size at the post-junk-filter best price. If the
    clamp bites, the residual intended stake is *not* spilled to
    the next level (no ladder walking).

14. **If there is no open position to close, the close action is
    a no-op.** It does not open a new naked position under any
    circumstance.

15. **Close-position action is additive.** Existing open-leg
    actions continue to work unchanged. Agents trained before
    Session 04 remain loadable; the new action head initialises
    fresh.

## Documentation

16. **Every session updates `progress.md`.**
17. **Every surprising finding goes in `lessons_learnt.md`.**
18. **Reward-scale changes are called out in commit messages and
    in the session progress entry.** Previously-trained models'
    reward numbers are no longer comparable after Sessions 01 and
    02 — flag this loudly so operators don't chase ghosts.
