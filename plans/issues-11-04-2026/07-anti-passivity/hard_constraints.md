# Hard Constraints — Anti-Passivity

1. **The penalty must not force specific actions.** It penalises
   inaction, not the choice of runner/stake/side. The model retains
   full freedom over what to bet on — just not whether to bet at all.

2. **Penalty=0.0 must reproduce current behaviour exactly.** The
   gene defaults to 0.0, so existing models and configs are
   unaffected.

3. **The penalty is shaped reward, not raw.** It goes in the
   `shaped` accumulator, not `raw`. The invariant
   `raw + shaped ≈ total_reward` must hold.

4. **No observation or action schema changes.**

5. **No changes to existing shaped terms.** The inactivity penalty
   is additive alongside early_pick, precision, efficiency,
   drawdown, and spread_cost.

6. **Every session updates `progress.md`.**
