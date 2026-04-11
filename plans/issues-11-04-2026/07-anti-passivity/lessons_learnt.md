# Lessons Learnt — Anti-Passivity

Append-only. Date each entry.

---

## 2026-04-11 — Analysis of the passivity equilibrium

When a model places zero bets, every reward component is exactly
zero: race_pnl=0, early_pick=0, precision=0, efficiency=0,
drawdown=0, spread_cost=0, terminal_bonus=0. There is literally
no gradient signal to encourage betting.

The precision bonus was explicitly redesigned (from `precision ×
bonus` to `(precision - 0.5) × bonus`) to prevent a "participation
trophy" where any betting was rewarded. That fix was correct — but
it also removed the last remaining incentive for a zero-bet model
to try betting.

The inactivity penalty fills this gap without reintroducing the
participation trophy. It's a nudge toward exploration ("try
something") not a reward for blind betting ("bet more = good").

Force-bet was rejected because picking a forced bet requires either
random selection (terrible for learning) or using the model's best
signal (which it already decided against). Both are worse than the
penalty approach.

Making the penalty a gene (range 0.0–2.0) is important: some
architectures may benefit from a stronger nudge, others may do
better with minimal pressure. Let evolution figure it out.
