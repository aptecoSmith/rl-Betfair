# Lessons Learnt — Scalping Active Management

Append-only. Surprising findings, wrong assumptions, near-misses.
Most recent at the top.

---

## Seed observations (2026-04-16) — pre-work

From the Gen 1 training run analysis:

- **"Arbs naked" is a terminology trap.** In this codebase it
  does not mean "the agent deliberately placed an unhedged
  bet." It means "the agent placed a pair, but the passive
  counter-order never filled before race-off." The aggressive
  leg still settled — directionally, by accident. 85.5 % of
  `ef453cd9`'s pair attempts ended this way. That's not
  reckless behaviour; it's timing-out scalps. The fix is to
  give the agent active management (re-quote) and fill-probability
  awareness, not to punish naked exposure harder.

- **The per-runner `arb_spread` action already exists.** It is
  the 5th per-runner action dimension, indexed by `slot_idx`.
  So the network can in principle condition `arb_spread` on
  the runner's market state at the current tick. In practice
  the gradient reaching that output is weak: it only flows
  through the long credit chain `passive fills → locked_pnl →
  reward`. Adding a fill-probability auxiliary head gives this
  output a direct, supervised training signal on every fill or
  non-fill, which should make arb-spread choices much sharper
  much faster.

- **Continuous action heads initialise near the centre of
  their range.** With arb_raw defaulting to N(0, σ), arb_frac
  starts around 0.5 — i.e. 8 ticks (mid-range). The agents
  explored a little but mostly stayed near 8. That's why fill
  rates cluster similar across agents despite `arb_spread_scale`
  varying 0.5–1.5. Stronger signal to that head should break
  the herd.

- **A "properly sized" pair is not the same as a "completed"
  pair.** The scalping-asymmetric-hedging fix (commit
  `c218bfb`) ensures completed pairs are properly sized. But
  most pair attempts don't complete — so the sizing fix only
  applies to 14.5 % of attempts. The rest become directional
  at the aggressive leg's original stake. Active management is
  the lever to move that 14.5 % up.
