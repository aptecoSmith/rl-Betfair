# Lessons Learnt — Scalping Active Management

Append-only. Surprising findings, wrong assumptions, near-misses.
Most recent at the top.

---

## Session 01 findings (2026-04-16)

- **Re-quote must run OUTSIDE the main per-slot loop.** First
  cut placed the re-quote check at the bottom of the main
  placement branch. But the main loop has several `continue`
  branches (no bet signal, below-min stake, below
  min_seconds_before_off) that skip the re-quote too. The agent
  needs to re-quote even on ticks where it isn't also placing a
  new aggressive bet, so the re-quote is now a dedicated second
  pass over `range(max_runners)` after the main loop finishes.
  Kept the per-slot bookkeeping (`action_debug[sid]`) but made
  the pass stateless w.r.t. the main loop's per-slot decisions.

- **Paired `PassiveOrderBook.place` bypasses the junk filter —
  the re-quote path must re-add it.** The auto-paired path
  intentionally skips the LTP-relative junk check because a
  large-tick offset can legitimately sit outside ±max_dev from
  the fill-time LTP (see the docstring inside
  `PassiveOrderBook.place`). But an active re-quote is posting
  relative to the CURRENT LTP — sitting outside the window IS
  the stale-parked-order risk. Added an explicit junk check in
  `_attempt_requote` that cancels the existing passive and sets
  `requote_reason="junk_band"` without placing a new leg. This
  leaves the aggressive leg naked by design; hard_constraints
  §5 still holds (no new naked exposure created by the
  re-quote; the existing exposure simply reverts to naked).

- **Synthetic-market tests need queue_ahead > 0 to prevent
  auto-fill.** Every paired passive placed via `_maybe_place_paired`
  uses an explicit price that doesn't match the lone ladder
  level in `_make_runner_snap`, so `queue_ahead_at_placement=0`.
  Combined with `total_matched` being constant across synthetic
  ticks, the fill threshold (`queue_ahead + already_filled`) is
  0, and `traded_volume_since_placement >= 0` is trivially true —
  every paired passive fills on the NEXT `on_tick`, defeating
  any test that tries to mutate it from a later step. The
  Session 01 re-quote tests patch
  `queue_ahead_at_placement = 1e12` on the resting order after
  initial placement to fence against this. Anyone writing a
  multi-step scalping test should do the same or craft
  synthetic ticks with realistic total_matched deltas.

- **Checkpoint migration is narrow and opt-in.** The cleanest
  shape migration for pre-Session-01 scalping state dicts is a
  standalone helper (`migrate_scalping_action_head`) that
  widens the actor-head final linear layer + `action_log_std`.
  Bumping `ACTION_SCHEMA_VERSION` to 3 invalidates strict
  validation as before, so any warm-start code path that wants
  to reuse old weights has to explicitly call the migration
  helper. Leaving it opt-in keeps the default "invalidate and
  retrain from scratch" behaviour that sessions 28, 29, and 30
  already established.

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
