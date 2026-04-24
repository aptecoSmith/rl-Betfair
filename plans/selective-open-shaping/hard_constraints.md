---
plan: selective-open-shaping
status: draft
---

# Hard constraints — selective-open-shaping

Lock rules for the implementation session. Any violation rolls back
the commit.

## §1 — Gene default must be byte-identical

`open_cost = 0.0` is the default. When unset on a per-agent gene
draw OR missing from the training plan entirely, the env's
reward-shaping branch must be **exactly equivalent** to the
pre-plan shape. No new branches executed, no float accumulator
touched when the gene is zero. The existing
`test_scalping_reward_invariant_under_default_genes`-style guard
(if present in the suite) must still pass.

Rationale: the 2026-04-19 `reward-densification` mark-to-market
mechanism set the bar — any new shaping term ships with a default
that preserves scoreboard comparability on pre-plan rows.

## §2 — Refund ONLY on favourable resolution

The cost refund fires only when a pair's settled classification is:

- `matured` (both legs filled naturally), OR
- `closed` (agent-initiated via `close_signal`, includes both
  profit-lock and loss-lock per `plans/ppo-kl-fix/` Session 02
  terminology).

It MUST NOT fire on:

- `force_closed` (env-initiated at T−N — the whole point of this
  plan is that these don't refund).
- `naked` (only one leg filled — the cost stays).
- `mid-race evicted` pairs whose aggressive leg never matched (the
  open cost shouldn't have been charged in the first place; see §4).

A pair's classification flows from `bm.bets[*].pair_id` +
`close_leg` + `force_close` flags at settle. The refund walk uses
the same grouping the `scalping_closed_pnl` / `force_closed_pnl`
attribution uses (`env/betfair_env.py:_settle_current_race`, the
`pair_bets` dict built post-2026-04-24 `da05332`).

## §3 — Cost charged at OPEN, not at aggressive-leg-placement

"Open" means: **the aggressive leg matched AND the paired passive
was posted to the book.** Bets that:

- fail to match on the aggressive leg,
- match the aggressive but fail to post the passive (budget exhaust,
  price cap, etc.),

are NOT opens for this plan's purposes. No cost, no refund. The
hook point is the successful return of `_maybe_place_paired` that
results in a `pair_id` being stamped on both legs.

Rationale: charging on aggressive-match alone would include
naked-by-design bets (the agent deliberately placing a one-sided
position). Those are a different behaviour and have their own
shaping term (`naked_penalty_weight`).

## §4 — Cost lives in the SHAPED channel only, never raw

`scalping_closed_pnl`, `naked_pnl`, `scalping_force_closed_pnl`,
`race_pnl`, and any field that feeds `raw_pnl_reward` must NOT see
the open-cost term. Raw is truthful cashflow; shaped carries
training-signal adjustments. The separation is load-bearing for
the "raw + shaped ≈ total_reward" invariant documented in
CLAUDE.md "Reward function: raw vs shaped".

The accumulator is a new scalar inside `_compute_scalping_reward_terms`
(or adjacent), added to the `shaped_bonus` return value and
exposed on `EpisodeStats` under a new `open_cost_shaped` field (or
similar) for observability.

## §5 — Interaction with close_signal_bonus is additive, not substitutive

Existing: agent-closed pairs get `+close_signal_bonus` (+£1) per
success. Under this plan: agent-closed pairs also get
`+open_cost` refund. Combined contribution on a close_signal
success: `+£1 + open_cost`. This is **by design**, not a bug —
both signals point at "close_signal is good", so compounding them
strengthens the learning signal.

Document this in the shaping code comment + episode log header so
a future reviewer reading the shaped total doesn't flag it as
double-counting.

## §6 — Zero-mean invariant under the "always mature or close" policy

Under a policy whose every opened pair resolves as matured or
agent-closed, the net shaped contribution from this plan must
sum to zero across any race (no open-cost residual). This is the
mathematical contract that prevents reward-hacking by the GA:

- Open N pairs → −N × open_cost
- All N mature or close → +N × open_cost
- Net = 0.

A unit test in `tests/test_forced_arbitrage.py` (or adjacent) must
assert this. Failing this test means the refund and charge are
not balanced in magnitude.

## §7 — Gene range bounds

- `open_cost` gene: `[0.0, 2.0]`, type `float`.
- Default in `hp_ranges`: `{min: 0.0, max: 1.0}` — narrower than
  the hard bound so the gene sweep converges faster.
- Cap at 2.0 enforced in `PPOTrainer.__init__` (or wherever gene
  clamps live) — values above saturate the silence-agent failure
  mode observed in cohort-A bottom-6.

## §8 — Settlement attribution is per-pair, not per-bet

The refund walk iterates over `pair_bets` (dict of pair_id → [agg,
close] or [agg, passive]), not over `bm.bets`. A pair with both
legs of its close settled gets ONE refund, not two. A pair with
partial-fill close gets one refund (the close_leg flag is binary).

## §9 — Don't touch force-close mechanics

The force-close code path (`_force_close_open_pairs`,
`_attempt_close` with `force_close=True`, the relaxed matcher
path in `ExchangeMatcher._match`) is off-limits. Any change to
what force-close DOES belongs in
`plans/force-close-sizing-review/`.

## §10 — Don't interfere with the running probe

At plan-implementation time, check `registry/training_plans/*.json`
for any `status: running`. If one is live, wait until it finishes
OR coordinate with the operator before the env change. The smoke
probe for this plan MUST use a separate seed (not 8301) from the
`post-kl-fix-reference` baseline.

## §11 — Observability minimum

The new per-episode JSONL row must surface:

- `open_cost_active` — the gene value the agent used this episode
  (float).
- `open_cost_shaped_pnl` — the net shaped contribution from this
  plan's mechanism (float; sum of charges minus refunds across the
  episode).
- `pairs_opened` — count of `_maybe_place_paired` successes (the
  denominator for force-close-rate analysis).

Without these, the gene sweep can't be analysed. Downstream readers
must default-tolerate absence on pre-plan rows (same pattern as
`alpha` / `log_alpha` in entropy-control-v2).

## §12 — CLAUDE.md entry required on landing

New subsection under "Reward function: raw vs shaped":
"Selective-open shaping (2026-04-NN)". Must cover:

- The open-cost / refund mechanism and its zero-mean property.
- The three classifications that DON'T refund (force, naked,
  aggressive-fail) and why.
- The gene range and default.
- Scoreboard comparability — pre-plan rows are comparable on
  `raw_pnl_reward`; `shaped_bonus` magnitudes change.

Match the writing style of existing subsections under that
heading (per-step mark-to-market, naked-loss annealing, matured-
arb bonus).

## §13 — Regression test is integration-level

The shaping correctness test must exercise the real env settle
path, not a mocked `RaceRecord` with synthetic field values. Build
a 2-race day with enough runners and ticks that at least one pair
matures, one is closed, one force-closes, and one goes naked —
then assert `open_cost_shaped_pnl` equals the expected sum of
charges and refunds across those four outcomes.

Follow the `test_real_ppo_update_feeds_per_step_mean_to_baseline`
pattern from `tests/test_ppo_trainer.py`: real components, no
forward-pass mocks, load-bearing.
