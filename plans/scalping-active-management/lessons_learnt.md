# Lessons Learnt — Scalping Active Management

Append-only. Surprising findings, wrong assumptions, near-misses.
Most recent at the top.

---

## Session 03 findings (2026-04-16)

- **Clamping log-var at forward-pass boundary beats "let the NLL
  handle it".** Initial sketch had the risk head return raw log-var
  and the NLL was responsible for keeping `exp(log_var)` finite
  (either via a tight clamp at loss-compute time or via `clamp` inside
  `_compute_risk_nll`). Switched to clamping inside each architecture's
  `forward` so `PolicyOutput` exposes a log-var that's *already* safe.
  Three independent benefits fell out: (a) parquet values can't store
  a silently-NaN stddev because `exp(0.5 * log_var)` is well-defined;
  (b) the NLL helper stays minimal — no defensive clamping duplicated
  across modules; (c) `test_log_var_clamped_in_forward` — the one that
  forces the head bias to ±100 and asserts the output stays in band —
  is a direct test of the single clamp site. If the clamp moved, the
  test would still catch it, but only because the UI and parquet paths
  depend on the same tensor. Cheap guarantee at one site.

- **Writing the realised locked_pnl in the backfill meant inlining
  `get_paired_positions` math, not calling it.** The backfill runs on
  `env.all_settled_bets` (episode-wide bet history — per CLAUDE.md
  "realised_pnl is last-race-only"), but `get_paired_positions` is a
  method on the per-race `BetManager` and each race gets a fresh
  instance which is gone by backfill time. Options were: (a) index
  `all_settled_bets` by market_id and reconstruct a minimal BetManager
  per market; (b) inline the pair classification + locked_pnl math
  directly in the trainer. Picked (b): it's 15 lines, the commission
  stays hard-coded at 0.05 to match the reference helper, and the math
  is already covered by `TestPairedPositions` tests in
  `test_forced_arbitrage.py`. Option (a) would have been a second
  abstraction layer to maintain and there's no caller that needs it.

- **Gaussian NLL without the `log(2π)/2` constant gives a clean
  analytic target for the zero-loss test.** The "full" NLL is
  `0.5 * (log_var + residual^2 / var + log(2π))`. The `log(2π)/2`
  additive constant cancels in every gradient — and adding it would
  mean the "perfect prediction at clamp-min log_var" test has to
  assert against `0.5 * (-8 + log(2π))` ≈ -3.0810, an opaque number.
  Omitting the constant makes the target `0.5 * -8.0 = -4.0` and
  every other test's gradient behaviour is unchanged. Pick the form
  whose test assertions are the easiest to reason about; the
  optimiser doesn't care either way.

- **`torch.exp(-log_vars)` is cleaner than `1.0 / torch.exp(log_vars)`.**
  First cut computed `var = log_vars.exp()` and then `(resid ** 2 /
  var)`. Switched to `inv_var = torch.exp(-log_vars)` and `(resid **
  2) * inv_var`. Two reasons: (a) one `exp` call instead of one `exp`
  + one division, marginal but it's in the hot loop; (b) when
  `log_var = RISK_LOG_VAR_MAX = 4.0`, `exp(log_var)` ≈ 54.6 and the
  division is a regular float op — not a near-zero-denominator
  gotcha, but `exp(-4.0)` ≈ 0.018 is the obviously-small multiplier
  that matches the "wide variance down-weights the residual" mental
  model for the reader. Tiny ergonomic win on a hot line.

- **Inheritance for paired passives: fold the risk fields into the
  existing Session-02 lookup.** The naive implementation added a
  second `pair_id` scan for risk — walk `bm.bets` twice on every
  passive fill. Instead, widened the Session-02 lookup to fetch all
  three fields in the single pass: one loop, one check, three reads.
  Also caught a would-be bug: if the aggressive carries risk
  predictions but no fill-prob (impossible in practice given the
  capture site, but possible in unit-test setups where the two are
  stamped independently), the scan still inherits whatever's set.
  Single-scan means the three fields can't desync on the passive
  leg.

- **Schema-check test needed updating — lesson for future additive
  columns.** `test_parquet_schema_correct` in
  `tests/test_model_store.py` hard-codes the expected column set. Any
  new nullable column trips it. The test is intentionally strict
  (catches *accidental* schema drift), so rather than loosening it to
  "at least these columns", session adds simply update the expected
  set. Mental model: adds are cheap, the test is an audit trail of
  what shipped when. Same pattern as the Session-01 schema-version
  tests that had to be loosened from `== 5` to `>= 5` — but here we
  *want* strict equality because silent column drift would poison the
  UI.

---

## Session 02 findings (2026-04-16)

- **BCE-with-probabilities beat BCE-with-logits for this shape of
  problem.** The spec required `PolicyOutput.fill_prob_per_runner` to
  default to a `0.5`-tensor (the "unsure" prior), which is the
  probability-space identity, not the logit-space identity (which would
  be `0.0`). Exposing sigmoid-applied probabilities on the output and
  using `F.binary_cross_entropy`-style math with an ε-clamp on the
  predictions gave us: (a) a human-readable `[0, 1]` number written
  directly into `Bet.fill_prob_at_placement` for UI / calibration plots
  with no extra sigmoid at the consumer; (b) clean interop with the
  "default 0.5 = unsure" contract; (c) identical gradient behaviour to
  the logit variant for the valid `[ε, 1-ε]` range. The marginal
  numerical-stability edge of `BCEWithLogits` at exact 0 / 1 predictions
  didn't materialise as a problem — the ε-clamp is enough and the
  ergonomics won.

- **Rollout label storage: NaN-as-mask beats a separate mask array.**
  Initial sketch had `Transition.fill_prob_labels` as `(max_runners,)`
  plus a parallel `Transition.fill_prob_mask` boolean array. Collapsed
  both into a single float32 array with `NaN` = "no label". Pros:
  half the memory, no chance of the two arrays drifting out of sync, and
  `~torch.isnan(labels)` is the one-liner mask inside the BCE helper.
  Cons: `np.nan` operations need care (`torch.where` to replace NaN
  before the `log` calls, or `log(NaN)` propagates through the masked
  cells and poisons the sum). Mitigated by computing `safe_labels =
  torch.where(mask, labels, zeros)` and multiplying the per-element BCE
  by `mask.float()` before the sum — so even if the ε-clamp somehow
  let a NaN through, the mask zero'd it out.

- **`create_policy(name=..., ...)` vs `architecture_name=...`.** First
  draft of the test helpers called `create_policy(architecture_name=)`
  — the rollout/evaluator layers name it `architecture_name` as a field
  on other dataclasses, so the naming was plausible but wrong. The
  registry function takes `name` positionally. Cost 3 test failures
  before I re-grepped the signature. Lesson: for helpers I've never
  personally called before, `grep 'def <name>\\('` the signature
  before writing the test, not after.

- **`test_gradients_flow_through_actor` was over-broad.** It asserted
  every non-critic / non-log-std parameter receives gradient from a
  synthetic `action_mean.sum()` loss. The new `fill_prob_head` is a
  sibling aux head — by design it gets zero gradient from that loss
  (hard_constraints §8: shares the backbone, not the actor head).
  Narrowed the filter to skip `fill_prob_head.*` using the same pattern
  as the existing `critic` / `action_log_std` exceptions. Future aux
  heads (risk head in §03) will hit the same skip list.

- **`.detach().cpu().numpy().reshape(-1)` is the safe rollout-capture
  idiom.** `out.fill_prob_per_runner` is a `(1, max_runners)` tensor on
  the training device. Flattening to a 1-D numpy array on CPU before
  the `for sid, entry in action_debug.items()` loop means the
  per-tick stamp is a plain float read — no accidental device transfer
  per stamp, no lingering autograd graph, no risk of a stale reference
  into the training device's scratch buffer. Three independent gotchas
  avoided by the single idiom.

- **Migration helper vs `strict=False`.** Picked the explicit
  `migrate_fill_prob_head(state_dict, fresh_policy)` helper over
  loosening `load_state_dict(strict=False)` for the session-02 keys.
  `strict=False` would have silently swallowed ANY missing-key error —
  including future migrations that forgot to run. The helper injects
  fresh weights for exactly the `fill_prob_head.*` keys and lets
  strict-load catch everything else. Added cost: one more helper in
  `policy_network.py`. Added value: audit trail stays intact.

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
