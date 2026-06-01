# training-speedup-v2 — hard constraints

These are the rules that prevent "a basic error sailed into." Violating any
one invalidates the work, regardless of the speedup it bought.

## 1. Bit-identical validation gate (the spine)

No speedup ships, and no result from it is trusted, until it passes the
golden-trajectory harness (Step 1): same inputs (day, seed, policy weights)
→ same per-tick stream as the current env.

- **Discrete quantities match EXACTLY:** actions, stake bins, bet counts,
  pair_ids, matched/unmatched, force-close/naked/stop classifications,
  settle outcomes.
- **Continuous quantities match within a DECLARED per-quantity tolerance**
  justified as float-reordering only (e.g. a GPU reduction reorders a sum →
  max abs diff ≤ 1e-5 on values/P&L). The harness must **distinguish
  acceptable float reordering from logic divergence** — one blanket
  tolerance that's too loose hides bugs, too tight false-alarms on
  legitimate GPU reductions. Set tolerances per quantity, explicitly, and
  document the justification for each.

## 2. No silent feature drops (born from the BC-under-batched bug, 2026-06-01)

The new path must, for EACH feature below, either preserve identical
behaviour OR emit a loud, logged, operator-visible decision — **never a
silent no-op**: `bc_pretrain`, `per_transition_credit`,
`per_pair_reward_at_resolution`, `force_close_before_off_seconds`, the aux
heads (`fill_prob`, `mature_prob`, `risk`, `direction`), `monitor`-eval,
`input_norm`, and the `reward_overrides` passthrough keys. A regression test
asserts each flag's effect is present under the new path (or that its drop
is logged). The whole point of this plan's existence-via-failure is that
`--batched` dropping BC went unnoticed for two cohorts.

## 3. Profile before optimizing; re-profile per config

Optimize the **measured** hot path on the **actual** config, never an
assumed breakdown or a profile from a different config. The Phase-3 lean-obs
profile does NOT transfer to full-obs + predictors. Three of today's
mistakes (cost-model guess, eval-σ miss, BC-under-batched) share one root:
acting on a plausible assumption when the real number/doc was available.

## 4. Correctness-risky optimizations ONLY behind the gate

The prior plan correctly **refused** incremental-bet-aggregates and
extract-vectorization for small gains because "a silent aggregate drift
poisons risk gating and reward shaping with no obvious signal." Under this
plan such optimizations are permitted **only because the bit-identical gate
makes the drift non-silent.** No optimization is trusted on "it should be
equivalent" reasoning — only on a passing golden diff.

## 5. Stage isolation

Each stage (0, 1, 2, 3A, 3B, 3C, 4) is independently validated and
independently revertable. No stage depends on an unvalidated downstream
stage. If 3C (env-core rewrite) stalls, 2+3A+3B still ship.

## 6. Preserve the matcher vendorability contract

`env/exchange_matcher.py` stays dependency-light (dataclasses + typing + the
`PriceLevel` protocol) so it vendors into `ai-betfair` unchanged. A
GPU-vectorized matcher, if built, is a **separate fast-path module** selected
at construction time — the canonical single-level matcher remains the golden
reference and the vendored artifact.

## 7. The current env is golden, not the fast path

When the fast path and the current env disagree, the **current env is right
by definition** until proven otherwise. The fast path earns trust by
matching golden; golden is never adjusted to make the fast path pass.

## 8. No training-dynamics change smuggled in as a "speedup"

This plan changes *how fast* we compute, never *what* we compute. Any change
that alters the reward, the gradient, the action distribution, or the
exploration is OUT OF SCOPE here and goes through a normal experiment plan
with held-out eval — not the speedup gate. (Tuning levers — PPO epochs,
mini-batch size, train-day count — are explicitly excluded; we keep those
options open elsewhere.)
