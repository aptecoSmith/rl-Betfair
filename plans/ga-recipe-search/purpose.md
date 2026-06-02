# GA recipe search — purpose

**Status:** scaffolded 2026-05-30; thesis revised same day after two
operator corrections (the reward was NOT matured-only; the run was
starved of data + obs width). **Not yet launched** — staged behind a
cheap-validation + smoke + single-config canary (see master_todo).

## The thesis (revised)

The campaign has spent weeks tuning mechanics (close-walk, force-close,
gates, reward shaping) while the model was **starved on two axes we
never questioned**:

1. **Data:** we have **49 days** of processed data (2026-04-06 →
   05-29). Every recipe trained on **3** (Apr 6/8/9) — 6% of it — a
   fast-probe shortcut that calcified into "the config."
2. **Observation width:** all 18 recipe scripts forced
   `--predictor-lean-obs` — the policy saw a **23-feature-per-runner**
   reduction + a few predictor scalars, never the **143-feature-per-
   runner full obs**. The direction-head sweep's "ceiling is data/
   signal, AUC ≈ 0.70" was measured on that *same 23-d reduction* — it
   does NOT bound what a recurrent policy can extract from the full
   feature set over a full race sequence.

The opportunity is real and present in THIS data: the **arb oracle
labels profitable scalps**, and **real traders extract them from the
same Betfair feed**. So the question was never "is the signal there" —
it's "is it reachable," and we made it unreachable by funnelling
everything through a lossy compression before the policy saw it, on
almost no data.

**This plan stops starving the model.** Train on the bulk of the data
at full obs width, warm-start from the oracle that proves the scalps
exist, and let PPO + GA find the way in. The GA is the *last* step,
gated behind a single-config canary that proves the full-obs + full-
data setup learns at all.

## The four changes (vs the campaign's standing config)

1. **Data: train 42 / hold out 7.** Train on Apr 6 → May 19 (42 days,
   ~3,360 races); hold out the **latest 7** (May 20,21,22,25,27,28,29
   — verified 75-91 markets each). Train strictly *before* holdout =
   no leak + deployment-realistic (learn the past, trade the future).
   Explicit day lists, never `select_days(n)` (the leak foot-gun).
2. **Obs: full, not lean.** Drop `--predictor-lean-obs` → 143-d/runner.
   Keep the predictor bundle (free extra features). Normalization is
   handled (feature_engineer log-norms volumes/sizes/depths; policy
   applies per-runner LayerNorm at input) — verified, low risk.
3. **Reward: operator's maturation shape** (env code, §R) — positive
   raw channel = naturally-matured locked P&L + agent-closes-at-a-
   profit; force-close/naked/loss-closes excluded; `open_cost` toll
   prevents the spam degeneracy.
4. **Selection: `locked_per_std`** (locked ÷ naked-σ) — NOT the default
   `total_reward` (the E7 naked-overfit trap).

## The reward-sparsity problem, sharpened by scale (the "500k" concern)

A rollout is **one day** (~5-13k transitions; verified in Round T
logs), so more days = more *episodes*, not a bigger single buffer —
credit assignment stays within a day. The real issue is **sparsity**,
per `plans/reward-densification/purpose.md` (verbatim): *"the reward
signal arrives once per race at settle, hundreds-to-thousands of steps
after the decisions that caused it … the policy gradient on quiet
steps is ≈ 0; there's nothing for PPO to optimise against for 99% of
the actions."* Maturation is ~5%; the matured-locked reward is sparse.

**Full obs sharpens this:** 143 features/runner to attribute the same
sparse signal across. So the densification mechanisms are not optional
here — they are how the signal becomes learnable at this width:

- **BC pretrain on the oracle** — the bridge from "oracle knows the
  scalps" to "policy initialised toward them." Dense supervised signal
  at init; PPO refines. MUST be ON and substantial for full obs (the
  head now has 143-d/runner to learn the oracle mapping from). This is
  the single most important sparsity mitigation.
- **Per-tick credit delivery** — the maturation reward and `open_cost`
  must land at the OPEN tick (charge) and the RESOLUTION tick (refund),
  not smeared across 13k steps by GAE (selective-open-shaping per-tick
  design, already in env). Confirm the maturation-reward mode delivers
  per-tick, not settle-only.
- **Mark-to-market shaping** (`mark_to_market_weight`) — per-tick
  position valuation; keep ON for density.

## The existential context (Round T)

Round T (2026-05-30) showed the inline `mature_prob` gate did NOT lift
mat% above ~5% — BUT that was on lean obs (23-d) and 3 training days.
This plan removes both of those handicaps. If full obs + 42 days + BC
*still* can't lift maturation, that's a far stronger negative than
Round T — but it's the honest test, and we haven't run it.

See `hard_constraints.md` (pins, data split, full-obs, sparsity, canary
gate) and `master_todo.md` (the staged validate → smoke → canary → GA
rollout).
